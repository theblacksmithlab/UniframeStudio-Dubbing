import os
import json
import subprocess
from pathlib import Path
import shutil


class VideoProcessor:
    def __init__(self, input_video_path, json_path, output_video_path, intro_outro_path, target_fps=25):
        """
        Video Processor initialization

        :param input_video_path: Path to original video file
        :param json_path: Path to segments' info JSON-file
        :param output_video_path: Output file path
        :param intro_outro_path: Path to intro/outro file
        :param target_fps: Target FPS (25 by default)
        """

        self.input_video_path = input_video_path
        self.json_path = json_path
        self.output_video_path = output_video_path
        self.intro_outro_path = intro_outro_path
        self.target_fps = target_fps

        self._gpu_available = True

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        self.temp_dir = Path(os.path.join(project_dir, "temp_processing"))
        self.temp_dir.mkdir(exist_ok=True)
        self.segments_dir = self.temp_dir / "segments"
        self.processed_segments_dir = self.temp_dir / "processed_segments"
        self.gaps_dir = self.temp_dir / "gaps"
        self.segments_dir.mkdir(exist_ok=True)
        self.processed_segments_dir.mkdir(exist_ok=True)
        self.gaps_dir.mkdir(exist_ok=True)

        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.input_fps = self._get_video_fps(input_video_path)
        self.needs_fps_conversion = abs(self.input_fps - target_fps) > 0.01
        self.converted_video_path = self.temp_dir / "converted_input.mp4"

    def _run_command(self, cmd, **kwargs):
        """Safely execute external command with log output"""
        try:
            print(f"Executing a command: {' '.join(map(str, cmd))}")
            result = subprocess.run(cmd, text=True, **kwargs)

            if result.returncode != 0:
                print(f"Error executing command. Return code: {result.returncode}")
                print(f"Standard output: {result.stdout}")
                print(f"Error output: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd)
            return result
        except Exception as e:
            print(f"Exception while executing command: {e}")
            raise

    def _check_gpu_availability(self):
        """Checks for NVIDIA GPU availability for video encoding"""
        # If we've already checked and have the result, use it
        if hasattr(self, '_gpu_available'):
            return self._gpu_available

        try:
            # Run FFmpeg to list available encoders
            cmd = ['ffmpeg', '-encoders']
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Check if h264_nvenc is among available encoders
            if 'h264_nvenc' in result.stdout:
                # Try a test encoding
                test_cmd = [
                    'ffmpeg',
                    '-f', 'lavfi',
                    '-i', 'nullsrc=s=640x480:d=1',
                    '-c:v', 'h264_nvenc',
                    '-f', 'null',
                    '-'
                ]
                test_result = subprocess.run(test_cmd, capture_output=True, text=True)

                # If the test succeeded without errors, GPU is available
                if test_result.returncode == 0:
                    self._gpu_available = True
                    return True

            # If we're here, the GPU is unavailable or unsupported
            self._gpu_available = False
            return False
        except Exception as e:
            print(f"Error checking GPU availability: {e}")
            self._gpu_available = False
            return False

    def _get_video_fps(self, video_path):
        """Get video FPS"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            fps_str = result.stdout.strip()

            if fps_str and '/' in fps_str:
                numerator, denominator = map(int, fps_str.split('/'))
                return numerator / denominator

            print(f"Warning: Could not determine video FPS, using default: {self.target_fps}")
            return self.target_fps
        except Exception as e:
            print(f"Error while getting FPS: {e}")
            return self.target_fps

    def _get_video_duration(self, video_path):
        """Retrieve video duration in seconds"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json',
                str(video_path)  # Convert to string for safety
            ]
            result = self._run_command(cmd)
            data = json.loads(result.stdout)

            if not data.get('format') or 'duration' not in data['format']:
                raise ValueError(f"Failed to get duration info: {result.stdout}")

            return float(data['format']['duration'])
        except Exception as e:
            print(f"Error while getting duration: {e}")
            # In case of error, try alternative method
            try:
                cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-i', str(video_path),
                    '-show_entries', 'format=duration',
                    '-sexagesimal',
                    '-of', 'default=noprint_wrappers=1:nokey=1'
                ]
                result = self._run_command(cmd)
                time_str = result.stdout.strip()

                # Parse time in HH:MM:SS.MS format
                if ':' in time_str:
                    parts = time_str.split(':')
                    if len(parts) == 3:
                        hours, minutes, seconds = parts
                        seconds = float(seconds)
                        return float(hours) * 3600 + float(minutes) * 60 + seconds

                # If parsing fails, try converting to float directly
                return float(time_str)
            except Exception as e2:
                print(f"Alternative method of getting duration also failed: {e2}")
                # Return assumed value
                return 0.0

    def _adjust_duration_for_fps(self, duration):
        """Adjust duration to match the target frame rate"""
        frame_duration = 1.0 / self.target_fps
        frames = round(duration / frame_duration)
        return frames * frame_duration

    def convert_to_target_fps(self):
        """Convert source video to target FPS without sound"""
        try:
            input_path = str(self.input_video_path)
            output_path = str(self.converted_video_path)

            # Принудительно проверяем FPS входного видео
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                   '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
                   input_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            fps_str = result.stdout.strip()

            if '/' in fps_str:
                try:
                    numerator, denominator = map(int, fps_str.split('/'))
                    actual_fps = numerator / denominator
                    self.input_fps = actual_fps
                    print(f"Detected input video FPS: {actual_fps}")
                    self.needs_fps_conversion = abs(actual_fps - self.target_fps) > 0.01
                except ValueError:
                    print(f"Could not parse FPS fraction: {fps_str}")
                    # Fallback to stored value

            # Проверка настроек конвертации
            if self.needs_fps_conversion:
                print(f"Convert video from {self.input_fps} FPS to {self.target_fps} FPS")
            else:
                print(f"The original video already has the required frame rate ({self.target_fps} FPS)")

            has_gpu = self._check_gpu_availability()

            if has_gpu:
                print("Using NVIDIA GPU to speed up conversion with maximum quality")
                encoder = 'h264_nvenc'
                pixel_format = 'yuv420p'
                quality_params = [
                    '-b:v', '100M',  # Very high bitrate
                    '-bufsize', '100M',  # Large buffer
                    '-rc', 'vbr',  # Changed from vbr_hq to vbr (supported mode)
                    '-rc-lookahead', '32',  # Maximum lookahead window
                    '-spatial_aq', '1',  # Spatial adaptive quantization
                    '-temporal_aq', '1',  # Temporal adaptive quantization
                    '-aq-strength', '15',  # Maximum adaptive quantization strength
                    '-nonref_p', '0',  # All P-frames are reference frames
                    # Completely removed weighted_pred and b_ref_mode
                ]
                preset = 'p7'  # Highest quality NVENC preset
                extra_params = ['-tune', 'hq']  # High quality tuning
            else:
                print("GPU not detected or not supported. Using CPU")
                encoder = 'libx264'
                pixel_format = 'yuv444p'
                quality_params = ['-crf', '0']  # Lossless quality
                preset = 'veryslow'  # Highest quality preset
                extra_params = ['-tune', 'film']  # Film tuning for better quality

            if not self.needs_fps_conversion:
                print(f"The original video already has the required frame rate ({self.target_fps} FPS)")
                cmd = [
                          'ffmpeg',
                          '-i', input_path,
                          '-an',  # Remove audio
                          '-c:v', encoder,
                      ] + quality_params + [
                          '-preset', preset,
                          '-pix_fmt', pixel_format,
                      ] + extra_params + [
                          output_path
                      ]
            else:
                print(f"Convert video from {self.input_fps} FPS to {self.target_fps} FPS")
                cmd = [
                          'ffmpeg',
                          '-i', input_path,
                          '-an',  # Remove audio
                          '-c:v', encoder,
                      ] + quality_params + [
                          '-preset', preset,
                          '-pix_fmt', pixel_format,
                      ] + extra_params + [
                          '-r', str(self.target_fps),
                          output_path
                      ]

            self._run_command(cmd)

            if not os.path.exists(output_path):
                raise FileNotFoundError(f"The converted file was not created: {output_path}")

            actual_fps = self._get_video_fps(output_path)
            print(f"Checking the FPS of the converted file: {actual_fps}")

            if abs(actual_fps - self.target_fps) > 0.1:
                print(
                    f"Warning: FPS of converted file ({actual_fps}) is different from target ({self.target_fps})")

            return output_path
        except Exception as e:
            print(f"Error converting video: {e}")
            # If the error is related to the GPU, try CPU fallback
            if has_gpu and ("NVENC" in str(e) or "GPU" in str(e) or "nvenc" in str(e)):
                print("Error using GPU. Trying conversion on CPU...")
                # Set the flag that the GPU is unavailable
                self._gpu_available = False
                # Recursively call the same function, which will now use the CPU
                return self.convert_to_target_fps()
            raise

    def extract_segments(self):
        """Extract segments and gaps from video using GPU if available"""
        segments = self.data.get('segments', [])
        video_path = self.converted_video_path

        has_gpu = self._check_gpu_availability()

        if has_gpu:
            print("Using NVIDIA GPU to speed up segment extraction with maximum quality")
            encoder = 'h264_nvenc'
            pixel_format = 'yuv420p'
            quality_params = [
                '-b:v', '100M',  # Very high bitrate
                '-bufsize', '100M',  # Large buffer
                '-rc', 'vbr',  # Changed from vbr_hq to vbr
                '-rc-lookahead', '32',  # Maximum lookahead window
                '-spatial_aq', '1',  # Spatial adaptive quantization
                '-temporal_aq', '1',  # Temporal adaptive quantization
                '-aq-strength', '15',  # Maximum adaptive quantization strength
                '-nonref_p', '0'  # All P-frames are reference frames
                # Removed weighted_pred parameter
            ]
            preset = 'p7'  # Highest quality NVENC preset
            extra_params = ['-tune', 'hq']  # High quality tuning
        else:
            print("GPU not detected or not supported. Using CPU")
            encoder = 'libx264'
            pixel_format = 'yuv444p'
            quality_params = ['-crf', '0']  # Lossless quality
            preset = 'veryslow'  # Highest quality preset
            extra_params = ['-tune', 'film']

        for i, segment in enumerate(segments):
            start = segment['start']
            end = segment['end']
            output_path = self.segments_dir / f"segment_{i:04d}.mp4"

            try:
                cmd = [
                          'ffmpeg',
                          '-i', video_path,
                          '-ss', str(start),
                          '-to', str(end),
                          '-c:v', encoder,
                      ] + quality_params + [
                          '-preset', preset,
                          '-pix_fmt', pixel_format,
                      ] + extra_params + [
                          '-an',
                          str(output_path)
                      ]

                self._run_command(cmd)

            except Exception as e:
                print(f"Error using GPU for segment {i}: {e}")
                if has_gpu:
                    print("Trying to extract the segment using CPU...")
                    fallback_cmd = [
                        'ffmpeg',
                        '-i', video_path,
                        '-ss', str(start),
                        '-to', str(end),
                        '-c:v', 'libx264',
                        '-crf', '0',
                        '-preset', 'veryslow',
                        '-pix_fmt', 'yuv444p',
                        '-tune', 'film',
                        '-an',
                        str(output_path)
                    ]
                    self._run_command(fallback_cmd)

            if i + 1 < len(segments):
                next_start = segments[i + 1]['start']
                if next_start > end:
                    gap_start = end
                    gap_end = next_start
                    gap_path = self.gaps_dir / f"gap_{i:04d}_{i + 1:04d}.mp4"

                    try:
                        if has_gpu:
                            gap_cmd = [
                                'ffmpeg',
                                '-i', video_path,
                                '-ss', str(gap_start),
                                '-to', str(gap_end),
                                '-c:v', encoder,
                                '-b:v', '50M',
                                '-rc', 'vbr',
                                '-preset', 'p5',
                                '-pix_fmt', pixel_format,
                                '-an',
                                str(gap_path)
                            ]
                        else:
                            gap_cmd = [
                                'ffmpeg',
                                '-i', video_path,
                                '-ss', str(gap_start),
                                '-to', str(gap_end),
                                '-c:v', 'libx264',
                                '-crf', '18',
                                '-preset', 'medium',
                                '-pix_fmt', 'yuv420p',
                                '-an',
                                str(gap_path)
                            ]

                        self._run_command(gap_cmd)

                    except Exception as e:
                        print(f"Error while extracting gap {i}-{i + 1}: {e}")
                        fallback_gap_cmd = [
                            'ffmpeg',
                            '-i', video_path,
                            '-ss', str(gap_start),
                            '-to', str(gap_end),
                            '-c:v', 'libx264',
                            '-crf', '18',
                            '-preset', 'medium',
                            '-an',
                            str(gap_path)
                        ]
                        self._run_command(fallback_gap_cmd)

    def process_segments(self):
        """Processing all segments with changes in their duration"""
        segments = self.data.get('segments', [])

        has_gpu = self._check_gpu_availability()

        if has_gpu:
            print("Using NVIDIA GPU to speed up conversion with maximum quality")
            encoder = 'h264_nvenc'
            pixel_format = 'yuv420p'
            quality_params = [
                '-b:v', '100M',
                '-bufsize', '100M',
                '-rc', 'vbr_hq',
                '-rc-lookahead', '32',
                '-spatial_aq', '1',
                '-temporal_aq', '1',
                '-aq-strength', '15',
                '-nonref_p', '0',
                '-weighted_pred', '1'
            ]
            preset = 'p7'
            extra_params = ['-tune', 'hq']
        else:
            print("GPU not detected or not supported. Using CPU")
            encoder = 'libx264'
            pixel_format = 'yuv444p'
            quality_params = ['-crf', '0']
            preset = 'veryslow'
            extra_params = ['-tune', 'film']

        for i, segment in enumerate(segments):
            try:
                input_path = self.segments_dir / f"segment_{i:04d}.mp4"
                output_path = self.processed_segments_dir / f"processed_segment_{i:04d}.mp4"

                if not os.path.exists(input_path):
                    print(f"Warning: Segment file not found: {input_path}")
                    continue

                original_duration = self._get_video_duration(input_path)
                target_duration = segment['tts_duration']
                adjusted_target_duration = self._adjust_duration_for_fps(target_duration)

                # Check for negative or zero duration
                if original_duration <= 0:
                    print(f"Error: Original duration of segment {i} is equal to or less than zero: {original_duration}")
                    continue

                if adjusted_target_duration <= 0:
                    print(
                        f"Error: Target duration of segment {i} is equal to or less than zero: {adjusted_target_duration}")
                    continue

                # If the original duration is approximately equal to the target, simply copy the file
                if abs(original_duration - adjusted_target_duration) < 0.04:  # Difference less than 1 frame
                    print(f"Minor change in duration. Copying file.")
                    shutil.copy(str(input_path), str(output_path))
                    actual_duration = self._get_video_duration(output_path)
                else:
                    # We use the direct method with setting the exact duration for all segments
                    print(f"Set exact duration for segment {i}...")
                    speed_factor = adjusted_target_duration / original_duration

                    try:
                        cmd = [
                                  'ffmpeg',
                                  '-i', str(input_path),
                                  '-filter:v', f'setpts={speed_factor}*PTS,fps={self.target_fps}',
                                  # Changing video segment's speed and guarantee accurate FPS
                                  '-r', str(self.target_fps),
                                  '-c:v', encoder,
                              ] + quality_params + [
                                  '-preset', preset,
                                  '-pix_fmt', pixel_format,
                              ] + extra_params + [
                                  '-an',
                                  '-t', str(adjusted_target_duration),  # Force the duration
                                  str(output_path)
                              ]

                        self._run_command(cmd)

                        if not os.path.exists(output_path):
                            raise FileNotFoundError(f"The processed file was not created: {output_path}")

                        actual_duration = self._get_video_duration(output_path)

                    except Exception as e:
                        if has_gpu:
                            print(f"Error processing segment {i} with GPU: {e}")
                            print(f"Trying to process segment {i} with CPU...")
                            # CPU fallback
                            cmd = [
                                'ffmpeg',
                                '-i', str(input_path),
                                '-filter:v', f'setpts={speed_factor}*PTS,fps={self.target_fps}',
                                '-r', str(self.target_fps),
                                '-c:v', 'libx264',
                                '-crf', '0',
                                '-preset', 'veryslow',
                                '-pix_fmt', 'yuv444p',
                                '-tune', 'film',
                                '-an',
                                '-t', str(adjusted_target_duration),
                                str(output_path)
                            ]
                            self._run_command(cmd)
                            actual_duration = self._get_video_duration(output_path)
                        else:
                            raise

                duration_diff = abs(actual_duration - adjusted_target_duration)

                print(f"Segment {i}: original duration = {original_duration:.4f} sec, "
                      f"target = {adjusted_target_duration:.4f} sec (tts = {target_duration:.4f} sec), "
                      f"actual = {actual_duration:.4f} sec, "
                      f"speed coefficient = {adjusted_target_duration / original_duration:.4f}")

                if duration_diff > 0.04:  # If the deviation is more than 1 frame
                    print(f"Warning: Deviation from target duration: {duration_diff:.4f} sec")

            except Exception as e:
                print(f"Error processing segment {i}: {e}")
                # Continue with the next segment instead of stopping completely

    def combine_final_video_reliable(self):
        """A Robust Method for Combining Videos via Intermediate Images"""
        segments = self.data.get('segments', [])
        frames_dir = self.temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        # Temporary path for output file
        temp_output = self.temp_dir / "temp_output.mp4"

        # Step 1: Get a list of all input files in the correct order
        input_files = []

        # Adding an intro
        input_files.append((str(self.intro_outro_path), "intro"))

        # Add segments and gaps
        for i in range(len(segments)):
            # Add the processed segment
            segment_path = self.processed_segments_dir / f"processed_segment_{i:04d}.mp4"
            if os.path.exists(segment_path):
                input_files.append((str(segment_path), f"segment_{i:04d}"))
            else:
                print(f"Warning: Processed segment not found: {segment_path}")

            # Check for a gap after a segment
            gap_path = self.gaps_dir / f"gap_{i:04d}_{i + 1:04d}.mp4"
            if gap_path.exists():
                input_files.append((str(gap_path), f"gap_{i:04d}_{i + 1:04d}"))

        # Adding an outro
        input_files.append((str(self.intro_outro_path), "outro"))

        print(f"Total files to merge: {len(input_files)}")

        # Step 2: Create a list file for direct frame merging
        frame_list_path = self.temp_dir / "frame_list.txt"
        frame_count = 0
        last_frame = None

        with open(frame_list_path, 'w') as frame_list:
            for idx, (file_path, file_id) in enumerate(input_files):
                print(f"Processing file {idx + 1}/{len(input_files)}: {file_id}")

                # Check the duration and fps of the file
                try:
                    # Getting an info about the file
                    file_fps = self._get_video_fps(file_path)
                    file_duration = self._get_video_duration(file_path)
                    frame_count_in_file = int(file_duration * self.target_fps)

                    print(
                        f"Duration: {file_duration:.4f} sec, FPS: {file_fps}, approximate number of frames: {frame_count_in_file}")

                    has_gpu = self._check_gpu_availability()

                    # Extract each frame from the file
                    output_pattern = frames_dir / f"{file_id}_%05d.png"

                    if has_gpu:
                        print(f"  Using GPU for frame extraction from {file_id}")
                        cmd = [
                            'ffmpeg',
                            '-i', file_path,
                            '-vf', f'fps={self.target_fps}',
                            '-q:v', '1',  # Maximum quality
                            str(output_pattern)
                        ]
                    else:
                        print(f"  Using CPU for frame extraction from {file_id}")
                        cmd = [
                            'ffmpeg',
                            '-i', file_path,
                            '-vf', f'fps={self.target_fps}',
                            '-q:v', '1',  # Maximum quality
                            str(output_pattern)
                        ]

                    self._run_command(cmd)

                    # Find all the extracted frames and add them to the list
                    extracted_frames = sorted(list(frames_dir.glob(f"{file_id}_*.png")))

                    if not extracted_frames:
                        print(f"Warning: No footage was extracted from {file_id}")
                        continue

                    print(f"Frames extracted: {len(extracted_frames)}")

                    # Add frames to the list
                    for frame_path in extracted_frames:
                        frame_list.write(f"file '{frame_path}'\n")
                        frame_list.write(f"duration {1.0 / self.target_fps}\n")
                        frame_count += 1
                        last_frame = frame_path

                except Exception as e:
                    print(f"Error processing file {file_id}: {e}")

            # Add the last frame without duration (FFmpeg requirement)
            if last_frame:
                frame_list.write(f"file '{last_frame}'\n")

        # Step 3: Assemble the video from the frames
        print(f"Assembling video from {frame_count} frames...")

        has_gpu = self._check_gpu_availability()

        if has_gpu:
            print("Using NVIDIA GPU for final video assembly with maximum quality")
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(frame_list_path),
                '-vsync', 'vfr',
                '-pix_fmt', 'yuv420p',  # Compatible format for NVENC
                '-c:v', 'h264_nvenc',
                '-b:v', '100M',  # Very high bitrate
                '-bufsize', '100M',  # Large buffer
                '-rc', 'vbr_hq',  # High quality variable bitrate
                '-rc-lookahead', '32',  # Maximum lookahead window
                '-spatial_aq', '1',  # Spatial adaptive quantization
                '-temporal_aq', '1',  # Temporal adaptive quantization
                '-aq-strength', '15',  # Maximum adaptive quantization strength
                '-nonref_p', '0',  # All P-frames are reference frames
                '-weighted_pred', '1',  # Weighted prediction for better transitions
                '-preset', 'p7',  # Highest quality NVENC preset
                '-tune', 'hq',  # High quality tuning
                '-movflags', '+faststart',
                str(temp_output)
            ]
        else:
            print("GPU not detected or not supported. Using CPU for maximum quality")
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(frame_list_path),
                '-vsync', 'vfr',
                '-pix_fmt', 'yuv444p',
                '-c:v', 'libx264',
                '-crf', '0',
                '-preset', 'veryslow',
                '-tune', 'film',
                '-movflags', '+faststart',
                str(temp_output)
            ]

        try:
            self._run_command(cmd)

            if not os.path.exists(temp_output) and has_gpu:
                print("Error using GPU. Trying CPU fallback...")
                # CPU fallback
                cpu_cmd = [
                    'ffmpeg',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', str(frame_list_path),
                    '-vsync', 'vfr',
                    '-pix_fmt', 'yuv444p',
                    '-c:v', 'libx264',
                    '-crf', '0',
                    '-preset', 'veryslow',
                    '-tune', 'film',
                    '-movflags', '+faststart',
                    str(temp_output)
                ]
                self._run_command(cpu_cmd)
        except Exception as e:
            print(f"Error during video assembly: {e}")
            if has_gpu:
                print("Trying CPU fallback for final assembly...")
                # CPU fallback
                cpu_cmd = [
                    'ffmpeg',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', str(frame_list_path),
                    '-vsync', 'vfr',
                    '-pix_fmt', 'yuv444p',
                    '-c:v', 'libx264',
                    '-crf', '0',
                    '-preset', 'veryslow',
                    '-tune', 'film',
                    '-movflags', '+faststart',
                    str(temp_output)
                ]
                self._run_command(cpu_cmd)

        if os.path.exists(temp_output):
            video_duration = self._get_video_duration(temp_output)
            print(f"Video successfully created! Duration: {video_duration:.4f} sec")

            # Copy video to output directory
            shutil.copy(temp_output, self.output_video_path)
            print(f"Result saved to {self.output_video_path}")

            # Verify file was copied correctly
            if os.path.exists(self.output_video_path):
                print(f"Verification: file successfully copied to {self.output_video_path}")
                print(f"File size: {os.path.getsize(self.output_video_path) / (1024 * 1024):.2f} MB")
            else:
                print(f"Error: file was not copied to {self.output_video_path}")
                # Try alternative copy method
                try:
                    os.system(f'cp "{temp_output}" "{self.output_video_path}"')
                    if os.path.exists(self.output_video_path):
                        print(f"File successfully copied using alternative method")
                    else:
                        print(f"Error: could not copy file even with alternative method")
                except Exception as e:
                    print(f"Error with alternative copying: {e}")

            return True
        else:
            print("Error: Failed to create final video!")
            return False

    def cleanup(self):
        """Deleting temp files"""
        try:
            shutil.rmtree(self.temp_dir)
            print(f"Temporary files have been deleted: {self.temp_dir}")
        except Exception as e:
            print(f"Error deleting temporary files: {e}")

    def process(self):
        """Executing the complete processing sequence"""
        try:
            print("1. Convert video to target FPS...")
            self.convert_to_target_fps()

            print("2. Extracting segments and gaps...")
            self.extract_segments()

            print("3. Processing segments with changing duration...")
            self.process_segments()

            print("4. Combining final video using the reliable method...")
            success = self.combine_final_video_reliable()

            if success and os.path.exists(self.output_video_path):
                print(f"Done! Result saved to {self.output_video_path}")
                self.cleanup()
            else:
                print(f"Warning: final file was not found at {self.output_video_path}")
                print("Temporary files will not be deleted for debugging purposes")

            return success
        except Exception as e:
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    current_dir = os.path.abspath(os.getcwd())
    input_dir = os.path.join(current_dir, "video_input")
    output_dir = os.path.join(current_dir, "video_output")
    resources_dir = os.path.join(current_dir, "resources")

    input_video = os.path.join(input_dir, "input.mp4")
    json_file = os.path.join(current_dir, "output", "timestamped_transcriptions", "input_timestamped_corrected_cleaned_optimized_adjusted_translated.json")
    output_video = os.path.join(output_dir, "output.mp4")
    intro_outro = os.path.join(resources_dir, "intro_outro_converted.mp4")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Work dir: {current_dir}")
    print(f"Input file dir: {input_dir}")
    print(f"Resources dir: {resources_dir}")

    if not os.path.exists(input_video):
        print(f"Error: Input video not found: {input_video}")
        return

    if not os.path.exists(json_file):
        print(f"Error: JSON-file not found: {json_file}")
        return

    if not os.path.exists(intro_outro):
        print(f"Error: Intro/outro file not found: {intro_outro}")
        print(f"Checking the contents of the resource directory:")
        try:
            for file in os.listdir(resources_dir):
                print(f"  - {file}")
        except Exception as e:
            print(f"Error reading directory: {e}")
        return

    print(f"All necessary files found:")
    print(f"  - Input video: {input_video}")
    print(f"  - JSON-file: {json_file}")
    print(f"  - Intro/outro: {intro_outro}")

    processor = VideoProcessor(
        input_video_path=input_video,
        json_path=json_file,
        output_video_path=output_video,
        intro_outro_path=intro_outro,
        target_fps=25
    )

    processor.process()


if __name__ == "__main__":
    main()