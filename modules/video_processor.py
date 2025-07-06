import os
import json
import subprocess
from pathlib import Path
import shutil
from utils.logger_config import setup_logger, get_job_logger

logger = setup_logger(name=__name__, log_file="logs/app.log")


class VideoProcessor:
    def __init__(
            self,
            job_id,
            input_video_path,
            json_path,
            output_video_path,
            target_fps=25,
    ):
        """
        Video Processor initialization

        :param input_video_path: Path to original video file
        :param json_path: Path to segments' info JSON-file
        :param output_video_path: Output file path
        :param target_fps: Target FPS (25 by default)
        """

        self.input_video_path = input_video_path
        self.json_path = json_path
        self.output_video_path = output_video_path
        self.target_fps = target_fps

        self._gpu_available = True

        self.temp_dir = Path(f"jobs/{job_id}/temp_video_processing")

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

        self.log = get_job_logger(logger, job_id)
        self.job_id = job_id

        self.log.info("Video Processor initialized for job")

    def _run_command(self, cmd, description=None, **kwargs):
        try:
            kwargs.setdefault('capture_output', True)
            kwargs.setdefault('text', True)

            result = subprocess.run(cmd, **kwargs)

            if result.returncode != 0:
                command_name = os.path.basename(cmd[0]) if cmd else "unknown"

                context = description or f"'{command_name}'"
                self.log.error(f"Command {context} failed (exit code {result.returncode})")

                if result.stderr and not self._is_ffmpeg_noise(result.stderr):
                    self.log.error(f"Error details: {result.stderr.strip()}")

                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

            return result

        except subprocess.CalledProcessError:
            raise
        except Exception as e:
            self.log.exception(f"Unexpected error running command: {e}")
            raise

    def _is_ffmpeg_noise(self, stderr_text):
        noise_patterns = [
            "ffmpeg version", "built with", "configuration:",
            "Duration:", "Stream #", "Press [q]", "Copyright"
        ]
        return any(pattern in stderr_text for pattern in noise_patterns)

    def _check_gpu_availability(self):
        """Checks for NVIDIA GPU availability for video encoding"""
        if hasattr(self, '_gpu_available'):
            return self._gpu_available

        try:
            cmd = ['ffmpeg', '-encoders']
            result = subprocess.run(cmd, capture_output=True, text=True)

            if 'h264_nvenc' in result.stdout:
                test_cmd = [
                    'ffmpeg',
                    '-f', 'lavfi',
                    '-i', 'nullsrc=s=640x480:d=1',
                    '-c:v', 'h264_nvenc',
                    '-f', 'null',
                    '-'
                ]
                test_result = subprocess.run(test_cmd, capture_output=True, text=True)

                if test_result.returncode == 0:
                    self._gpu_available = True
                    return True

            self._gpu_available = False
            return False
        except Exception as e:
            self.log.error(f"Error checking GPU availability: {e}")
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

            self.log.warning(f"Warning: Could not determine video FPS, using default: {self.target_fps}")
            return self.target_fps
        except Exception as e:
            self.log.warning(f"Error while getting FPS: {e}")
            return self.target_fps

    def _get_video_duration(self, video_path):
        """Retrieve video duration in seconds"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json',
                str(video_path)
            ]
            result = self._run_command(cmd, description="Getting video duration (JSON)")
            data = json.loads(result.stdout)

            if not data.get('format') or 'duration' not in data['format']:
                raise ValueError(f"Failed to get duration info: {result.stdout}")

            return float(data['format']['duration'])
        except Exception as e:
            self.log.warning(f"Error while getting duration: {e}")

            try:
                cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-i', str(video_path),
                    '-show_entries', 'format=duration',
                    '-sexagesimal',
                    '-of', 'default=noprint_wrappers=1:nokey=1'
                ]
                result = self._run_command(cmd, description="Getting video duration (fallback)")
                time_str = result.stdout.strip()

                if ':' in time_str:
                    parts = time_str.split(':')
                    if len(parts) == 3:
                        hours, minutes, seconds = parts
                        seconds = float(seconds)
                        return float(hours) * 3600 + float(minutes) * 60 + seconds

                return float(time_str)
            except Exception as e2:
                self.log.error(f"Alternative method of getting duration also failed: {e2}")
                return 0.0

    def _get_video_resolution(self, video_path):
        """Get video resolution (width, height)"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0',
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout and 'x' in result.stdout:
                width, height = map(int, result.stdout.strip().split('x'))
                return width, height
            return None, None
        except Exception as e:
            self.log.error(f"Error getting video resolution: {e}")
            return None, None

    def _adjust_duration_for_fps(self, duration):
        """Adjust duration to match the target frame rate"""
        frame_duration = 1.0 / self.target_fps
        frames = round(duration / frame_duration)
        return frames * frame_duration

    def convert_to_target_fps(self):
        has_gpu = self._check_gpu_availability()
        try:
            input_path = str(self.input_video_path)
            output_path = str(self.converted_video_path)

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
                    self.log.info(f"Detected input video FPS: {actual_fps}")
                    self.needs_fps_conversion = abs(actual_fps - self.target_fps) > 0.01
                except ValueError:
                    self.log.warning(f"Could not parse FPS fraction: {fps_str}")
                    # Fallback to stored value

            if self.needs_fps_conversion:
                self.log.info(f"Convert video from {self.input_fps} FPS to {self.target_fps} FPS")
            else:
                self.log.info(f"The original video already has the required frame rate ({self.target_fps} FPS)")

            # has_gpu = self._check_gpu_availability()

            if has_gpu:
                self.log.info("Using NVIDIA GPU to speed up conversion with maximum quality")
                encoder = 'h264_nvenc'
                pixel_format = 'yuv420p'
                quality_params = [
                    '-b:v', '100M',
                    '-bufsize', '100M',
                    '-rc', 'vbr',
                    '-rc-lookahead', '32',
                    '-spatial_aq', '1',
                    '-temporal_aq', '1',
                    '-aq-strength', '15',
                    '-nonref_p', '0',
                ]
                preset = 'p7'
                extra_params = ['-tune', 'hq']
            else:
                self.log.info("GPU not detected or not supported. Using CPU")
                encoder = 'libx264'
                pixel_format = 'yuv444p'
                quality_params = ['-crf', '0']
                preset = 'veryslow'
                extra_params = ['-tune', 'film']

            if not self.needs_fps_conversion:
                self.log.info(f"The original video already has the required frame rate ({self.target_fps} FPS)")
                cmd = [
                          'ffmpeg',
                          '-y',
                          '-i', input_path,
                          '-an',
                          '-c:v', encoder,
                      ] + quality_params + [
                          '-preset', preset,
                          '-pix_fmt', pixel_format,
                      ] + extra_params + [
                          output_path
                      ]
            else:
                self.log.info(f"Convert video from {self.input_fps} FPS to {self.target_fps} FPS")
                cmd = [
                          'ffmpeg',
                          '-y',
                          '-i', input_path,
                          '-an',
                          '-c:v', encoder,
                      ] + quality_params + [
                          '-preset', preset,
                          '-pix_fmt', pixel_format,
                      ] + extra_params + [
                          '-r', str(self.target_fps),
                          output_path
                      ]

            self._run_command(cmd, description="Converting video fps")

            if not os.path.exists(output_path):
                raise FileNotFoundError(f"The converted file was not created: {output_path}")

            actual_fps = self._get_video_fps(output_path)
            self.log.info(f"Checking the FPS of the converted file: {actual_fps}")

            if abs(actual_fps - self.target_fps) > 0.1:
                self.log.warning(
                    f"FPS of converted file ({actual_fps}) is different from target ({self.target_fps})")

            return output_path
        except Exception as e:
            self.log.warning(f"Error converting video: {e}")
            # If the error is related to the GPU, try CPU fallback
            if has_gpu and ("NVENC" in str(e) or "GPU" in str(e) or "nvenc" in str(e)):
                self.log.warning("Error using GPU. Trying conversion on CPU...")
                # Set the flag that the GPU is unavailable
                self._gpu_available = False
                # Recursively call the same function, which will now use the CPU
                return self.convert_to_target_fps()
            raise

    def extract_segments(self):
        segments = self.data.get('segments', [])
        video_path = self.converted_video_path

        total_duration = self._get_video_duration(video_path)
        self.log.info(f"Total video duration: {total_duration:.4f} seconds")

        has_gpu = self._check_gpu_availability()

        if has_gpu:
            self.log.info("Using NVIDIA GPU to speed up segment extraction with maximum quality")
            encoder = 'h264_nvenc'
            pixel_format = 'yuv420p'
            quality_params = [
                '-b:v', '100M',
                '-bufsize', '100M',
                '-rc', 'vbr',
                '-rc-lookahead', '32',
                '-spatial_aq', '1',
                '-temporal_aq', '1',
                '-aq-strength', '15',
                '-nonref_p', '0'
            ]
            preset = 'p7'
            extra_params = ['-tune', 'hq']
        else:
            self.log.warning("GPU not detected or not supported. Using CPU")
            encoder = 'libx264'
            pixel_format = 'yuv444p'
            quality_params = ['-crf', '0']
            preset = 'veryslow'
            extra_params = ['-tune', 'film']

        if segments and segments[0]['start'] > 0.01:
            initial_gap_start = 0.0
            initial_gap_end = segments[0]['start']
            initial_gap_path = self.gaps_dir / f"gap_start_0000.mp4"

            self.log.info(f"Extracting initial gap: {initial_gap_start} - {initial_gap_end}")

            try:
                if has_gpu:
                    gap_cmd = [
                        'ffmpeg',
                        '-i', str(video_path),
                        '-ss', str(initial_gap_start),
                        '-to', str(initial_gap_end),
                        '-c:v', encoder,
                        '-b:v', '50M',
                        '-rc', 'vbr',
                        '-preset', 'p5',
                        '-pix_fmt', pixel_format,
                        '-an',
                        str(initial_gap_path)
                    ]
                else:
                    gap_cmd = [
                        'ffmpeg',
                        '-i', str(video_path),
                        '-ss', str(initial_gap_start),
                        '-to', str(initial_gap_end),
                        '-c:v', 'libx264',
                        '-crf', '18',
                        '-preset', 'medium',
                        '-pix_fmt', 'yuv420p',
                        '-an',
                        str(initial_gap_path)
                    ]

                self._run_command(gap_cmd, description="Extracting initial gap from video")

            except Exception as e:
                self.log.warning(f"Error while extracting initial gap: {e}")

                fallback_gap_cmd = [
                    'ffmpeg',
                    '-i', str(video_path),
                    '-ss', str(initial_gap_start),
                    '-to', str(initial_gap_end),
                    '-c:v', 'libx264',
                    '-crf', '18',
                    '-preset', 'medium',
                    '-an',
                    str(initial_gap_path)
                ]

                self._run_command(fallback_gap_cmd, description="Extracting initial gap from video")


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

                self._run_command(cmd, description="Extracting segments from video")

            except Exception as e:
                self.log.warning(f"Error using GPU for segment {i}: {e}")
                if has_gpu:
                    self.log.info("Trying to extract the segment using CPU...")
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
                    self._run_command(fallback_cmd, description="Extracting segments from video with CPU (fallback)")

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

                        self._run_command(gap_cmd, description="Extracting gap between video segments")

                    except Exception as e:
                        self.log.warning(f"Error while extracting gap {i}-{i + 1}: {e}")

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

                        self._run_command(fallback_gap_cmd, description="Extracting gap between video segments with CPU (fallback)")

        if segments and segments[-1]['end'] < total_duration - 0.01:
            final_gap_start = segments[-1]['end']
            final_gap_end = total_duration
            final_gap_path = self.gaps_dir / f"gap_{len(segments) - 1:04d}_end.mp4"

            self.log.info(f"Extracting final gap: {final_gap_start} - {final_gap_end}")

            try:
                if has_gpu:
                    gap_cmd = [
                        'ffmpeg',
                        '-i', str(video_path),
                        '-ss', str(final_gap_start),
                        '-to', str(final_gap_end),
                        '-c:v', encoder,
                        '-b:v', '50M',
                        '-rc', 'vbr',
                        '-preset', 'p5',
                        '-pix_fmt', pixel_format,
                        '-an',
                        str(final_gap_path)
                    ]
                else:
                    gap_cmd = [
                        'ffmpeg',
                        '-i', str(video_path),
                        '-ss', str(final_gap_start),
                        '-to', str(final_gap_end),
                        '-c:v', 'libx264',
                        '-crf', '18',
                        '-preset', 'medium',
                        '-pix_fmt', 'yuv420p',
                        '-an',
                        str(final_gap_path)
                    ]

                self._run_command(gap_cmd, description="Extracting final gap from video")

            except Exception as e:
                self.log.warning(f"Error while extracting final gap: {e}")

                fallback_gap_cmd = [
                    'ffmpeg',
                    '-i', str(video_path),
                    '-ss', str(final_gap_start),
                    '-to', str(final_gap_end),
                    '-c:v', 'libx264',
                    '-crf', '18',
                    '-preset', 'medium',
                    '-an',
                    str(final_gap_path)
                ]

                self._run_command(fallback_gap_cmd, description="Extracting final gap from video with CPU (fallback)")

    def process_segments(self):
        segments = self.data.get('segments', [])

        has_gpu = self._check_gpu_availability()

        if has_gpu:
            self.log.info("Using NVIDIA GPU for segment processing with maximum quality")
            encoder = 'h264_nvenc'
            pixel_format = 'yuv420p'
            quality_params = [
                '-b:v', '100M',
                '-bufsize', '100M',
                '-rc', 'vbr',
                '-rc-lookahead', '32',
                '-spatial_aq', '1',
                '-temporal_aq', '1',
                '-aq-strength', '15',
                '-nonref_p', '0',
            ]
            preset = 'p7'
            extra_params = ['-tune', 'hq']
        else:
            self.log.warning("GPU not detected or not supported. Using CPU")
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
                    self.log.error(f"Segment file not found: {input_path}")
                    continue

                original_duration = self._get_video_duration(input_path)
                target_duration = segment['tts_duration']
                adjusted_target_duration = self._adjust_duration_for_fps(target_duration)

                if original_duration <= 0:
                    self.log.warning(f"Original duration of segment {i} is equal to or less than zero: {original_duration}")
                    continue

                if adjusted_target_duration <= 0:
                    self.log.warning(
                        f"Target duration of segment {i} is equal to or less than zero: {adjusted_target_duration}")
                    continue

                if abs(original_duration - adjusted_target_duration) < 0.04:  # Difference less than 1 frame
                    self.log.info(f"Minor change in duration. Copying file.")
                    shutil.copy(str(input_path), str(output_path))
                    actual_duration = self._get_video_duration(output_path)
                else:
                    self.log.info(f"Set exact duration for segment {i}...")
                    speed_factor = adjusted_target_duration / original_duration

                    try:
                        cmd = [
                                  'ffmpeg',
                                  '-i', str(input_path),
                                  '-filter:v', f'setpts={speed_factor}*PTS,fps={self.target_fps}',
                                  '-r', str(self.target_fps),
                                  '-c:v', encoder,
                              ] + quality_params + [
                                  '-preset', preset,
                                  '-pix_fmt', pixel_format,
                              ] + extra_params + [
                                  '-an',
                                  '-t', str(adjusted_target_duration),
                                  str(output_path)
                              ]

                        self._run_command(cmd, description="Processing video segment")

                        if not os.path.exists(output_path):
                            raise FileNotFoundError(f"The processed file was not created: {output_path}")

                        actual_duration = self._get_video_duration(output_path)

                    except Exception as e:
                        if has_gpu:
                            self.log.warning(f"Error processing segment {i} with GPU: {e}")
                            self.log.info(f"Trying to process segment {i} with CPU...")

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
                            self._run_command(cmd, description="Processing video segment with CPU (fallback)")
                            actual_duration = self._get_video_duration(output_path)
                        else:
                            raise

                duration_diff = abs(actual_duration - adjusted_target_duration)

                self.log.info(f"Segment {i}: original duration = {original_duration:.4f} sec, "
                      f"target = {adjusted_target_duration:.4f} sec (tts = {target_duration:.4f} sec), "
                      f"actual = {actual_duration:.4f} sec, "
                      f"speed coefficient = {adjusted_target_duration / original_duration:.4f}")

                if duration_diff > 0.04:  # If the deviation is more than 1 frame
                    self.log.warning(f"Deviation from target duration: {duration_diff:.4f} sec")

            except Exception as e:
                self.log.error(f"Error processing segment {i}: {e}")

    def combine_final_video_reliable(self):
        segments = self.data.get('segments', [])

        input_files = []

        initial_gap_path = self.gaps_dir / "gap_start_0000.mp4"
        if initial_gap_path.exists():
            input_files.append((str(initial_gap_path), "gap_start_0000"))

        for i in range(len(segments)):
            segment_path = self.processed_segments_dir / f"processed_segment_{i:04d}.mp4"
            if os.path.exists(segment_path):
                input_files.append((str(segment_path), f"segment_{i:04d}"))
            else:
                self.log.error(f"Processed segment not found: {segment_path}")

            gap_path = self.gaps_dir / f"gap_{i:04d}_{i + 1:04d}.mp4"
            if gap_path.exists():
                input_files.append((str(gap_path), f"gap_{i:04d}_{i + 1:04d}"))

        final_gap_path = self.gaps_dir / f"gap_{len(segments) - 1:04d}_end.mp4"
        if final_gap_path.exists():
            input_files.append((str(final_gap_path), f"gap_{len(segments) - 1:04d}_end"))

        has_gpu = self._check_gpu_availability()

        self.log.info("Creating final video...")
        temp_output = self.temp_dir / "temp_output.mp4"

        self.log.info(f"Files to merge: {len(input_files)}")

        filter_parts = []
        for i in range(len(input_files)):
            filter_parts.append(f"[{i}:v]")

        filter_graph = f"{''.join(filter_parts)}concat=n={len(input_files)}:v=1:a=0[outv]"

        ffmpeg_inputs = []
        for file_path, file_id in input_files:
            ffmpeg_inputs.extend(['-i', file_path])

        if has_gpu:
            self.log.info("Using NVIDIA GPU for video creation...")
            cmd = [
                'ffmpeg',
                *ffmpeg_inputs,
                '-filter_complex', filter_graph,
                '-map', '[outv]',
                '-c:v', 'h264_nvenc',
                '-b:v', '200M',
                '-bufsize', '200M',
                '-rc', 'vbr',
                '-rc-lookahead', '32',
                '-spatial_aq', '1',
                '-temporal_aq', '1',
                '-aq-strength', '15',
                '-qmin', '0',
                '-qmax', '25',
                '-profile:v', 'high',
                '-level', '5.1',
                '-preset', 'p7',
                '-tune', 'hq',
                '-pix_fmt', 'yuv444p',
                '-movflags', '+faststart',
                str(temp_output)
            ]
        else:
            self.log.warning("Using CPU for video creation...")
            cmd = [
                'ffmpeg',
                *ffmpeg_inputs,
                '-filter_complex', filter_graph,
                '-map', '[outv]',
                '-c:v', 'libx264',
                '-crf', '0',
                '-preset', 'veryslow',
                '-tune', 'film',
                '-pix_fmt', 'yuv444p',
                '-movflags', '+faststart',
                str(temp_output)
            ]

        try:
            self._run_command(cmd, description="Combining processed video segments")
            if not os.path.exists(temp_output) and has_gpu:
                self.log.warning("Error using GPU. Trying CPU fallback...")
                cpu_cmd = [
                    'ffmpeg',
                    *ffmpeg_inputs,
                    '-filter_complex', filter_graph,
                    '-map', '[outv]',
                    '-c:v', 'libx264',
                    '-crf', '0',
                    '-preset', 'veryslow',
                    '-tune', 'film',
                    '-pix_fmt', 'yuv444p',
                    '-movflags', '+faststart',
                    str(temp_output)
                ]
                self._run_command(cpu_cmd, description="Combining processed video segments with CPU (fallback)")
        except Exception as e:
            self.log.warning(f"Error creating video: {e}")
            if has_gpu:
                self.log.warning("Trying CPU fallback...")
                cpu_cmd = [
                    'ffmpeg',
                    *ffmpeg_inputs,
                    '-filter_complex', filter_graph,
                    '-map', '[outv]',
                    '-c:v', 'libx264',
                    '-crf', '0',
                    '-preset', 'veryslow',
                    '-tune', 'film',
                    '-pix_fmt', 'yuv444p',
                    '-movflags', '+faststart',
                    str(temp_output)
                ]
                self._run_command(cpu_cmd, description="Combining processed video segments with CPU (fallback)")

        if os.path.exists(temp_output):
            video_duration = self._get_video_duration(temp_output)
            self.log.info(f"Video created! Duration: {video_duration:.4f} sec")
            shutil.copy(temp_output, self.output_video_path)
            self.log.info(f"Final video saved to: {self.output_video_path}")
            return True
        else:
            self.log.error("Error: Failed to create video!")
            return False


    def cleanup(self):
        try:
            shutil.rmtree(self.temp_dir)
            self.log.info("Video Processor temporary files have been deleted")
        except Exception as e:
            self.log.warning(f"Error deleting temporary files: {e}")

    def process(self):
        try:
            self.log.info("VideoProcessor | Step 1/4. Convert video to target FPS...")
            self.convert_to_target_fps()

            self.log.info("VideoProcessor | Step 2/4. Extracting segments and gaps...")
            self.extract_segments()

            self.log.info("VideoProcessor | Step 3/4. Processing segments with changing duration...")
            self.process_segments()

            self.log.info("VideoProcessor | Step 4/4. Combining final video using the reliable method...")
            is_success = self.combine_final_video_reliable()

            if is_success:
                if os.path.exists(self.output_video_path):
                    self.log.info("Video Processor done the work successfully!")
                    self.cleanup()
                else:
                    self.log.warning(f"Final file not found at {self.output_video_path}")
                    self.log.warning("Temporary files will not be deleted for debugging purposes")
                    is_success = False
            else:
                self.log.error("Video Processor failed")
                self.log.error("Temporary files will not be deleted for debugging purposes")

            return is_success
        except Exception as e:
            self.log.error(f"VideoProcessor error: {e}")
            import traceback
            traceback.print_exc()
            return False
