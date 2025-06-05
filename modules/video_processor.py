import os
import json
import subprocess
from pathlib import Path
import shutil
from utils.logger_config import setup_logger


logger = setup_logger(name=__name__, log_file="logs/app.log")


class VideoProcessor:
    def __init__(
            self,
            job_id,
            input_video_path,
            json_path,
            output_video_path,
            output_video_path_premium,
            intro_outro_path,
            target_fps=25,
            is_premium=False
    ):
        """
        Video Processor initialization

        :param input_video_path: Path to original video file
        :param json_path: Path to segments' info JSON-file
        :param output_video_path: Output file path
        :param intro_outro_path: Path to intro/outro files directory
        :param target_fps: Target FPS (25 by default)
        :param is_premium: If True, no intro/outro will be added (default: False)
        """

        self.input_video_path = input_video_path
        self.json_path = json_path
        self.output_video_path = output_video_path
        self.output_video_path_premium = output_video_path_premium
        self.resources_dir = intro_outro_path
        self.target_fps = target_fps
        self.is_premium = is_premium

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

    def _run_command(self, cmd, **kwargs):
        try:
            if 'capture_output' not in kwargs:
                kwargs['capture_output'] = True

            command_str = ' '.join(map(str, cmd))
            if len(command_str) > 100:
                command_str = command_str[:97] + "..."
            logger.info(f"Executing: {command_str}")

            result = subprocess.run(cmd, text=True, **kwargs)

            if result.returncode != 0:
                logger.error(f"Command failed with code {result.returncode}")
                if hasattr(result, 'stderr') and result.stderr:
                    logger.error(f"Error: {result.stderr.strip()}")
                if hasattr(result, 'stdout') and result.stdout:
                    logger.info(f"STDOUT: {result.stdout.strip()}")
                raise subprocess.CalledProcessError(result.returncode, cmd)

            return result
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise

    def _check_gpu_availability(self):
        """Checks for NVIDIA GPU availability for video encoding"""
        if hasattr(self, '_gpu_available'):
            return self._gpu_available

        try:
            cmd = ['ffmpeg', '-encoders']
            result = subprocess.run(cmd, capture_output=True, text=True)

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

                if test_result.returncode == 0:
                    self._gpu_available = True
                    return True

            self._gpu_available = False
            return False
        except Exception as e:
            logger.error(f"Error checking GPU availability: {e}")
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

            logger.warning(f"Warning: Could not determine video FPS, using default: {self.target_fps}")
            return self.target_fps
        except Exception as e:
            logger.warning(f"Error while getting FPS: {e}")
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
            result = self._run_command(cmd)
            data = json.loads(result.stdout)

            if not data.get('format') or 'duration' not in data['format']:
                raise ValueError(f"Failed to get duration info: {result.stdout}")

            return float(data['format']['duration'])
        except Exception as e:
            logger.warning(f"Error while getting duration: {e}")

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

                if ':' in time_str:
                    parts = time_str.split(':')
                    if len(parts) == 3:
                        hours, minutes, seconds = parts
                        seconds = float(seconds)
                        return float(hours) * 3600 + float(minutes) * 60 + seconds

                return float(time_str)
            except Exception as e2:
                logger.error(f"Alternative method of getting duration also failed: {e2}")
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
            logger.error(f"Error getting video resolution: {e}")
            return None, None

    def _adjust_duration_for_fps(self, duration):
        """Adjust duration to match the target frame rate"""
        frame_duration = 1.0 / self.target_fps
        frames = round(duration / frame_duration)
        return frames * frame_duration

    def convert_to_target_fps(self):
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
                    logger.info(f"Detected input video FPS: {actual_fps}")
                    self.needs_fps_conversion = abs(actual_fps - self.target_fps) > 0.01
                except ValueError:
                    logger.warning(f"Could not parse FPS fraction: {fps_str}")
                    # Fallback to stored value

            if self.needs_fps_conversion:
                logger.info(f"Convert video from {self.input_fps} FPS to {self.target_fps} FPS")
            else:
                logger.info(f"The original video already has the required frame rate ({self.target_fps} FPS)")

            has_gpu = self._check_gpu_availability()

            if has_gpu:
                logger.info("Using NVIDIA GPU to speed up conversion with maximum quality")
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
                logger.info("GPU not detected or not supported. Using CPU")
                encoder = 'libx264'
                pixel_format = 'yuv444p'
                quality_params = ['-crf', '0']
                preset = 'veryslow'
                extra_params = ['-tune', 'film']

            if not self.needs_fps_conversion:
                logger.info(f"The original video already has the required frame rate ({self.target_fps} FPS)")
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
                logger.info(f"Convert video from {self.input_fps} FPS to {self.target_fps} FPS")
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

            self._run_command(cmd)

            if not os.path.exists(output_path):
                raise FileNotFoundError(f"The converted file was not created: {output_path}")

            actual_fps = self._get_video_fps(output_path)
            logger.info(f"Checking the FPS of the converted file: {actual_fps}")

            if abs(actual_fps - self.target_fps) > 0.1:
                logger.warning(
                    f"FPS of converted file ({actual_fps}) is different from target ({self.target_fps})")

            return output_path
        except Exception as e:
            logger.warning(f"Error converting video: {e}")
            # If the error is related to the GPU, try CPU fallback
            if has_gpu and ("NVENC" in str(e) or "GPU" in str(e) or "nvenc" in str(e)):
                logger.warning("Error using GPU. Trying conversion on CPU...")
                # Set the flag that the GPU is unavailable
                self._gpu_available = False
                # Recursively call the same function, which will now use the CPU
                return self.convert_to_target_fps()
            raise

    def extract_segments(self):
        segments = self.data.get('segments', [])
        video_path = self.converted_video_path

        total_duration = self._get_video_duration(video_path)
        logger.info(f"Total video duration: {total_duration:.4f} seconds")

        has_gpu = self._check_gpu_availability()

        if has_gpu:
            logger.info("Using NVIDIA GPU to speed up segment extraction with maximum quality")
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
            logger.warning("GPU not detected or not supported. Using CPU")
            encoder = 'libx264'
            pixel_format = 'yuv444p'
            quality_params = ['-crf', '0']
            preset = 'veryslow'
            extra_params = ['-tune', 'film']

        if segments and segments[0]['start'] > 0.01:
            initial_gap_start = 0.0
            initial_gap_end = segments[0]['start']
            initial_gap_path = self.gaps_dir / f"gap_start_0000.mp4"

            logger.info(f"Extracting initial gap: {initial_gap_start} - {initial_gap_end}")

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

                self._run_command(gap_cmd)

            except Exception as e:
                logger.warning(f"Error while extracting initial gap: {e}")

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

                self._run_command(fallback_gap_cmd)


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
                logger.warning(f"Error using GPU for segment {i}: {e}")
                if has_gpu:
                    logger.info("Trying to extract the segment using CPU...")
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
                        logger.warning(f"Error while extracting gap {i}-{i + 1}: {e}")

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

        if segments and segments[-1]['end'] < total_duration - 0.01:
            final_gap_start = segments[-1]['end']
            final_gap_end = total_duration
            final_gap_path = self.gaps_dir / f"gap_{len(segments) - 1:04d}_end.mp4"

            logger.info(f"Extracting final gap: {final_gap_start} - {final_gap_end}")

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

                self._run_command(gap_cmd)

            except Exception as e:
                logger.warning(f"Error while extracting final gap: {e}")

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

                self._run_command(fallback_gap_cmd)

    def process_segments(self):
        segments = self.data.get('segments', [])

        has_gpu = self._check_gpu_availability()

        if has_gpu:
            logger.info("Using NVIDIA GPU for segment processing with maximum quality")
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
            logger.warning("GPU not detected or not supported. Using CPU")
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
                    logger.error(f"Segment file not found: {input_path}")
                    continue

                original_duration = self._get_video_duration(input_path)
                target_duration = segment['tts_duration']
                adjusted_target_duration = self._adjust_duration_for_fps(target_duration)

                if original_duration <= 0:
                    logger.warning(f"Original duration of segment {i} is equal to or less than zero: {original_duration}")
                    continue

                if adjusted_target_duration <= 0:
                    logger.warning(
                        f"Target duration of segment {i} is equal to or less than zero: {adjusted_target_duration}")
                    continue

                if abs(original_duration - adjusted_target_duration) < 0.04:  # Difference less than 1 frame
                    logger.info(f"Minor change in duration. Copying file.")
                    shutil.copy(str(input_path), str(output_path))
                    actual_duration = self._get_video_duration(output_path)
                else:
                    logger.info(f"Set exact duration for segment {i}...")
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

                        self._run_command(cmd)

                        if not os.path.exists(output_path):
                            raise FileNotFoundError(f"The processed file was not created: {output_path}")

                        actual_duration = self._get_video_duration(output_path)

                    except Exception as e:
                        if has_gpu:
                            logger.warning(f"Error processing segment {i} with GPU: {e}")
                            logger.info(f"Trying to process segment {i} with CPU...")

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

                logger.info(f"Segment {i}: original duration = {original_duration:.4f} sec, "
                      f"target = {adjusted_target_duration:.4f} sec (tts = {target_duration:.4f} sec), "
                      f"actual = {actual_duration:.4f} sec, "
                      f"speed coefficient = {adjusted_target_duration / original_duration:.4f}")

                if duration_diff > 0.04:  # If the deviation is more than 1 frame
                    logger.warning(f"Deviation from target duration: {duration_diff:.4f} sec")

            except Exception as e:
                logger.error(f"Error processing segment {i}: {e}")

    def combine_final_video_reliable(self):
        segments = self.data.get('segments', [])

        main_width, main_height = self._get_video_resolution(self.converted_video_path)
        logger.info(f"Main video resolution: {main_width}x{main_height}")

        base_input_files = []

        initial_gap_path = self.gaps_dir / "gap_start_0000.mp4"
        if initial_gap_path.exists():
            base_input_files.append((str(initial_gap_path), "gap_start_0000"))

        for i in range(len(segments)):
            segment_path = self.processed_segments_dir / f"processed_segment_{i:04d}.mp4"
            if os.path.exists(segment_path):
                base_input_files.append((str(segment_path), f"segment_{i:04d}"))
            else:
                logger.error(f"Processed segment not found: {segment_path}")

            gap_path = self.gaps_dir / f"gap_{i:04d}_{i + 1:04d}.mp4"
            if gap_path.exists():
                base_input_files.append((str(gap_path), f"gap_{i:04d}_{i + 1:04d}"))

        final_gap_path = self.gaps_dir / f"gap_{len(segments) - 1:04d}_end.mp4"
        if final_gap_path.exists():
            base_input_files.append((str(final_gap_path), f"gap_{len(segments) - 1:04d}_end"))

        selected_intro_path = None

        if not self.is_premium:
            logger.info("Finding intro/outro for common version...")
            resources_dir = self.resources_dir

            intro_4k_path = os.path.join(resources_dir, "intro_outro_4k.mp4")
            intro_2k_path = os.path.join(resources_dir, "intro_outro_2k.mp4")
            intro_fullhd_path = os.path.join(resources_dir, "intro_outro_full_hd.mp4")

            selected_intro_path = intro_fullhd_path

            if main_width is not None and main_height is not None:
                if main_width >= 3840 or main_height >= 2160:
                    if os.path.exists(intro_4k_path):
                        selected_intro_path = intro_4k_path
                        logger.info(f"Using 4K intro/outro: {intro_4k_path}")
                    else:
                        logger.warning(f"4K intro not found, using fallback")
                elif main_width >= 2560 or main_height >= 1440:
                    if os.path.exists(intro_2k_path):
                        selected_intro_path = intro_2k_path
                        logger.info(f"Using 2K intro/outro: {intro_2k_path}")
                    else:
                        logger.warning(f"2K intro not found, using fallback")

            if not os.path.exists(selected_intro_path):
                logger.warning(f"Selected intro file {selected_intro_path} not found!")
                intro_files = [f for f in os.listdir(resources_dir) if f.startswith("intro_outro_")]
                if intro_files:
                    selected_intro_path = os.path.join(resources_dir, intro_files[0])
                    logger.info(f"Using alternative intro: {selected_intro_path}")
                else:
                    logger.warning("No intro files found in resources directory!")
                    selected_intro_path = None

            if selected_intro_path:
                logger.info(f"Selected intro/outro: {selected_intro_path}")

        has_gpu = self._check_gpu_availability()

        logger.info("Creating Premium version...")
        temp_output_premium = self.temp_dir / "temp_output_premium.mp4"

        premium_input_files = base_input_files.copy()
        logger.info(f"Premium version files to merge: {len(premium_input_files)}")

        filter_parts_premium = []
        for i in range(len(premium_input_files)):
            filter_parts_premium.append(f"[{i}:v]")

        filter_graph_premium = f"{''.join(filter_parts_premium)}concat=n={len(premium_input_files)}:v=1:a=0[outv]"

        ffmpeg_inputs_premium = []
        for file_path, file_id in premium_input_files:
            ffmpeg_inputs_premium.extend(['-i', file_path])

        if has_gpu:
            logger.info("Using NVIDIA GPU for creating premium video...")
            cmd_premium = [
                'ffmpeg',
                *ffmpeg_inputs_premium,
                '-filter_complex', filter_graph_premium,
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
                str(temp_output_premium)
            ]
        else:
            logger.warning("Using CPU for creating premium video...")
            cmd_premium = [
                'ffmpeg',
                *ffmpeg_inputs_premium,
                '-filter_complex', filter_graph_premium,
                '-map', '[outv]',
                '-c:v', 'libx264',
                '-crf', '0',
                '-preset', 'veryslow',
                '-tune', 'film',
                '-pix_fmt', 'yuv444p',
                '-movflags', '+faststart',
                str(temp_output_premium)
            ]

        try:
            self._run_command(cmd_premium)
            if not os.path.exists(temp_output_premium) and has_gpu:
                logger.warning("Error using GPU for creating premium version. Trying CPU fallback...")
                cpu_cmd_premium = [
                    'ffmpeg',
                    *ffmpeg_inputs_premium,
                    '-filter_complex', filter_graph_premium,
                    '-map', '[outv]',
                    '-c:v', 'libx264',
                    '-crf', '0',
                    '-preset', 'veryslow',
                    '-tune', 'film',
                    '-pix_fmt', 'yuv444p',
                    '-movflags', '+faststart',
                    str(temp_output_premium)
                ]
                self._run_command(cpu_cmd_premium)
        except Exception as e:
            logger.warning(f"Error creating premium video: {e}")
            if has_gpu:
                logger.warning("Trying CPU fallback for creating premium version...")
                cpu_cmd_premium = [
                    'ffmpeg',
                    *ffmpeg_inputs_premium,
                    '-filter_complex', filter_graph_premium,
                    '-map', '[outv]',
                    '-c:v', 'libx264',
                    '-crf', '0',
                    '-preset', 'veryslow',
                    '-tune', 'film',
                    '-pix_fmt', 'yuv444p',
                    '-movflags', '+faststart',
                    str(temp_output_premium)
                ]
                self._run_command(cpu_cmd_premium)

        if os.path.exists(temp_output_premium):
            premium_duration = self._get_video_duration(temp_output_premium)
            logger.info(f"Premium video created! Duration: {premium_duration:.4f} sec")
            shutil.copy(temp_output_premium, self.output_video_path_premium)
            logger.info(f"Premium video saved to: {self.output_video_path_premium}")
        else:
            logger.error("Error: Failed to create premium video!")
            return False

        if not self.is_premium and selected_intro_path:
            logger.info("Creating Common version")
            temp_output_regular = self.temp_dir / "temp_output_regular.mp4"

            regular_input_files = []
            regular_input_files.append((str(selected_intro_path), "intro"))
            regular_input_files.extend(base_input_files)
            regular_input_files.append((str(selected_intro_path), "outro"))

            logger.info(f"Common version files to merge: {len(regular_input_files)}")

            filter_parts_regular = []
            for i in range(len(regular_input_files)):
                filter_parts_regular.append(f"[{i}:v]")

            filter_graph_regular = f"{''.join(filter_parts_regular)}concat=n={len(regular_input_files)}:v=1:a=0[outv]"

            ffmpeg_inputs_regular = []
            for file_path, file_id in regular_input_files:
                ffmpeg_inputs_regular.extend(['-i', file_path])

            if has_gpu:
                logger.info("Using NVIDIA GPU for creating common version...")
                cmd_regular = [
                    'ffmpeg',
                    *ffmpeg_inputs_regular,
                    '-filter_complex', filter_graph_regular,
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
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    str(temp_output_regular)
                ]
            else:
                logger.warning("Using CPU for creating common version...")
                cmd_regular = [
                    'ffmpeg',
                    *ffmpeg_inputs_regular,
                    '-filter_complex', filter_graph_regular,
                    '-map', '[outv]',
                    '-c:v', 'libx264',
                    '-crf', '0',
                    '-preset', 'veryslow',
                    '-tune', 'film',
                    '-pix_fmt', 'yuv444p',
                    '-movflags', '+faststart',
                    str(temp_output_regular)
                ]

            try:
                self._run_command(cmd_regular)
                if not os.path.exists(temp_output_regular) and has_gpu:
                    logger.warning("Error using GPU for creating common version. Trying CPU fallback...")
                    cpu_cmd_regular = [
                        'ffmpeg',
                        *ffmpeg_inputs_regular,
                        '-filter_complex', filter_graph_regular,
                        '-map', '[outv]',
                        '-c:v', 'libx264',
                        '-crf', '0',
                        '-preset', 'veryslow',
                        '-tune', 'film',
                        '-pix_fmt', 'yuv444p',
                        '-movflags', '+faststart',
                        str(temp_output_regular)
                    ]
                    self._run_command(cpu_cmd_regular)
            except Exception as e:
                logger.warning(f"Error creating common version: {e}")
                if has_gpu:
                    logger.info("Trying CPU fallback for creating common version...")
                    cpu_cmd_regular = [
                        'ffmpeg',
                        *ffmpeg_inputs_regular,
                        '-filter_complex', filter_graph_regular,
                        '-map', '[outv]',
                        '-c:v', 'libx264',
                        '-crf', '0',
                        '-preset', 'veryslow',
                        '-tune', 'film',
                        '-pix_fmt', 'yuv444p',
                        '-movflags', '+faststart',
                        str(temp_output_regular)
                    ]
                    self._run_command(cpu_cmd_regular)

            if os.path.exists(temp_output_regular):
                common_version_duration = self._get_video_duration(temp_output_regular)
                logger.info(f"Common video created! Duration: {common_version_duration:.4f} sec")
                shutil.copy(temp_output_regular, self.output_video_path)
                logger.info(f"Regular video saved to: {self.output_video_path}")
            else:
                logger.error("Error: Failed to create regular video!")
                return False
        else:
            if self.is_premium:
                logger.info("Premium user - only premium version created")
            else:
                logger.warning("No intro/outro found - only premium version created")

        premium_exists = os.path.exists(self.output_video_path_premium)
        regular_exists = os.path.exists(self.output_video_path) if not self.is_premium else True

        if premium_exists and regular_exists:
            logger.info(f"Final video successfully created!")
            logger.info(f"Premium video: {self.output_video_path_premium}")
            if not self.is_premium:
                logger.info(f"Common video: {self.output_video_path}")
            return True
        else:
            logger.error("Error: One or both videos failed to create!")
            return False

    def cleanup(self):
        try:
            shutil.rmtree(self.temp_dir)
            logger.info(f"Temporary files have been deleted: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Error deleting temporary files: {e}")

    def process(self):
        try:
            logger.info("VideoProcessor | Step 1/4. Convert video to target FPS...")
            self.convert_to_target_fps()

            logger.info("VideoProcessor | Step 2/4. Extracting segments and gaps...")
            self.extract_segments()

            logger.info("VideoProcessor | Step 3/4. Processing segments with changing duration...")
            self.process_segments()

            logger.info("VideoProcessor | Step 4/4. Combining final video using the reliable method...")
            is_success = self.combine_final_video_reliable()

            if is_success:
                if self.is_premium:
                    if os.path.exists(self.output_video_path_premium):
                        logger.info(f"Done! Premium result saved to {self.output_video_path_premium}")
                        self.cleanup()
                    else:
                        logger.warning(f"Warning: premium file was not found at {self.output_video_path_premium}")
                        logger.warning("Temporary files will not be deleted for debugging purposes")
                        is_success = False
                else:
                    if os.path.exists(self.output_video_path) and os.path.exists(self.output_video_path_premium):
                        logger.info(f"Done! Regular result saved to {self.output_video_path}")
                        logger.info(f"Done! Premium result saved to {self.output_video_path_premium}")
                        self.cleanup()
                    else:
                        logger.warning(f"Final files not found")
                        if not os.path.exists(self.output_video_path):
                            logger.warning(f"Missing: {self.output_video_path}")
                        if not os.path.exists(self.output_video_path_premium):
                            logger.warning(f"Missing: {self.output_video_path_premium}")
                        logger.warning("Temporary files will not be deleted for debugging purposes")
                        is_success = False
            else:
                logger.error("Processing failed.")
                logger.error("Temporary files will not be deleted for debugging purposes")

            return is_success
        except Exception as e:
            logger.error(f"VideoProcessor error: {e}")
            import traceback
            traceback.print_exc()
            return False
