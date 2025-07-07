import os
import subprocess
from pathlib import Path
import shutil
from utils.logger_config import setup_logger, get_job_logger

logger = setup_logger(name=__name__, log_file="logs/app.log")


class AudioProcessor:
    def __init__(self, job_id, input_audio_path, segments_data):
        self.job_id = job_id
        self.input_audio_path = input_audio_path
        self.segments_data = segments_data

        self.temp_dir = Path(f"jobs/{job_id}/temp_audio_processing")
        self.audio_segments_dir = self.temp_dir / "audio_segments"
        self.processed_audio_segments_dir = self.temp_dir / "processed_audio_segments"
        self.audio_gaps_dir = self.temp_dir / "audio_gaps"

        self.temp_dir.mkdir(exist_ok=True)
        self.audio_segments_dir.mkdir(exist_ok=True)
        self.processed_audio_segments_dir.mkdir(exist_ok=True)
        self.audio_gaps_dir.mkdir(exist_ok=True)

        self.log = get_job_logger(logger, job_id)
        self.job_id = job_id

        self.log.info("AudioProcessor initialized for job")

    def _get_audio_duration(self, audio_path):
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path)
            ]

            result = self._run_command(cmd, description="Getting audio duration")

            return float(result.stdout.strip())

        except Exception as e:
            self.log.error(f"Failed to get audio duration for {audio_path}: {e}")
            return 0.0

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

    # def _run_command(self, cmd):
    #     try:
    #         result = subprocess.run(cmd, capture_output=True, text=True)
    #         if result.returncode != 0:
    #             logger.error(f"Command failed: {' '.join(cmd)}")
    #             logger.error(f"Error: {result.stderr}")
    #             raise subprocess.CalledProcessError(result.returncode, cmd)
    #         return True
    #     except Exception as e:
    #         logger.error(f"Command execution failed: {e}")
    #         raise

    def extract_audio_segments(self):
        segments = self.segments_data.get('segments', [])

        total_duration = self._get_audio_duration(self.input_audio_path)
        self.log.info(f"Total audio duration: {total_duration:.4f} seconds")

        if segments and segments[0]['start'] > 0.01:
            initial_gap_start = 0.0
            initial_gap_end = segments[0]['start']
            initial_gap_path = self.audio_gaps_dir / "audio_gap_start_0000.wav"

            self.log.info(f"Extracting initial audio gap: {initial_gap_start} - {initial_gap_end}")

            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.input_audio_path),
                '-ss', str(initial_gap_start),
                '-to', str(initial_gap_end),
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                str(initial_gap_path)
            ]
            self._run_command(cmd, description="Extracting initial gap from original audio")

        for i, segment in enumerate(segments):
            start = segment['start']
            end = segment['end']
            output_path = self.audio_segments_dir / f"audio_segment_{i:04d}.wav"  # WAV

            self.log.info(f"Extracting audio segment {i}: {start} - {end}")

            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.input_audio_path),
                '-ss', str(start),
                '-to', str(end),
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                str(output_path)
            ]
            self._run_command(cmd, description="Extracting audio segment from original audio")

            if i + 1 < len(segments):
                next_start = segments[i + 1]['start']
                if next_start > end:
                    gap_start = end
                    gap_end = next_start
                    gap_path = self.audio_gaps_dir / f"audio_gap_{i:04d}_{i + 1:04d}.wav"  # WAV

                    self.log.info(f"Extracting audio gap {i}-{i + 1}: {gap_start} - {gap_end}")

                    cmd = [
                        'ffmpeg', '-y',
                        '-i', str(self.input_audio_path),
                        '-ss', str(gap_start),
                        '-to', str(gap_end),
                        '-acodec', 'pcm_s16le',
                        '-ar', '44100',
                        str(gap_path)
                    ]
                    self._run_command(cmd, description="Extracting gaps between audio segments from original audio")

        if segments and segments[-1]['end'] < total_duration - 0.01:
            final_gap_start = segments[-1]['end']
            final_gap_end = total_duration
            final_gap_path = self.audio_gaps_dir / f"audio_gap_{len(segments) - 1:04d}_end.wav"  # WAV

            self.log.info(f"Extracting final audio gap: {final_gap_start} - {final_gap_end}")

            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.input_audio_path),
                '-ss', str(final_gap_start),
                '-to', str(final_gap_end),
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                str(final_gap_path)
            ]
            self._run_command(cmd, description="Extracting final gap from original audio")

    def process_audio_segments(self):
        segments = self.segments_data.get('segments', [])

        for i, segment in enumerate(segments):
            input_path = self.audio_segments_dir / f"audio_segment_{i:04d}.wav"
            output_path = self.processed_audio_segments_dir / f"processed_audio_segment_{i:04d}.wav"

            if not os.path.exists(input_path):
                self.log.error(f"Audio segment file not found: {input_path}")
                continue

            original_duration = self._get_audio_duration(input_path)
            target_duration = segment['tts_duration']

            if original_duration <= 0 or target_duration <= 0:
                self.log.warning(f"Invalid durations for segment {i}: orig={original_duration}, target={target_duration}")
                continue

            if abs(original_duration - target_duration) < 0.01:
                self.log.info(f"Audio segment {i}: minimal duration change, copying file")
                shutil.copy(str(input_path), str(output_path))
                continue

            self.log.info(
                f"Audio segment {i}: precise stretching from {original_duration:.6f}s to {target_duration:.6f}s")

            try:
                speed_factor = original_duration / target_duration

                cmd_sox = [
                    'sox', str(input_path), str(output_path),
                    # 'speed', str(speed_factor)
                    'tempo', str(speed_factor)
                ]

                try:
                    self._run_command(cmd_sox, description="Using SOX to stretch audio segment extracted from original audio")

                    actual_duration = self._get_audio_duration(output_path)
                    duration_diff = abs(actual_duration - target_duration)

                    self.log.info(f"Audio segment {i}: SOX result = {actual_duration:.6f}s (diff: {duration_diff:.6f}s)")

                    if duration_diff > 0.005:
                        self.log.info(f"Applying micro-correction for segment {i}")
                        self._apply_micro_correction_wav(output_path, target_duration, i)

                        final_duration = self._get_audio_duration(output_path)
                        self.log.info(f"Audio segment {i}: After micro-correction = {final_duration:.6f}s")

                except FileNotFoundError:
                    self.log.warning(f"SOX not found, using ffmpeg for segment {i}")
                    self._precise_ffmpeg_stretch_wav(input_path, output_path, speed_factor)

                final_duration = self._get_audio_duration(output_path)
                final_diff = final_duration - target_duration
                self.log.info(f"Audio segment {i}: FINAL RESULT = {final_duration:.6f}s (diff: {final_diff:+.6f}s)")

            except Exception as e:
                self.log.error(f"Error processing segment {i}: {e}")
                shutil.copy(str(input_path), str(output_path))

    def _precise_ffmpeg_stretch_wav(self, input_wav, output_wav, speed_factor):
        if speed_factor < 0.5:
            filter_chain = f'atempo=0.5,atempo={speed_factor / 0.5}'
        elif speed_factor > 2.0:
            filter_chain = f'atempo=2.0,atempo={speed_factor / 2.0}'
        else:
            filter_chain = f'atempo={speed_factor}'

        cmd = [
            'ffmpeg', '-y',
            '-i', str(input_wav),
            '-filter:a', filter_chain,
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            str(output_wav)
        ]
        self._run_command(cmd, description="Stretching audio segment extracted from original audio")

    def _apply_micro_correction_wav(self, wav_file, target_duration, segment_id):
        actual_duration = self._get_audio_duration(wav_file)
        duration_diff = target_duration - actual_duration

        if abs(duration_diff) < 0.001:
            return

        temp_corrected = self.temp_dir / f"micro_corrected_{segment_id}.wav"

        if duration_diff > 0:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(wav_file),
                '-filter:a', f'apad=pad_dur={duration_diff}',
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                str(temp_corrected)
            ]
        else:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(wav_file),
                '-t', str(target_duration),
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                str(temp_corrected)
            ]

        self._run_command(cmd,
                          description="Applying micro correction for stretched audio segment extracted from original audio")

        shutil.move(str(temp_corrected), str(wav_file))

    def combine_background_audio(self):
        segments = self.segments_data.get('segments', [])

        input_files = []

        initial_gap_path = self.audio_gaps_dir / "audio_gap_start_0000.wav"  # WAV
        if initial_gap_path.exists():
            input_files.append(str(initial_gap_path))

        for i in range(len(segments)):
            segment_path = self.processed_audio_segments_dir / f"processed_audio_segment_{i:04d}.wav"  # WAV
            if os.path.exists(segment_path):
                input_files.append(str(segment_path))

            gap_path = self.audio_gaps_dir / f"audio_gap_{i:04d}_{i + 1:04d}.wav"  # WAV
            if gap_path.exists():
                input_files.append(str(gap_path))

        final_gap_path = self.audio_gaps_dir / f"audio_gap_{len(segments) - 1:04d}_end.wav"  # WAV
        if final_gap_path.exists():
            input_files.append(str(final_gap_path))

        if not input_files:
            self.log.error("No audio files to combine")
            return None

        concat_file = self.temp_dir / "audio_concat_list.txt"
        with open(concat_file, 'w') as f:
            for file_path in input_files:
                abs_path = os.path.abspath(file_path)
                f.write(f"file '{abs_path}'\n")

        background_audio_wav = self.temp_dir / "background_audio.wav"
        background_audio_mp3 = self.temp_dir / "background_audio.mp3"

        self.log.info(f"Combining {len(input_files)} audio parts into background track")

        cmd_concat = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            str(background_audio_wav)
        ]
        self._run_command(cmd_concat, description="Concatenating processed audio segments extracted from original audio (WAV)")

        cmd_mp3 = [
            'ffmpeg', '-y',
            '-i', str(background_audio_wav),
            '-acodec', 'mp3',
            '-b:a', '320k',
            str(background_audio_mp3)
        ]
        self._run_command(cmd_mp3, description="Concatenating processed audio segments extracted from original audio (MP3)")

        if os.path.exists(background_audio_mp3):
            duration = self._get_audio_duration(background_audio_mp3)

            tts_duration = sum(seg['tts_duration'] for seg in segments)

            gaps_duration = 0.0

            if segments and segments[0]['start'] > 0.01:
                gaps_duration += segments[0]['start']

            for i in range(len(segments) - 1):
                current_end = segments[i]['end']
                next_start = segments[i + 1]['start']
                if next_start > current_end:
                    gaps_duration += (next_start - current_end)

            total_original_duration = self._get_audio_duration(self.input_audio_path)
            if segments and segments[-1]['end'] < total_original_duration - 0.01:
                gaps_duration += (total_original_duration - segments[-1]['end'])

            expected_duration = tts_duration + gaps_duration
            duration_diff = duration - expected_duration

            self.log.info(f"Background audio created! Duration: {duration:.4f} sec")
            self.log.info(f"TTS duration: {tts_duration:.4f} sec")
            self.log.info(f"Gaps duration: {gaps_duration:.4f} sec")
            self.log.info(f"Expected total: {expected_duration:.4f} sec")
            self.log.info(f"Duration difference: {duration_diff:+.4f} sec ({duration_diff * 1000:+.1f} ms)")

            if abs(duration_diff) > 0.1:
                self.log.warning(f"WARNING: Background audio duration differs by more than 100ms!")

            return str(background_audio_mp3)
        else:
            self.log.error("Failed to create background audio")
            return None

    def cleanup(self):
        try:
            shutil.rmtree(self.temp_dir)
            self.log.info("AudioProcessor temporary files cleaned up")
        except Exception as e:
            self.log.warning(f"Failed to cleanup AudioProcessor temp files: {e}")
