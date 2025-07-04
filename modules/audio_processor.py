import os
import json
import subprocess
from pathlib import Path
import shutil
from utils.logger_config import setup_logger

logger = setup_logger(name=__name__, log_file="logs/app.log")

class AudioProcessor:
    def __init__(self, job_id, input_video_path, segments_data):
        """
        Обработчик аудио для создания фоновой подложки

        :param job_id: ID задачи
        :param input_video_path: Путь к исходному видео
        :param segments_data: Данные сегментов с tts_duration
        """
        self.job_id = job_id
        self.input_video_path = input_video_path
        self.segments_data = segments_data

        # Создаем структуру папок
        self.temp_dir = Path(f"jobs/{job_id}/temp_audio_processing")
        self.audio_segments_dir = self.temp_dir / "audio_segments"
        self.processed_audio_segments_dir = self.temp_dir / "processed_audio_segments"
        self.audio_gaps_dir = self.temp_dir / "audio_gaps"

        # Создаем папки
        self.temp_dir.mkdir(exist_ok=True)
        self.audio_segments_dir.mkdir(exist_ok=True)
        self.processed_audio_segments_dir.mkdir(exist_ok=True)
        self.audio_gaps_dir.mkdir(exist_ok=True)

        logger.info(f"AudioProcessor initialized for job {job_id}")

    def _get_audio_duration(self, audio_path):
        """Получить длительность аудио файла"""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                   '-of', 'csv=p=0', str(audio_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 0.0

    def _run_command(self, cmd):
        """Выполнить команду ffmpeg"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Command failed: {' '.join(cmd)}")
                logger.error(f"Error: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd)
            return True
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise

    def extract_audio_segments(self):
        """Извлекаем аудио сегменты по тем же временным меткам что и видео"""
        segments = self.segments_data.get('segments', [])

        total_duration = self._get_audio_duration(self.input_video_path)
        logger.info(f"Total audio duration: {total_duration:.4f} seconds")

        # Извлекаем начальный gap если есть
        if segments and segments[0]['start'] > 0.01:
            initial_gap_start = 0.0
            initial_gap_end = segments[0]['start']
            initial_gap_path = self.audio_gaps_dir / "audio_gap_start_0000.mp3"

            logger.info(f"Extracting initial audio gap: {initial_gap_start} - {initial_gap_end}")

            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.input_video_path),
                '-ss', str(initial_gap_start),
                '-to', str(initial_gap_end),
                '-vn',  # без видео
                '-acodec', 'mp3',
                '-b:a', '192k',
                str(initial_gap_path)
            ]
            self._run_command(cmd)

        # Извлекаем аудио сегменты
        for i, segment in enumerate(segments):
            start = segment['start']
            end = segment['end']
            output_path = self.audio_segments_dir / f"audio_segment_{i:04d}.mp3"

            logger.info(f"Extracting audio segment {i}: {start} - {end}")

            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.input_video_path),
                '-ss', str(start),
                '-to', str(end),
                '-vn',  # без видео
                '-acodec', 'mp3',
                '-b:a', '192k',
                str(output_path)
            ]
            self._run_command(cmd)

            # Извлекаем gap между сегментами
            if i + 1 < len(segments):
                next_start = segments[i + 1]['start']
                if next_start > end:
                    gap_start = end
                    gap_end = next_start
                    gap_path = self.audio_gaps_dir / f"audio_gap_{i:04d}_{i + 1:04d}.mp3"

                    logger.info(f"Extracting audio gap {i}-{i + 1}: {gap_start} - {gap_end}")

                    cmd = [
                        'ffmpeg', '-y',
                        '-i', str(self.input_video_path),
                        '-ss', str(gap_start),
                        '-to', str(gap_end),
                        '-vn',
                        '-acodec', 'mp3',
                        '-b:a', '192k',
                        str(gap_path)
                    ]
                    self._run_command(cmd)

        # Извлекаем финальный gap если есть
        if segments and segments[-1]['end'] < total_duration - 0.01:
            final_gap_start = segments[-1]['end']
            final_gap_end = total_duration
            final_gap_path = self.audio_gaps_dir / f"audio_gap_{len(segments) - 1:04d}_end.mp3"

            logger.info(f"Extracting final audio gap: {final_gap_start} - {final_gap_end}")

            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.input_video_path),
                '-ss', str(final_gap_start),
                '-to', str(final_gap_end),
                '-vn',
                '-acodec', 'mp3',
                '-b:a', '192k',
                str(final_gap_path)
            ]
            self._run_command(cmd)

    def process_audio_segments(self):
        """Растягиваем/сжимаем аудио сегменты под tts_duration"""
        segments = self.segments_data.get('segments', [])

        for i, segment in enumerate(segments):
            input_path = self.audio_segments_dir / f"audio_segment_{i:04d}.mp3"
            output_path = self.processed_audio_segments_dir / f"processed_audio_segment_{i:04d}.mp3"

            if not os.path.exists(input_path):
                logger.error(f"Audio segment file not found: {input_path}")
                continue

            original_duration = self._get_audio_duration(input_path)
            target_duration = segment['tts_duration']

            if original_duration <= 0 or target_duration <= 0:
                logger.warning(f"Invalid durations for segment {i}: orig={original_duration}, target={target_duration}")
                continue

            if abs(original_duration - target_duration) < 0.04:  # меньше 1 кадра
                logger.info(f"Audio segment {i}: minimal duration change, copying file")
                shutil.copy(str(input_path), str(output_path))
            else:
                # Растягиваем/сжимаем аудио
                speed_factor = original_duration / target_duration  # обратный коэффициент для atempo

                logger.info(
                    f"Audio segment {i}: stretching from {original_duration:.4f}s to {target_duration:.4f}s (factor: {speed_factor:.4f})")

                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(input_path),
                    '-filter:a', f'atempo={speed_factor}',
                    '-acodec', 'mp3',
                    '-b:a', '192k',
                    str(output_path)
                ]
                self._run_command(cmd)

                actual_duration_after = self._get_audio_duration(str(output_path))
                logger.info(
                    f"Audio segment {i}: RESULT actual={actual_duration_after:.4f}s (diff from target: {actual_duration_after - target_duration:+.4f}s)")

    def combine_background_audio(self):
        """Склеиваем все обработанные аудио сегменты в одну фоновую подложку"""
        segments = self.segments_data.get('segments', [])

        # Собираем список файлов для конкатенации
        input_files = []

        # Начальный gap
        initial_gap_path = self.audio_gaps_dir / "audio_gap_start_0000.mp3"
        if initial_gap_path.exists():
            input_files.append(str(initial_gap_path))

        # Сегменты и gap'ы между ними
        for i in range(len(segments)):
            segment_path = self.processed_audio_segments_dir / f"processed_audio_segment_{i:04d}.mp3"
            if os.path.exists(segment_path):
                input_files.append(str(segment_path))

            gap_path = self.audio_gaps_dir / f"audio_gap_{i:04d}_{i + 1:04d}.mp3"
            if gap_path.exists():
                input_files.append(str(gap_path))

        # Финальный gap
        final_gap_path = self.audio_gaps_dir / f"audio_gap_{len(segments) - 1:04d}_end.mp3"
        if final_gap_path.exists():
            input_files.append(str(final_gap_path))

        if not input_files:
            logger.error("No audio files to combine")
            return None

        # Создаем временный файл со списком файлов для concat
        concat_file = self.temp_dir / "audio_concat_list.txt"
        with open(concat_file, 'w') as f:
            for file_path in input_files:
                abs_path = os.path.abspath(file_path)
                f.write(f"file '{abs_path}'\n")

        # Выходной файл
        background_audio_path = self.temp_dir / "background_audio.mp3"

        logger.info(f"Combining {len(input_files)} audio parts into background track")

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-acodec', 'mp3',
            '-b:a', '192k',
            str(background_audio_path)
        ]
        self._run_command(cmd)

        if os.path.exists(background_audio_path):
            duration = self._get_audio_duration(background_audio_path)
            logger.info(f"Background audio created! Duration: {duration:.4f} sec")
            return str(background_audio_path)
        else:
            logger.error("Failed to create background audio")
            return None

    def cleanup(self):
        """Очистка временных файлов"""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("AudioProcessor temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup AudioProcessor temp files: {e}")
