import os
import json
import subprocess
from pathlib import Path
import shutil


class VideoProcessor:
    def __init__(self, input_video_path, json_path, output_video_path, intro_outro_path, target_fps=25):
        """
        Инициализация процессора видео

        :param input_video_path: Путь к исходному видео
        :param json_path: Путь к JSON-файлу с сегментами
        :param output_video_path: Путь для сохранения результата
        :param intro_outro_path: Путь к файлу с интро/аутро
        :param target_fps: Целевой FPS (по умолчанию 25)
        """
        self.input_video_path = input_video_path
        self.json_path = json_path
        self.output_video_path = output_video_path
        self.intro_outro_path = intro_outro_path
        self.target_fps = target_fps

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
        """Безопасное выполнение внешней команды с выводом журнала"""
        try:
            print(f"Выполнение команды: {' '.join(map(str, cmd))}")
            result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)

            if result.returncode != 0:
                print(f"Ошибка при выполнении команды. Код возврата: {result.returncode}")
                print(f"Стандартный вывод: {result.stdout}")
                print(f"Вывод ошибок: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd,
                                                    result.stdout, result.stderr)
            return result
        except Exception as e:
            print(f"Исключение при выполнении команды: {e}")
            raise

    def _check_gpu_availability(self):
        """Проверяет доступность NVIDIA GPU для кодирования видео"""
        # Если мы уже проверяли и знаем результат, используем его
        if hasattr(self, '_gpu_available'):
            return self._gpu_available

        try:
            # Запускаем FFmpeg с запросом доступных кодеков
            cmd = ['ffmpeg', '-encoders']
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Проверяем наличие h264_nvenc среди доступных кодеков
            if 'h264_nvenc' in result.stdout:
                # Пытаемся выполнить тестовое кодирование
                test_cmd = [
                    'ffmpeg',
                    '-f', 'lavfi',
                    '-i', 'nullsrc=s=640x480:d=1',
                    '-c:v', 'h264_nvenc',
                    '-f', 'null',
                    '-'
                ]
                test_result = subprocess.run(test_cmd, capture_output=True, text=True)

                # Если тест прошел без ошибок, значит GPU доступен
                if test_result.returncode == 0:
                    self._gpu_available = True
                    return True

            # Если мы здесь, значит GPU недоступен или не поддерживается
            self._gpu_available = False
            return False
        except Exception as e:
            print(f"Ошибка при проверке GPU: {e}")
            self._gpu_available = False
            return False

    def _get_video_fps(self, video_path):
        """Получение FPS видео"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'json',
                str(video_path)
            ]
            result = self._run_command(cmd)
            data = json.loads(result.stdout)

            if not data.get('streams') or len(data['streams']) == 0:
                raise ValueError(f"Не удалось получить информацию о потоке: {result.stdout}")

            fps_str = data['streams'][0]['r_frame_rate']
            numerator, denominator = map(int, fps_str.split('/'))
            return numerator / denominator
        except Exception as e:
            print(f"Ошибка при получении FPS: {e}")
            # Возвращаем значение по умолчанию в случае ошибки
            print(f"Используем значение FPS по умолчанию: 25")
            return 25.0

    def _get_video_duration(self, video_path):
        """Получение длительности видео в секундах"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json',
                str(video_path)  # Преобразуем в строку для безопасности
            ]
            result = self._run_command(cmd)
            data = json.loads(result.stdout)

            if not data.get('format') or 'duration' not in data['format']:
                raise ValueError(f"Не удалось получить информацию о длительности: {result.stdout}")

            return float(data['format']['duration'])
        except Exception as e:
            print(f"Ошибка при получении длительности: {e}")
            # В случае ошибки попробуем альтернативный метод
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

                # Разбор времени в формате HH:MM:SS.MS
                if ':' in time_str:
                    parts = time_str.split(':')
                    if len(parts) == 3:
                        hours, minutes, seconds = parts
                        seconds = float(seconds)
                        return float(hours) * 3600 + float(minutes) * 60 + seconds

                # Если не удалось разобрать, просто преобразуем в число
                return float(time_str)
            except Exception as e2:
                print(f"Альтернативный метод получения длительности также не удался: {e2}")
                # Возвращаем предполагаемое значение
                return 0.0

    def _adjust_duration_for_fps(self, duration):
        """Регулировка длительности для соответствия кадровой частоте"""
        frame_duration = 1.0 / self.target_fps
        frames = round(duration / frame_duration)
        return frames * frame_duration

    def convert_to_target_fps(self):
        """Конвертация исходного видео в целевой FPS без звука"""
        try:
            input_path = str(self.input_video_path)
            output_path = str(self.converted_video_path)

            has_gpu = self._check_gpu_availability()

            if has_gpu:
                print("Используем NVIDIA GPU для ускорения конвертации")
                encoder = 'h264_nvenc'
                # NVENC не поддерживает yuv444p, используем совместимый формат
                pixel_format = 'yuv420p'
                # NVENC не поддерживает crf, используем высокий битрейт
                quality_params = ['-b:v', '20M']
                preset = 'slow'  # NVENC использует другие пресеты
            else:
                print("GPU не обнаружен или не поддерживается. Используем CPU")
                encoder = 'libx264'
                pixel_format = 'yuv444p'
                quality_params = ['-crf', '0']
                preset = 'veryslow'

            if not self.needs_fps_conversion:
                print(f"Исходное видео уже имеет нужную частоту кадров ({self.target_fps} FPS)")
                # Копируем оригинал, но удаляем аудио
                cmd = [
                          'ffmpeg',
                          '-i', input_path,
                          '-an',  # Удаление аудио
                          '-c:v', encoder,
                      ] + quality_params + [
                          '-preset', preset,
                          '-pix_fmt', pixel_format,
                          output_path
                      ]
            else:
                print(f"Конвертация видео из {self.input_fps} FPS в {self.target_fps} FPS")
                cmd = [
                          'ffmpeg',
                          '-i', input_path,
                          '-an',  # Удаление аудио
                          '-c:v', encoder,
                      ] + quality_params + [
                          '-preset', preset,
                          '-pix_fmt', pixel_format,
                          '-r', str(self.target_fps),
                          output_path
                      ]

            self._run_command(cmd)

            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Конвертированный файл не был создан: {output_path}")

            actual_fps = self._get_video_fps(output_path)
            print(f"Проверка FPS конвертированного файла: {actual_fps}")

            if abs(actual_fps - self.target_fps) > 0.1:
                print(
                    f"Предупреждение: FPS конвертированного файла ({actual_fps}) отличается от целевого ({self.target_fps})")

            return output_path
        except Exception as e:
            print(f"Ошибка при конвертации видео: {e}")
            # Если ошибка связана с GPU, пробуем fallback на CPU
            if has_gpu and ("NVENC" in str(e) or "GPU" in str(e) or "nvenc" in str(e)):
                print("Ошибка при использовании GPU. Пробуем конвертацию на CPU...")
                # Устанавливаем флаг, что GPU недоступен
                self._gpu_available = False
                # Рекурсивно вызываем ту же функцию, которая теперь будет использовать CPU
                return self.convert_to_target_fps()
            raise

    def extract_segments(self):
        """Извлечение сегментов и промежутков (gaps) из видео"""
        segments = self.data.get('segments', [])
        video_path = self.converted_video_path

        for i, segment in enumerate(segments):
            start = segment['start']
            end = segment['end']
            output_path = self.segments_dir / f"segment_{i:04d}.mp4"

            # Извлечение сегмента
            # # Faster, but with lower quality
            # cmd = [
            #     'ffmpeg',
            #     '-i', video_path,
            #     '-ss', str(start),
            #     '-to', str(end),
            #     '-c:v', 'libx264',
            #     '-crf', '18',
            #     '-preset', 'veryfast',
            #     '-an',
            #     output_path
            # ]

            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(start),
                '-to', str(end),
                '-c:v', 'libx264',
                '-crf', '0',
                '-preset', 'veryslow',
                '-pix_fmt', 'yuv444p',
                '-an',
                output_path
            ]

            subprocess.run(cmd, check=True)

            # Проверка на наличие промежутка (gap) перед следующим сегментом
            if i + 1 < len(segments):
                next_start = segments[i + 1]['start']
                if next_start > end:
                    gap_start = end
                    gap_end = next_start
                    gap_path = self.gaps_dir / f"gap_{i:04d}_{i + 1:04d}.mp4"

                    # Извлечение промежутка
                    cmd = [
                        'ffmpeg',
                        '-i', video_path,
                        '-ss', str(gap_start),
                        '-to', str(gap_end),
                        '-c:v', 'libx264',
                        '-crf', '18',
                        '-preset', 'veryfast',
                        '-an',
                        gap_path
                    ]
                    subprocess.run(cmd, check=True)

    def process_segments(self):
        """Обработка всех сегментов с изменением их длительности"""
        segments = self.data.get('segments', [])

        for i, segment in enumerate(segments):
            try:
                input_path = self.segments_dir / f"segment_{i:04d}.mp4"
                output_path = self.processed_segments_dir / f"processed_segment_{i:04d}.mp4"

                # Проверка существования входного файла
                if not os.path.exists(input_path):
                    print(f"Предупреждение: Файл сегмента не найден: {input_path}")
                    continue

                # Получение оригинальной длительности из JSON и корректировка tts_duration
                original_duration = self._get_video_duration(input_path)
                target_duration = segment['tts_duration']
                adjusted_target_duration = self._adjust_duration_for_fps(target_duration)

                # Проверка на отрицательную или нулевую длительность
                if original_duration <= 0:
                    print(f"Ошибка: Оригинальная длительность сегмента {i} равна или меньше нуля: {original_duration}")
                    continue

                if adjusted_target_duration <= 0:
                    print(
                        f"Ошибка: Целевая длительность сегмента {i} равна или меньше нуля: {adjusted_target_duration}")
                    continue

                # Если исходная длительность примерно равна целевой, просто копируем файл
                if abs(original_duration - adjusted_target_duration) < 0.04:  # Разница меньше 1 кадра
                    print(f"  Незначительное изменение длительности. Копирование файла.")
                    shutil.copy(str(input_path), str(output_path))
                    actual_duration = self._get_video_duration(output_path)
                else:
                    # Используем прямой метод с установкой точной длительности для всех сегментов
                    print(f"  Установка точной длительности для сегмента {i}...")
                    speed_factor = adjusted_target_duration / original_duration

                    # Используем комбинацию изменения скорости и точной длительности для лучшего качества
                    # # Faster, but with lower quality
                    # cmd = [
                    #     'ffmpeg',
                    #     '-i', str(input_path),
                    #     '-filter:v', f'setpts={speed_factor}*PTS,fps={self.target_fps}',
                    #     # Изменяем скорость и гарантируем точный FPS
                    #     '-r', str(self.target_fps),
                    #     '-c:v', 'libx264',
                    #     '-crf', '18',
                    #     '-preset', 'medium',
                    #     '-an',
                    #     '-t', str(adjusted_target_duration),
                    #     str(output_path)
                    # ]

                    cmd = [
                        'ffmpeg',
                        '-i', str(input_path),
                        '-filter:v', f'setpts={speed_factor}*PTS,fps={self.target_fps}',
                        '-r', str(self.target_fps),
                        '-c:v', 'libx264',
                        '-crf', '0',
                        '-preset', 'veryslow',
                        '-pix_fmt', 'yuv444p',
                        '-an',
                        '-t', str(adjusted_target_duration),
                        str(output_path)
                    ]

                    self._run_command(cmd)

                    # Проверка, что файл был создан
                    if not os.path.exists(output_path):
                        print(f"Ошибка: Обработанный файл не был создан: {output_path}")
                        continue

                    actual_duration = self._get_video_duration(output_path)

                # Вывод информации о результатах
                duration_diff = abs(actual_duration - adjusted_target_duration)

                print(f"Сегмент {i}: исходная длительность = {original_duration:.4f} сек, "
                      f"целевая = {adjusted_target_duration:.4f} сек (tts = {target_duration:.4f} сек), "
                      f"фактическая = {actual_duration:.4f} сек, "
                      f"коэффициент скорости = {adjusted_target_duration / original_duration:.4f}")

                if duration_diff > 0.04:  # Если отклонение больше 1 кадра
                    print(f"  Предупреждение: Отклонение от целевой длительности: {duration_diff:.4f} сек")

            except Exception as e:
                print(f"Ошибка при обработке сегмента {i}: {e}")
                # Продолжаем со следующим сегментом вместо полной остановки

    def combine_final_video_reliable(self):
        """Надежный метод объединения видео через промежуточные изображения"""
        segments = self.data.get('segments', [])
        frames_dir = self.temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        # Временный путь для выходного файла
        temp_output = self.temp_dir / "temp_output.mp4"

        print("Применение надежного метода конкатенации видео...")
        print("Этот метод займет больше времени, но гарантирует отсутствие артефактов")

        # Шаг 1: Получаем список всех входных файлов в правильном порядке
        input_files = []

        # Добавляем интро
        input_files.append((str(self.intro_outro_path), "intro"))

        # Добавляем сегменты и промежутки
        for i in range(len(segments)):
            # Добавляем обработанный сегмент
            segment_path = self.processed_segments_dir / f"processed_segment_{i:04d}.mp4"
            if os.path.exists(segment_path):
                input_files.append((str(segment_path), f"segment_{i:04d}"))
            else:
                print(f"Предупреждение: Обработанный сегмент не найден: {segment_path}")

            # Проверка на наличие промежутка (gap) после сегмента
            gap_path = self.gaps_dir / f"gap_{i:04d}_{i + 1:04d}.mp4"
            if gap_path.exists():
                input_files.append((str(gap_path), f"gap_{i:04d}_{i + 1:04d}"))

        # Добавляем аутро
        input_files.append((str(self.intro_outro_path), "outro"))

        print(f"Всего файлов для объединения: {len(input_files)}")

        # Шаг 2: Создаем файл списка для прямого объединения кадров
        frame_list_path = self.temp_dir / "frame_list.txt"
        frame_count = 0
        last_frame = None

        with open(frame_list_path, 'w') as frame_list:
            for idx, (file_path, file_id) in enumerate(input_files):
                print(f"Обработка файла {idx + 1}/{len(input_files)}: {file_id}")

                # Проверяем длительность и fps файла
                try:
                    # Получаем информацию о файле
                    file_fps = self._get_video_fps(file_path)
                    file_duration = self._get_video_duration(file_path)
                    frame_count_in_file = int(file_duration * self.target_fps)

                    print(
                        f"  Длительность: {file_duration:.4f} сек, FPS: {file_fps}, примерное количество кадров: {frame_count_in_file}")

                    # Извлекаем каждый кадр из файла
                    output_pattern = frames_dir / f"{file_id}_%05d.png"

                    cmd = [
                        'ffmpeg',
                        '-i', file_path,
                        '-vf', f'fps={self.target_fps}',
                        '-q:v', '1',  # Максимальное качество
                        str(output_pattern)
                    ]

                    self._run_command(cmd)

                    # Находим все извлеченные кадры и добавляем их в список
                    extracted_frames = sorted(list(frames_dir.glob(f"{file_id}_*.png")))

                    if not extracted_frames:
                        print(f"  Предупреждение: Кадры не были извлечены из {file_id}")
                        continue

                    print(f"  Извлечено кадров: {len(extracted_frames)}")

                    # Добавляем кадры в список
                    for frame_path in extracted_frames:
                        frame_list.write(f"file '{frame_path}'\n")
                        frame_list.write(f"duration {1.0 / self.target_fps}\n")
                        frame_count += 1
                        last_frame = frame_path

                except Exception as e:
                    print(f"  Ошибка при обработке файла {file_id}: {e}")

            # Добавляем последний кадр без длительности (требование FFmpeg)
            if last_frame:
                frame_list.write(f"file '{last_frame}'\n")

        # Шаг 3: Собираем видео из кадров
        print(f"Собираем видео из {frame_count} кадров...")

        # # Faster, but with lower quality
        # cmd = [
        #     'ffmpeg',
        #     '-f', 'concat',
        #     '-safe', '0',
        #     '-i', str(frame_list_path),
        #     '-vsync', 'vfr',
        #     '-pix_fmt', 'yuv420p',
        #     '-c:v', 'libx264',
        #     '-crf', '18',
        #     '-preset', 'medium',
        #     '-movflags', '+faststart',
        #     str(temp_output)
        # ]

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
            '-movflags', '+faststart',
            str(temp_output)
        ]

        self._run_command(cmd)

        if os.path.exists(temp_output):
            video_duration = self._get_video_duration(temp_output)
            print(f"Видео успешно создано! Длительность: {video_duration:.4f} сек")

            shutil.copy(temp_output, self.output_video_path)
            print(f"Результат сохранен в {self.output_video_path}")
            return True
        else:
            print("Ошибка: Не удалось создать итоговое видео!")
            return False

    def cleanup(self):
        """Очистка временных файлов"""
        try:
            shutil.rmtree(self.temp_dir)
            print(f"Временные файлы удалены: {self.temp_dir}")
        except Exception as e:
            print(f"Ошибка при удалении временных файлов: {e}")

    def process(self):
        """Выполнение полной последовательности обработки"""
        try:
            print("1. Конвертация видео в целевой FPS...")
            self.convert_to_target_fps()

            print("2. Извлечение сегментов и промежутков...")
            self.extract_segments()

            print("3. Обработка сегментов с изменением длительности...")
            self.process_segments()

            print("4. Объединение финального видео надежным методом...")
            self.combine_final_video_reliable()

            print(f"Готово! Результат сохранен в {self.output_video_path}")

            self.cleanup()

            return True
        except Exception as e:
            print(f"Ошибка при обработке: {e}")
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

    print(f"Рабочая директория: {current_dir}")
    print(f"Директория с входными файлами: {input_dir}")
    print(f"Директория с ресурсами: {resources_dir}")

    if not os.path.exists(input_video):
        print(f"Ошибка: Входное видео не найдено: {input_video}")
        return

    if not os.path.exists(json_file):
        print(f"Ошибка: JSON-файл не найден: {json_file}")
        return

    if not os.path.exists(intro_outro):
        print(f"Ошибка: Файл интро/аутро не найден: {intro_outro}")
        print(f"Проверка содержимого директории ресурсов:")
        try:
            for file in os.listdir(resources_dir):
                print(f"  - {file}")
        except Exception as e:
            print(f"  Ошибка при чтении директории: {e}")
        return

    print(f"Все необходимые файлы найдены:")
    print(f"  - Входное видео: {input_video}")
    print(f"  - JSON-файл: {json_file}")
    print(f"  - Интро/аутро: {intro_outro}")

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