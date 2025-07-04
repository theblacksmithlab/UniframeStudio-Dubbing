#!/usr/bin/env python3
"""
Тест AudioProcessor независимо от основного проекта
"""

import os
import json
import sys
from pathlib import Path

# Добавляем корневую папку проекта в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.audio_processor import AudioProcessor


def create_test_segments_data():
    """Создает тестовые данные сегментов"""
    return {
        "segments": [
            {
                "id": 0,
                "start": 1.0,
                "end": 4.0,
                "text": "Test segment 1",
                "translated_text": "Тестовый сегмент 1",
                "tts_duration": 2.5  # Короче оригинала (3.0)
            },
            {
                "id": 1,
                "start": 5.0,
                "end": 8.0,
                "text": "Test segment 2",
                "translated_text": "Тестовый сегмент 2",
                "tts_duration": 4.2  # Длиннее оригинала (3.0)
            },
            {
                "id": 2,
                "start": 10.0,
                "end": 12.5,
                "text": "Test segment 3",
                "translated_text": "Тестовый сегмент 3",
                "tts_duration": 2.5  # Точно как оригинал
            }
        ]
    }


def test_audio_processor(wav_file_path, test_job_id="test_job_001"):
    """
    Тестирует AudioProcessor

    :param wav_file_path: Путь к тестовому WAV файлу
    :param test_job_id: ID тестовой задачи
    """

    # Проверяем существование файла
    if not os.path.exists(wav_file_path):
        print(f"❌ Файл не найден: {wav_file_path}")
        print("Создайте тестовый WAV файл или укажите путь к существующему")
        return False

    print(f"🎵 Тестируем AudioProcessor с файлом: {wav_file_path}")
    print(f"📁 Тест job ID: {test_job_id}")

    # Создаем тестовые данные сегментов
    segments_data = create_test_segments_data()

    print("\n📊 Тестовые сегменты:")
    for seg in segments_data["segments"]:
        original_duration = seg["end"] - seg["start"]
        print(f"  Сегмент {seg['id']}: {seg['start']:.1f}s - {seg['end']:.1f}s")
        print(f"    Оригинал: {original_duration:.1f}s → TTS: {seg['tts_duration']:.1f}s")

    # Создаем папку для тестов
    test_jobs_dir = "test_jobs"
    os.makedirs(test_jobs_dir, exist_ok=True)

    try:
        print("\n🔧 Инициализируем AudioProcessor...")
        audio_processor = AudioProcessor(test_job_id, wav_file_path, segments_data)

        print("📤 Извлекаем аудио сегменты...")
        audio_processor.extract_audio_segments()

        print("⚙️  Обрабатываем сегменты (растяжение/сжатие)...")
        audio_processor.process_audio_segments()

        print("🔗 Склеиваем background audio...")
        background_audio_path = audio_processor.combine_background_audio()

        if background_audio_path:
            print(f"✅ Тест успешно завершен!")
            print(f"📁 Background audio создан: {background_audio_path}")

            # Дополнительная информация
            duration = audio_processor._get_audio_duration(background_audio_path)
            expected_duration = sum(seg['tts_duration'] for seg in segments_data['segments'])

            print(f"\n📏 Результаты:")
            print(f"  Финальная длительность: {duration:.4f}s")
            print(f"  Ожидаемая длительность: {expected_duration:.4f}s")
            print(f"  Разница: {duration - expected_duration:+.4f}s")

            if abs(duration - expected_duration) < 0.1:
                print("🎉 Точность отличная! (< 100мс)")
            else:
                print("⚠️  Есть расхождение > 100мс")

            return True
        else:
            print("❌ Не удалось создать background audio")
            return False

    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Опционально - очистка
        try:
            audio_processor.cleanup()
            print("🧹 Временные файлы очищены")
        except:
            pass


def create_test_wav_file(output_path="test_audio.wav", duration=15):
    """Создает тестовый WAV файл"""
    print(f"🎵 Создаем тестовый WAV файл: {output_path}")

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration}",
        "-ar", "44100",
        "-acodec", "pcm_s16le",
        output_path
    ]

    import subprocess
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"✅ Тестовый файл создан: {output_path}")
            return output_path
        else:
            print(f"❌ Ошибка создания файла: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def main():
    print("🧪 Тестирование AudioProcessor")
    print("=" * 50)

    # Опция 1: Создать тестовый файл автоматически
    print("1. Создать тестовый WAV файл автоматически")
    print("2. Использовать существующий файл")

    choice = input("Выберите опцию (1 или 2): ").strip()

    if choice == "1":
        test_wav_path = create_test_wav_file("test_audio.wav", 15)
        if not test_wav_path:
            print("❌ Не удалось создать тестовый файл")
            return
    elif choice == "2":
        test_wav_path = input("Введите путь к WAV файлу: ").strip()
        if not test_wav_path:
            print("❌ Путь не указан")
            return
    else:
        print("❌ Неверный выбор")
        return

    success = test_audio_processor(test_wav_path)

    if success:
        print("\n🎉 Тест завершен успешно!")
        if choice == "1":
            print(f"🗑️  Можете удалить тестовый файл: {test_wav_path}")
    else:
        print("\n💥 Тест провален!")


if __name__ == "__main__":
    main()