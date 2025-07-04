#!/usr/bin/env python3
"""
Тест SOX: сравнение speed vs tempo для растяжения аудио
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Выполнить команду и показать результат"""
    print(f"🔧 {description}")
    print(f"   Команда: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Успешно")
            return True
        else:
            print(f"   ❌ Ошибка: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Исключение: {e}")
        return False


def get_audio_duration(file_path):
    """Получить длительность аудио файла"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except:
        return 0.0


def create_test_audio(output_path="test_voice.wav"):
    """Создать тестовый аудио файл с разными частотами (имитация голоса)"""
    print(f"🎵 Создаем тестовый аудио файл: {output_path}")

    # Создаем более сложный звук похожий на голос (несколько частот)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "sine=frequency=220:duration=3,sine=frequency=440:duration=3,sine=frequency=880:duration=3",
        "-filter_complex", "[0][1][2]amix=inputs=3:duration=longest",
        "-ar", "44100",
        "-acodec", "pcm_s16le",
        output_path
    ]

    return run_command(cmd, "Создание тестового аудио")


def test_sox_methods(input_file):
    """Тестируем разные методы SOX"""

    if not os.path.exists(input_file):
        print(f"❌ Файл не найден: {input_file}")
        return

    original_duration = get_audio_duration(input_file)
    print(f"📏 Оригинальная длительность: {original_duration:.3f}s")

    # Тестовые коэффициенты
    test_factors = [
        (0.75, "замедление на 30%"),
        (1.25, "ускорение на 40%")
    ]

    results = []

    for factor, description in test_factors:
        print(f"\n{'=' * 50}")
        print(f"🧪 Тестируем {description} (фактор: {factor})")
        print(f"{'=' * 50}")

        expected_duration = original_duration / factor
        print(f"📐 Ожидаемая длительность: {expected_duration:.3f}s")

        # Тест 1: SOX speed (изменяет тон)
        speed_output = f"test_speed_{factor}.wav"
        cmd_speed = ["sox", input_file, speed_output, "speed", str(factor)]

        if run_command(cmd_speed, f"SOX speed {factor}"):
            speed_duration = get_audio_duration(speed_output)
            speed_diff = abs(speed_duration - expected_duration)
            print(f"   📏 Результат: {speed_duration:.3f}s (точность: ±{speed_diff:.3f}s)")
            results.append(("speed", factor, speed_duration, speed_diff, "ИЗМЕНЯЕТ ТОН"))

        # Тест 2: SOX tempo (сохраняет тон)
        tempo_output = f"test_tempo_{factor}.wav"
        cmd_tempo = ["sox", input_file, tempo_output, "tempo", str(factor)]

        if run_command(cmd_tempo, f"SOX tempo {factor}"):
            tempo_duration = get_audio_duration(tempo_output)
            tempo_diff = abs(tempo_duration - expected_duration)
            print(f"   📏 Результат: {tempo_duration:.3f}s (точность: ±{tempo_diff:.3f}s)")
            results.append(("tempo", factor, tempo_duration, tempo_diff, "СОХРАНЯЕТ ТОН"))

        # Тест 3: ffmpeg atempo (для сравнения)
        atempo_output = f"test_atempo_{factor}.wav"
        cmd_atempo = [
            "ffmpeg", "-y", "-i", input_file,
            "-filter:a", f"atempo={factor}",
            atempo_output
        ]

        if run_command(cmd_atempo, f"ffmpeg atempo {factor}"):
            atempo_duration = get_audio_duration(atempo_output)
            atempo_diff = abs(atempo_duration - expected_duration)
            print(f"   📏 Результат: {atempo_duration:.3f}s (точность: ±{atempo_diff:.3f}s)")
            results.append(("atempo", factor, atempo_duration, atempo_diff, "СОХРАНЯЕТ ТОН"))

    # Итоговое сравнение
    print(f"\n{'=' * 60}")
    print("📊 ИТОГОВОЕ СРАВНЕНИЕ")
    print(f"{'=' * 60}")
    print(f"{'Метод':<10} {'Фактор':<8} {'Длительность':<12} {'Точность':<10} {'Тон'}")
    print("-" * 60)

    for method, factor, duration, diff, tone_info in results:
        print(f"{method:<10} {factor:<8} {duration:<12.3f} ±{diff:<9.3f} {tone_info}")

    print(f"\n🎧 РЕКОМЕНДАЦИЯ ДЛЯ ПРОСЛУШИВАНИЯ:")
    print(f"   Откройте созданные файлы и сравните качество звука:")
    for method, factor, _, _, _ in results:
        filename = f"test_{method}_{factor}.wav"
        if os.path.exists(filename):
            print(f"   - {filename}")


def main():
    print("🧪 Тестирование методов растяжения аудио")
    print("=" * 50)

    # Создаем тестовый файл
    test_file = "audio_test.wav"

    if not os.path.exists(test_file):
        if not create_test_audio(test_file):
            print("❌ Не удалось создать тестовый файл")
            return

    print(f"\n✅ Используем файл: {test_file}")

    # Тестируем методы
    test_sox_methods(test_file)

    print(f"\n🎉 Тестирование завершено!")
    print(f"💡 Прослушайте созданные файлы и выберите лучший метод")


if __name__ == "__main__":
    main()