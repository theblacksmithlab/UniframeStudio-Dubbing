#!/usr/bin/env python3
"""
Определение пола спикера по частоте голоса (F0)
"""

import json
import librosa
import numpy as np
from pathlib import Path
import argparse
import warnings

warnings.filterwarnings("ignore")


def analyze_voice_frequency(audio_path: str, start_time: float, end_time: float, sr: int = 16000):
    """
    Анализирует основную частоту голоса (F0) для определения пола

    Args:
        audio_path: путь к аудио файлу
        start_time: начало сегмента в секундах
        end_time: конец сегмента в секундах
        sr: частота дискретизации

    Returns:
        dict: информация о частоте и предполагаемом поле
    """
    try:
        # Загружаем аудио
        y, sr = librosa.load(audio_path, sr=sr)

        # Вырезаем нужный сегмент
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        # Извлекаем основную частоту (F0)
        f0 = librosa.yin(segment,
                         fmin=50,  # минимальная частота (Hz)
                         fmax=400,  # максимальная частота (Hz)
                         sr=sr)

        # Убираем нулевые значения (безголосые участки)
        voiced_f0 = f0[f0 > 0]

        if len(voiced_f0) == 0:
            return {
                "f0_mean": 0,
                "f0_median": 0,
                "f0_std": 0,
                "voiced_ratio": 0,
                "gender": "unknown",
                "confidence": 0
            }

        # Вычисляем статистики
        f0_mean = np.mean(voiced_f0)
        f0_median = np.median(voiced_f0)
        f0_std = np.std(voiced_f0)
        voiced_ratio = len(voiced_f0) / len(f0)

        # Определяем пол на основе частоты
        # Типичные диапазоны:
        # Мужчины: 85-180 Hz (среднее ~120 Hz)
        # Женщины: 165-265 Hz (среднее ~210 Hz)

        gender = "unknown"
        confidence = 0

        if f0_median < 140:
            gender = "male"
            # Чем ниже частота, тем выше уверенность
            confidence = min(1.0, (140 - f0_median) / 55)
        elif f0_median > 160:
            gender = "female"
            # Чем выше частота, тем выше уверенность
            confidence = min(1.0, (f0_median - 160) / 105)
        else:
            # Переходная зона 140-160 Hz
            gender = "uncertain"
            confidence = 0.3

        return {
            "f0_mean": float(f0_mean),
            "f0_median": float(f0_median),
            "f0_std": float(f0_std),
            "voiced_ratio": float(voiced_ratio),
            "gender": gender,
            "confidence": float(confidence)
        }

    except Exception as e:
        print(f"Ошибка анализа сегмента {start_time}-{end_time}: {e}")
        return {
            "f0_mean": 0,
            "f0_median": 0,
            "f0_std": 0,
            "voiced_ratio": 0,
            "gender": "error",
            "confidence": 0
        }


def add_gender_to_segments(audio_path: str, segments: list, output_path: str = None):
    """
    Добавляет информацию о поле к сегментам транскрипции

    Args:
        audio_path: путь к аудио файлу
        segments: список сегментов с временными метками
        output_path: путь для сохранения результата
    """
    try:
        print(f"🎵 Анализирую пол спикеров в {len(segments)} сегментах...")

        # Анализируем каждый сегмент
        for i, segment in enumerate(segments):
            if i % 10 == 0:
                print(f"  Обработано: {i}/{len(segments)}")

            # Анализируем частоту голоса
            gender_info = analyze_voice_frequency(
                audio_path,
                segment["start"],
                segment["end"]
            )

            # Добавляем информацию о поле
            segment["gender_analysis"] = gender_info
            segment["predicted_gender"] = gender_info["gender"]
            segment["gender_confidence"] = gender_info["confidence"]

        # Выводим статистику
        print("\n" + "=" * 50)
        print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА ПОЛА")
        print("=" * 50)

        gender_stats = {}
        for segment in segments:
            gender = segment["predicted_gender"]
            if gender not in gender_stats:
                gender_stats[gender] = {"count": 0, "total_duration": 0}
            gender_stats[gender]["count"] += 1
            gender_stats[gender]["total_duration"] += segment.get("original_duration", 0)

        for gender, stats in gender_stats.items():
            print(f"{gender.upper()}: {stats['count']} сегментов, "
                  f"{stats['total_duration']:.1f}s общей длительности")

        # Показываем примеры
        print(f"\n📋 Примеры анализа (первые 10 сегментов):")
        for i, segment in enumerate(segments[:10]):
            gender_info = segment["gender_analysis"]
            print(f"  {i + 1:2d}. {segment['start']:6.2f}s - {segment['end']:6.2f}s | "
                  f"F0: {gender_info['f0_median']:.1f}Hz | "
                  f"Пол: {segment['predicted_gender']} ({segment['gender_confidence']:.2f})")

        # Сохраняем результат
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Результат сохранён в: {output_path}")

        return segments

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def test_gender_detection(audio_path: str, segments_path: str, output_path: str = None):
    """
    Тестирует определение пола на реальных данных
    """
    try:
        print(f"🔄 Загружаю сегменты из: {segments_path}")

        # Загружаем сегменты
        with open(segments_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ищем сегменты в разных форматах
        segments = None
        if "segments" in data:
            segments = data["segments"]
        elif isinstance(data, list):
            segments = data
        else:
            print("❌ Не удалось найти сегменты в файле")
            return None

        print(f"✅ Найдено {len(segments)} сегментов")

        # Анализируем пол
        result = add_gender_to_segments(audio_path, segments, output_path)

        if result:
            print("\n✅ Анализ пола завершён успешно!")
            print("💡 Теперь можно использовать predicted_gender для выбора голоса TTS")

        return result

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Определение пола по частоте голоса")
    parser.add_argument("audio_path", help="Путь к аудио файлу")
    parser.add_argument("segments_path", help="Путь к JSON файлу с сегментами")
    parser.add_argument("-o", "--output", help="Путь для сохранения результата")

    args = parser.parse_args()

    # Проверяем наличие файлов
    if not Path(args.audio_path).exists():
        print(f"❌ Аудио файл не найден: {args.audio_path}")
        return

    if not Path(args.segments_path).exists():
        print(f"❌ Файл сегментов не найден: {args.segments_path}")
        return

    # Тестируем определение пола
    result = test_gender_detection(args.audio_path, args.segments_path, args.output)


if __name__ == "__main__":
    main()