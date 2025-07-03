import json
import librosa
import numpy as np
from pathlib import Path
import argparse
import warnings

warnings.filterwarnings("ignore")


def analyze_voice_frequency_tuned(audio_path: str, start_time: float, end_time: float, sr: int = 16000):
    """
    Анализирует основную частоту голоса с улучшенными порогами
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
                "f0_min": 0,
                "f0_max": 0,
                "voiced_ratio": 0,
                "gender": "unknown",
                "confidence": 0,
                "decision_reason": "no_voice_detected"
            }

        # Вычисляем статистики
        f0_mean = np.mean(voiced_f0)
        f0_median = np.median(voiced_f0)
        f0_std = np.std(voiced_f0)
        f0_min = np.min(voiced_f0)
        f0_max = np.max(voiced_f0)
        voiced_ratio = len(voiced_f0) / len(f0)

        # НОВЫЕ УЛУЧШЕННЫЕ ПОРОГИ
        # Смещаем в пользу женских голосов

        gender = "unknown"
        confidence = 0
        decision_reason = ""

        # Очень низкие голоса - точно мужские
        if f0_median < 110:
            gender = "male"
            confidence = min(1.0, (110 - f0_median) / 40)  # чем ниже, тем увереннее
            decision_reason = f"very_low_f0_{f0_median:.1f}Hz"

        # Низкие голоса - скорее мужские
        elif f0_median < 130:
            gender = "male"
            confidence = min(0.8, (130 - f0_median) / 25)
            decision_reason = f"low_f0_{f0_median:.1f}Hz"

        # Средне-низкие голоса - может быть низкий женский
        elif f0_median < 145:
            # Дополнительная проверка по вариативности
            if f0_std > 25:  # высокая вариативность -> женский
                gender = "female"
                confidence = 0.4
                decision_reason = f"medium_low_f0_{f0_median:.1f}Hz_high_variability_{f0_std:.1f}"
            else:
                gender = "male"
                confidence = 0.5
                decision_reason = f"medium_low_f0_{f0_median:.1f}Hz_low_variability_{f0_std:.1f}"

        # Средние голоса - серая зона, больше проверок
        elif f0_median < 165:
            # Анализируем диапазон и вариативность
            f0_range = f0_max - f0_min

            # Если есть высокие пики - вероятно женский
            if f0_max > 200:
                gender = "female"
                confidence = 0.6
                decision_reason = f"medium_f0_{f0_median:.1f}Hz_high_peaks_{f0_max:.1f}Hz"
            # Высокая вариативность - скорее женский
            elif f0_std > 30:
                gender = "female"
                confidence = 0.5
                decision_reason = f"medium_f0_{f0_median:.1f}Hz_high_variability_{f0_std:.1f}"
            # Большой диапазон - скорее женский
            elif f0_range > 80:
                gender = "female"
                confidence = 0.4
                decision_reason = f"medium_f0_{f0_median:.1f}Hz_wide_range_{f0_range:.1f}Hz"
            else:
                gender = "uncertain"
                confidence = 0.2
                decision_reason = f"medium_f0_{f0_median:.1f}Hz_ambiguous"

        # Средне-высокие голоса - скорее женские
        elif f0_median < 185:
            gender = "female"
            confidence = 0.7
            decision_reason = f"medium_high_f0_{f0_median:.1f}Hz"

        # Высокие голоса - женские
        elif f0_median < 220:
            gender = "female"
            confidence = 0.9
            decision_reason = f"high_f0_{f0_median:.1f}Hz"

        # Очень высокие голоса - точно женские
        else:
            gender = "female"
            confidence = 1.0
            decision_reason = f"very_high_f0_{f0_median:.1f}Hz"

        return {
            "f0_mean": float(f0_mean),
            "f0_median": float(f0_median),
            "f0_std": float(f0_std),
            "f0_min": float(f0_min),
            "f0_max": float(f0_max),
            "f0_range": float(f0_max - f0_min),
            "voiced_ratio": float(voiced_ratio),
            "gender": gender,
            "confidence": float(confidence),
            "decision_reason": decision_reason
        }

    except Exception as e:
        print(f"Ошибка анализа сегмента {start_time}-{end_time}: {e}")
        return {
            "f0_mean": 0,
            "f0_median": 0,
            "f0_std": 0,
            "f0_min": 0,
            "f0_max": 0,
            "f0_range": 0,
            "voiced_ratio": 0,
            "gender": "error",
            "confidence": 0,
            "decision_reason": "processing_error"
        }


def add_tuned_gender_to_segments(audio_path: str, segments: list, output_path: str = None):
    """
    Добавляет информацию о поле с настроенными порогами
    """
    try:
        print(f"🎵 Анализирую пол с настроенными порогами для {len(segments)} сегментов...")

        # Анализируем каждый сегмент
        for i, segment in enumerate(segments):
            if i % 10 == 0:
                print(f"  Обработано: {i}/{len(segments)}")

            # Анализируем частоту голоса
            gender_info = analyze_voice_frequency_tuned(
                audio_path,
                segment["start"],
                segment["end"]
            )

            # Добавляем информацию о поле
            segment["tuned_gender_analysis"] = gender_info
            segment["predicted_gender"] = gender_info["gender"]
            segment["gender_confidence"] = gender_info["confidence"]
            segment["decision_reason"] = gender_info["decision_reason"]

        # Выводим статистику
        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТЫ С НАСТРОЕННЫМИ ПОРОГАМИ")
        print("=" * 60)

        gender_stats = {}
        confidence_stats = {"high": 0, "medium": 0, "low": 0}

        for segment in segments:
            gender = segment["predicted_gender"]
            confidence = segment["gender_confidence"]

            if gender not in gender_stats:
                gender_stats[gender] = {
                    "count": 0,
                    "total_duration": 0,
                    "avg_confidence": []
                }

            gender_stats[gender]["count"] += 1
            gender_stats[gender]["total_duration"] += segment.get("original_duration",
                                                                  segment["end"] - segment["start"])
            gender_stats[gender]["avg_confidence"].append(confidence)

            # Статистика по уверенности
            if confidence > 0.7:
                confidence_stats["high"] += 1
            elif confidence > 0.4:
                confidence_stats["medium"] += 1
            else:
                confidence_stats["low"] += 1

        # Выводим результаты
        for gender, stats in gender_stats.items():
            avg_conf = np.mean(stats["avg_confidence"]) if stats["avg_confidence"] else 0
            percentage = (stats["count"] / len(segments)) * 100
            print(f"{gender.upper()}: {stats['count']} сегментов ({percentage:.1f}%) | "
                  f"Длительность: {stats['total_duration']:.1f}s | "
                  f"Средняя уверенность: {avg_conf:.2f}")

        print(f"\nРаспределение по уверенности:")
        print(f"  Высокая (>0.7): {confidence_stats['high']} сегментов")
        print(f"  Средняя (0.4-0.7): {confidence_stats['medium']} сегментов")
        print(f"  Низкая (<0.4): {confidence_stats['low']} сегментов")

        # Показываем примеры решений
        print(f"\n📋 Примеры решений:")
        examples_shown = 0
        for segment in segments:
            if examples_shown >= 10:
                break
            if segment["predicted_gender"] in ["male", "female"]:
                analysis = segment["tuned_gender_analysis"]
                print(f"  {segment['start']:5.1f}s-{segment['end']:5.1f}s | "
                      f"F0: {analysis['f0_median']:.1f}Hz "
                      f"(±{analysis['f0_std']:.1f}, range: {analysis['f0_range']:.1f}) | "
                      f"{segment['predicted_gender']} ({segment['gender_confidence']:.2f}) | "
                      f"{segment['decision_reason']}")
                examples_shown += 1

        # Сохраняем результат
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Результат сохранён в: {output_path}")

        return segments

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Определение пола с настроенными порогами")
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

    # Загружаем сегменты
    try:
        with open(args.segments_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ищем сегменты в разных форматах
        if "segments" in data:
            segments = data["segments"]
        elif isinstance(data, list):
            segments = data
        else:
            print("❌ Не удалось найти сегменты в файле")
            return

        print(f"✅ Найдено {len(segments)} сегментов")

        # Анализируем пол
        result = add_tuned_gender_to_segments(args.audio_path, segments, args.output)

        if result:
            print("\n✅ Анализ с настроенными порогами завершён!")
            print("🎯 Должно быть лучше распознавание женских голосов")

    except Exception as e:
        print(f"❌ Ошибка загрузки сегментов: {e}")


if __name__ == "__main__":
    main()