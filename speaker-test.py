# test!
import json
import librosa
import numpy as np
from pathlib import Path
import argparse
import warnings
from scipy import stats

warnings.filterwarnings("ignore")


def extract_voice_features(audio_path: str, start_time: float, end_time: float, sr: int = 16000):
    """
    Извлекает расширенные голосовые признаки для определения пола

    Args:
        audio_path: путь к аудио файлу
        start_time: начало сегмента в секундах
        end_time: конец сегмента в секундах
        sr: частота дискретизации

    Returns:
        dict: набор признаков для классификации пола
    """
    try:
        # Загружаем аудио
        y, sr = librosa.load(audio_path, sr=sr)

        # Вырезаем нужный сегмент
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        if len(segment) < sr * 0.1:  # Слишком короткий сегмент
            return None

        # 1. ОСНОВНАЯ ЧАСТОТА (F0)
        f0 = librosa.yin(segment, fmin=50, fmax=400, sr=sr)
        voiced_f0 = f0[f0 > 0]

        if len(voiced_f0) == 0:
            return None

        f0_mean = np.mean(voiced_f0)
        f0_median = np.median(voiced_f0)
        f0_std = np.std(voiced_f0)
        f0_range = np.max(voiced_f0) - np.min(voiced_f0)

        # 2. СПЕКТРАЛЬНЫЕ ПРИЗНАКИ
        # Спектральный центроид (центр масс спектра)
        spectral_centroids = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
        centroid_mean = np.mean(spectral_centroids)

        # Спектральная ширина полосы (bandwidth)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)[0]
        bandwidth_mean = np.mean(spectral_bandwidth)

        # Спектральный rolloff (частота, ниже которой 85% энергии)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)[0]
        rolloff_mean = np.mean(spectral_rolloff)

        # 3. МFCC ПРИЗНАКИ
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)

        # 4. ФОРМАНТНЫЕ ПРИЗНАКИ (приблизительные)
        # Анализ через спектрограмму
        D = librosa.stft(segment)
        magnitude = np.abs(D)

        # Ищем пики в спектре (приблизительные форманты)
        freq_bins = librosa.fft_frequencies(sr=sr)
        avg_magnitude = np.mean(magnitude, axis=1)

        # Находим пики в диапазоне формант
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(avg_magnitude, height=np.max(avg_magnitude) * 0.1)
        formant_freqs = freq_bins[peaks]

        # Берем первые 3 форманты
        f1 = formant_freqs[0] if len(formant_freqs) > 0 else 0
        f2 = formant_freqs[1] if len(formant_freqs) > 1 else 0
        f3 = formant_freqs[2] if len(formant_freqs) > 2 else 0

        # 5. ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ
        # Джиттер (вариация периода)
        if len(voiced_f0) > 1:
            periods = 1.0 / voiced_f0
            jitter = np.std(periods) / np.mean(periods) if np.mean(periods) > 0 else 0
        else:
            jitter = 0

        # Шиммер (вариация амплитуды)
        rms = librosa.feature.rms(y=segment)[0]
        shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0

        # Соотношение голос/шум
        voiced_ratio = len(voiced_f0) / len(f0)

        return {
            # Основная частота
            "f0_mean": float(f0_mean),
            "f0_median": float(f0_median),
            "f0_std": float(f0_std),
            "f0_range": float(f0_range),

            # Спектральные признаки
            "spectral_centroid": float(centroid_mean),
            "spectral_bandwidth": float(bandwidth_mean),
            "spectral_rolloff": float(rolloff_mean),

            # MFCC
            "mfcc_1": float(mfcc_mean[0]),
            "mfcc_2": float(mfcc_mean[1]),
            "mfcc_3": float(mfcc_mean[2]),

            # Форманты
            "f1": float(f1),
            "f2": float(f2),
            "f3": float(f3),

            # Качество голоса
            "jitter": float(jitter),
            "shimmer": float(shimmer),
            "voiced_ratio": float(voiced_ratio),
        }

    except Exception as e:
        print(f"Ошибка извлечения признаков {start_time}-{end_time}: {e}")
        return None


def improved_gender_classification(features):
    """
    Улучшенная классификация пола с использованием нескольких признаков

    Args:
        features: словарь с извлеченными признаками

    Returns:
        tuple: (predicted_gender, confidence_score)
    """
    if not features:
        return "unknown", 0.0

    # Инициализируем счетчики
    male_score = 0
    female_score = 0
    total_weight = 0

    # 1. ОСНОВНАЯ ЧАСТОТА (F0) - основной признак
    f0_median = features["f0_median"]
    f0_weight = 0.4

    if f0_median < 110:
        male_score += f0_weight * 1.0
    elif f0_median < 130:
        male_score += f0_weight * 0.8
    elif f0_median < 150:
        male_score += f0_weight * 0.4
    elif f0_median < 170:
        # Серая зона - полагаемся на другие признаки
        pass
    elif f0_median < 200:
        female_score += f0_weight * 0.4
    elif f0_median < 230:
        female_score += f0_weight * 0.8
    else:
        female_score += f0_weight * 1.0

    total_weight += f0_weight

    # 2. СПЕКТРАЛЬНЫЙ ЦЕНТРОИД
    centroid = features["spectral_centroid"]
    centroid_weight = 0.2

    if centroid < 1500:
        male_score += centroid_weight * 0.8
    elif centroid < 2000:
        male_score += centroid_weight * 0.4
    elif centroid < 2500:
        pass  # нейтральная зона
    elif centroid < 3000:
        female_score += centroid_weight * 0.4
    else:
        female_score += centroid_weight * 0.8

    total_weight += centroid_weight

    # 3. ФОРМАНТЫ (если доступны)
    if features["f1"] > 0 and features["f2"] > 0:
        f1, f2 = features["f1"], features["f2"]
        formant_weight = 0.2

        # Женские форманты обычно выше
        if f1 > 500 and f2 > 1500:
            female_score += formant_weight * 0.6
        elif f1 < 400 and f2 < 1200:
            male_score += formant_weight * 0.6

        total_weight += formant_weight

    # 4. ВАРИАТИВНОСТЬ F0
    f0_std = features["f0_std"]
    variability_weight = 0.1

    # Женские голоса обычно более вариативны
    if f0_std > 20:
        female_score += variability_weight * 0.5
    elif f0_std < 10:
        male_score += variability_weight * 0.5

    total_weight += variability_weight

    # 5. MFCC ПРИЗНАКИ
    mfcc_weight = 0.1
    if features["mfcc_2"] > 0:
        female_score += mfcc_weight * 0.3
    else:
        male_score += mfcc_weight * 0.3

    total_weight += mfcc_weight

    # Нормализуем оценки
    if total_weight > 0:
        male_score /= total_weight
        female_score /= total_weight

    # Принимаем решение
    if male_score > female_score:
        confidence = male_score - female_score
        return "male", min(confidence, 1.0)
    elif female_score > male_score:
        confidence = female_score - male_score
        return "female", min(confidence, 1.0)
    else:
        return "uncertain", 0.3


def analyze_improved_gender(audio_path: str, segments: list, output_path: str = None):
    """
    Анализирует пол с улучшенным алгоритмом
    """
    try:
        print(f"🎵 Улучшенный анализ пола для {len(segments)} сегментов...")

        # Анализируем каждый сегмент
        for i, segment in enumerate(segments):
            if i % 10 == 0:
                print(f"  Обработано: {i}/{len(segments)}")

            # Извлекаем признаки
            features = extract_voice_features(
                audio_path,
                segment["start"],
                segment["end"]
            )

            if features:
                # Классифицируем пол
                gender, confidence = improved_gender_classification(features)

                # Добавляем результаты
                segment["voice_features"] = features
                segment["predicted_gender"] = gender
                segment["gender_confidence"] = confidence

                # Добавляем объяснение
                segment["gender_explanation"] = {
                    "f0_hz": features["f0_median"],
                    "spectral_centroid": features["spectral_centroid"],
                    "primary_indicators": []
                }

                # Объясняем решение
                if features["f0_median"] < 130:
                    segment["gender_explanation"]["primary_indicators"].append("низкая F0")
                elif features["f0_median"] > 200:
                    segment["gender_explanation"]["primary_indicators"].append("высокая F0")

                if features["spectral_centroid"] > 2500:
                    segment["gender_explanation"]["primary_indicators"].append("высокий спектральный центроид")
                elif features["spectral_centroid"] < 1500:
                    segment["gender_explanation"]["primary_indicators"].append("низкий спектральный центроид")

            else:
                segment["predicted_gender"] = "unknown"
                segment["gender_confidence"] = 0.0
                segment["voice_features"] = None

        # Выводим статистику
        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТЫ УЛУЧШЕННОГО АНАЛИЗА ПОЛА")
        print("=" * 60)

        gender_stats = {"male": 0, "female": 0, "uncertain": 0, "unknown": 0}
        high_conf_stats = {"male": 0, "female": 0}

        for segment in segments:
            gender = segment["predicted_gender"]
            confidence = segment["gender_confidence"]

            gender_stats[gender] += 1

            if confidence > 0.6:
                if gender in ["male", "female"]:
                    high_conf_stats[gender] += 1

        for gender, count in gender_stats.items():
            percentage = (count / len(segments)) * 100
            print(f"{gender.upper()}: {count} сегментов ({percentage:.1f}%)")

        print(f"\nВысокая уверенность (>0.6):")
        for gender, count in high_conf_stats.items():
            print(f"  {gender.upper()}: {count} сегментов")

        # Показываем примеры
        print(f"\n📋 Примеры с объяснениями:")
        for i, segment in enumerate(segments[:8]):
            if segment["predicted_gender"] != "unknown":
                expl = segment.get("gender_explanation", {})
                indicators = ", ".join(expl.get("primary_indicators", []))
                print(f"  {i + 1}. {segment['start']:5.1f}s-{segment['end']:5.1f}s | "
                      f"F0: {expl.get('f0_hz', 0):.1f}Hz | "
                      f"{segment['predicted_gender']} ({segment['gender_confidence']:.2f}) | "
                      f"{indicators}")

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
    parser = argparse.ArgumentParser(description="Улучшенное определение пола")
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
    with open(args.segments_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = data.get("segments", data) if isinstance(data, dict) else data

    # Анализируем пол
    result = analyze_improved_gender(args.audio_path, segments, args.output)

    if result:
        print("\n✅ Улучшенный анализ завершён!")
        print("💡 Теперь классификация учитывает больше признаков")


if __name__ == "__main__":
    main()