import json
import librosa
import numpy as np
import os
from utils.logger_config import setup_logger

logger = setup_logger(name=__name__, log_file="logs/app.log")


def get_openai_voice_mapping(f0_median: float, gender: str, confidence: float):
    """
    Маппинг характеристик голоса на голоса OpenAI TTS

    OpenAI Voices доступные:
    МУЖСКИЕ:
    - onyx: глубокий, авторитетный (низкий голос)
    - alloy: естественный, гладкий (средний голос)
    - echo: артикулированный, точный (средний голос)
    - fable: тёплый, вовлекающий (средний голос)

    ЖЕНСКИЕ:
    - nova: яркий, энергичный (высокий голос)
    - shimmer: мягкий, нежный (средний женский голос)

    НОВЫЕ (с октября 2024):
    - ash: мужской
    - ballad: женский
    - coral: женский
    - sage: мужской
    - verse: женский
    """

    # Определяем рекомендуемый голос на основе частоты и пола
    if gender == "male":
        if f0_median < 100:
            # Очень низкий мужской голос
            suggested_voice = "onyx"
            voice_description = "deep_male"
        elif f0_median < 125:
            # Низкий мужской голос
            suggested_voice = "onyx"
            voice_description = "low_male"
        elif f0_median < 140:
            # Средний мужской голос
            if confidence > 0.7:
                suggested_voice = "alloy"  # более натуральный
            else:
                suggested_voice = "fable"  # тёплый, если не уверены
            voice_description = "medium_male"
        else:
            # Высокий мужской голос
            suggested_voice = "echo"  # артикулированный
            voice_description = "high_male"

    elif gender == "female":
        if f0_median < 160:
            # Низкий женский голос
            suggested_voice = "shimmer"  # мягкий, нежный
            voice_description = "low_female"
        elif f0_median < 190:
            # Средний женский голос
            if confidence > 0.7:
                suggested_voice = "nova"  # яркий, энергичный
            else:
                suggested_voice = "shimmer"  # более безопасный выбор
            voice_description = "medium_female"
        elif f0_median < 220:
            # Высокий женский голос
            suggested_voice = "nova"  # яркий, энергичный
            voice_description = "high_female"
        else:
            # Очень высокий женский голос
            suggested_voice = "nova"
            voice_description = "very_high_female"
    else:
        # Неопределённый пол - выбираем нейтральный голос
        if f0_median < 150:
            suggested_voice = "alloy"  # может звучать и мужским, и женским
            voice_description = "neutral_low"
        else:
            suggested_voice = "shimmer"  # мягкий, менее выраженный
            voice_description = "neutral_high"

    return suggested_voice, voice_description


def analyze_voice_with_openai_mapping(audio_path: str, start_time: float, end_time: float, sr: int = 16000):
    """
    Анализирует голос и предлагает подходящий голос OpenAI
    """
    try:
        # Загружаем аудио
        y, sr = librosa.load(audio_path, sr=sr)

        # Вырезаем нужный сегмент
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        # Проверяем длину сегмента
        if len(segment) < sr * 0.1:  # Слишком короткий сегмент
            return None

        # Извлекаем основную частоту (F0)
        f0 = librosa.yin(segment, fmin=50, fmax=400, sr=sr)

        # Убираем нулевые значения
        voiced_f0 = f0[f0 > 0]

        if len(voiced_f0) == 0:
            return None

        # Вычисляем статистики и преобразуем в обычные float
        f0_median = float(np.median(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_max = float(np.max(voiced_f0))
        f0_range = float(f0_max - np.min(voiced_f0))

        # Определяем пол (используем настроенные пороги)
        if f0_median < 110:
            gender = "male"
            confidence = min(1.0, (110 - f0_median) / 40)
        elif f0_median < 130:
            gender = "male"
            confidence = min(0.8, (130 - f0_median) / 25)
        elif f0_median < 145:
            if f0_std > 25:
                gender = "female"
                confidence = 0.4
            else:
                gender = "male"
                confidence = 0.5
        elif f0_median < 165:
            if f0_max > 200:
                gender = "female"
                confidence = 0.6
            elif f0_std > 30:
                gender = "female"
                confidence = 0.5
            elif f0_range > 80:
                gender = "female"
                confidence = 0.4
            else:
                gender = "uncertain"
                confidence = 0.2
        elif f0_median < 185:
            gender = "female"
            confidence = 0.7
        elif f0_median < 220:
            gender = "female"
            confidence = 0.9
        else:
            gender = "female"
            confidence = 1.0

        # Получаем рекомендуемый голос OpenAI
        suggested_voice, voice_description = get_openai_voice_mapping(f0_median, gender, confidence)

        return {
            "predicted_gender": gender,
            "gender_confidence": float(confidence),
            "f0_median": float(f0_median),
            "suggested_voice": suggested_voice,
            "voice_description": voice_description
        }

    except Exception as e:
        logger.warning(f"Error analyzing segment {start_time}-{end_time}: {e}")
        return None


def add_gender_and_voice_mapping_to_segments(audio_path: str, translated_json_path: str):
    """
    Добавляет анализ пола и рекомендацию голоса OpenAI к переведённым сегментам
    """
    try:
        logger.info(f"Adding gender analysis and OpenAI voice mapping...")
        logger.info(f"Audio file: {audio_path}")
        logger.info(f"Translation file: {translated_json_path}")

        # Проверяем наличие файлов
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return False

        if not os.path.exists(translated_json_path):
            logger.error(f"Translation file not found: {translated_json_path}")
            return False

        # Загружаем JSON с переводом
        with open(translated_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Находим сегменты
        segments = None
        if "segments" in data:
            segments = data["segments"]
        elif isinstance(data, list):
            segments = data
        else:
            logger.error("No segments found in translation file")
            return False

        if not segments:
            logger.error("Empty segments list")
            return False

        logger.info(f"Found {len(segments)} segments to analyze")

        # Анализируем каждый сегмент
        successful_analyses = 0
        voice_stats = {}

        for i, segment in enumerate(segments):
            if i % 20 == 0:
                logger.info(f"Processing segment {i + 1}/{len(segments)}")

            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)

            # Анализируем голос и получаем рекомендацию
            analysis = analyze_voice_with_openai_mapping(audio_path, start_time, end_time)

            if analysis:
                # Добавляем результаты к сегменту
                segment["predicted_gender"] = analysis["predicted_gender"]
                segment["gender_confidence"] = analysis["gender_confidence"]
                segment["suggested_voice"] = analysis["suggested_voice"]

                # Считаем статистику по голосам
                voice = analysis["suggested_voice"]
                voice_stats[voice] = voice_stats.get(voice, 0) + 1

                successful_analyses += 1
            else:
                # Если анализ не удался, ставим значения по умолчанию
                segment["predicted_gender"] = "unknown"
                segment["gender_confidence"] = 0.0
                segment["suggested_voice"] = "alloy"  # нейтральный голос по умолчанию

        # Сохраняем обновлённый JSON
        with open(translated_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Логируем статистику
        gender_stats = {}
        for segment in segments:
            gender = segment.get("predicted_gender", "unknown")
            gender_stats[gender] = gender_stats.get(gender, 0) + 1

        logger.info(f"Gender analysis and voice mapping completed!")
        logger.info(f"Successfully analyzed: {successful_analyses}/{len(segments)} segments")
        logger.info(f"Gender distribution: {gender_stats}")
        logger.info(f"Suggested OpenAI voices: {voice_stats}")

        # Логируем детали по каждому голосу
        voice_descriptions = {
            "onyx": "deep, authoritative male",
            "alloy": "natural, smooth (neutral)",
            "echo": "articulate, precise male",
            "fable": "warm, engaging male",
            "nova": "bright, energetic female",
            "shimmer": "soft, gentle female"
        }

        for voice, count in voice_stats.items():
            description = voice_descriptions.get(voice, "unknown voice")
            logger.info(f"  {voice}: {count} segments ({description})")

        return True

    except Exception as e:
        logger.error(f"Error in gender analysis and voice mapping: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_gender_and_voice_analysis_step(job_id, audio_path, translated_json_path, tts_provider):
    """
    Запускает шаг анализа пола и маппинга голосов для OpenAI TTS
    """
    try:
        # Проверяем нужно ли выполнять анализ
        if tts_provider != "openai":
            logger.info(f"Skipping gender/voice analysis for TTS provider: {tts_provider}")
            return True

        logger.info(f"Running gender analysis and OpenAI voice mapping for job {job_id}")
        logger.info(f"TTS provider: {tts_provider}")

        # Выполняем анализ пола и маппинг голосов
        success = add_gender_and_voice_mapping_to_segments(audio_path, translated_json_path)

        if success:
            logger.info(f"Gender and voice analysis completed successfully for job {job_id}")
        else:
            logger.error(f"Gender and voice analysis failed for job {job_id}")

        return success

    except Exception as e:
        logger.error(f"Error in gender/voice analysis step for job {job_id}: {e}")
        return False