import json
import librosa
import numpy as np
import os
from utils.logger_config import setup_logger

logger = setup_logger(name=__name__, log_file="logs/app.log")


def get_simple_voice_mapping(gender: str):
    if gender == "male":
        return "onyx"
    elif gender == "female":
        return "shimmer"
    else:
        return "onyx"


def analyze_voice_gender_simple(audio_path: str, start_time: float, end_time: float, sr: int = 16000):
    try:
        y, sr = librosa.load(audio_path, sr=sr)

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        if len(segment) < sr * 0.1:
            return None

        f0 = librosa.yin(segment, fmin=50, fmax=400, sr=sr)

        voiced_f0 = f0[f0 > 0]

        if len(voiced_f0) == 0:
            return None

        f0_median = float(np.median(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_max = float(np.max(voiced_f0))
        f0_range = float(f0_max - np.min(voiced_f0))

        gender = "unknown"
        confidence = 0.0

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

        suggested_voice = get_simple_voice_mapping(gender)

        return {
            "predicted_gender": gender,
            "gender_confidence": float(confidence),
            "suggested_voice": suggested_voice
        }

    except Exception as e:
        logger.warning(f"Error analyzing segment {start_time}-{end_time}: {e}")
        return None


def add_gender_and_voice_mapping_to_segments(audio_path: str, translated_json_path: str):
    try:
        logger.info(f"Adding simple gender analysis and OpenAI voice mapping...")
        logger.info(f"Audio file: {audio_path}")
        logger.info(f"Translation file: {translated_json_path}")

        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return False

        if not os.path.exists(translated_json_path):
            logger.error(f"Translation file not found: {translated_json_path}")
            return False

        with open(translated_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

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

        successful_analyses = 0
        voice_stats = {}

        for i, segment in enumerate(segments):
            if i % 20 == 0:
                logger.info(f"Processing segment {i + 1}/{len(segments)}")

            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)

            analysis = analyze_voice_gender_simple(audio_path, start_time, end_time)

            if analysis:
                segment["predicted_gender"] = analysis["predicted_gender"]
                segment["gender_confidence"] = analysis["gender_confidence"]
                segment["suggested_voice"] = analysis["suggested_voice"]

                voice = analysis["suggested_voice"]
                voice_stats[voice] = voice_stats.get(voice, 0) + 1

                successful_analyses += 1
            else:
                segment["predicted_gender"] = "unknown"
                segment["gender_confidence"] = 0.0
                segment["suggested_voice"] = "onyx"

        with open(translated_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        gender_stats = {}
        for segment in segments:
            gender = segment.get("predicted_gender", "unknown")
            gender_stats[gender] = gender_stats.get(gender, 0) + 1

        logger.info(f"Simple gender analysis and voice mapping completed!")
        logger.info(f"Successfully analyzed: {successful_analyses}/{len(segments)} segments")
        logger.info(f"Gender distribution: {gender_stats}")
        logger.info(f"Voice mapping:")

        total_segments = len(segments)
        for voice, count in voice_stats.items():
            percentage = (count / total_segments) * 100 if total_segments > 0 else 0
            voice_type = "male" if voice == "onyx" else "female"
            logger.info(f"  {voice} ({voice_type}): {count} segments ({percentage:.1f}%)")

        return True

    except Exception as e:
        logger.error(f"Error in simple gender analysis and voice mapping: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_gender_and_voice_analysis_step(job_id, audio_path, translated_json_path, tts_provider):
    try:
        if tts_provider != "openai":
            logger.info(f"Skipping gender/voice analysis for TTS provider: {tts_provider}")
            return True

        logger.info(f"Running simple gender analysis and OpenAI voice mapping for job {job_id}")
        logger.info(f"TTS provider: {tts_provider}")

        success = add_gender_and_voice_mapping_to_segments(audio_path, translated_json_path)

        if success:
            logger.info(f"Simple gender and voice analysis completed successfully for job {job_id}")
        else:
            logger.error(f"Simple gender and voice analysis failed for job {job_id}")

        return success

    except Exception as e:
        logger.error(f"Error in simple gender/voice analysis step for job {job_id}: {e}")
        return False