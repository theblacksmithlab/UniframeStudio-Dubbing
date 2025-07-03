import json
import os
import tempfile
import torch
import librosa
import soundfile as sf
from modules.voice_gender_classifier_model import ECAPA_gender
from utils.logger_config import setup_logger
from typing import Optional, Dict, Any

logger = setup_logger(name=__name__, log_file="logs/app.log")


class JaesungGenderClassifier:

    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing JaesungHuh gender classifier on device: {self.device}")

    def load_model(self):
        try:
            logger.info("Loading JaesungHuh/voice-gender-classifier model...")

            self.model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
            self.model.eval()
            self.model.to(self.device)

            logger.info("JaesungHuh gender classification model loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to load JaesungHuh gender classification model: {e}")
            return False

    def predict_gender(self, audio_path: str, start_time: float, end_time: float) -> Optional[Dict[str, Any]]:
        try:
            if self.model is None:
                if not self.load_model():
                    return None

            audio, sr = librosa.load(audio_path, sr=16000)

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = audio[start_sample:end_sample]

            if len(segment) < sr * 0.1:
                return None

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, segment, sr)
                temp_path = temp_file.name

            try:
                with torch.no_grad():
                    output = self.model.predict(temp_path, device=self.device)

                logger.info(f"Model output: {output}")

                if output:
                    gender_str = str(output).lower().strip()

                    if 'male' in gender_str and 'female' not in gender_str:
                        gender = "male"
                        confidence = 0.85
                    elif 'female' in gender_str:
                        gender = "female"
                        confidence = 0.85
                    else:
                        gender = "unknown"
                        confidence = 0.3

                    return {
                        "predicted_gender": gender,
                        "gender_confidence": confidence,
                        "raw_output": str(output)
                    }
                else:
                    logger.warning(f"No output from model for segment {start_time}-{end_time}")
                    return None

            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Error predicting gender for segment {start_time}-{end_time}: {e}")
            return None


def get_simple_voice_mapping(gender: str):
    if gender == "male":
        return "onyx"
    elif gender == "female":
        return "shimmer"
    else:
        return "onyx"


def add_gender_and_voice_mapping_to_segments(audio_path: str, translated_json_path: str):
    try:
        logger.info(f"Adding JaesungHuh gender analysis and OpenAI voice mapping...")
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

        segments = data.get("segments", data) if isinstance(data, dict) else data
        if not segments:
            logger.error("No segments found")
            return False

        logger.info(f"Found {len(segments)} segments to analyze")

        gender_classifier = JaesungGenderClassifier()

        successful_analyses = 0
        voice_stats = {}

        for i, segment in enumerate(segments):
            if i % 5 == 0:
                logger.info(f"Processing segment {i + 1}/{len(segments)}")

            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)

            analysis = gender_classifier.predict_gender(audio_path, start_time, end_time)

            if analysis:
                suggested_voice = get_simple_voice_mapping(analysis["predicted_gender"])

                segment["predicted_gender"] = analysis["predicted_gender"]
                segment["gender_confidence"] = analysis["gender_confidence"]
                segment["suggested_voice"] = suggested_voice

                voice_stats[suggested_voice] = voice_stats.get(suggested_voice, 0) + 1
                successful_analyses += 1

                logger.info(
                    f"Segment {i + 1}: {analysis['predicted_gender']} ({analysis['gender_confidence']:.2f}) -> {suggested_voice}")
            else:
                segment["predicted_gender"] = "unknown"
                segment["gender_confidence"] = 0.0
                segment["suggested_voice"] = "onyx"
                logger.warning(f"Segment {i + 1}: failed to analyze")

        with open(translated_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        gender_stats = {}
        for segment in segments:
            gender = segment.get("predicted_gender", "unknown")
            gender_stats[gender] = gender_stats.get(gender, 0) + 1

        logger.info(f"JaesungHuh gender analysis completed!")
        logger.info(f"Successfully analyzed: {successful_analyses}/{len(segments)} segments")
        logger.info(f"Gender distribution: {gender_stats}")
        logger.info(f"Voice mapping: {voice_stats}")

        return True

    except Exception as e:
        logger.error(f"Error in JaesungHuh gender analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_gender_and_voice_analysis_step(job_id, audio_path, translated_json_path, tts_provider):
    try:
        if tts_provider != "openai":
            logger.info(f"Skipping gender/voice analysis for TTS provider: {tts_provider}")
            return True

        logger.info(f"Running JaesungHuh gender analysis for job {job_id}")

        success = add_gender_and_voice_mapping_to_segments(audio_path, translated_json_path)

        if success:
            logger.info(f"JaesungHuh gender analysis completed successfully for job {job_id}")
        else:
            logger.error(f"JaesungHuh gender analysis failed for job {job_id}")

        return success

    except Exception as e:
        logger.error(f"Error in JaesungHuh gender analysis step for job {job_id}: {e}")
        return False
