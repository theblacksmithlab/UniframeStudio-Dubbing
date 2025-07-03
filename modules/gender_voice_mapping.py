import json
import os
import tempfile
import torch
import librosa
import soundfile as sf
from transformers import pipeline
from utils.logger_config import setup_logger

logger = setup_logger(name=__name__, log_file="logs/app.log")


class HuggingFaceGenderClassifier:
    """
    Класс для определения пола с помощью предобученной модели HuggingFace
    """

    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing gender classifier on device: {self.device}")

    def load_model(self):
        """Загружает модель определения пола"""
        try:
            logger.info("Loading alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech model...")

            # Используем более простую модель, которая точно работает с pipeline
            self.model = pipeline(
                "audio-classification",
                model="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
                device=0 if torch.cuda.is_available() else -1
            )

            logger.info("Gender classification model loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to load gender classification model: {e}")
            logger.info("Trying fallback model...")

            # Fallback на другую модель
            try:
                self.model = pipeline(
                    "audio-classification",
                    model="MIT/ast-finetuned-speech-commands-v2",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Fallback model loaded successfully!")
                return True
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
                return False

    def predict_gender(self, audio_path: str, start_time: float, end_time: float):
        """
        Предсказывает пол для аудио сегмента

        Args:
            audio_path: путь к аудио файлу
            start_time: начало сегмента в секундах
            end_time: конец сегмента в секундах

        Returns:
            dict: результат предсказания
        """
        try:
            if self.model is None:
                if not self.load_model():
                    return None

            # Загружаем и обрезаем аудио
            audio, sr = librosa.load(audio_path, sr=16000)  # Модель ожидает 16kHz

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = audio[start_sample:end_sample]

            # Проверяем длину сегмента
            if len(segment) < sr * 0.1:  # Слишком короткий
                return None

            # Создаем временный файл для сегмента
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, segment, sr)
                temp_path = temp_file.name

            try:
                # Получаем предсказание от модели
                results = self.model(temp_path)

                # Обрабатываем результат
                if results and len(results) > 0:
                    # Модель возвращает список с результатами
                    best_result = max(results, key=lambda x: x['score'])

                    # Определяем пол и уверенность
                    predicted_label = best_result['label'].lower()
                    confidence = float(best_result['score'])

                    # Маппим на наши стандартные лейблы
                    if 'male' in predicted_label or 'm' in predicted_label or 'man' in predicted_label:
                        gender = "male"
                    elif 'female' in predicted_label or 'f' in predicted_label or 'woman' in predicted_label:
                        gender = "female"
                    else:
                        # Если не удалось определить, используем простую эвристику
                        # Если confidence низкая, лучше сказать unknown
                        if confidence < 0.6:
                            gender = "unknown"
                        else:
                            gender = "male"  # fallback по умолчанию

                    return {
                        "predicted_gender": gender,
                        "gender_confidence": confidence,
                        "raw_prediction": predicted_label,
                        "all_results": results
                    }
                else:
                    logger.warning(f"No results from model for segment {start_time}-{end_time}")
                    return None

            finally:
                # Удаляем временный файл
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Error predicting gender for segment {start_time}-{end_time}: {e}")
            return None


def get_simple_voice_mapping(gender: str):
    """
    Простой маппинг пола на голос OpenAI
    """
    if gender == "male":
        return "onyx"  # Мужской голос
    elif gender == "female":
        return "shimmer"  # Женский голос
    else:
        return "onyx"  # По умолчанию мужской


def add_gender_and_voice_mapping_to_segments(audio_path: str, translated_json_path: str):
    """
    Добавляет анализ пола с помощью HuggingFace модели к переведённым сегментам
    """
    try:
        logger.info(f"Adding HuggingFace gender analysis and OpenAI voice mapping...")
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

        # Инициализируем классификатор
        gender_classifier = HuggingFaceGenderClassifier()

        # Анализируем каждый сегмент
        successful_analyses = 0
        voice_stats = {}
        confidence_stats = {"high": 0, "medium": 0, "low": 0}

        for i, segment in enumerate(segments):
            if i % 10 == 0:
                logger.info(f"Processing segment {i + 1}/{len(segments)}")

            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)

            # Анализируем пол с помощью HuggingFace модели
            analysis = gender_classifier.predict_gender(audio_path, start_time, end_time)

            if analysis:
                # Получаем простой голос (onyx или shimmer)
                suggested_voice = get_simple_voice_mapping(analysis["predicted_gender"])

                # Добавляем результаты к сегменту
                segment["predicted_gender"] = analysis["predicted_gender"]
                segment["gender_confidence"] = analysis["gender_confidence"]
                segment["suggested_voice"] = suggested_voice

                # Считаем статистику по голосам
                voice_stats[suggested_voice] = voice_stats.get(suggested_voice, 0) + 1

                # Статистика по уверенности
                confidence = analysis["gender_confidence"]
                if confidence > 0.8:
                    confidence_stats["high"] += 1
                elif confidence > 0.5:
                    confidence_stats["medium"] += 1
                else:
                    confidence_stats["low"] += 1

                successful_analyses += 1
            else:
                # Если анализ не удался, ставим значения по умолчанию
                segment["predicted_gender"] = "unknown"
                segment["gender_confidence"] = 0.0
                segment["suggested_voice"] = "onyx"  # мужской голос по умолчанию

        # Сохраняем обновлённый JSON
        with open(translated_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Логируем статистику
        gender_stats = {}
        for segment in segments:
            gender = segment.get("predicted_gender", "unknown")
            gender_stats[gender] = gender_stats.get(gender, 0) + 1

        logger.info(f"HuggingFace gender analysis and voice mapping completed!")
        logger.info(f"Successfully analyzed: {successful_analyses}/{len(segments)} segments")
        logger.info(f"Gender distribution: {gender_stats}")
        logger.info(f"Voice mapping:")

        total_segments = len(segments)
        for voice, count in voice_stats.items():
            percentage = (count / total_segments) * 100 if total_segments > 0 else 0
            voice_type = "male" if voice == "onyx" else "female"
            logger.info(f"  {voice} ({voice_type}): {count} segments ({percentage:.1f}%)")

        logger.info(f"Confidence distribution:")
        for level, count in confidence_stats.items():
            percentage = (count / successful_analyses) * 100 if successful_analyses > 0 else 0
            logger.info(f"  {level}: {count} segments ({percentage:.1f}%)")

        return True

    except Exception as e:
        logger.error(f"Error in HuggingFace gender analysis and voice mapping: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_gender_and_voice_analysis_step(job_id, audio_path, translated_json_path, tts_provider):
    """
    Запускает шаг анализа пола с HuggingFace моделью для OpenAI TTS
    """
    try:
        # Проверяем нужно ли выполнять анализ
        if tts_provider != "openai":
            logger.info(f"Skipping gender/voice analysis for TTS provider: {tts_provider}")
            return True

        logger.info(f"Running HuggingFace gender analysis and OpenAI voice mapping for job {job_id}")
        logger.info(f"TTS provider: {tts_provider}")

        # Выполняем анализ пола и маппинг голосов
        success = add_gender_and_voice_mapping_to_segments(audio_path, translated_json_path)

        if success:
            logger.info(f"HuggingFace gender and voice analysis completed successfully for job {job_id}")
        else:
            logger.error(f"HuggingFace gender and voice analysis failed for job {job_id}")

        return success

    except Exception as e:
        logger.error(f"Error in HuggingFace gender/voice analysis step for job {job_id}: {e}")
        return False