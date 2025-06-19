import json
import os
import shutil
import subprocess
from pydub import AudioSegment
from utils.audio_utils import split_audio
import openai
from utils.logger_config import setup_logger
import whisper
from typing import Optional, Dict, Any


logger = setup_logger(name=__name__, log_file="logs/app.log")

_whisper_model = None
_current_model_size = None


def load_whisper_model(model_size: str = "large-v2", device: str = "auto"):
    global _whisper_model, _current_model_size

    if _whisper_model is None or _current_model_size != model_size:
        logger.info(f"Loading Whisper model: {model_size}")
        _whisper_model = whisper.load_model(model_size, device=device)
        _current_model_size = model_size
        logger.info(f"Whisper model {model_size} loaded successfully")

    return _whisper_model


def transcribe_local(file_path: str, source_language: Optional[str] = None,
                     model_size: str = "large-v2", device: str = "cuda",
                     transcription_keywords: Optional[str] = None) -> Dict[str, Any]:
    try:
        logger.info(f"Downloading Whisper model: {model_size}...")
        model = load_whisper_model("large-v2", device)
        logger.info(f"Whisper model: {model_size} is ready!")

        base_prompt = "Add proper punctuation."
        if transcription_keywords:
            prompt = f"{transcription_keywords}. {base_prompt}"
            logger.info(f"Using keywords: {transcription_keywords}")
        else:
            prompt = base_prompt

        logger.info(f"Prompt: {prompt}")

        transcribe_params = {
            "word_timestamps": True,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "prompt": prompt,
        }

        logger.info(f"Starting transcription of {file_path}")

        if source_language:
            result = model.transcribe(file_path, language=source_language, **transcribe_params)
        else:
            result = model.transcribe(file_path, **transcribe_params)

        formatted_result = {
            "text": result["text"],
            "segments": [],
            "words": []
        }

        for i, segment in enumerate(result["segments"]):
            seg_dict = {
                "id": i,
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": segment["text"]
            }
            formatted_result["segments"].append(seg_dict)

            if "words" in segment:
                for word_info in segment["words"]:
                    word_dict = {
                        "word": word_info["word"],
                        "start": float(word_info["start"]),
                        "end": float(word_info["end"])
                    }
                    formatted_result["words"].append(word_dict)

        if not formatted_result.get("text") and not formatted_result.get("segments"):
            raise ValueError(f"Empty transcription result for {file_path}")

        logger.info(f"Transcription completed for {file_path}. Text length: {len(formatted_result['text'])}")

        return formatted_result

    except Exception as e:
        logger.error(f"Error transcribing {file_path} with local Whisper: {str(e)}")
        raise ValueError(f"Failed to transcribe audio with local Whisper: {str(e)}")


def transcribe_audio_with_timestamps(
        input_audio,
        job_id,
        source_language=None,
        output_file=None,
        openai_api_key=None,
        transcription_keywords=None
):
    def get_precise_audio_duration(file_path):
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())

    initial_audio_duration = get_precise_audio_duration(input_audio)
    logger.info(f"Initial audio duration: {initial_audio_duration:.3f} seconds")

    temp_audio_chunks_dir = f"jobs/{job_id}/output/temp_audio_chunks"
    os.makedirs(temp_audio_chunks_dir, exist_ok=True)
    os.makedirs(temp_audio_chunks_dir, exist_ok=True)

    chunk_paths = split_audio(input_audio, temp_audio_chunks_dir)

    full_result = {
        "text": "",
        "segments": [],
        "words": []
    }

    time_offset = 0
    segment_id = 0
    word_id = 0

    for i, chunk_path in enumerate(chunk_paths):
        logger.info(f"Processing chunk {i + 1}/{len(chunk_paths)}: {chunk_path}")

        audio_chunk = AudioSegment.from_file(chunk_path)
        chunk_duration = len(audio_chunk) / 1000

        # transcription_api = transcribe(chunk_path, source_language=source_language, openai_api_key=openai_api_key)

        transcription = transcribe_local(
            file_path=chunk_path,
            source_language=source_language,
            transcription_keywords=transcription_keywords
        )

        logger.info(f"Chunk {i + 1}/{len(chunk_paths)} transcribed. Text: {transcription.get('text', '')[:50]}...")
        logger.info(f"Total number of segments: {len(transcription.get('segments', []))}")
        logger.info(f"Total number of words: {len(transcription.get('words', []))}")

        full_result["text"] += (
            " " + transcription.get("text", "") if full_result["text"] else transcription.get("text", ""))

        for segment in transcription.get("segments", []):
            simplified_segment = {
                "id": segment_id,
                "start": segment.get("start", 0) + time_offset,
                "end": segment.get("end", 0) + time_offset,
                "text": segment.get("text", "")
            }
            segment_id += 1
            full_result["segments"].append(simplified_segment)

        for word in transcription.get("words", []):
            simplified_word = {
                "id": word_id,
                "start": word.get("start", 0) + time_offset,
                "end": word.get("end", 0) + time_offset,
                "word": word.get("word", "")
            }
            word_id += 1
            full_result["words"].append(simplified_word)

        time_offset += chunk_duration

        if chunk_path != input_audio:
            os.remove(chunk_path)

    outro_gap_duration = 0.0
    if full_result["segments"]:
        last_segment_end = full_result["segments"][-1]["end"]
        outro_gap_duration = max(0.0, initial_audio_duration - last_segment_end)
        logger.info(f"Last segment ends at: {last_segment_end:.3f}s")
        logger.info(f"Outro gap duration: {outro_gap_duration:.3f}s")

    full_result["outro_gap_duration"] = outro_gap_duration

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2)

    if os.path.exists(temp_audio_chunks_dir) and len(chunk_paths) > 1:
        shutil.rmtree(temp_audio_chunks_dir)

    logger.info(f"Timestamped transcription successfully finished! Result: {output_file}")
    return output_file


def transcribe(file_path, source_language=None, openai_api_key=None):
    if not openai_api_key:
        raise ValueError("OpenAI API key is required for transcription step but not provided")

    if source_language:
        language = source_language
    else:
        language = None

    try:
        client = openai.OpenAI(api_key=openai_api_key)

        with open(file_path, "rb") as audio_file:
            api_params = {
                "model": "whisper-1",
                "file": audio_file,
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment", "word"],
                "temperature": 0.1
            }

            if language:
                api_params["language"] = language

            response = client.audio.transcriptions.create(**api_params)

        if hasattr(response, "model_dump"):
            result = response.model_dump()
        elif hasattr(response, "to_dict"):
            result = response.to_dict()
        else:
            result = {
                "text": getattr(response, "text", ""),
                "segments": [],
                "words": []
            }

            segments = getattr(response, "segments", [])
            for segment in segments:
                seg_dict = {
                    "id": getattr(segment, "id", 0),
                    "start": getattr(segment, "start", 0),
                    "end": getattr(segment, "end", 0),
                    "text": getattr(segment, "text", "")
                }
                result["segments"].append(seg_dict)

            words = getattr(response, "words", [])
            for word in words:
                word_dict = {
                    "word": getattr(word, "word", ""),
                    "start": getattr(word, "start", 0),
                    "end": getattr(word, "end", 0)
                }
                result["words"].append(word_dict)

        if not result.get("text") and not result.get("segments") and not result.get("words"):
            raise ValueError(f"Empty transcription result for {file_path}")

        return result
    except Exception as e:
        logger.error(f"Error transcribing {file_path}: {str(e)}")
        raise ValueError(f"Failed to transcribe audio: {str(e)}")
