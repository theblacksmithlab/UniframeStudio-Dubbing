import json
import os
import shutil

from pydub import AudioSegment
from utils.audio_utils import split_audio
import openai


def transcribe_audio_with_timestamps(input_audio):
    base_name = os.path.splitext(os.path.basename(input_audio))[0]

    timestamped_transcriptions_dir = "output/timestamped_transcriptions"
    temp_audio_chunks_dir = "output/temp_audio_chunks"
    os.makedirs(timestamped_transcriptions_dir, exist_ok=True)
    os.makedirs(temp_audio_chunks_dir, exist_ok=True)
    output_json = os.path.join(timestamped_transcriptions_dir, f"{base_name}_transcribed.json")

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
        print(f"Processing chunk {i + 1}/{len(chunk_paths)}: {chunk_path}")

        audio_chunk = AudioSegment.from_file(chunk_path)
        chunk_duration = len(audio_chunk) / 1000

        transcription = transcribe(chunk_path)

        print(f"Chunk {i + 1}/{len(chunk_paths)} transcribed. Text: {transcription.get('text', '')[:50]}...")
        print(f"Total number of segments: {len(transcription.get('segments', []))}")
        print(f"Total number of words: {len(transcription.get('words', []))}")

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
                "word": word.get("word", ""),
                "probability": word.get("probability", 0)
            }
            word_id += 1
            full_result["words"].append(simplified_word)

        time_offset += chunk_duration
        os.remove(chunk_path)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2)

    shutil.rmtree(temp_audio_chunks_dir)
    print(f"Temporary chunks directory {temp_audio_chunks_dir} removed")

    print(f"Timestamped word-level transcription successfully finished! Result: {output_json}")
    return output_json


def transcribe(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment", "word"],
                language="ru",
                temperature=0.1
            )

        if hasattr(response, "model_dump"):
            return response.model_dump()
        elif hasattr(response, "to_dict"):
            return response.to_dict()
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
                    "end": getattr(word, "end", 0),
                    "probability": getattr(word, "probability", 0)
                }
                result["words"].append(word_dict)

            return result
    except Exception as e:
        print(f"Error transcribing {file_path}: {str(e)}")
        return {"text": "", "segments": [], "words": []}
