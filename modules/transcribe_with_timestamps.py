import json
import os
from pydub import AudioSegment
from utils.audio_utils import split_audio
import openai


def transcribe_audio_with_timestamps(input_audio):
    base_name = os.path.splitext(os.path.basename(input_audio))[0]

    timestamped_transcriptions_dir = "output/timestamped_transcriptions"
    temp_audio_chunks_dir = "output/temp_audio_chunks"
    os.makedirs(timestamped_transcriptions_dir, exist_ok=True)
    os.makedirs(temp_audio_chunks_dir, exist_ok=True)
    output_json = os.path.join(timestamped_transcriptions_dir, f"{base_name}_timestamped.json")

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

    print(f"Timestamped word-level transcription successfully finished! Result: {output_json}")
    return output_json


def transcribe(file_path):
    with open(file_path, "rb") as audio_file:
        response = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
            language="ru",
            temperature=0.1
        )

    try:
        if hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        elif hasattr(response, "to_dict"):
            response_dict = response.to_dict()
        else:
            response_dict = vars(response)
    except:
        response_dict = {
            "text": response.text if hasattr(response, "text") else "",
            "segments": [],
            "words": []
        }

        if hasattr(response, "segments"):
            for seg in response.segments:
                try:
                    if hasattr(seg, "model_dump"):
                        segment_dict = seg.model_dump()
                    elif hasattr(seg, "to_dict"):
                        segment_dict = seg.to_dict()
                    else:
                        segment_dict = vars(seg)

                    response_dict["segments"].append(segment_dict)
                except:
                    segment_dict = {
                        "id": seg.id if hasattr(seg, "id") else 0,
                        "start": seg.start if hasattr(seg, "start") else 0,
                        "end": seg.end if hasattr(seg, "end") else 0,
                        "text": seg.text if hasattr(seg, "text") else ""
                    }
                    response_dict["segments"].append(segment_dict)

        if hasattr(response, "words"):
            for word in response.words:
                try:
                    if hasattr(word, "model_dump"):
                        word_dict = word.model_dump()
                        if "probability" in word_dict:
                            del word_dict["probability"]
                    elif hasattr(word, "to_dict"):
                        word_dict = word.to_dict()
                        if "probability" in word_dict:
                            del word_dict["probability"]
                    else:
                        word_dict = vars(word)
                        if "probability" in word_dict:
                            del word_dict["probability"]

                    response_dict["words"].append(word_dict)
                except:
                    word_dict = {
                        "word": word.word if hasattr(word, "word") else "",
                        "start": word.start if hasattr(word, "start") else 0,
                        "end": word.end if hasattr(word, "end") else 0
                    }
                    response_dict["words"].append(word_dict)

    return response_dict
