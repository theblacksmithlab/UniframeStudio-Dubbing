import os
import json
import whisper
import torch
from pydub import AudioSegment

from utils.audio_utils import split_audio


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
        "segments": []
    }

    time_offset = 0
    segment_id = 0

    # Загружаем модель один раз перед обработкой чанков
    model = load_whisper_model()

    for i, chunk_path in enumerate(chunk_paths):
        print(f"Processing chunk {i + 1}/{len(chunk_paths)}: {chunk_path}")

        audio_chunk = AudioSegment.from_file(chunk_path)
        chunk_duration = len(audio_chunk) / 1000

        transcription = transcribe(chunk_path, model)

        print(f"Chunk {i + 1}/{len(chunk_paths)} transcribed. Text: {transcription.get('text', '')[:50]}...")
        print(f"Total number of segments: {len(transcription.get('segments', []))}")

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

        time_offset += chunk_duration
        os.remove(chunk_path)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2)

    print(f"Timestamped transcription successfully finished! Result: {output_json}")
    return output_json


def load_whisper_model(model_size="base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = whisper.load_model(model_size).to(device)
    return model


def transcribe(file_path, model):
    options = {
        "language": "ru",  # Аналог вашего параметра language
        "verbose": True,  # Для получения дополнительной информации
        "temperature": 0.2  # Аналог вашего параметра temperature
    }

    # Выполняем транскрибацию
    result = model.transcribe(file_path, **options)

    # Форматируем результат аналогично формату API OpenAI
    response_dict = {
        "text": result.get("text", ""),
        "segments": []
    }

    # Обрабатываем сегменты и добавляем временные метки
    for i, seg in enumerate(result.get("segments", [])):
        segment_dict = {
            "id": i,
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "text": seg.get("text", "")
        }
        response_dict["segments"].append(segment_dict)

    return response_dict

