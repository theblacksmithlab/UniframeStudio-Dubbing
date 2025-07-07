import shutil
import torch
import torchaudio
import subprocess
import os
import json

from pydub import AudioSegment

from modules.transcribe_with_timestamps import transcribe
from utils.audio_utils import split_audio
from utils.logger_config import get_job_logger, setup_logger

logger = setup_logger(name=__name__, log_file="logs/app.log")


def detect_speech_start_with_vad(audio_path, job_id):
    """
    Определяет начало речи в аудио файле с помощью Silero VAD
    """
    log = get_job_logger(logger, job_id)

    try:
        # Загружаем Silero VAD модель с правильным репозиторием
        log.info("Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True,
            force_reload=False
        )

        # Загружаем аудио с правильным sample rate
        log.info(f"Analyzing audio file: {audio_path}")
        wav, original_sr = torchaudio.load(audio_path)

        # Silero VAD требует 16kHz или 8kHz
        target_sr = 16000
        if original_sr != target_sr:
            log.info(f"Resampling from {original_sr}Hz to {target_sr}Hz for VAD")
            resampler = torchaudio.transforms.Resample(original_sr, target_sr)
            wav = resampler(wav)

        # Получаем speech timestamps
        speech_timestamps = utils[0](wav, model, sampling_rate=target_sr)

        if speech_timestamps:
            speech_start = speech_timestamps[0]['start'] / target_sr  # используем target_sr
            speech_end = speech_timestamps[-1]['end'] / target_sr

            log.info(f"✅ Speech detected from {speech_start:.2f}s to {speech_end:.2f}s")
            log.info(f"📈 Total speech segments found: {len(speech_timestamps)}")

            return {
                'speech_start': speech_start,
                'speech_end': speech_end,
                'segments_count': len(speech_timestamps),
                'all_segments': speech_timestamps
            }
        else:
            log.warning("❌ No speech detected in audio file")
            return {
                'speech_start': 0.0,
                'speech_end': 0.0,
                'segments_count': 0,
                'all_segments': []
            }

    except Exception as e:
        log.error(f"Error in VAD analysis: {e}")
        # Fallback - возвращаем начало файла
        return {
            'speech_start': 0.0,
            'speech_end': 0.0,
            'segments_count': 0,
            'all_segments': []
        }


def trim_audio_to_speech(audio_path, speech_start, job_id, min_trim_threshold=5.0):
    """
    Обрезает аудио до начала речи, если тишина больше порога
    """
    log = get_job_logger(logger, job_id)

    if speech_start < min_trim_threshold:
        log.info(f"Speech starts at {speech_start:.2f}s - no trimming needed (threshold: {min_trim_threshold}s)")
        return audio_path

    log.info(f"Trimming audio: removing first {speech_start:.2f} seconds")

    # Создаем путь для обрезанного файла
    base_name = os.path.splitext(audio_path)[0]
    trimmed_path = f"{base_name}_speech_trimmed.mp3"

    # Обрезаем аудио с помощью ffmpeg
    cmd = [
        'ffmpeg', '-y',  # -y для перезаписи
        '-ss', f'{speech_start:.3f}',  # начать с speech_start
        '-i', audio_path,
        '-c', 'copy',  # копировать без перекодирования
        trimmed_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        log.info(f"✅ Audio trimmed successfully: {trimmed_path}")
        return trimmed_path, speech_start  # возвращаем путь и offset

    except subprocess.CalledProcessError as e:
        log.error(f"Error trimming audio: {e}")
        log.error(f"FFmpeg stderr: {e.stderr}")
        return audio_path, 0.0  # возвращаем оригинал


def adjust_transcription_timestamps(transcription_result, speech_offset):
    """
    Корректирует timestamps в результате транскрипции с учетом обрезки
    """
    if speech_offset <= 0:
        return transcription_result

    # Корректируем timestamps в segments
    for segment in transcription_result.get("segments", []):
        segment["start"] += speech_offset
        segment["end"] += speech_offset

    # Корректируем timestamps в words
    for word in transcription_result.get("words", []):
        word["start"] += speech_offset
        word["end"] += speech_offset

    return transcription_result


# Обновленная функция transcribe с интеграцией VAD
def transcribe_with_vad(job_id, file_path, source_language=None, openai_api_key=None,
                        transcription_keywords=None, enable_vad=True, min_trim_threshold=5.0):
    log = get_job_logger(logger, job_id)

    original_audio_path = file_path
    speech_offset = 0.0

    if enable_vad:
        log.info("🔍 Analyzing audio for speech detection...")

        vad_result = detect_speech_start_with_vad(file_path, job_id)
        speech_start = vad_result['speech_start']

        if speech_start > min_trim_threshold:
            trimmed_audio, speech_offset = trim_audio_to_speech(
                file_path, speech_start, job_id, min_trim_threshold
            )
            file_path = trimmed_audio

            log.info(f"📊 VAD Results:")
            log.info(f"  - Original file: {original_audio_path}")
            log.info(f"  - Trimmed file: {file_path}")
            log.info(f"  - Speech offset: {speech_offset:.2f}s")
        else:
            log.info(f"💡 No significant silence detected, using original audio")

    log.info("🎯 Starting transcription...")
    transcription = transcribe(
        job_id, file_path, source_language, openai_api_key, transcription_keywords
    )

    if speech_offset > 0:
        log.info(f"⏰ Adjusting timestamps by +{speech_offset:.2f}s")
        transcription = adjust_transcription_timestamps(transcription, speech_offset)

    if file_path != original_audio_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            log.info(f"🗑️ Cleaned up temporary file: {file_path}")
        except:
            log.warning(f"⚠️ Could not remove temporary file: {file_path}")

    return transcription


# Интеграция в основную функцию transcribe_audio_with_timestamps
def transcribe_audio_with_timestamps_vad(
        input_audio,
        job_id,
        source_language=None,
        output_file=None,
        openai_api_key=None,
        transcription_keywords=None,
        enable_vad=True,
        min_trim_threshold=5.0
):
    log = get_job_logger(logger, job_id)

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
    log.info(f"Initial audio duration: {initial_audio_duration:.3f} seconds")

    temp_audio_chunks_dir = f"jobs/{job_id}/output/temp_audio_chunks"
    os.makedirs(temp_audio_chunks_dir, exist_ok=True)

    chunk_paths = split_audio(job_id, input_audio, temp_audio_chunks_dir)

    full_result = {
        "text": "",
        "segments": [],
        "words": [],
        "vad_info": {}
    }

    time_offset = 0
    segment_id = 0
    word_id = 0

    for i, chunk_path in enumerate(chunk_paths):
        log.info(f"Processing chunk {i + 1}/{len(chunk_paths)}: {chunk_path}")

        audio_chunk = AudioSegment.from_file(chunk_path)
        chunk_duration = len(audio_chunk) / 1000

        use_vad = enable_vad and (i == 0)

        if use_vad:
            transcription = transcribe_with_vad(
                job_id, chunk_path, source_language, openai_api_key,
                transcription_keywords, enable_vad=True, min_trim_threshold=min_trim_threshold
            )
        else:
            transcription = transcribe(
                job_id, chunk_path, source_language, openai_api_key, transcription_keywords
            )

        log.info(f"Chunk {i + 1}/{len(chunk_paths)} transcribed. Text: {transcription.get('text', '')[:50]}...")
        log.info(f"Total number of segments: {len(transcription.get('segments', []))}")
        log.info(f"Total number of words: {len(transcription.get('words', []))}")

        # Сохраняем VAD info для первого чанка
        if i == 0 and 'vad_info' in transcription:
            full_result['vad_info'] = transcription['vad_info']

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

    # Подсчитываем outro gap
    outro_gap_duration = 0.0
    if full_result["segments"]:
        last_segment_end = full_result["segments"][-1]["end"]
        outro_gap_duration = max(0.0, initial_audio_duration - last_segment_end)
        log.info(f"Last segment ends at: {last_segment_end:.3f}s")
        log.info(f"Outro gap duration: {outro_gap_duration:.3f}s")

    full_result["outro_gap_duration"] = outro_gap_duration

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2)

    if os.path.exists(temp_audio_chunks_dir) and len(chunk_paths) > 1:
        shutil.rmtree(temp_audio_chunks_dir)

    return output_file