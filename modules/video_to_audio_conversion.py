import os
import subprocess
from utils.logger_config import setup_logger


logger = setup_logger(name=__name__, log_file="logs/app.log")


def extract_audio_legacy(input_video_path, extracted_audio_path=None):
    if extracted_audio_path is None:
        base_name = os.path.splitext(os.path.basename(input_video_path))[0]
        output_dir = os.path.dirname(input_video_path)
        extracted_audio_path = os.path.join(output_dir, f"{base_name}.mp3")

    command = [
        "ffmpeg", "-y",
        "-i", input_video_path,
        "-vn",
        "-codec:a", "libmp3lame",
        "-qscale:a", "2",
        "-ac", "1",
        "-ar", "24000",
        extracted_audio_path
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
    except Exception as e:
        raise RuntimeError(f"Error executing ffmpeg command: {e}")

    if not os.path.exists(extracted_audio_path):
        raise FileNotFoundError(f"Audio file was not created at {extracted_audio_path}")

    file_size_mb = os.path.getsize(extracted_audio_path) / (1024 * 1024)
    logger.info(f"Audio successfully extracted: {extracted_audio_path} (Size: {file_size_mb:.2f} MB)")

    if file_size_mb > 25:
        logger.warning(f"WARNING: Audio file is larger than 25MB ({file_size_mb:.2f} MB). "
              f"It will be split into chunks at the transcription step.")

    return extracted_audio_path


def extract_audio(input_video_path, extracted_audio_path=None):
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_dir = os.path.dirname(input_video_path)

    if extracted_audio_path is None:
        extracted_audio_path = os.path.join(output_dir, f"{base_name}.mp3")

    # Пути для дополнительных файлов
    hq_audio_path = os.path.join(output_dir, f"{base_name}_44100.mp3")
    wav_audio_path = os.path.join(output_dir, f"{base_name}_44100.wav")

    # 1. Для транскрипции (экономный, 24kHz моно)
    command_transcription = [
        "ffmpeg", "-y",
        "-i", input_video_path,
        "-vn",
        "-codec:a", "libmp3lame",
        "-qscale:a", "2",
        "-ac", "1",
        "-ar", "24000",
        extracted_audio_path
    ]

    # 2. Для обработки и микширования (качественный MP3, 44.1kHz)
    command_hq_mp3 = [
        "ffmpeg", "-y",
        "-i", input_video_path,
        "-vn",
        "-acodec", "mp3",
        "-b:a", "320k",
        "-ar", "44100",
        hq_audio_path
    ]

    # 3. Для точной обработки длительности (WAV без потерь, 44.1kHz)
    command_wav = [
        "ffmpeg", "-y",
        "-i", input_video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        wav_audio_path
    ]

    def run_extraction(command, output_path, description):
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error for {description}: {result.stderr}")
        except Exception as e:
            raise RuntimeError(f"Error executing ffmpeg command for {description}: {e}")

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"{description} file was not created at {output_path}")

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"{description} successfully extracted: {output_path} (Size: {file_size_mb:.2f} MB)")

        return file_size_mb

    # Извлекаем все три формата
    transcription_size = run_extraction(command_transcription, extracted_audio_path, "Transcription audio")
    hq_mp3_size = run_extraction(command_hq_mp3, hq_audio_path, "High-quality MP3 audio")
    wav_size = run_extraction(command_wav, wav_audio_path, "WAV audio")

    # Предупреждение о размере только для файла транскрипции
    if transcription_size > 25:
        logger.warning(f"WARNING: Transcription audio file is larger than 25MB ({transcription_size:.2f} MB). "
                       f"It will be split into chunks at the transcription step.")

    logger.info(f"Audio extraction completed:")
    logger.info(f"  - Transcription: {os.path.basename(extracted_audio_path)} ({transcription_size:.2f} MB)")
    logger.info(f"  - High-quality MP3: {os.path.basename(hq_audio_path)} ({hq_mp3_size:.2f} MB)")
    logger.info(f"  - WAV processing: {os.path.basename(wav_audio_path)} ({wav_size:.2f} MB)")

    # Сохраняем дополнительные файлы как атрибуты основного пути
    # Для обратной совместимости возвращаем только основной файл
    # Дополнительные файлы доступны по предсказуемым именам

    return extracted_audio_path
