import os
import subprocess
from utils.logger_config import setup_logger, get_job_logger

logger = setup_logger(name=__name__, log_file="logs/app.log")


def extract_audio(
        input_video_path,
        job_id,
        extracted_audio_path=None,
        original_hq_audio_path=None,
        original_wav_audio_path=None,
):
    log = get_job_logger(logger, job_id)

    def check_cuda_available():
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    cuda_available = check_cuda_available()

    if cuda_available:
        log.info("CUDA GPU detected, using hardware acceleration")
        hwaccel_args = ["-hwaccel", "cuda"]
    else:
        log.info("CUDA not available, using CPU")
        hwaccel_args = []

    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_dir = os.path.dirname(input_video_path)

    if extracted_audio_path is None:
        extracted_audio_path = os.path.join(output_dir, f"{base_name}.mp3")

    if original_hq_audio_path is None:
        original_hq_audio_path = os.path.join(output_dir, f"{base_name}_44100.mp3")

    if original_wav_audio_path is None:
        original_wav_audio_path = os.path.join(output_dir, f"{base_name}_44100.wav")

    command_transcription = [
                                "ffmpeg", "-y"
                            ] + hwaccel_args + [
                                "-i", input_video_path,
                                "-vn",
                                "-codec:a", "libmp3lame",
                                "-qscale:a", "2",
                                "-ac", "1",
                                "-ar", "24000",
                                extracted_audio_path
                            ]

    command_hq_mp3 = [
                         "ffmpeg", "-y"
                     ] + hwaccel_args + [
                         "-i", input_video_path,
                         "-vn",
                         "-acodec", "mp3",
                         "-b:a", "320k",
                         "-ar", "44100",
                         original_hq_audio_path
                     ]

    command_wav = [
                      "ffmpeg", "-y"
                  ] + hwaccel_args + [
                      "-i", input_video_path,
                      "-vn",
                      "-acodec", "pcm_s16le",
                      "-ar", "44100",
                      original_wav_audio_path
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
        log.info(f"{description} successfully extracted: {output_path} (Size: {file_size_mb:.2f} MB)")

        return file_size_mb

    transcription_size = run_extraction(command_transcription, extracted_audio_path, "Extracted audio")
    hq_mp3_size = run_extraction(command_hq_mp3, original_hq_audio_path, "High-quality MP3 audio")
    wav_size = run_extraction(command_wav, original_wav_audio_path, "WAV audio")

    if transcription_size > 25:
        log.warning(f"WARNING: Transcription audio file is larger than 25MB ({transcription_size:.2f} MB). "
                       f"It will be split into chunks at the transcription step.")

    log.info(f"Audio extraction completed:")
    log.info(f"  - Transcription: {os.path.basename(extracted_audio_path)} ({transcription_size:.2f} MB)")
    log.info(f"  - High-quality MP3: {os.path.basename(original_hq_audio_path)} ({hq_mp3_size:.2f} MB)")
    log.info(f"  - WAV processing: {os.path.basename(original_wav_audio_path)} ({wav_size:.2f} MB)")

    return extracted_audio_path
