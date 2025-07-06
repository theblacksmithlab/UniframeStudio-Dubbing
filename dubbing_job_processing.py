import json
import os
import shutil
import subprocess
import sys
from modules.audio_processor import AudioProcessor
from modules.gender_voice_mapping import run_gender_and_voice_analysis_step
from modules.job_status import update_job_status
from modules.video_to_audio_conversion import extract_audio
from utils.logger_config import setup_logger, get_job_logger
from utils.transcription_review import get_review_result
from modules.job_status import STEP_DESCRIPTIONS


logger = setup_logger(name=__name__, log_file="logs/app.log")


def run_command(command, job_id=None, description=None, **kwargs):
    if job_id:
        from utils.logger_config import get_job_logger
        log = get_job_logger(logger, job_id)
    else:
        log = logger

    try:
        kwargs.setdefault('capture_output', True)
        kwargs.setdefault('text', True)

        result = subprocess.run(command, **kwargs)

        if result.returncode != 0:
            command_name = os.path.basename(command[0]) if command else "unknown"

            context = description or f"'{command_name}'"
            log.error(f"Command {context} failed (exit code {result.returncode})")

            if result.stderr:
                log.error(f"Error details: {result.stderr.strip()}")

            return False

        return True

    except Exception as e:
        log.exception(f"Error executing command: {e}")
        return False


def create_stereo_version(input_file, output_file):
    cmd = [
        'ffmpeg', '-y',
        '-i', input_file,
        '-filter_complex', '[0]pan=stereo|c0=c0|c1=c0',
        output_file
    ]

    subprocess.run(cmd, capture_output=True, text=True)
    return os.path.exists(output_file)


def combine_video_and_audio(video_path, audio_path, output_path, job_id, step_description):
    if job_id:
        from utils.logger_config import get_job_logger
        log = get_job_logger(logger, job_id)
    else:
        log = logger

    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '320k',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_path
        ]

        log.info(f"Combining: {os.path.basename(video_path)} + {os.path.basename(audio_path)}")

        if run_command(cmd, job_id=job_id, description=step_description):
            log.info(f"Successfully combined: {os.path.basename(output_path)}")
            return True
        else:
            return False

    except Exception as e:
        log.error(f"Combining video and audio failed with error: {e}")
        return False


def mix_audio_tracks(tts_audio_path, background_audio_path, output_path, tts_volume=1.0, bg_volume=0.25, job_id=None):
    if job_id:
        from utils.logger_config import get_job_logger
        log = get_job_logger(logger, job_id)
    else:
        log = logger

    try:
        log.info(f"Mixing TTS (vol: {tts_volume}) + Background (vol: {bg_volume})")

        cmd = [
            'ffmpeg', '-y',
            '-i', tts_audio_path,
            '-i', background_audio_path,
            '-filter_complex',
            f'[0:a]volume={tts_volume}[tts];[1:a]volume={bg_volume}[bg];[tts][bg]amix=inputs=2:duration=shortest[out]',
            '-map', '[out]',
            '-acodec', 'mp3',
            '-b:a', '320k',
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            log.info(f"Successfully mixed audio: {os.path.basename(output_path)}")
            return True
        else:
            log.error(f"Audio mixing failed: {result.stderr}")
            return False

    except Exception as e:
        log.error(f"Audio mixing failed: {e}")
        return False


def process_job(job_id, source_language=None, target_language=None, tts_provider=None, tts_voice=None,
                elevenlabs_api_key=None, openai_api_key=None, transcription_keywords=None):
    job_logger = get_job_logger(logger, job_id)

    job_dir = f"jobs/{job_id}"
    input_video_dir = f"{job_dir}/video_input"
    processing_jsons_dir = f"{job_dir}/timestamped_transcriptions"
    audio_input_dir = f"{job_dir}/audio_input"
    result_output_dir = f"{job_dir}/output"

    os.makedirs(input_video_dir, exist_ok=True)
    os.makedirs(processing_jsons_dir, exist_ok=True)
    os.makedirs(audio_input_dir, exist_ok=True)
    os.makedirs(result_output_dir, exist_ok=True)

    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.3gp', '.m4v']:
        video_files.extend([f for f in os.listdir(input_video_dir) if f.lower().endswith(ext)])

    if not video_files:
        return {
            "status": "error",
            "message": f"No video files found in {input_video_dir}. Please upload a video file first.",
            "job_id": job_id
        }

    if len(video_files) > 1:
        job_logger.warning(f"Multiple video files found in {input_video_dir}. Using the first one: {video_files[0]}")

    video_path = os.path.join(input_video_dir, video_files[0])
    base_name = os.path.splitext(video_files[0])[0]

    job_logger.info(f"{'=' * 50}")
    job_logger.info(f"Processing job: {job_id}")
    if source_language:
        job_logger.info(f"Source language: {source_language}")
    job_logger.info(f"Target language: {target_language}")
    job_logger.info(f"TTS provider: {tts_provider}")
    job_logger.info(f"TTS voice: {tts_voice}")
    job_logger.info(f"{'=' * 50}")

    # [Step 1]
    current_step = 3
    step_description = STEP_DESCRIPTIONS[current_step].rstrip('.')
    update_job_status(job_id=job_id, step=current_step)
    job_logger.info(f"{'=' * 25}")
    job_logger.info(f"[Step 1] {STEP_DESCRIPTIONS[current_step]}")
    original_audio_path = f"{audio_input_dir}/{base_name}.mp3"
    original_hq_audio_path = f"{audio_input_dir}/{base_name}_44100.mp3"
    original_wav_audio_path = f"{audio_input_dir}/{base_name}_44100.wav"
    extract_cmd = [sys.executable, "cli.py", "extract_audio", "--job_id", job_id, "--input", video_path, "--output",
                   original_audio_path, "--hq_output", original_hq_audio_path, "--wav_output", original_wav_audio_path]

    if not run_command(extract_cmd, job_id=job_id, description=step_description):
        return {
            "status": "error",
            "message": f"Failed to extract audio from video-file",
            "job_id": job_id,
            "step": current_step
        }

    if not os.path.exists(original_audio_path):
        return {
            "status": "error",
            "message": "Failed to extract audio from video-file",
            "job_id": job_id,
            "step": current_step
        }


    # [Step 2]
    current_step = 4
    step_description = STEP_DESCRIPTIONS[current_step].rstrip('.')
    update_job_status(job_id=job_id, step=current_step)
    job_logger.info(f"{'=' * 25}")
    job_logger.info(f"[Step 2] {STEP_DESCRIPTIONS[current_step]}")
    transcription_path = os.path.join(processing_jsons_dir, f"{base_name}_transcribed.json")
    transcribe_cmd = [sys.executable, "cli.py", "transcribe",
                      "--input", original_audio_path,
                      "--output", transcription_path,
                      "--job_id", job_id,
                      "--openai_api_key", openai_api_key]

    if source_language:
        transcribe_cmd.extend(["--source_language", source_language])

    if transcription_keywords:
        transcribe_cmd.extend(["--transcription_keywords", transcription_keywords])

    if not run_command(transcribe_cmd, job_id=job_id, description=step_description):
        return {
            "status": "error",
            "message": f"Failed to transcribe audio",
            "job_id": job_id,
            "step": current_step
        }

    if not os.path.exists(transcription_path):
        return {
            "status": "error",
            "message": f"Failed to transcribe audio",
            "job_id": job_id,
            "step": current_step
        }


    # [Step 3]
    current_step = 5
    step_description = STEP_DESCRIPTIONS[current_step].rstrip('.')
    update_job_status(job_id=job_id, step=current_step)
    job_logger.info(f"{'=' * 25}")
    job_logger.info(f"[Step 3] {STEP_DESCRIPTIONS[current_step]}")
    corrected_path = os.path.join(processing_jsons_dir, f"{base_name}_transcribed_corrected.json")
    correct_cmd = [sys.executable, "cli.py", "correct", "--job_id", job_id, "--input", transcription_path, "--output",
                   corrected_path]

    if not run_command(correct_cmd, job_id=job_id, description=step_description):
        return {
            "status": "error",
            "message": f"Failed to structure transcription",
            "job_id": job_id,
            "step": current_step
        }

    if not os.path.exists(corrected_path):
        return {
            "status": "error",
            "message": f"Failed to structure transcription",
            "job_id": job_id,
            "step": current_step
        }


    # [Step 4]
    current_step = 6
    step_description = STEP_DESCRIPTIONS[current_step].rstrip('.')
    update_job_status(job_id=job_id, step=current_step)
    job_logger.info(f"{'=' * 25}")
    job_logger.info(f"[Step 4] {STEP_DESCRIPTIONS[current_step]}")
    cleaned_path = os.path.join(processing_jsons_dir, f"{base_name}_transcribed_corrected_cleaned.json")
    cleanup_cmd = [sys.executable, "cli.py", "cleanup", "--job_id", job_id, "--input", corrected_path, "--output",
                   cleaned_path]

    if not run_command(cleanup_cmd, job_id=job_id, description=step_description):
        return {
            "status": "error",
            "message": f"Failed to clean transcription",
            "job_id": job_id,
            "step": current_step
        }

    if not os.path.exists(cleaned_path):
        return {
            "status": "error",
            "message": f"Failed to clean transcription",
            "job_id": job_id,
            "step": current_step
        }


    # [Step 5]
    current_step = 7
    step_description = STEP_DESCRIPTIONS[current_step].rstrip('.')
    update_job_status(job_id=job_id, step=current_step)
    job_logger.info(f"{'=' * 25}")
    job_logger.info(f"[Step 5] {STEP_DESCRIPTIONS[current_step]}")
    optimized_path = os.path.join(processing_jsons_dir, f"{base_name}_transcribed_corrected_cleaned_optimized.json")
    optimize_cmd = [sys.executable, "cli.py", "optimize", "--job_id", job_id, "--input", cleaned_path, "--output",
                    optimized_path]

    if openai_api_key:
        optimize_cmd.extend(["--openai_api_key", openai_api_key])

    if not run_command(optimize_cmd, job_id=job_id, description=step_description):
        return {
            "status": "error",
            "message": f"Failed to optimize transcription segments",
            "job_id": job_id,
            "step": current_step
        }

    if not os.path.exists(optimized_path):
        return {
            "status": "error",
            "message": f"Failed to optimize transcription segments",
            "job_id": job_id,
            "step": current_step
        }


    # [Step 6]
    current_step = 8
    step_description = STEP_DESCRIPTIONS[current_step].rstrip('.')
    update_job_status(job_id=job_id, step=current_step)
    job_logger.info(f"{'=' * 25}")
    job_logger.info(f"[Step 6] {STEP_DESCRIPTIONS[current_step]}")
    adjusted_path = os.path.join(
        processing_jsons_dir,
        f"{base_name}_transcribed_corrected_cleaned_optimized_adjusted.json"
    )
    adjust_cmd = [sys.executable, "cli.py", "adjust_timing", "--job_id", job_id, "--input", optimized_path, "--output",
                  adjusted_path]

    if not run_command(adjust_cmd, job_id=job_id, description=step_description):
        return {
            "status": "error",
            "message": f"Failed to adjust timing for transcription segments",
            "job_id": job_id,
            "step": current_step
        }

    if not os.path.exists(adjusted_path):
        return {
            "status": "error",
            "message": f"Failed to adjust timing for transcription segments",
            "job_id": job_id,
            "step": current_step
        }


    # [Step 7]
    current_step = 9
    step_description = STEP_DESCRIPTIONS[current_step].rstrip('.')
    update_job_status(job_id=job_id, step=current_step)
    job_logger.info(f"{'=' * 25}")
    job_logger.info(f"[Step 7] {STEP_DESCRIPTIONS[current_step]}")
    translated_path = os.path.join(
        processing_jsons_dir,
        f"{base_name}_transcribed_corrected_cleaned_optimized_adjusted_translated.json"
    )
    translate_cmd = [sys.executable, "cli.py", "translate", "--job_id", job_id, "--input", adjusted_path, "--output",
                     translated_path]

    translate_cmd.extend(["--target_language", target_language])

    if openai_api_key:
        translate_cmd.extend(["--openai_api_key", openai_api_key])

    if not run_command(translate_cmd, job_id=job_id, description=step_description):
        return {
            "status": "error",
            "message": f"Failed to translate transcription segments",
            "job_id": job_id,
            "step": current_step
        }

    if not os.path.exists(translated_path):
        return {
            "status": "error",
            "message": f"Failed to translate transcription segments",
            "job_id": job_id,
            "step": current_step
        }


    # [Step 8] Gender Analysis & Voice Mapping
    if tts_provider == "openai":
        current_step = 10
        update_job_status(job_id=job_id, step=current_step)
        job_logger.info(f"{'=' * 25}")
        job_logger.info(f"[Step 8] {STEP_DESCRIPTIONS[current_step]}")

        if not os.path.exists(original_audio_path):
            job_logger.warning(f"Audio file not found at {original_audio_path}")
            job_logger.info(f"Extracting audio from video: {video_path}")

            try:
                extracted_audio_path = extract_audio(video_path, original_audio_path, original_hq_audio_path,
                                                     original_wav_audio_path, job_id=job_id)
                job_logger.info(f"Audio successfully extracted to: {extracted_audio_path}")
            except Exception as e:
                job_logger.error(f"Failed to extract audio: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to extract audio for gender analysis: {str(e)}",
                    "job_id": job_id,
                    "step": current_step
                }
        else:
            job_logger.info(f"Using existing audio file: {original_audio_path}")

        if not run_gender_and_voice_analysis_step(job_id, original_audio_path, translated_path, tts_provider):
            return {
                "status": "error",
                "message": "Failed to analyze gender and map voices",
                "job_id": job_id,
                "step": current_step
            }

    else:
        job_logger.info(f"Skipping gender/voice analysis for TTS provider: {tts_provider}")

    # [review step]
    job_logger.info(f"{'=' * 25}")
    job_logger.info(f"[Review step] Translated transcription review required...")
    original_translated_transcription = translated_path

    get_review_result(job_id, original_translated_transcription)

    # [Step 9]
    current_step = 12
    step_description = STEP_DESCRIPTIONS[current_step].rstrip('.')
    update_job_status(job_id=job_id, step=current_step)
    job_logger.info(f"{'=' * 25}")
    job_logger.info(f"[Step 9] {STEP_DESCRIPTIONS[current_step]}")

    tts_cmd = [sys.executable, "cli.py", "tts",
               "--input", translated_path,
               "--dealer", tts_provider,
               "--voice", tts_voice,
               "--job_id", job_id]

    if tts_provider == "elevenlabs":
        if elevenlabs_api_key:
            tts_cmd.extend(["--elevenlabs_api_key", elevenlabs_api_key])
        else:
            return {
                "status": "error",
                "message": "ElevenLabs API key is required but not provided",
                "job_id": job_id,
                "step": current_step
            }

    if tts_provider == "openai":
        if openai_api_key:
            tts_cmd.extend(["--openai_api_key", openai_api_key])
        else:
            return {
                "status": "error",
                "message": "OpenAI API key is required but not provided",
                "job_id": job_id,
                "step": current_step
            }

    if not run_command(tts_cmd, job_id=job_id, description=step_description):
        return {
            "status": "error",
            "message": f"Failed to generate TTS audio",
            "job_id": job_id,
            "step": current_step
        }

    expected_audio_dir = os.path.join(result_output_dir, "audio_result")
    expected_audio_path = os.path.join(expected_audio_dir, "new_audio.mp3")

    if not os.path.exists(expected_audio_path):
        return {
            "status": "error",
            "message": f"Failed to generate TTS audio",
            "job_id": job_id,
            "step": current_step
        }

    audio_segments_dir = os.path.join(result_output_dir, "audio_segments")
    if not os.path.exists(audio_segments_dir) or not any(f.endswith('.mp3') for f in os.listdir(audio_segments_dir)):
        job_logger.error(f"Warning: Audio segments directory {audio_segments_dir} is empty or missing")
        return {
            "status": "error",
            "message": f"Failed to generate TTS audio",
            "job_id": job_id,
            "step": current_step
        }


    # [Step 10]
    current_step = 13
    step_description = STEP_DESCRIPTIONS[current_step].rstrip('.')
    update_job_status(job_id=job_id, step=current_step)
    job_logger.info(f"{'=' * 25}")
    job_logger.info(f"[Step 10] {STEP_DESCRIPTIONS[current_step]}")
    auto_correct_cmd = [sys.executable, "cli.py", "auto-correct",
                        "--input", translated_path,
                        "--job_id", job_id,
                        "--dealer", tts_provider,
                        "--voice", tts_voice,
                        "--attempts", "5",
                        "--threshold", "0.2"]

    if tts_provider == "elevenlabs" and elevenlabs_api_key:
        auto_correct_cmd.extend(["--elevenlabs_api_key", elevenlabs_api_key])
    if tts_provider == "openai" and openai_api_key:
        auto_correct_cmd.extend(["--openai_api_key", openai_api_key])

    if not run_command(auto_correct_cmd, job_id=job_id, description=step_description):
        return {
            "status": "error",
            "message": f"Failed to auto-correct transcription segments",
            "job_id": job_id,
            "step": current_step
        }

    audio_result_dir = os.path.join(result_output_dir, "audio_result")
    final_audio_path = os.path.join(audio_result_dir, "new_audio.mp3")
    final_stereo_path = os.path.join(audio_result_dir, "new_audio_stereo.mp3")

    job_logger.info(f"Audio processing completed!")
    job_logger.info(f"Clean audio: {final_audio_path}")
    job_logger.info(f"Clean stereo audio: {final_stereo_path}")

    # [Step 11]
    current_step = 14
    update_job_status(job_id=job_id, step=current_step)
    job_logger.info(f"{'=' * 25}")
    job_logger.info(f"[Step 11] {STEP_DESCRIPTIONS[current_step]}")

    with open(translated_path, 'r') as f:
        segments_data = json.load(f)

    original_hq_audio_path = os.path.join(audio_input_dir, f"{base_name}_44100.wav")

    audio_processor = AudioProcessor(job_id, original_hq_audio_path, segments_data)
    audio_processor.extract_audio_segments()
    audio_processor.process_audio_segments()
    background_audio_path = audio_processor.combine_background_audio()

    mixed_audio_path = None
    mixed_stereo_path = None

    if background_audio_path and os.path.exists(background_audio_path):
        # Step [12]
        current_step = 15
        update_job_status(job_id=job_id, step=current_step)
        job_logger.info(f"{'=' * 25}")
        job_logger.info(f"[Step 12] {STEP_DESCRIPTIONS[current_step]}")

        mixed_audio_path = os.path.join(audio_result_dir, "mixed_audio_with_bg.mp3")
        mixed_stereo_path = os.path.join(audio_result_dir, "mixed_stereo_with_bg.mp3")

        if mix_audio_tracks(final_audio_path, background_audio_path, mixed_audio_path, job_id):
            job_logger.info("Successfully created mixed audio with background")
            if not create_stereo_version(mixed_audio_path, mixed_stereo_path):
                job_logger.warning("Failed to create stereo version of mixed audio, will use mono")
                mixed_stereo_path = mixed_audio_path
        else:
            job_logger.error("Failed to mix audio tracks, skipping background audio versions")
            mixed_audio_path = None
            mixed_stereo_path = None

        audio_processor.cleanup()
    else:
        job_logger.warning("Background audio creation failed, skipping mixed audio versions")

    # [Step 13]
    current_step = 16
    step_description = STEP_DESCRIPTIONS[current_step].rstrip('.')
    update_job_status(job_id=job_id, step=current_step)
    job_logger.info(f"{'=' * 25}")
    job_logger.info(f"[Step 13] {STEP_DESCRIPTIONS[current_step]}")
    video_result_dir = os.path.join(result_output_dir, "video_result")
    os.makedirs(video_result_dir, exist_ok=True)

    tts_based_video_path = os.path.join(video_result_dir, f"{base_name}_tts_based.mp4")

    process_video_cmd = [sys.executable, "cli.py", "process_video",
                         "--job_id", job_id,
                         "--input_video", video_path,
                         "--json_file", translated_path,
                         "--output_video", tts_based_video_path]

    if not run_command(process_video_cmd, job_id=job_id, description=step_description):
        return {
            "status": "error",
            "message": f"Failed to process video",
            "job_id": job_id,
            "step": current_step
        }

    if not os.path.exists(tts_based_video_path):
        return {
            "status": "error",
            "message": f"Expected tts-based video file not created",
            "job_id": job_id,
            "step": current_step
        }

    job_logger.info(f"Video processing completed!")
    job_logger.info(f"TTS-based videos created successfully")

    # [Step 14]
    current_step = 17
    step_description = STEP_DESCRIPTIONS[current_step].rstrip('.')
    update_job_status(job_id=job_id, step=current_step)
    job_logger.info(f"{'=' * 25}")
    job_logger.info(f"[Step 14] {STEP_DESCRIPTIONS[current_step]}")

    final_video_path = os.path.join(video_result_dir, f"{base_name}_processed.mp4")

    if os.path.exists(final_stereo_path):
        new_audio_track = final_stereo_path
    else:
        job_logger.warning(f"Stereo audio not found, using mono audio: {os.path.basename(final_audio_path)}")
        new_audio_track = final_audio_path

    if not combine_video_and_audio(tts_based_video_path, new_audio_track, final_video_path, job_id, step_description):
        return {
            "status": "error",
            "message": f"Failed to create final video file",
            "job_id": job_id,
            "step": current_step
        }

    job_logger.info(f"Main video result created successfully: {final_video_path}")

    if mixed_audio_path and os.path.exists(mixed_audio_path):
        final_video_with_bg_path = os.path.join(video_result_dir, f"{base_name}_processed_with_bg.mp4")

        if mixed_stereo_path and os.path.exists(mixed_stereo_path):
            new_audio_track_with_background = mixed_stereo_path
        else:
            job_logger.warning(
                f"Stereo audio with background not found, using mono audio: {os.path.basename(mixed_audio_path)}")
            new_audio_track_with_background = mixed_audio_path

        if combine_video_and_audio(tts_based_video_path, new_audio_track_with_background, final_video_with_bg_path,
                                   job_id, step_description):
            job_logger.info(f"Video with background created successfully: {final_video_with_bg_path}")
        else:
            job_logger.error("Failed to create video with background, but continuing")
            final_video_with_bg_path = None
    else:
        job_logger.info("Skipping video with background creation - no mixed audio available")
        final_video_with_bg_path = None

    job_logger.info(f"Video-audio combination completed!")

    result = {
        "status": "success",
        "job_id": job_id,
        "input_file": video_path,
        "params": {
            "source_language": source_language,
            "target_language": target_language,
            "tts_provider": tts_provider,
            "tts_voice": tts_voice
        },
        "output_files": {
        "audio": original_audio_path,
        "transcription": transcription_path,
        "corrected": corrected_path,
        "cleaned": cleaned_path,
        "optimized": optimized_path,
        "adjusted": adjusted_path,
        "translated": translated_path,
        "final_audio": final_audio_path,
        "final_audio_stereo": final_stereo_path,
        "final_video": final_video_path,
        "final_video_tts_based": tts_based_video_path

        },
        "steps_completed": ["extract", "transcribe", "correct", "cleanup", "optimize", "adjust", "translate",
                            "tts", "auto-correct", "final_audio_files_creation", "process_video",
                            "combine_video_audio"]
    }

    # [Cleanup]
    job_logger.info(f"{'=' * 25}")
    job_logger.info(f"[Cleanup] Moving final files and cleaning up temporary data...")

    job_result_dir = os.path.join(f"jobs/{job_id}", "job_result")
    os.makedirs(job_result_dir, exist_ok=True)

    files_to_move = [
        (os.path.join(f"jobs/{job_id}", "job_params.json"),
         os.path.join(job_result_dir, "job_params.json")),
        (translated_path,
         os.path.join(job_result_dir, os.path.basename(translated_path))),
        (final_audio_path,
         os.path.join(job_result_dir, os.path.basename(final_audio_path))),
        (final_stereo_path,
         os.path.join(job_result_dir, os.path.basename(final_stereo_path))),
        (final_video_path,
         os.path.join(job_result_dir, os.path.basename(final_video_path))),
        (tts_based_video_path,
         os.path.join(job_result_dir, os.path.basename(tts_based_video_path)))
    ]

    if mixed_audio_path and os.path.exists(mixed_audio_path):
        files_to_move.append((mixed_audio_path, os.path.join(job_result_dir, os.path.basename(mixed_audio_path))))

    if mixed_stereo_path and os.path.exists(mixed_stereo_path) and mixed_stereo_path != mixed_audio_path:
        files_to_move.append((mixed_stereo_path, os.path.join(job_result_dir, os.path.basename(mixed_stereo_path))))

    if final_video_with_bg_path and os.path.exists(final_video_with_bg_path):
        files_to_move.append(
            (final_video_with_bg_path, os.path.join(job_result_dir, os.path.basename(final_video_with_bg_path))))

    moved_files = []
    for src, dst in files_to_move:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            moved_files.append(os.path.basename(dst))
            job_logger.info(f"Copied: {os.path.basename(src)} to result directory")
        else:
            job_logger.warning(f"Warning: File not found: {src}")

    result["output_files"]["translated"] = os.path.join(job_result_dir, os.path.basename(translated_path))
    result["output_files"]["final_audio"] = os.path.join(job_result_dir, os.path.basename(final_audio_path))
    result["output_files"]["final_audio_stereo"] = os.path.join(job_result_dir, os.path.basename(final_stereo_path))
    result["output_files"]["final_video"] = os.path.join(job_result_dir, os.path.basename(final_video_path))
    result["output_files"]["final_video_tts_based"] = os.path.join(job_result_dir,
                                                                   os.path.basename(tts_based_video_path))

    if mixed_audio_path and os.path.exists(os.path.join(job_result_dir, os.path.basename(mixed_audio_path))):
        result["output_files"]["final_audio_with_bg"] = os.path.join(job_result_dir, os.path.basename(mixed_audio_path))

    if mixed_stereo_path and os.path.exists(os.path.join(job_result_dir, os.path.basename(mixed_stereo_path))):
        result["output_files"]["final_audio_stereo_with_bg"] = os.path.join(job_result_dir,
                                                                            os.path.basename(mixed_stereo_path))

    if final_video_with_bg_path and os.path.exists(
            os.path.join(job_result_dir, os.path.basename(final_video_with_bg_path))):
        result["output_files"]["final_video_with_bg"] = os.path.join(job_result_dir,
                                                                     os.path.basename(final_video_with_bg_path))

    with open(os.path.join(job_result_dir, "pipeline_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    job_logger.info(f"Files copied to result directory: {job_result_dir}")
    job_logger.info(f"Result files: {moved_files}")

    job_logger.info(f"{'=' * 50}")
    job_logger.info(f"DUBBING JOB MAIN PIPELINE FINISHED SUCCESSFULLY!")
    job_logger.info(f"Job ID: {job_id}")
    job_logger.info(f"Original video: {video_path}")
    job_logger.info(f"Final video: {final_video_path}")
    job_logger.info(f"Total steps completed: {len(result['steps_completed'])}")
    job_logger.info(f"{'=' * 50}")

    return result


# def main():
#     parser = argparse.ArgumentParser(description="Complete video processing pipeline")
#     parser.add_argument("--job_id", required=True, help="Unique job identifier")
#     parser.add_argument("--source_language", help="Source language of the video (optional)")
#     parser.add_argument("--target_language", required=True, help="Target language for translation")
#     parser.add_argument("--tts_provider", choices=["openai", "elevenlabs"],
#                         required=True, help="TTS service provider")
#     parser.add_argument("--tts_voice", required=True, help="Voice identifier for TTS")
#     parser.add_argument("--elevenlabs_api_key", help="ElevenLabs API key (optional)")
#     parser.add_argument("--openai_api_key", required=True, help="OpenAI API key")
#
#     args = parser.parse_args()
#
#     if args.tts_provider == "elevenlabs" and not args.elevenlabs_api_key:
#         logger.error("Error: ElevenLabs API key is required for ElevenLabs TTS provider.")
#         return 1
#
#     os.makedirs("jobs", exist_ok=True)
#
#     result = process_job(
#         args.job_id,
#         source_language=args.source_language,
#         target_language=args.target_language,
#         tts_provider=args.tts_provider,
#         tts_voice=args.tts_voice,
#         elevenlabs_api_key=args.elevenlabs_api_key,
#         openai_api_key=args.openai_api_key,
#     )
#
#     if result["status"] == "error":
#         logger.error(f"Error processing job {args.job_id}: {result['message']}")
#         return 1
#
#     return 0
#
# if __name__ == "__main__":
#     exit_code = main()
#     sys.exit(exit_code)
