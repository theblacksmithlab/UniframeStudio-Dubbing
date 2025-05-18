#!/usr/bin/env python3
import json
import os
import argparse
import shutil
import subprocess
import sys
from modules.job_status import update_job_status


def run_command(command):
    print(f"Executing: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command execution error: {e}")
        return False


def add_intro_outro_audio(input_audio, output_audio, resources_dir):
    intro_outro_path = os.path.join(resources_dir, "intro_outro_audio.mp3")

    if not os.path.exists(intro_outro_path):
        print(f"Warning: Intro/outro audio not found: {intro_outro_path}")
        return False

    cmd = [
        'ffmpeg', '-y',
        '-i', intro_outro_path,
        '-i', input_audio,
        '-i', intro_outro_path,
        '-filter_complex', '[0:a][1:a][2:a]concat=n=3:v=0:a=1',
        output_audio
    ]

    subprocess.run(cmd, capture_output=True, text=True)
    return os.path.exists(output_audio)


def create_stereo_version(input_file, output_file):
    cmd = [
        'ffmpeg', '-y',
        '-i', input_file,
        '-filter_complex', '[0]pan=stereo|c0=c0|c1=c0',
        output_file
    ]

    subprocess.run(cmd, capture_output=True, text=True)
    return os.path.exists(output_file)


def combine_video_and_audio(video_path, audio_path, output_path):
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

        print(f"Combining: {os.path.basename(video_path)} + {os.path.basename(audio_path)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Successfully combined: {os.path.basename(output_path)}")
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False

    except Exception as e:
        print(f"Exception combining video and audio: {e}")
        return False


def process_job(job_id, source_language=None, target_language=None, tts_provider=None, tts_voice=None,
                elevenlabs_api_key=None, openai_api_key=None, is_premium=False):
    job_dir = f"jobs/{job_id}"
    input_video_dir = f"{job_dir}/video_input"
    processing_jsons_dir = f"{job_dir}/timestamped_transcriptions"
    audio_input_dir = f"{job_dir}/audio_input"
    result_output_dir = f"{job_dir}/output"

    print(f"Creating directory structure for job {job_id}...")
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
        print(f"Warning: Multiple video files found in {input_video_dir}. Using the first one: {video_files[0]}")

    video_path = os.path.join(input_video_dir, video_files[0])
    base_name = os.path.splitext(video_files[0])[0]

    print(f"\n{'=' * 50}")
    print(f"Processing job: {job_id}")
    print(f"Video file: {video_path}")
    if source_language:
        print(f"Source language: {source_language}")
    print(f"Target language: {target_language}")
    print(f"TTS provider: {tts_provider}")
    print(f"TTS voice: {tts_voice}")

    # [Step 1]
    current_step = 3
    update_job_status(job_id=job_id, step=current_step)
    print(f"\n{'=' * 25}")
    print(f"\n[Step 1] Extracting audio from video {video_path}...")
    audio_path = f"{audio_input_dir}/{base_name}.mp3"
    extract_cmd = [sys.executable, "cli.py", "extract_audio", "--input", video_path, "--output", audio_path]

    if not run_command(extract_cmd):
        return {
            "status": "error",
            "message": f"Failed to extract audio from video-file",
            "job_id": job_id,
            "step": current_step
        }

    if not os.path.exists(audio_path):
        return {
            "status": "error",
            "message": f"Failed to extract audio from video-file",
            "job_id": job_id,
            "step": current_step
        }

    print(f"Audio file created: {audio_path}")
    print(f"\n")

    # [Step 2]
    current_step = 4
    update_job_status(job_id=job_id, step=current_step)
    print(f"\n{'=' * 25}")
    print(f"\n[Step 2] Transcribing audio {os.path.basename(audio_path)}...")
    transcription_path = os.path.join(processing_jsons_dir, f"{base_name}_transcribed.json")
    transcribe_cmd = [sys.executable, "cli.py", "transcribe",
                      "--input", audio_path,
                      "--output", transcription_path,
                      "--job_id", job_id,
                      "--openai_api_key", openai_api_key]

    if source_language:
        transcribe_cmd.extend(["--source_language", source_language])

    if not run_command(transcribe_cmd):
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

    print(f"Transcription file created: {transcription_path}")
    print(f"\n")

    # [Step 3]
    current_step = 5
    update_job_status(job_id=job_id, step=current_step)
    print(f"\n{'=' * 25}")
    print(f"\n[Step 3] Structuring transcription {os.path.basename(transcription_path)}...")
    corrected_path = os.path.join(processing_jsons_dir, f"{base_name}_transcribed_corrected.json")
    correct_cmd = [sys.executable, "cli.py", "correct", "--input", transcription_path, "--output", corrected_path]

    if not run_command(correct_cmd):
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

    print(f"Corrected transcription file created: {corrected_path}")
    print(f"\n")

    # [Step 4]
    current_step = 6
    update_job_status(job_id=job_id, step=current_step)
    print(f"\n{'=' * 25}")
    print(f"\n[Step 4] Cleaning transcription {os.path.basename(corrected_path)}...")
    cleaned_path = os.path.join(processing_jsons_dir, f"{base_name}_transcribed_corrected_cleaned.json")
    cleanup_cmd = [sys.executable, "cli.py", "cleanup", "--input", corrected_path, "--output", cleaned_path]

    if not run_command(cleanup_cmd):
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

    print(f"Cleaned transcription file created: {cleaned_path}")
    print(f"\n")

    # [Step 5]
    current_step = 7
    update_job_status(job_id=job_id, step=current_step)
    print(f"\n{'=' * 25}")
    print(f"\n[Step 5] Optimizing segments in transcription {os.path.basename(cleaned_path)}...")
    optimized_path = os.path.join(processing_jsons_dir, f"{base_name}_transcribed_corrected_cleaned_optimized.json")
    optimize_cmd = [sys.executable, "cli.py", "optimize", "--input", cleaned_path, "--output", optimized_path]

    if not run_command(optimize_cmd):
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

    print(f"Optimized transcription file created: {optimized_path}")
    print(f"\n")

    # [Step 6]
    current_step = 8
    update_job_status(job_id=job_id, step=current_step)
    print(f"\n{'=' * 25}")
    print(f"\n[Step 6] Adjusting segments timing in transcription {os.path.basename(optimized_path)}...")
    adjusted_path = os.path.join(
        processing_jsons_dir,
        f"{base_name}_transcribed_corrected_cleaned_optimized_adjusted.json"
    )
    adjust_cmd = [sys.executable, "cli.py", "adjust_timing", "--input", optimized_path, "--output", adjusted_path]

    if not run_command(adjust_cmd):
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

    print(f"Adjusted transcription file created: {adjusted_path}")
    print(f"\n")

    # [Step 7]
    current_step = 9
    update_job_status(job_id=job_id, step=current_step)
    print(f"\n{'=' * 25}")
    print(f"\n[Step 7] Translating transcription segments in {os.path.basename(adjusted_path)}...")
    translated_path = os.path.join(
        processing_jsons_dir,
        f"{base_name}_transcribed_corrected_cleaned_optimized_adjusted_translated.json"
    )
    translate_cmd = [sys.executable, "cli.py", "translate", "--input", adjusted_path, "--output", translated_path]

    translate_cmd.extend(["--target_language", target_language])

    if openai_api_key:
        translate_cmd.extend(["--openai_api_key", openai_api_key])

    if not run_command(translate_cmd):
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

    print(f"Segments translation file created: {translated_path}")
    print(f"\n")

    # [Step 8]
    current_step = 10
    update_job_status(job_id=job_id, step=current_step)
    print(f"\n{'=' * 25}")
    print(f"\n[Step 8] Generating audio using {tts_provider} with voice {tts_voice}...")

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

    print(f"Running TTS command: {' '.join(tts_cmd[:6])}...")

    if not run_command(tts_cmd):
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
        print(f"Warning: Audio segments directory {audio_segments_dir} is empty or missing")
        return {
            "status": "error",
            "message": f"Failed to generate TTS audio",
            "job_id": job_id,
            "step": current_step
        }

    print(f"TTS generation completed!")
    print(f"Audio segments saved to: {audio_segments_dir}")
    print(f"Final audio saved to: {expected_audio_path}")
    print(f"\n")

    # [Step 9]
    current_step = 11
    update_job_status(job_id=job_id, step=current_step)
    print(f"\n{'=' * 25}")
    print(f"\n[Step 9] Auto-correcting segment durations...")
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

    if not run_command(auto_correct_cmd):
        return {
            "status": "error",
            "message": f"Failed to auto-correct transcription segments",
            "job_id": job_id,
            "step": current_step
        }

    audio_result_dir = os.path.join(result_output_dir, "audio_result")
    final_audio_path = os.path.join(audio_result_dir, "new_audio.mp3")
    final_stereo_path = os.path.join(audio_result_dir, "new_audio_stereo.mp3")

    print(f"Auto-correction completed!")
    print(f"Final audio file: {final_audio_path}")
    if os.path.exists(final_stereo_path):
        print(f"Final stereo file: {final_stereo_path}")
    print(f"\n")

    # [Step 10]
    current_step = 12
    update_job_status(job_id=job_id, step=current_step)
    print(f"\n{'=' * 25}")
    print(f"\n[Step 10] Creating audio versions with intro/outro...")

    final_audio_with_ads = os.path.join(audio_result_dir, "new_audio_ads.mp3")
    final_stereo_with_ads = os.path.join(audio_result_dir, "new_audio_stereo_ads.mp3")

    if not is_premium:
        print("Creating audio with intro/outro...")
        if not add_intro_outro_audio(final_audio_path, final_audio_with_ads, "resources"):
            return {
                "status": "error",
                "message": f"Failed to create final audio file",
                "job_id": job_id,
                "step": current_step
            }

        if not os.path.exists(final_audio_with_ads):
            return {
                "status": "error",
                "message": f"Failed to create final audio file",
                "job_id": job_id,
                "step": current_step
            }

        if not create_stereo_version(final_audio_with_ads, final_stereo_with_ads):
            print(f"Warning: Failed to create stereo version of audio with ads, but continuing...")
        elif not os.path.exists(final_stereo_with_ads):
            print(f"Warning: Stereo audio with ads file {final_stereo_with_ads} not created, but continuing...")
        else:
            print(f"Successfully created stereo version with ads: {final_stereo_with_ads}")

    else:
        print("Premium user - skipping intro/outro for audio")

    print(f"Audio processing completed!")
    print(f"Clean audio: {final_audio_path}")
    print(f"Clean stereo: {final_stereo_path}")
    if not is_premium:
        print(f"Audio with ads: {final_audio_with_ads}")
        if os.path.exists(final_stereo_with_ads):
            print(f"Stereo with ads: {final_stereo_with_ads}")
        else:
            print("Stereo audio with ads: not created")
    print("\n")

    # [Step 11]
    current_step = 13
    update_job_status(job_id=job_id, step=13)
    print(f"\n{'=' * 25}")
    print(f"\n[Step 11] Processing video with new audio...")
    video_result_dir = os.path.join(result_output_dir, "video_result")
    os.makedirs(video_result_dir, exist_ok=True)

    final_video_path_mute = os.path.join(video_result_dir, f"{base_name}_tts_based_mute.mp4")
    final_video_path_mute_premium = os.path.join(video_result_dir, f"{base_name}_tts_based_mute_premium.mp4")

    process_video_cmd = [sys.executable, "cli.py", "process_video",
                         "--job_id", job_id,
                         "--input_video", video_path,
                         "--json_file", translated_path,
                         "--output_video", final_video_path_mute,
                         "--output_video_premium", final_video_path_mute_premium,
                         "--resources_dir", "resources"]

    if is_premium:
        process_video_cmd.append("--is_premium")

    if not run_command(process_video_cmd):
        return {
            "status": "error",
            "message": f"Failed to process video",
            "job_id": job_id,
            "step": current_step
        }

    if is_premium:
        if not os.path.exists(final_video_path_mute_premium):
            return {
                "status": "error",
                "message": f"Expected muted video file not created",
                "job_id": job_id,
                "step": current_step
            }
    else:
        if not os.path.exists(final_video_path_mute):
            return {
                "status": "error",
                "message": f"Expected muted video file not created",
                "job_id": job_id,
                "step": current_step
            }
        if not os.path.exists(final_video_path_mute_premium):
            return {
                "status": "error",
                "message": f"Expected muted video file not created",
                "job_id": job_id,
                "step": current_step
            }

    print(f"Video processing completed!")
    print(f"Mute videos created successfully")
    print(f"\n")

    # [Step 12]
    current_step = 14
    update_job_status(job_id=job_id, step=current_step)
    print(f"\n{'=' * 25}")
    print(f"\n[Step 12] Combining mute videos with stereo audio...")

    final_video_path = os.path.join(video_result_dir, f"{base_name}_tts_based.mp4")
    final_video_path_premium = os.path.join(video_result_dir, f"{base_name}_tts_based_premium.mp4")

    if is_premium:
        print("Premium user: combining premium mute video with audio...")

        if os.path.exists(final_stereo_path):
            audio_for_premium = final_stereo_path
            print(f"Using stereo audio for premium video: {os.path.basename(audio_for_premium)}")
        else:
            audio_for_premium = final_audio_path
            print(
                f"Stereo audio not found, using regular audio for premium video: {os.path.basename(audio_for_premium)}")

        if not combine_video_and_audio(final_video_path_mute_premium, audio_for_premium, final_video_path_premium):
            return {
                "status": "error",
                "message": f"Failed to create final video file",
                "job_id": job_id,
                "step": current_step
            }
    else:
        print("Regular user: combining standard mute video with audio (with ads)...")

        if os.path.exists(final_stereo_with_ads):
            audio_for_standard = final_stereo_with_ads
            print(f"Using stereo audio with ads: {os.path.basename(audio_for_standard)}")
        elif os.path.exists(final_audio_with_ads):
            audio_for_standard = final_audio_with_ads
            print(
                f"Stereo audio with ads not found, using regular audio with ads: {os.path.basename(audio_for_standard)}")
        else:
            return {
                "status": "error",
                "message": f"Failed to create final video file",
                "job_id": job_id,
                "step": current_step
            }

        if not combine_video_and_audio(final_video_path_mute, audio_for_standard, final_video_path):
            return {
                "status": "error",
                "message": f"Failed to create final video file",
                "job_id": job_id,
                "step": current_step
            }

        print("Regular user: combining premium mute video with clean audio...")

        if os.path.exists(final_stereo_path):
            audio_for_premium = final_stereo_path
            print(f"Using clean stereo audio for premium version: {os.path.basename(audio_for_premium)}")
        else:
            audio_for_premium = final_audio_path
            print(f"Stereo audio not found, using regular clean audio: {os.path.basename(audio_for_premium)}")

        if not combine_video_and_audio(final_video_path_mute_premium, audio_for_premium, final_video_path_premium):
            return {
                "status": "error",
                "message": f"Failed to create final video file",
                "job_id": job_id,
                "step": current_step
            }

    print(f"Video-audio combination completed!")
    if is_premium:
        print(f"Premium video (with stereo): {final_video_path_premium}")
    else:
        print(f"Video with ads (with stereo): {final_video_path}")
        print(f"Premium video (with stereo): {final_video_path_premium}")
    print(f"\n")

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
            "audio": audio_path,
            "transcription": transcription_path,
            "corrected": corrected_path,
            "cleaned": cleaned_path,
            "optimized": optimized_path,
            "adjusted": adjusted_path,
            "translated": translated_path,
            "final_audio": final_audio_path,
            "final_audio_stereo": final_stereo_path,
            "final_audio_with_ads": final_audio_with_ads,
            "final_audio_stereo_with_ads": final_stereo_with_ads,
            "final_video": final_video_path,
            "final_video_premium": final_video_path_premium,
            "final_video_mute": final_video_path_mute,
            "final_video_mute_premium": final_video_path_mute_premium
        },
        "steps_completed": ["extract", "transcribe", "correct", "cleanup", "optimize", "adjust", "translate",
                            "tts", "auto-correct", "final_audio_files_creation", "process_video",
                            "combine_video_audio"]
    }

    # [Cleanup]
    print(f"\n{'=' * 25}")
    print(f"\n[Cleanup] Moving final files and cleaning up temporary data...")

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

        (final_video_path_premium,
         os.path.join(job_result_dir, os.path.basename(final_video_path_premium))),
        (final_video_path_mute_premium,
         os.path.join(job_result_dir, os.path.basename(final_video_path_mute_premium)))
    ]

    if not is_premium:
        files_to_move.extend([
            (final_video_path,
             os.path.join(job_result_dir, os.path.basename(final_video_path))),
            (final_video_path_mute,
             os.path.join(job_result_dir, os.path.basename(final_video_path_mute))),

            (final_audio_with_ads,
             os.path.join(job_result_dir, os.path.basename(final_audio_with_ads))),
            (final_stereo_with_ads,
             os.path.join(job_result_dir, os.path.basename(final_stereo_with_ads)))
        ])

    moved_files = []
    for src, dst in files_to_move:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            moved_files.append(os.path.basename(dst))
            print(f"Copied: {os.path.basename(src)} to result directory")
        else:
            print(f"Warning: File not found: {src}")

    result["output_files"]["final_audio"] = os.path.join(job_result_dir, os.path.basename(final_audio_path))
    result["output_files"]["final_audio_stereo"] = os.path.join(job_result_dir, os.path.basename(final_stereo_path))
    result["output_files"]["translated"] = os.path.join(job_result_dir, os.path.basename(translated_path))
    result["output_files"]["final_video_premium"] = os.path.join(job_result_dir,
                                                                 os.path.basename(final_video_path_premium))
    result["output_files"]["final_video_mute_premium"] = os.path.join(job_result_dir,
                                                                      os.path.basename(final_video_path_mute_premium))

    if not is_premium:
        result["output_files"]["final_video"] = os.path.join(job_result_dir, os.path.basename(final_video_path))
        result["output_files"]["final_video_mute"] = os.path.join(job_result_dir,
                                                                  os.path.basename(final_video_path_mute))
        result["output_files"]["final_audio_with_ads"] = os.path.join(job_result_dir,
                                                                      os.path.basename(final_audio_with_ads))
        result["output_files"]["final_audio_stereo_with_ads"] = os.path.join(job_result_dir,
                                                                             os.path.basename(final_stereo_with_ads))


    with open(os.path.join(job_result_dir, "pipeline_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"Files copied to result directory: {job_result_dir}")
    print(f"Result files: {moved_files}")
    print(f"\n")

    print(f"\n{'=' * 50}")
    print("\n\n=================================================================")
    print(f"COMPLETE VIDEO PROCESSING FINISHED SUCCESSFULLY!")
    print(f"Job ID: {job_id}")
    print(f"Original video: {video_path}")
    print(f"Final video: {final_video_path_mute}")
    print(f"Total steps completed: {len(result['steps_completed'])}")
    print("=================================================================\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Complete video processing pipeline")
    parser.add_argument("--job_id", required=True, help="Unique job identifier")
    parser.add_argument("--source_language", help="Source language of the video (optional)")
    parser.add_argument("--target_language", required=True, help="Target language for translation")
    parser.add_argument("--tts_provider", choices=["openai", "elevenlabs"],
                        required=True, help="TTS service provider")
    parser.add_argument("--tts_voice", required=True, help="Voice identifier for TTS")
    parser.add_argument("--elevenlabs_api_key", help="ElevenLabs API key (optional)")
    parser.add_argument("--openai_api_key", required=True, help="OpenAI API key")
    parser.add_argument("--is_premium", action="store_true", help="User's subscription status")

    args = parser.parse_args()

    if args.tts_provider == "elevenlabs" and not args.elevenlabs_api_key:
        print("Error: ElevenLabs API key is required for ElevenLabs TTS provider.")
        return 1

    os.makedirs("jobs", exist_ok=True)

    result = process_job(
        args.job_id,
        source_language=args.source_language,
        target_language=args.target_language,
        tts_provider=args.tts_provider,
        tts_voice=args.tts_voice,
        elevenlabs_api_key=args.elevenlabs_api_key,
        openai_api_key=args.openai_api_key,
        is_premium=args.is_premium,
    )

    if result["status"] == "error":
        print(f"Error processing job {args.job_id}: {result['message']}")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
