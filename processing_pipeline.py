#!/usr/bin/env python3
import json
import os
import argparse
import shutil
import subprocess
import sys
from datetime import datetime


def run_command(command):
    print(f"Executing: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command execution error: {e}")
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

    job_params = {
        "job_id": job_id,
        "source_language": source_language,
        "target_language": target_language,
        "tts_provider": tts_provider,
        "tts_voice": tts_voice,
        "video_file": video_path,
        "created_at": datetime.now().isoformat()
    }

    with open(f"{job_dir}/job_params.json", "w") as f:
        json.dump(job_params, f)

    print(f"\n{'=' * 50}")
    print(f"Processing job: {job_id}")
    print(f"Video file: {video_path}")
    if source_language:
        print(f"Source language: {source_language}")
    print(f"Target language: {target_language}")
    print(f"TTS provider: {tts_provider}")
    print(f"TTS voice: {tts_voice}")

    # [Step 1]
    print(f"\n{'=' * 25}")
    print(f"\n[Step 1] Extracting audio from video {video_path}...")
    audio_path = f"{audio_input_dir}/{base_name}.mp3"
    extract_cmd = [sys.executable, "cli.py", "extract_audio", "--input", video_path, "--output", audio_path]

    if not run_command(extract_cmd):
        return {
            "status": "error",
            "message": f"Failed to extract audio from {video_path}",
            "job_id": job_id,
            "step": "extract_audio"
        }

    if not os.path.exists(audio_path):
        return {
            "status": "error",
            "message": f"Expected audio file {audio_path} not created",
            "job_id": job_id,
            "step": "extract_audio"
        }

    print(f"Audio file created: {audio_path}")
    print(f"\n")

    # [Step 2]
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
            "message": f"Failed to transcribe audio {audio_path}",
            "job_id": job_id,
            "step": "transcribe"
        }

    if not os.path.exists(transcription_path):
        return {
            "status": "error",
            "message": f"Expected transcription file {transcription_path} not created",
            "job_id": job_id,
            "step": "transcribe"
        }

    print(f"Transcription file created: {transcription_path}")
    print(f"\n")

    # [Step 3]
    print(f"\n{'=' * 25}")
    print(f"\n[Step 3] Structuring transcription {os.path.basename(transcription_path)}...")
    corrected_path = os.path.join(processing_jsons_dir, f"{base_name}_transcribed_corrected.json")
    correct_cmd = [sys.executable, "cli.py", "correct", "--input", transcription_path, "--output", corrected_path]

    if not run_command(correct_cmd):
        return {
            "status": "error",
            "message": f"Failed to structure transcription {transcription_path}",
            "job_id": job_id,
            "step": "correct"
        }

    if not os.path.exists(corrected_path):
        return {
            "status": "error",
            "message": f"Expected corrected file {corrected_path} not created",
            "job_id": job_id,
            "step": "correct"
        }

    print(f"Corrected transcription file created: {corrected_path}")
    print(f"\n")

    # [Step 4]
    print(f"\n{'=' * 25}")
    print(f"\n[Step 4] Cleaning transcription {os.path.basename(corrected_path)}...")
    cleaned_path = os.path.join(processing_jsons_dir, f"{base_name}_transcribed_corrected_cleaned.json")
    cleanup_cmd = [sys.executable, "cli.py", "cleanup", "--input", corrected_path, "--output", cleaned_path]

    if not run_command(cleanup_cmd):
        return {
            "status": "error",
            "message": f"Failed to clean transcription {corrected_path}",
            "job_id": job_id,
            "step": "cleanup"
        }

    if not os.path.exists(cleaned_path):
        return {
            "status": "error",
            "message": f"Expected cleaned file {cleaned_path} not created",
            "job_id": job_id,
            "step": "cleanup"
        }

    print(f"Cleaned transcription file created: {cleaned_path}")
    print(f"\n")

    # [Step 5]
    print(f"\n{'=' * 25}")
    print(f"\n[Step 5] Optimizing segments in transcription {os.path.basename(cleaned_path)}...")
    optimized_path = os.path.join(processing_jsons_dir, f"{base_name}_transcribed_corrected_cleaned_optimized.json")
    optimize_cmd = [sys.executable, "cli.py", "optimize", "--input", cleaned_path, "--output", optimized_path]

    if not run_command(optimize_cmd):
        return {
            "status": "error",
            "message": f"Failed to optimize segments in {cleaned_path}",
            "job_id": job_id,
            "step": "optimize"
        }

    if not os.path.exists(optimized_path):
        return {
            "status": "error",
            "message": f"Expected optimized file {optimized_path} not created",
            "job_id": job_id,
            "step": "optimize"
        }

    print(f"Optimized transcription file created: {optimized_path}")
    print(f"\n")

    # [Step 6]
    print(f"\n{'=' * 25}")
    print(f"\n[Step 6] Adjusting segment timing in transcription {os.path.basename(optimized_path)}...")
    adjusted_path = os.path.join(processing_jsons_dir, f"{base_name}_transcribed_corrected_cleaned_optimized_adjusted.json")
    adjust_cmd = [sys.executable, "cli.py", "adjust_timing", "--input", optimized_path, "--output", adjusted_path]

    if not run_command(adjust_cmd):
        return {
            "status": "error",
            "message": f"Failed to adjust timing in {optimized_path}",
            "job_id": job_id,
            "step": "adjust_timing"
        }

    if not os.path.exists(adjusted_path):
        return {
            "status": "error",
            "message": f"Expected adjusted file {adjusted_path} not created",
            "job_id": job_id,
            "step": "adjust_timing"
        }

    print(f"Adjusted transcription file created: {adjusted_path}")
    print(f"\n")

    # [Step 7]
    print(f"\n{'=' * 25}")
    print(f"\n[Step 7] Translating segments in {os.path.basename(adjusted_path)}...")
    translated_path = os.path.join(processing_jsons_dir, f"{base_name}_transcribed_corrected_cleaned_optimized_adjusted_translated.json")
    translate_cmd = [sys.executable, "cli.py", "translate", "--input", adjusted_path, "--output", translated_path]

    translate_cmd.extend(["--target_language", target_language])

    if openai_api_key:
        translate_cmd.extend(["--openai_api_key", openai_api_key])

    if not run_command(translate_cmd):
        return {
            "status": "error",
            "message": f"Failed to translate segments in {adjusted_path}",
            "job_id": job_id,
            "step": "translation"
        }

    if not os.path.exists(translated_path):
        return {
            "status": "error",
            "message": f"Expected translation file {translated_path} not created",
            "job_id": job_id,
            "step": "translation"
        }

    print(f"Segments translation file created: {translated_path}")
    print(f"\n")

    # [Step 8]
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
                "step": "tts"
            }

    if tts_provider == "openai":
        if openai_api_key:
            tts_cmd.extend(["--openai_api_key", openai_api_key])
        else:
            return {
                "status": "error",
                "message": "OpenAI API key is required but not provided",
                "job_id": job_id,
                "step": "tts"
            }

    print(f"Running TTS command: {' '.join(tts_cmd[:6])}...")

    if not run_command(tts_cmd):
        return {
            "status": "error",
            "message": f"Failed to generate TTS audio for {translated_path}",
            "job_id": job_id,
            "step": "tts"
        }

    expected_audio_dir = os.path.join(result_output_dir, "audio_result")
    expected_audio_path = os.path.join(expected_audio_dir, "new_audio.mp3")

    if not os.path.exists(expected_audio_path):
        return {
            "status": "error",
            "message": f"Expected TTS audio file {expected_audio_path} not created",
            "job_id": job_id,
            "step": "tts"
        }

    audio_segments_dir = os.path.join(result_output_dir, "audio_segments")
    if not os.path.exists(audio_segments_dir) or not any(f.endswith('.mp3') for f in os.listdir(audio_segments_dir)):
        print(f"Warning: Audio segments directory {audio_segments_dir} is empty or missing")

    tts_audio_path = expected_audio_path

    print(f"TTS generation completed!")
    print(f"Audio segments saved to: {audio_segments_dir}")
    print(f"Final audio saved to: {expected_audio_path}")
    print(f"\n")

    # [Step 9]
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
            "message": f"Failed to auto-correct segments in {translated_path}",
            "job_id": job_id,
            "step": "auto-correct"
        }

    audio_result_dir = os.path.join(result_output_dir, "audio_result")
    reassembled_audio_path = os.path.join(audio_result_dir, "new_audio_reassembled.mp3")
    reassembled_stereo_path = os.path.join(audio_result_dir, "new_audio_reassembled_stereo.mp3")
    original_stereo_path = os.path.join(audio_result_dir, "new_audio_stereo.mp3")

    if os.path.exists(reassembled_audio_path):
        print(f"Reassembled audio created: {reassembled_audio_path}")
        final_audio_path = reassembled_audio_path
        final_stereo_path = reassembled_stereo_path if os.path.exists(reassembled_stereo_path) else None
    else:
        print(f"No reassembled audio found, using original TTS audio: {tts_audio_path}")
        final_audio_path = tts_audio_path
        final_stereo_path = original_stereo_path if os.path.exists(original_stereo_path) else None

    print(f"Auto-correction completed!")
    print(f"Final audio file: {final_audio_path}")
    if final_stereo_path:
        print(f"Final stereo file: {final_stereo_path}")
    print(f"\n")

    # [Step 10]
    print(f"\n{'=' * 25}")
    print(f"\n[Step 10] Processing video with new audio...")
    video_result_dir = os.path.join(result_output_dir, "video_result")
    os.makedirs(video_result_dir, exist_ok=True)
    final_video_path = os.path.join(video_result_dir, f"{base_name}_tts_based.mp4")

    process_video_cmd = [sys.executable, "cli.py", "process_video",
                         "--job_id", job_id,
                         "--is_premium", is_premium,
                         "--input_video", video_path,
                         "--json_file", translated_path,
                         "--output_video", final_video_path,
                         "--resources_dir", "resources"]

    if not run_command(process_video_cmd):
        return {
            "status": "error",
            "message": f"Failed to process video {video_path}",
            "job_id": job_id,
            "step": "process_video"
        }

    if not os.path.exists(final_video_path):
        return {
            "status": "error",
            "message": f"Expected final video file {final_video_path} not created",
            "job_id": job_id,
            "step": "process_video"
        }

    print(f"Video processing completed!")
    print(f"Final video saved to: {final_video_path}")
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
            "final_video": final_video_path
        },
        "steps_completed": ["extract", "transcribe", "correct", "cleanup", "optimize",
                            "adjust", "translate", "tts", "auto-correct", "process_video"]
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

        (final_video_path,
         os.path.join(job_result_dir, os.path.basename(final_video_path)))
    ]

    if final_stereo_path and os.path.exists(final_stereo_path):
        files_to_move.append((final_stereo_path,
                              os.path.join(job_result_dir, os.path.basename(final_stereo_path))))

    moved_files = []
    for src, dst in files_to_move:
        if os.path.exists(src):
            shutil.move(src, dst)
            moved_files.append(os.path.basename(dst))
            print(f"Moved: {os.path.basename(src)}")
        else:
            print(f"Warning: File not found: {src}")

    result["output_files"]["final_audio"] = os.path.join(job_result_dir, os.path.basename(final_audio_path))
    result["output_files"]["final_video"] = os.path.join(job_result_dir, os.path.basename(final_video_path))
    result["output_files"]["translated"] = os.path.join(job_result_dir, os.path.basename(translated_path))
    if final_stereo_path:
        result["output_files"]["final_audio_stereo"] = os.path.join(job_result_dir, os.path.basename(final_stereo_path))

    with open(os.path.join(job_result_dir, "pipeline_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    job_base_dir = f"jobs/{job_id}"
    cleanup_paths = [
        os.path.join(job_base_dir, "video_input"),
        os.path.join(job_base_dir, "audio_input"),
        os.path.join(job_base_dir, "timestamped_transcriptions"),
        os.path.join(job_base_dir, "output"),
        os.path.join(job_base_dir, "job_params.json")
    ]

    for path in cleanup_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Removed directory: {os.path.basename(path)}")
            else:
                os.remove(path)
                print(f"Removed file: {os.path.basename(path)}")

    print(f"Cleanup completed! Final files moved to: {job_result_dir}")
    print(f"Files in result: {moved_files}")
    print(f"\n")

    print(f"\n{'=' * 50}")
    print("\n\n=================================================================")
    print(f"COMPLETE VIDEO PROCESSING FINISHED SUCCESSFULLY!")
    print(f"Job ID: {job_id}")
    print(f"Original video: {video_path}")
    print(f"Final video: {final_video_path}")
    print(f"Total steps completed: {len(result['steps_completed'])}")
    print("=================================================================\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Complete video processing pipeline")
    parser.add_argument("--job_id", required=True, help="Unique job identifier")
    parser.add_argument("--source_language", help="Source language of the video (optional)")
    parser.add_argument("--target_language", required=True, help="Target language for translation")
    parser.add_argument("--tts_provider", choices=["openai", "elevenlabs"], required=True, help="TTS service provider")
    parser.add_argument("--tts_voice", required=True, help="Voice identifier for TTS")
    parser.add_argument("--elevenlabs_api_key", help="ElevenLabs API key (optional)")
    parser.add_argument("--openai_api_key", help="OpenAI API key (optional)")
    parser.add_argument("--is_premium", required=True, help="User's subscription status")

    args = parser.parse_args()

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
