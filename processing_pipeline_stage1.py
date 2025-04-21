#!/usr/bin/env python3
import os
import argparse
import subprocess
import glob
import sys


def run_command(command):
    print(f"Executing: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command execution error: {e}")
        return False


def process_single_file(video_path, args):
    video_filename = os.path.basename(video_path)
    base_name = os.path.splitext(video_filename)[0]
    output_dir = "output/timestamped_transcriptions"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"Processing file: {video_filename}")
    print(f"{'=' * 50}")

    # Step 1: Extract audio
    print(f"\n[Step 1/6] Extracting audio from video {video_filename}...")
    audio_path = f"audio_input/{base_name}.mp3"
    extract_cmd = [sys.executable, "cli.py", "extract_audio", "--input", video_path]

    if not run_command(extract_cmd):
        print(f"Error: Failed to extract audio from {video_path}")
        return False

    if not os.path.exists(audio_path):
        print(f"Error: Expected audio file {audio_path} not created")
        return False

    print(f"Audio file created: {audio_path}")

    # Step 2: Transcription
    print(f"\n[Step 2/6] Transcribing audio {os.path.basename(audio_path)}...")
    transcription_path = os.path.join(output_dir, f"{base_name}_transcribed.json")
    transcribe_cmd = [sys.executable, "cli.py", "transcribe", "--input", audio_path]

    if not run_command(transcribe_cmd):
        print(f"Error: Failed to transcribe audio {audio_path}")
        return False

    if not os.path.exists(transcription_path):
        print(f"Error: Expected transcription file {transcription_path} not created")
        return False

    print(f"Transcription file created: {transcription_path}")

    # Step 3: Structure transcription
    print(f"\n[Step 3/6] Structuring transcription {os.path.basename(transcription_path)}...")
    corrected_path = os.path.join(output_dir, f"{base_name}_transcribed_corrected.json")
    correct_cmd = [sys.executable, "cli.py", "correct", "--input", transcription_path]

    if args.start_timestamp is not None:
        correct_cmd.extend(["--start_timestamp", str(args.start_timestamp)])

    if not run_command(correct_cmd):
        print(f"Error: Failed to structure transcription {transcription_path}")
        return False

    if not os.path.exists(corrected_path):
        print(f"Error: Expected corrected file {corrected_path} not created")
        return False

    print(f"Corrected transcription file created: {corrected_path}")

    # Step 4: Clean transcription
    print(f"\n[Step 4/6] Cleaning transcription {os.path.basename(corrected_path)}...")
    cleaned_path = os.path.join(output_dir, f"{base_name}_transcribed_corrected_cleaned.json")
    cleanup_cmd = [sys.executable, "cli.py", "cleanup", "--input", corrected_path]

    if not run_command(cleanup_cmd):
        print(f"Error: Failed to clean transcription {corrected_path}")
        return False

    if not os.path.exists(cleaned_path):
        print(f"Error: Expected cleaned file {cleaned_path} not created")
        return False

    print(f"Cleaned transcription file created: {cleaned_path}")

    # Step 5: Optimize segments
    print(f"\n[Step 5/6] Optimizing segments in transcription {os.path.basename(cleaned_path)}...")
    optimized_path = os.path.join(output_dir, f"{base_name}_transcribed_corrected_cleaned_optimized.json")
    optimize_cmd = [sys.executable, "cli.py", "optimize", "--input", cleaned_path]

    if not run_command(optimize_cmd):
        print(f"Error: Failed to optimize segments in {cleaned_path}")
        return False

    if not os.path.exists(optimized_path):
        print(f"Error: Expected optimized file {optimized_path} not created")
        return False

    print(f"Optimized transcription file created: {optimized_path}")

    # Step 6: Adjust timing
    print(f"\n[Step 6/6] Adjusting segment timing in transcription {os.path.basename(optimized_path)}...")
    adjusted_path = os.path.join(output_dir, f"{base_name}_transcribed_corrected_cleaned_optimized_adjusted.json")
    adjust_cmd = [sys.executable, "cli.py", "adjust_timing", "--input", optimized_path]

    if not run_command(adjust_cmd):
        print(f"Error: Failed to adjust timing in {optimized_path}")
        return False

    if not os.path.exists(adjusted_path):
        print(f"Error: Expected adjusted file {adjusted_path} not created")
        return False

    print(f"Adjusted transcription file created: {adjusted_path}")

    print("\n=================================================================")
    print(f"First stage processing of file {video_filename} completed successfully!")
    print(f"Final result saved to {adjusted_path}")
    print(f"To translate, use the command:")
    print(f"python3 cli.py translate --input {adjusted_path}")

    # Continue with second stage
    print(f"Or run the full second stage processor with:")
    print(f"python3 processing_pipeline_stage2.py --input {adjusted_path} --dealer elevenlabs")
    print("=================================================================\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="Automatic video processing (first 6 stages): "
                                                 "audio extraction, transcription, correction, cleaning, "
                                                 "optimization, and timing adjustment")

    parser.add_argument("--input", "-i", help="Path to video file or directory with videos")
    parser.add_argument("--start_timestamp", "-st", type=float,
                        help="Set start time for the first segment (e.g., 0.0)")

    args = parser.parse_args()

    # If path is not specified, use "video_input" directory
    if not args.input:
        args.input = "video_input"

    # Determine if a file or directory is specified
    if os.path.isdir(args.input):
        print(f"Processing all videos in directory: {args.input}")
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.3gp', '.m4v']:
            video_files.extend(glob.glob(os.path.join(args.input, f"*{ext}")))
            video_files.extend(glob.glob(os.path.join(args.input, f"*{ext.upper()}")))

        if not video_files:
            print(f"No video files found in directory {args.input}.")
            return

        print(f"Found {len(video_files)} video files.")

        success_count = 0
        for video_path in video_files:
            if process_single_file(video_path, args):
                success_count += 1

        print(f"\nProcessing completed. Successfully processed {success_count} of {len(video_files)} files.")

        if success_count > 0:
            print("\nTo translate files, run the command with the appropriate adjusted file path:")
            print(f"python3 cli.py translate --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted.json")
            print("\nOr run the second stage processor:")
            print(f"python3 processing_pipeline_stage2.py --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted.json")

    elif os.path.isfile(args.input):
        success = process_single_file(args.input, args)
        if not success:
            print(f"Failed to process file {args.input}")

    else:
        print(f"Error: Path {args.input} does not exist.")


if __name__ == "__main__":
    main()