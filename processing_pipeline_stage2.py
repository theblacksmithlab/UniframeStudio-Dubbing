#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys


def run_command(command):
    print(f"Executing: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command execution error: {e}")
        return False


def process_translation_stage(json_path, args):
    """
    Process the second stage: translation, TTS, auto-correction, and reassembly
    """
    base_dir = os.path.dirname(json_path)
    filename = os.path.basename(json_path)
    base_name = os.path.splitext(filename)[0]

    print(f"\n{'=' * 50}")
    print(f"Starting second stage processing for: {filename}")
    print(f"{'=' * 50}")

    # Step 1: Translation
    print(f"\n[Step 1/4] Translating segments in {filename}...")
    translated_path = json_path.replace(".json", "_translated.json")
    translate_cmd = [sys.executable, "cli.py", "translate", "--input", json_path]

    if args.model:
        translate_cmd.extend(["--model", args.model])

    if not run_command(translate_cmd):
        print(f"Error: Failed to translate {json_path}")
        return False

    if not os.path.exists(translated_path):
        print(f"Error: Expected translated file {translated_path} not created")
        return False

    print(f"Translated file created: {translated_path}")

    # Step 2: TTS Generation
    print(f"\n[Step 2/4] Generating TTS for {os.path.basename(translated_path)}...")
    audio_result_dir = os.path.join(base_dir, "audio_result")
    audio_path = os.path.join(audio_result_dir, "en_audio.mp3")
    tts_cmd = [sys.executable, "cli.py", "tts", "--input", translated_path, "--dealer", args.dealer]

    if args.voice:
        tts_cmd.extend(["--voice", args.voice])
    if args.intro:
        tts_cmd.append("--intro")
    if args.outro:
        tts_cmd.append("--outro")

    if not run_command(tts_cmd):
        print(f"Error: Failed to generate TTS for {translated_path}")
        return False

    if not os.path.exists(audio_path):
        print(f"Warning: Expected audio file {audio_path} not created")
        print("This might be normal if there were issues during TTS generation.")
        print("Continuing with next steps anyway, as the JSON file should be updated.")

    # Step 3: Auto-correction of segment durations
    print(f"\n[Step 3/4] Auto-correcting segment durations in {os.path.basename(translated_path)}...")
    correct_cmd = [
        sys.executable, "cli.py", "auto-correct",
        "--input", translated_path,
        "--dealer", args.dealer
    ]

    if args.voice:
        correct_cmd.extend(["--voice", args.voice])
    if args.attempts:
        correct_cmd.extend(["--attempts", str(args.attempts)])
    if args.threshold:
        correct_cmd.extend(["--threshold", str(args.threshold)])

    if not run_command(correct_cmd):
        print(f"Error: Failed to auto-correct {translated_path}")
        return False

    # Step 4: Reassemble final audio
    print(f"\n[Step 4/4] Reassembling final audio from corrected segments...")
    reassembled_audio_path = os.path.join(audio_result_dir, "en_audio_reassembled.mp3")
    reassemble_cmd = [
        sys.executable, "cli.py", "reassemble",
        "--input", translated_path
    ]

    if args.intro:
        reassemble_cmd.append("--intro")
    if args.outro:
        reassemble_cmd.append("--outro")

    if not run_command(reassemble_cmd):
        print(f"Error: Failed to reassemble audio from {translated_path}")
        return False

    if not os.path.exists(reassembled_audio_path):
        print(f"Warning: Expected reassembled audio file {reassembled_audio_path} not created")
    else:
        print(f"Final audio created: {reassembled_audio_path}")

    print(f"\nSecond stage processing completed successfully!")
    print(f"Final translation result: {translated_path}")
    print(f"You can now use this file for video processing with the command:")
    print(f"python cli.py process_video")

    return True


def main():
    parser = argparse.ArgumentParser(description="Second stage of processing: "
                                                "translation, TTS generation, auto-correction, "
                                                "and audio reassembly")

    parser.add_argument("--input", "-i", required=True,
                        help="Path to the JSON file from the first stage (after adjustment)")
    parser.add_argument("--model", "-m", default="gpt-4o",
                        help="Translation model (default: gpt-4o)")
    parser.add_argument("--dealer", "-d", default="elevenlabs", choices=["openai", "elevenlabs"],
                        help="TTS service provider (default: elevenlabs)")
    parser.add_argument("--voice", "-v", default="onyx",
                        help="Voice for dubbing (only for OpenAI, default: onyx)")
    parser.add_argument("--attempts", "-a", type=int, default=5,
                        help="Maximum number of auto-correction attempts (default: 5)")
    parser.add_argument("--threshold", "-t", type=float, default=0.2,
                        help="Duration difference threshold for correction (default: 0.2 or 20%)")
    parser.add_argument("--intro", action="store_true",
                        help="Add intro audio at the beginning")
    parser.add_argument("--outro", action="store_true",
                        help="Add outro audio after the last segment")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return

    try:
        success = process_translation_stage(args.input, args)
        if not success:
            print(f"Second stage processing failed for {args.input}")
            sys.exit(1)
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()