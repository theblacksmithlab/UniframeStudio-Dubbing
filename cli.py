#!/usr/bin/env python3
import argparse
import os
import dotenv

from modules.cleaning_up_corrected_transcirption import cleanup_transcript_segments
from modules.transcribe_with_timestamps import transcribe_audio_with_timestamps
from modules.transcription_correction import correct_transcript_segments
from modules.translation import translate_transcript_segments
from modules.tts import generate_tts_for_segments


def main():
    dotenv.load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY must be set in .env file!")

    # CLI agrs parser initialization
    parser = argparse.ArgumentParser(description="Smart audio transcription and translation tools")

    # Sub-parsers for commands initialization
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Transcription command sub-parser
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribes audio with timestamps")
    transcribe_parser.add_argument("--input", "-i", required=True, help="Input audio-file path")

    # Segments correction command sub-parser
    correct_parser = subparsers.add_parser("correct", help="Structures raw transcription of segments")
    correct_parser.add_argument("--input", "-i", required=True, help="Path to transcription file")
    correct_parser.add_argument("--output", "-o",
                                help="Path to save structured transcription (optional)")

    # Sub-parse for clearing segments command
    cleanup_parser = subparsers.add_parser("cleanup",
                                           help="Cleans segments (deletes merged and removes extra spaces)")
    cleanup_parser.add_argument("--input", "-i", required=True,
                                help="Path to structured transcription file")
    cleanup_parser.add_argument("--output", "-o", help="Path to save cleaned transcription (optional)")

    # Sub-parser for the segment translation command
    translate_parser = subparsers.add_parser("translate", help="Translates transcription segments")
    translate_parser.add_argument("--input", "-i", required=True,
                                  help="Path to cleaned transcription file")
    translate_parser.add_argument("--output", "-o",
                                  help="Path to save translated transcription (optional)")
    translate_parser.add_argument("--model", "-m", default="gpt-4o-mini",
                                  help="Translation model (default: gpt-4o-mini)")

    # Sub-parser for the segment voicing over command
    tts_parser = subparsers.add_parser("tts", help="Voices over translated segments")
    tts_parser.add_argument("--input", "-i", required=True,
                            help="Path to translated transcription file")
    tts_parser.add_argument("--output", "-o", help="Path to save the audio file (optional)")
    tts_parser.add_argument("--dealer", "-d", default="openai", choices=["openai", "elevenlabs"],
                            help="TTS service provider (default: openai)")
    tts_parser.add_argument("--voice", "-v", default="onyx",
                            help="Voice for dubbing (only used for OpenAI, default: onyx)")

    # Parsing arguments
    args = parser.parse_args()

    # Check if the command was specified
    if not args.command:
        parser.print_help()
        return

    # Processing the commands
    if args.command == "transcribe":
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found.")
            return

        print(f"Transcribing the file: {args.input}")
        try:
            result_file = transcribe_audio_with_timestamps(args.input)
            print(f"Transcription completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error during transcription: {e}")
            return

    elif args.command == "correct":
        if not os.path.exists(args.input):
            print(f"Error: Transcription file {args.input} not found.")
            return

        print(f"Restructuring transcription segments: {args.input}")
        try:
            result_file = correct_transcript_segments(args.input, args.output)
            print(f"Restructuring completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error during restructuring: {e}")
            return

    elif args.command == "cleanup":
        if not os.path.exists(args.input):
            print(f"Error: Transcription file {args.input} not found.")
            return

        print(f"Cleaning-up transcription segments: {args.input}")
        try:
            result_file = cleanup_transcript_segments(args.input, args.output)
            print(f"Transcription segments cleaned-up successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error cleaning-up: {e}")
            return

    elif args.command == "translate":
        if not os.path.exists(args.input):
            print(f"Error: Transcription file {args.input} not found.")
            return

        print(f"Translating transcription sergments: {args.input}")
        try:
            import openai
            openai_model = args.model

            original_model = None
            if hasattr(openai, 'model'):
                original_model = openai.model

            if hasattr(openai, 'model'):
                openai.model = openai_model

            result_file = translate_transcript_segments(args.input, args.output)

            if original_model is not None and hasattr(openai, 'model'):
                openai.model = original_model

            print(f"Translation completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error translating segments: {e}")
            return


    elif args.command == "tts":
        if not os.path.exists(args.input):
            print(f"Error: Translated transcription file {args.input} not found.")
            return

        print(f"Voicing-over translated segments using {args.dealer}: {args.input}")
        try:
            result_file = generate_tts_for_segments(args.input, args.output, args.voice, args.dealer)
            print(f"Voicing-over completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error voicing-over segments: {e}")
            import traceback
            traceback.print_exc()
            return

if __name__ == "__main__":
    main()