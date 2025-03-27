#!/usr/bin/env python3
import argparse
import os
import dotenv

from modules.cleaning_up_corrected_transcirption import cleanup_transcript_segments
from modules.transcribe_with_timestamps import transcribe_audio_with_timestamps
from modules.transcription_correction import correct_transcript_segments
from modules.translation import translate_transcript_segments
from modules.tts import generate_tts_for_segments, reassemble_audio_file
from modules.tts_correction import regenerate_segment, replace_segment_in_audio
from modules.video_to_audio_conversion import extract_audio
from modules.optimized_segmentation import optimize_transcription_segments


def main():
    dotenv.load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY must be set in .env file!")

    # CLI agrs parser initialization
    parser = argparse.ArgumentParser(description="Smart audio transcription and translation tools")

    # Sub-parsers for commands initialization
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Audio extraction sub-parser
    extract_parser = subparsers.add_parser("extract_audio", help="Extracting audio from video file")
    extract_parser.add_argument("--input", "-i", required=True, help="Input video-file path")

    # Transcription command sub-parser
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribes audio with timestamps")
    transcribe_parser.add_argument("--input", "-i", required=True, help="Input audio-file path")

    # Segments correction command sub-parser
    correct_parser = subparsers.add_parser("correct", help="Structures raw transcription of segments")
    correct_parser.add_argument("--input", "-i", required=True, help="Path to transcription file")
    correct_parser.add_argument("--output", "-o",
                                help="Path to save structured transcription (optional)")
    correct_parser.add_argument("--start_timestamp", "-st", type=float,
                                help="Set specific start timestamp for the first segment (e.g. 0.0 or 4.0)")

    # Sub-parse for clearing segments command
    cleanup_parser = subparsers.add_parser("cleanup",
                                           help="Cleans segments (deletes merged and removes extra spaces)")
    cleanup_parser.add_argument("--input", "-i", required=True,
                                help="Path to structured transcription file")
    cleanup_parser.add_argument("--output", "-o", help="Path to save cleaned transcription (optional)")

    # Sub-parser for optimizing segments to sentences
    optimize_parser = subparsers.add_parser("optimize-segments",
                                            help="Optimize transcription by breaking segments into sentences")
    optimize_parser.add_argument("--input", "-i", required=True,
                                 help="Path to the transcription file with segments and words")
    optimize_parser.add_argument("--output", "-o",
                                 help="Path to save the optimized transcription (optional)")

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
    tts_parser.add_argument("--intro", action="store_true",
                            help="Add intro audio at the beginning (replaces first 4 seconds)")
    tts_parser.add_argument("--outro", action="store_true",
                            help="Add outro audio after the last segment")

    # Sub-parser for regenerating a single segment
    segment_tts_parser = subparsers.add_parser("segment-tts",
                                               help="Regenerate audio for a specific segment with updated translation")
    segment_tts_parser.add_argument("--input", "-i", required=True,
                                    help="Path to translated transcription file")
    segment_tts_parser.add_argument("--segment-id", "-s", required=True, type=int,
                                    help="ID of the segment to regenerate")
    segment_tts_parser.add_argument("--output", "-o",
                                    help="Path to save the segment audio file (optional)")
    segment_tts_parser.add_argument("--dealer", "-d", default="openai", choices=["openai", "elevenlabs"],
                                    help="TTS service provider (default: openai)")
    segment_tts_parser.add_argument("--voice", "-v", default="onyx",
                                    help="Voice for dubbing (only used for OpenAI, default: onyx)")

    # Add a new sub-parser for replacing a segment in the main audio
    replace_segment_parser = subparsers.add_parser("replace-segment",
                                                   help="Replace a segment in the main audio file with new audio")
    replace_segment_parser.add_argument("--main-audio", "-m", required=True,
                                        help="Path to the main audio file")
    replace_segment_parser.add_argument("--segment-audio", "-s", required=True,
                                        help="Path to the new segment audio file")
    replace_segment_parser.add_argument("--translation", "-t", required=True,
                                        help="Path to the translation file with segment timestamps")
    replace_segment_parser.add_argument("--segment-id", "-i", required=True, type=int,
                                        help="ID of the segment to replace")
    replace_segment_parser.add_argument("--output", "-o",
                                        help="Path to save the new audio file (optional)")

    # Add a new sub-parser for reassembling audio from existing segments
    reassemble_parser = subparsers.add_parser("reassemble",
                                              help="Reassemble audio file from existing segments")
    reassemble_parser.add_argument("--input", "-i", required=True,
                                   help="Path to the translation file with segments")
    reassemble_parser.add_argument("--output", "-o",
                                   help="Path to save the reassembled audio file (optional)")
    reassemble_parser.add_argument("--intro", action="store_true",
                                   help="Add intro audio at the beginning (replaces first 4 seconds)")
    reassemble_parser.add_argument("--outro", action="store_true",
                                   help="Add outro audio after the last segment")

    # Parsing arguments
    args = parser.parse_args()

    # Check if the command was specified
    if not args.command:
        parser.print_help()
        return

    # Processing the commands
    if args.command == "extract_audio":
        if not os.path.exists(args.input):
            print(f"Error: Video file {args.input} not found.")
        try:
            result_file = extract_audio(args.input)
            print(f"Extracting audio from video file completed successfully."
                  f"The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error during video conversion: {e}")
            return

    elif args.command == "transcribe":
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
            result_file = correct_transcript_segments(args.input, args.output, args.start_timestamp)
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

    elif args.command == "optimize-segments":
        if not os.path.exists(args.input):
            print(f"Error: Transcription file {args.input} not found.")
            return

        print(f"Optimizing segments in transcription file: {args.input}")
        try:
            result_file = optimize_transcription_segments(args.input, args.output)
            print(f"Segment optimization completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error optimizing segments: {e}")
            import traceback
            traceback.print_exc()
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
            result_file = generate_tts_for_segments(
                args.input, args.output, args.voice, args.dealer,
                intro=args.intro, outro=args.outro
            )
            print(f"Voicing-over completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error voicing-over segments: {e}")
            import traceback
            traceback.print_exc()
            return

    # And in the command processing section:
    elif args.command == "segment-tts":
        if not os.path.exists(args.input):
            print(f"Error: Translated transcription file {args.input} not found.")
            return

        print(f"Regenerating segment {args.segment_id} using {args.dealer}: {args.input}")
        try:
            result_file = regenerate_segment(
                args.input, args.segment_id, args.output, args.voice, args.dealer
            )
            if result_file:
                print(f"Segment regeneration completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error regenerating segment: {e}")
            import traceback
            traceback.print_exc()
            return

    # And in the command processing section:
    elif args.command == "replace-segment":
        if not os.path.exists(args.main_audio):
            print(f"Error: Main audio file {args.main_audio} not found.")
            return

        if not os.path.exists(args.segment_audio):
            print(f"Error: Segment audio file {args.segment_audio} not found.")
            return

        if not os.path.exists(args.translation):
            print(f"Error: Translation file {args.translation} not found.")
            return

        print(f"Replacing segment {args.segment_id} in main audio file: {args.main_audio}")
        try:
            result_file = replace_segment_in_audio(
                args.main_audio, args.segment_audio, args.translation, args.segment_id, args.output
            )
            if result_file:
                print(f"Segment replacement completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error replacing segment: {e}")
            import traceback
            traceback.print_exc()
            return

    elif args.command == "reassemble":
        if not os.path.exists(args.input):
            print(f"Error: Translation file {args.input} not found.")
            return

        print(f"Reassembling audio from segments using file: {args.input}")
        try:
            result_file = reassemble_audio_file(
                args.input, args.output,
                intro=args.intro, outro=args.outro
            )
            if result_file:
                print(f"Audio successfully reassembled. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error reassembling audio: {e}")
            import traceback
            traceback.print_exc()
            return

if __name__ == "__main__":
    main()