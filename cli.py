#!/usr/bin/env python3
import argparse
import os
import sys

# import dotenv

from modules.adjust_timing import adjust_segments_timing
from modules.cleaning_up_corrected_transcirption import cleanup_transcript_segments
from modules.transcribe_with_timestamps import transcribe_audio_with_timestamps
from modules.transcription_correction import correct_transcript_segments
from modules.translation import translate_transcribed_segments
from modules.tts import generate_tts_for_segments, reassemble_audio_file
# from modules.tts_correction import regenerate_segment
from modules.video_processor import VideoProcessor
from modules.video_to_audio_conversion import extract_audio
from modules.optimized_segmentation import optimize_transcription_segments
from modules.automatic_text_correction import correct_segment_durations


def main():
    parser = argparse.ArgumentParser(description="Smart dubbing system")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Audio extraction sub-parser
    extract_parser = subparsers.add_parser("extract_audio", help="Extract audio from video file")
    extract_parser.add_argument("--input", "-i", required=True, help="Input video-file path")
    extract_parser.add_argument("--output", "-o", help="Extracted audio-file path (optional)")

    # Transcription command sub-parser
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio with timestamps")
    transcribe_parser.add_argument("--job_id", required=True, help="Job identifier")
    transcribe_parser.add_argument("--input", "-i", required=True, help="Input audio-file path")
    transcribe_parser.add_argument("--output", "-o", help="Output transcription file path (optional)")
    transcribe_parser.add_argument("--source_language", "-sl", help="Source language code (e.g., 'en', 'ru', 'es')")
    transcribe_parser.add_argument("--openai_api_key", required=True, help="OpenAI API key")

    # Segments correction command sub-parser
    correct_parser = subparsers.add_parser("correct", help="Structure raw transcription")
    correct_parser.add_argument("--input", "-i", required=True, help="Path to raw transcription file")
    correct_parser.add_argument("--output", "-o", help="Path to save structured transcription (optional)")

    # Sub-parser for clearing segments command
    cleanup_parser = subparsers.add_parser("cleanup",
                                           help="Clean-up segments (deletes merged and removes extra spaces)")
    cleanup_parser.add_argument("--input", "-i", required=True,
                                help="Path to corrected transcription file")
    cleanup_parser.add_argument("--output", "-o", help="Path to save cleaned transcription (optional)")

    # Sub-parser for optimizing segments to sentences
    optimize_parser = subparsers.add_parser("optimize",
                                            help="Optimize transcription by breaking segments into sentences")
    optimize_parser.add_argument("--input", "-i", required=True,
                                 help="Path to the cleaned-up transcription")
    optimize_parser.add_argument("--output", "-o",
                                 help="Path to save the optimized transcription (optional)")

    # Sub-parser for adjusting segments' timing
    adjust_parser = subparsers.add_parser("adjust_timing",
                                          help="Adjust segment end times to match next segment start times")
    adjust_parser.add_argument("--input", "-i", required=True, help="Path to the optimized transcription file")
    adjust_parser.add_argument("--output", "-o", help="Path to save adjusted transcription (optional)")

    # Sub-parser for the segment translation command
    translate_parser = subparsers.add_parser("translate", help="Translates transcription segments")
    translate_parser.add_argument("--input", "-i", required=True,
                                  help="Path to time-adjusted transcription file")
    translate_parser.add_argument("--output", "-o",
                                  help="Path to save translated transcription (optional)")
    translate_parser.add_argument("--model", "-m", default="gpt-4o",
                                  help="Translation model (default: gpt-4o)")
    translate_parser.add_argument("--target_language", "-tl", required=True,
                                  help="Target language for translation")
    translate_parser.add_argument("--openai_api_key", required=True, help="OpenAI API key")

    # Sub-parser for the segment voicing over command
    tts_parser = subparsers.add_parser("tts", help="Voices over translated segments")
    tts_parser.add_argument("--input", "-i", required=True,
                            help="Path to translated transcription file")
    tts_parser.add_argument("--output", "-o", help="Path to save the audio file (optional)")
    tts_parser.add_argument("--dealer", "-d", required=True, choices=["openai", "elevenlabs"],
                            help="TTS service provider")
    tts_parser.add_argument("--voice", "-v", required=True,
                            help="Voice for dubbing")
    tts_parser.add_argument("--elevenlabs_api_key", help="ElevenLabs API key (required for ElevenLabs)")
    tts_parser.add_argument("--openai_api_key", required=True, help="OpenAI API key")
    tts_parser.add_argument("--job_id", required=True, help="Job identifier")

    # # Sub-parser for regenerating a single segment
    # segment_tts_parser = subparsers.add_parser("segment-tts",
    #                                            help="Regenerate audio for a specific segment with updated translation")
    # segment_tts_parser.add_argument("--input", "-i", required=True,
    #                                 help="Path to translated transcription file")
    # segment_tts_parser.add_argument("--segment-id", "-s", required=True, type=int,
    #                                 help="ID of the segment to regenerate")
    # segment_tts_parser.add_argument("--output", "-o",
    #                                 help="Path to save the segment audio file (optional)")
    # segment_tts_parser.add_argument("--dealer", "-d", default="openai", choices=["openai", "elevenlabs"],
    #                                 help="TTS service provider (default: openai)")
    # segment_tts_parser.add_argument("--voice", "-v", default="onyx",
    #                                 help="Voice for dubbing (only used for OpenAI, default: onyx)")

    text_correction_parser = subparsers.add_parser("auto-correct",
                                                   help="Automatically correct text segments to match original duration")
    text_correction_parser.add_argument("--input", "-i", required=True,
                                        help="Path to translated transcription file with TTS data")
    text_correction_parser.add_argument("--output", "-o", help="Path to save reassembled audio file (optional)")
    text_correction_parser.add_argument("--attempts", "-a", type=int, default=5,
                                        help="Maximum number of correction attempts (default: 5)")
    text_correction_parser.add_argument("--threshold", "-t", type=float, default=0.2,
                                        help="Duration difference threshold for correction (default: 0.2 or 20%)")
    text_correction_parser.add_argument("--dealer", "-d", required=True, choices=["openai", "elevenlabs"],
                                        help="TTS service provider")
    text_correction_parser.add_argument("--voice", "-v", required=True,
                                        help="Voice for dubbing")
    text_correction_parser.add_argument("--elevenlabs_api_key", help="ElevenLabs API key (optional)")
    text_correction_parser.add_argument("--openai_api_key", required=True,
                                        help="OpenAI API key (required for text correction)")
    text_correction_parser.add_argument("--job_id", required=True, help="Job identifier")

    # # Add a new sub-parser for reassembling audio from existing segments
    # reassemble_parser = subparsers.add_parser("reassemble",
    #                                           help="Reassemble audio file from existing segments")
    # reassemble_parser.add_argument("--input", "-i", required=True,
    #                                help="Path to translated transcription file")
    # reassemble_parser.add_argument("--output", "-o",
    #                                help="Path to save the reassembled audio file (optional)")
    # reassemble_parser.add_argument("--intro", action="store_true",
    #                                help="Add intro audio at the beginning (replaces first 4 seconds)")
    # reassemble_parser.add_argument("--outro", action="store_true",
    #                                help="Add outro audio after the last segment")

    # Sub-parser for processing input video
    video_parser = subparsers.add_parser("process_video", help="Process video according to TTS duration")
    video_parser.add_argument("--job_id", required=True, help="Job identifier")
    video_parser.add_argument("--input_video", required=True, help="Path to input video file")
    video_parser.add_argument("--json_file", required=True, help="Path to translated JSON with timing")
    video_parser.add_argument("--output_video", required=True, help="Path for output video")
    video_parser.add_argument("--output_video_premium", required=True, help="Path for premium output video")
    video_parser.add_argument("--resources_dir", required=True, help="Path to resources directory")
    video_parser.add_argument("--is_premium", action="store_true",
                              help="Premium user (no intro/outro)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "extract_audio":
        if not os.path.exists(args.input):
            print(f"Error: Video file {args.input} not found.")
            sys.exit(1)

        try:
            result_file = extract_audio(args.input, args.output)
            print(f"Extracting audio from video file completed successfully.")
            print(f"The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error during video conversion: {e}")
            sys.exit(1)

    elif args.command == "transcribe":
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found.")
            sys.exit(1)

        print(f"Transcribing the file: {args.input}")
        if args.source_language:
            print(f"Using source language: {args.source_language}")
        try:
            result_file = transcribe_audio_with_timestamps(
                args.input,
                args.job_id,
                source_language=args.source_language,
                output_file=args.output,
                openai_api_key=args.openai_api_key
            )

            print(f"Transcription completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error during transcription: {e}")
            sys.exit(1)

    elif args.command == "correct":
        if not os.path.exists(args.input):
            print(f"Error: Transcription file {args.input} not found.")
            sys.exit(1)

        print(f"Restructuring transcription segments: {args.input}")
        try:
            result_file = correct_transcript_segments(args.input, args.output)
            print(f"Restructuring completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error during restructuring: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.command == "cleanup":
        if not os.path.exists(args.input):
            print(f"Error: Transcription file {args.input} not found.")
            sys.exit(1)

        print(f"Cleaning-up transcription segments: {args.input}")
        try:
            result_file = cleanup_transcript_segments(args.input, args.output)
            print(f"Transcription segments cleaned-up successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error cleaning-up: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.command == "optimize":
        if not os.path.exists(args.input):
            print(f"Error: Transcription file {args.input} not found.")
            sys.exit(1)

        print(f"Optimizing segments in transcription file: {args.input}")
        try:
            result_file = optimize_transcription_segments(args.input, args.output)
            print(f"Segment optimization completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error optimizing segments: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.command == "adjust_timing":
        if not os.path.exists(args.input):
            print(f"Error: Input JSON file {args.input} not found.")
            sys.exit(1)

        print(f"Adjusting segment timing in file: {args.input}")

        try:
            result_file = adjust_segments_timing(args.input, args.output)
            print(f"Timing adjustment completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error during timing adjustment: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.command == "translate":
        if not os.path.exists(args.input):
            print(f"Error: Transcription file {args.input} not found.")
            sys.exit(1)

        print(f"Translating transcription segments: {args.input}")
        print(f"Target language: {args.target_language}")
        print(f"Using model: {args.model}")

        try:
            result_file = translate_transcribed_segments(
                args.input,
                args.output,
                target_language=args.target_language,
                model=args.model,
                openai_api_key=args.openai_api_key
            )

            print(f"Translation completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error translating segments: {e}")
            sys.exit(1)


    elif args.command == "tts":
        if not os.path.exists(args.input):
            print(f"Error: Translated transcription file {args.input} not found.")
            sys.exit(1)

        if args.dealer == "elevenlabs" and not args.elevenlabs_api_key:
            print("Error: ElevenLabs API key is required for ElevenLabs TTS.")
            sys.exit(1)

        print(f"Voicing-over translated segments using {args.dealer}: {args.input}")
        print(f"Using voice: {args.voice}")

        try:
            result_file = generate_tts_for_segments(
                args.input,
                args.job_id,
                args.output,
                args.voice,
                args.dealer,
                elevenlabs_api_key=args.elevenlabs_api_key,
                openai_api_key=args.openai_api_key
            )

            if result_file:
                print(f"Voicing-over completed successfully. The result was saved in file: {result_file}")
            else:
                print("Error: TTS generation failed.")
                sys.exit(1)

        except Exception as e:
            print(f"Error voicing-over segments: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # elif args.command == "segment-tts":
    #     if not os.path.exists(args.input):
    #         print(f"Error: Translated transcription file {args.input} not found.")
    #         return
    #
    #     print(f"Regenerating segment {args.segment_id} using {args.dealer}: {args.input}")
    #     try:
    #         result_file = regenerate_segment(
    #             args.input, args.segment_id, args.output, args.voice, args.dealer
    #         )
    #         if result_file:
    #             print(f"Segment regeneration completed successfully. The result was saved in file: {result_file}")
    #     except Exception as e:
    #         print(f"Error regenerating segment: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return

    elif args.command == "auto-correct":
        if not os.path.exists(args.input):
            print(f"Error: Transcription file {args.input} not found.")
            sys.exit(1)

        if args.dealer == "elevenlabs" and not args.elevenlabs_api_key:
            print("Error: ElevenLabs API key is required for ElevenLabs TTS.")
            sys.exit(1)

        if not args.openai_api_key:
            print(f"Error: OpenAI API key is required for all operations")
            sys.exit(1)

        print(f"Automatically correcting segment durations: {args.input}")
        print(f"Using TTS provider: {args.dealer} with voice: {args.voice}")

        try:
            result_file = correct_segment_durations(
                translation_file=args.input,
                job_id=args.job_id,
                max_attempts=args.attempts,
                threshold=args.threshold,
                voice=args.voice,
                dealer=args.dealer,
                elevenlabs_api_key=args.elevenlabs_api_key,
                openai_api_key=args.openai_api_key
            )

            if not result_file:
                print("Error: Segment correction failed.")
                sys.exit(1)

            print(f"Automatic text correction completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error during automatic text correction: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        print(f"Reassembling audio from segments using file: {args.input}")
        try:
            result_file = reassemble_audio_file(
                args.input,
                args.job_id,
                args.output
            )

            if not result_file:
                print("Error: Audio reassembly failed.")
                sys.exit(1)

            print(f"Audio successfully reassembled. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error reassembling audio: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.command == "process_video":
        print("Initializing Video Processor...")

        try:
            if not os.path.exists(args.input_video):
                print(f"Error: Input video not found: {args.input_video}")
                sys.exit(1)

            if not os.path.exists(args.json_file):
                print(f"Error: JSON file not found: {args.json_file}")
                sys.exit(1)

            if not os.path.exists(args.resources_dir):
                print(f"Error: Resources directory not found: {args.resources_dir}")
                sys.exit(1)

            os.makedirs(os.path.dirname(args.output_video), exist_ok=True)

            processor = VideoProcessor(
                job_id=args.job_id,
                input_video_path=args.input_video,
                json_path=args.json_file,
                output_video_path=args.output_video,
                output_video_path_premium=args.output_video_premium,
                intro_outro_path=args.resources_dir,
                target_fps=25,
                is_premium=args.is_premium
            )

            if processor.process():
                print(f"Video processing completed successfully. The result was saved in file: {args.output_video}")
            else:
                print("Video processing failed.")
                sys.exit(1)

        except Exception as e:
            print(f"Error during video processing: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()
