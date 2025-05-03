#!/usr/bin/env python3
import argparse
import os
import dotenv

from modules.adjust_timing import adjust_segments_timing
from modules.cleaning_up_corrected_transcirption import cleanup_transcript_segments
from modules.transcribe_with_timestamps import transcribe_audio_with_timestamps
from modules.transcription_correction import correct_transcript_segments
from modules.translation import translate_transcript_segments
from modules.tts import generate_tts_for_segments, reassemble_audio_file
from modules.tts_correction import regenerate_segment
from video_duration_edit_workflow import VideoProcessor
from modules.video_to_audio_conversion import extract_audio
from modules.optimized_segmentation import optimize_transcription_segments
from modules.automatic_text_correction import correct_segment_durations


def main():
    dotenv.load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY must be set in .env file to get full functionality!")

    if not os.getenv("ELEVENLABS_API_KEY"):
        print("ELEVENLABS_API_KEY must be set in .env file to get full functionality!")

    parser = argparse.ArgumentParser(description="Smart dubbing system")

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Audio extraction sub-parser
    extract_parser = subparsers.add_parser("extract_audio", help="Extract audio from video file")
    extract_parser.add_argument("--input", "-i", required=True, help="Input video-file path")

    # Transcription command sub-parser
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio with timestamps")
    transcribe_parser.add_argument("--input", "-i", required=True, help="Input audio-file path")

    # Segments correction command sub-parser
    correct_parser = subparsers.add_parser("correct", help="Structure raw transcription")
    correct_parser.add_argument("--input", "-i", required=True, help="Path to raw transcription file")
    correct_parser.add_argument("--output", "-o",
                                help="Path to save structured transcription (optional)")
    correct_parser.add_argument("--start_timestamp", "-st", type=float,
                                help="Set specific start timestamp for the first segment (e.g. 0.0 or 4.0)")

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

    # Sub-parser for the segment translation command
    translate_parser = subparsers.add_parser("translate", help="Translates transcription segments")
    translate_parser.add_argument("--input", "-i", required=True,
                                  help="Path to time-adjusted transcription file")
    translate_parser.add_argument("--output", "-o",
                                  help="Path to save translated transcription (optional)")
    translate_parser.add_argument("--model", "-m", default="gpt-4o",
                                  help="Translation model (default: gpt-4o)")

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

    text_correction_parser = subparsers.add_parser("auto-correct",
                                                   help="Automatically correct text segments to match original duration")
    text_correction_parser.add_argument("--input", "-i", required=True,
                                        help="Path to translated transcription file with TTS data")
    text_correction_parser.add_argument("--attempts", "-a", type=int, default=5,
                                        help="Maximum number of correction attempts (default: 5)")
    text_correction_parser.add_argument("--threshold", "-t", type=float, default=0.2,
                                        help="Duration difference threshold for correction (default: 0.2 or 20%)")
    text_correction_parser.add_argument("--dealer", "-d", default="openai", choices=["openai", "elevenlabs"],
                                        help="TTS service provider (default: openai)")
    text_correction_parser.add_argument("--voice", "-v", default="onyx",
                                        help="Voice for dubbing (only used for OpenAI, default: onyx)")

    # Add a new sub-parser for reassembling audio from existing segments
    reassemble_parser = subparsers.add_parser("reassemble",
                                              help="Reassemble audio file from existing segments")
    reassemble_parser.add_argument("--input", "-i", required=True,
                                   help="Path to translated transcription file")
    reassemble_parser.add_argument("--output", "-o",
                                   help="Path to save the reassembled audio file (optional)")
    reassemble_parser.add_argument("--intro", action="store_true",
                                   help="Add intro audio at the beginning (replaces first 4 seconds)")
    reassemble_parser.add_argument("--outro", action="store_true",
                                   help="Add outro audio after the last segment")

    # Sub-parser for processing input video
    subparsers.add_parser("process_video", help="Process video according to TTS duration")


    args = parser.parse_args()

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

    elif args.command == "optimize":
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

    elif args.command == "adjust_timing":
        if not os.path.exists(args.input):
            print(f"Error: Input JSON file {args.input} not found.")
            return

        print(f"Adjusting segment timing in file: {args.input}")
        try:
            result_file = adjust_segments_timing(args.input)
            print(f"Timing adjustment completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error during timing adjustment: {e}")
            return

    elif args.command == "translate":
        if not os.path.exists(args.input):
            print(f"Error: Transcription file {args.input} not found.")
            return

        print(f"Translating transcription segments: {args.input}")
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

    elif args.command == "auto-correct":
        if not os.path.exists(args.input):
            print(f"Error: Transcription file {args.input} not found.")
            return

        print(f"Automatically correcting segment durations: {args.input}")
        try:
            result_file = correct_segment_durations(
                translation_file=args.input,
                max_attempts=args.attempts,
                threshold=args.threshold,
                voice=args.voice,
                dealer=args.dealer
            )
            print(f"Automatic text correction completed successfully. The result was saved in file: {result_file}")
        except Exception as e:
            print(f"Error during automatic text correction: {e}")
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


    elif args.command == "process_video":
        print("Initializing Video Processor...")
        try:

            current_dir = os.path.abspath(os.getcwd())
            input_dir = os.path.join(current_dir, "video_input")
            output_dir = os.path.join(current_dir, "video_output")
            resources_dir = os.path.join(current_dir, "resources")
            input_video = os.path.join(input_dir, "input.mp4")
            json_file = os.path.join(current_dir, "output", "timestamped_transcriptions",
                                     "input_transcribed_corrected_cleaned_optimized_adjusted_translated.json")
            output_video = os.path.join(output_dir, "output.mp4")

            os.makedirs(output_dir, exist_ok=True)

            if not os.path.exists(input_video):
                print(f"Error: Input video not found: {input_video}")
                return

            if not os.path.exists(json_file):
                print(f"Error: JSON file not found: {json_file}")
                return

            if not os.path.exists(resources_dir):
                print(f"Error: Resources directory not found: {resources_dir}")
                return

            processor = VideoProcessor(
                input_video_path=input_video,
                json_path=json_file,
                output_video_path=output_video,
                intro_outro_path=resources_dir,
                target_fps=25
            )

            if processor.process():
                print(f"Video processing completed successfully. The result was saved in file: {output_video}")
            else:
                print("Video processing failed.")

        except Exception as e:
            print(f"Error during video processing: {e}")
            import traceback
            traceback.print_exc()
            return

if __name__ == "__main__":
    main()
