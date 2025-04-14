## Extract audio from video file:
python3 cli.py extract_audio --input video_input/input.mp4
or
# for processing all videos in the directory
python3 cli.py extract_audio --input video_input

## Timestamped transcription:
python3 cli.py transcribe --input audio_input/input.wav

## Timestamped transcription correction:
python3 cli.py correct --input output/timestamped_transcriptions/input_timestamped.json

# First segment start timestamp correction (FOR W3A ONLY):
python3 cli.py correct --input output/timestamped_transcriptions/input_timestamped.json --start_timestamp 4.0

## Corrected timestamped transcription cleaning up:
python3 cli.py cleanup --input output/timestamped_transcriptions/input_timestamped_corrected.json

## Transcription segments optimization:
python3 cli.py optimize-segments --input output/timestamped_transcriptions/input_timestamped_corrected_cleaned.json

## Adjusting segments timing:
python3 cli.py adjust_timing --input output/timestamped_transcriptions/input_timestamped_corrected_cleaned_optimized.json

## Translation:
python3 cli.py translate --input output/timestamped_transcriptions/input_timestamped_corrected_cleaned_optimized_adjusted.json 

## TTS (text-to-speech):
python3 cli.py tts --input output/timestamped_transcriptions/input_timestamped_corrected_cleaned_optimized_adjusted_translated.json

# TTS with determined API:
python3 cli.py tts --input output/timestamped_transcriptions/input_timestamped_corrected_cleaned_optimized_adjusted_translated.json --dealer elevenlabs

# TTS with added intro and outro from resources folder
python3 cli.py tts --input output/timestamped_transcriptions/input_timestamped_corrected_cleaned_optimized_adjusted_translated.json --dealer elevenlabs --outro --intro


## 6-stage (audio extraction, transcription, correction, cleaning-up, optimization, time-adjustment) pipeline:
# Process a specific video file
python3 processing_video_pipeline.py --input video_input/input.mp4

# Process a specific video with a start offset for intro
python3 processing_video_pipeline.py --input video_input/input.mp4 --start_timestamp 4.0

# WARNING! Process ALL videos in the default input directory with a start timestamp
python processing_video_pipeline.py --input video_input --start_timestamp 4.0


# Regenerating segment by id:
python3 cli.py segment-tts --input output/timestamped_transcriptions/input_timestamped_corrected_cleaned_optimized_adjusted_translated.json --segment-id 1 --dealer elevenlabs --output segment_1.mp3

# Reassemble output audio from prepared segments at output/temp_audio_segments:
python3 cli.py reassemble --input output/timestamped_transcriptions/input_timestamped_corrected_cleaned_optimized_adjusted_translated.json --intro --outro

# Edit original video duration with tts_duration info:
python3 cli.py process_video

python cli.py process_video --input path/to/tts_duration/file.json

python cli.py process_video --fps25
