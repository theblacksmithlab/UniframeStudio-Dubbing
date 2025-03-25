## Extract audio from video file:
python3 cli.py extract_audio --input video_input/sample.mp4
or
# for processing all videos in the directory
python3 cli.py extract_audio --input video_input

## Timestamped transcription:
python3 cli.py transcribe --input audio_input/sample.mp3

## Timestamped transcription correction:
python3 cli.py correct --input output/timestamped_transcriptions/sample_timestamped.json
or
# First segment start timestamp correction (for W3A only):
python3 cli.py correct --input sample_timestamped.json --start_timestamp 4.0
or
python3 cli.py correct -i transcription.json -st 4.0

## Corrected timestamped transcription cleaning up:
python3 cli.py cleanup --input output/timestamped_transcriptions/sample_timestamped_corrected.json

## Translation:
python3 cli.py translate --input output/timestamped_transcriptions/sample_timestamped_corrected_cleaned.json 

## TTS (text-to-speech):
python3 cli.py tts --input output/timestamped_transcriptions/sample_timestamped_corrected_cleaned_translated.json

# TTS with determined API:
python3 cli.py tts --input output/timestamped_transcriptions/sample_timestamped_corrected_cleaned_translated.json --dealer elevenlabs

# TTS with added intro and outro from resources folder
python3 cli.py tts --input output/timestamped_transcriptions/sample_timestamped_corrected_cleaned_translated.json --dealer elevenlabs --outro --intro


## 4-stage (audio extraction, transcription, correction, cleaning-up) pipeline:
# Process a specific video file
python3 processing_video_pipeline.py --input video_input/sample.mp4

# Process a specific video with a start timestamp
python processing_video_pipeline.py --input video_input/sample.mp4 --start_timestamp 4.0

# WARNING! Process ALL videos in the default input directory with a start timestamp
python processing_video_pipeline.py --input input --start_timestamp 4.0