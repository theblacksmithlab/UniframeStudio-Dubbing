## Timestamped transcription:
python cli.py transcribe --input audio_input/sample.mp3

## Timestamped transcription correction:
python cli.py correct --input output/timestamped_transcriptions/sample_timestamped.json

## Corrected timestamped transcription cleaning up:
python cli.py cleanup --input output/timestamped_transcriptions/sample_timestamped_corrected.json

## Translation:
python cli.py translate --input output/timestamped_transcriptions/sample_timestamped_corrected_cleaned.json 

## TTS (text-to-speech):
python cli.py tts --input output/timestamped_transcriptions/sample_timestamped_corrected_cleaned_translated.json
