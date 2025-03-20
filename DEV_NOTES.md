## Timestamped transcription:
python3 cli.py transcribe --input audio_input/sample.mp3

## Timestamped transcription correction:
python3 cli.py correct --input output/timestamped_transcriptions/sample_timestamped.json

## Corrected timestamped transcription cleaning up:
python3 cli.py cleanup --input output/timestamped_transcriptions/sample_timestamped_corrected.json

## Translation:
python3 cli.py translate --input output/timestamped_transcriptions/sample_timestamped_corrected_cleaned.json 

## TTS (text-to-speech):
python3 cli.py tts --input output/timestamped_transcriptions/sample_timestamped_corrected_cleaned_translated.json

python3 cli.py tts --input output/timestamped_transcriptions/output_timestamped_corrected_cleaned_translated.json --dealer elevenlabs

