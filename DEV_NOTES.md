## Extract audio from video file:
python3 cli.py extract_audio --input video_input/input.mp4
or
# for processing all videos in the directory
python3 cli.py extract_audio --input video_input

=================================================================

## Timestamped transcription:
python3 cli.py transcribe --input audio_input/input.mp3

=================================================================

## Timestamped transcription correction:
python3 cli.py correct --input output/timestamped_transcriptions/input_transcribed.json

=================================================================

# First segment start timestamp correction (adding intro gap if needed):
python3 cli.py correct --input output/timestamped_transcriptions/input_transcribed.json --start_timestamp 4.0

Args:
--start_timestamp int -> Set specific start timestamp for the first segment (e.g. 0.0 or 4.0), for example for adding intro

=================================================================

## Corrected timestamped transcription cleaning up:
python3 cli.py cleanup --input output/timestamped_transcriptions/input_transcribed_corrected.json

=================================================================

## Transcription segments optimization:
python3 cli.py optimize --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned.json

=================================================================

## Adjusting segments timing:
python3 cli.py adjust_timing --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized.json

=================================================================

## Translation:
python3 cli.py translate --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted.json

Args:
--model str -> Choose OpenAI's model for translation

=================================================================

## TTS (text-to-speech):
python3 cli.py tts --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted_translated.json

Args:
--dealer openai/elevenlabs -> Choose TTS provider
--voice str -> ONLY for OpenAI TTs provider
--intro -> Add intro from the 'resources' folder
--outro -> Add outro from the 'resources' folder

=================================================================

# Translated text automatic correction to get proper tts_duration:
python3 cli.py auto-correct --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted_translated.json

Args:
--dealer openai/elevenlabs -> Choose TTS provider

=================================================================

# Regenerating segment by id:
python3 cli.py segment-tts --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted_translated.json --segment-id 1 --dealer elevenlabs --output segment_1.mp3

# Reassemble output audio from prepared segments at output/temp_audio_segments:
python3 cli.py reassemble --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted_translated.json --intro --outro

=================================================================

## 6-step (audio extraction, transcription, correction, cleaning-up, optimization, time-adjustment) pipeline | STAGE 1:
# Process a specific video file
python3 processing_pipeline_stage1.py --input video_input/input.mp4

Args: 
--start_timestamp int -> Set specific start timestamp for the first segment (e.g. 0.0 or 4.0), for example for adding intro

=================================================================

## 4-step (translation, tts, text auto-correction, reassemble) pipeline | STAGE 2: 
python3 processing_pipeline_stage2.py --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted.json

Args:
--dealer openai/elevenlabs -> Choose TTS provider
--intro -> Add intro from the 'resources' folder
--outro -> Add outro from the 'resources' folder

=================================================================

# Edit original video duration with tts_duration info with 25 fps converting:
python3 cli.py process_video

=================================================================

## W3A processing steps:

python3 processing_pipeline_stage1.py --input video_input/input.mp4 --start_timestamp 4.0

python3 processing_pipeline_stage2.py --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted.json --dealer elevenlabs --intro --outro