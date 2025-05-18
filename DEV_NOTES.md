# DEV_NOTES

# Extract audio from video file:

```bash
python cli.py extract_audio --input video_input/input.mp4
```

or
## Extract audio from all videos in the directory:

python cli.py extract_audio --input video_input

=================================================================

# Timestamped transcription:

```bash
python cli.py transcribe --input audio_input/input.mp3
```

=================================================================

# Timestamped transcription correction:

```bash
python cli.py correct --input output/timestamped_transcriptions/input_transcribed.json
```

=================================================================

# Transcription segments correction:

```bash
python cli.py correct --input output/timestamped_transcriptions/input_transcribed.json --start_timestamp 4.0
```

## Args 
--start_timestamp 4.0 -> Set specific start timestamp for the first segment (e.g. 0.0 or 4.0), for example for adding
known length intro later (optional)

=================================================================

# Corrected timestamped transcription cleaning up:

```bash
python cli.py cleanup --input output/timestamped_transcriptions/input_transcribed_corrected.json
```

=================================================================

# Transcription segments optimization:

```bash
python cli.py optimize --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned.json
```

=================================================================

# Adjusting segments timing:

```bash
python cli.py adjust_timing --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized.json
```

=================================================================

# Translation:

```bash
python3 cli.py translate --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted.json
```

## Args 
--model str -> Choose OpenAI's model for translation (optional, default: 'gpt-4o')

=================================================================

# TTS (text-to-speech):

```bash
python cli.py tts --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted_translated.json
```

Args:
--dealer openai/elevenlabs -> Choose TTS provider (optional, default: 'openai')
--voice str -> only use for OpenAI TTs provider (optional, default: 'onyx')
--intro -> Add intro from the 'resources' folder (optional)
--outro -> Add outro from the 'resources' folder (optional)

=================================================================
ADDITIONAL POST-PROCESSING:
=================================================================

# Translated text automatic correction to get proper tts_duration:

```bash
python cli.py auto-correct --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted_translated.json
```

## Args 
--dealer openai/elevenlabs -> Choose TTS provider (optional, default: 'openai')

=================================================================

## Regenerating segment by id:

```bash
python cli.py segment-tts --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted_translated.json
```

## Args
-- segment-id int -> Specify transcription segment to regenerate
-- dealer openai/elevenlabs -> Choose TTS provider (optional, default: 'openai')
-- output str.mp3 -> Specify path to the regenerated file

=================================================================

## Reassemble output audio from prepared segments at output/temp_audio_segments:

```bash
python cli.py reassemble --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted_translated.json
```

Args
--intro -> Add intro from the 'resources' folder (optional)
--outro -> Add outro from the 'resources' folder (optional)

=================================================================

# 6-step (audio extraction, transcription, correction, cleaning-up, optimization, time-adjustment) pipeline | STAGE 1:
## Process input video

```bash
python processing_pipeline_stage1.py --input video_input/input.mp4
```

## Args: 
--start_timestamp int -> Set specific start timestamp for the first segment (e.g. 0.0 or 4.0), for example for adding intro

=================================================================

# 4-step (translation, tts, text auto-correction, reassemble) pipeline | STAGE 2: 

```bash
python processing_pipeline_stage2.py --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted.json
```

## Args:
--dealer openai/elevenlabs -> Choose TTS provider (optional, default: 'openai')
--intro -> Add intro from the 'resources' folder (optional)
--outro -> Add outro from the 'resources' folder (optional)

=================================================================

# Align the original video to match the timing of the new voiceover with 25 fps converting:

```bash
python cli.py process_video
```
