# W3A processing steps:

```bash
python processing_pipeline_stage1.py --input video_input/input.mp4 --start_timestamp 4.0
```

```bash
python processing_pipeline_stage2.py --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted.json --dealer elevenlabs --intro --outro
```

```bash
python cli.py segment-tts --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted_translated.json --segment-id 1 --dealer elevenlabs --output segment_1.mp3
```

```bash
python cli.py reassemble --input output/timestamped_transcriptions/input_transcribed_corrected_cleaned_optimized_adjusted_translated.json --intro --outro
```
