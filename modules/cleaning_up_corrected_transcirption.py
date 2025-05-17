import os
import json


def cleanup_transcript_segments(input_file, output_file=None):
    if output_file is None:
        base_dir = os.path.dirname(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(base_dir, f"{base_name}_cleaned.json")

    with open(input_file, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    outro_gap_duration = transcript.get('outro_gap_duration')

    segments = transcript.get('segments', [])

    cleaned_segments = []
    removed_count = 0
    cleaned_count = 0

    for segment in segments:
        if segment.get("merged", False):
            removed_count += 1
            continue

        original_text = segment["text"]
        segment["text"] = segment["text"].strip()

        if original_text != segment["text"]:
            cleaned_count += 1

        cleaned_segments.append(segment)

    transcript['segments'] = cleaned_segments

    full_text = " ".join([segment["text"] for segment in cleaned_segments])
    transcript['text'] = full_text

    if outro_gap_duration is not None:
        transcript['outro_gap_duration'] = outro_gap_duration

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    print(f"Cleanup complete. Removed {removed_count} merged segments, cleaned {cleaned_count} text entries.")
    print(f"Result saved to {output_file}")

    return output_file
