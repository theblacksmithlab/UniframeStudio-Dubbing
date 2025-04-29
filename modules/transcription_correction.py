import os
import json


def merge_segments(segments, current_idx, next_idx):
    if current_idx >= len(segments) or next_idx >= len(segments):
        return False

    current = segments[current_idx]
    next_seg = segments[next_idx]

    if next_seg.get("merged", False):
        return False

    current["text"] = current["text"] + next_seg["text"]

    current["end"] = next_seg["end"]

    if "merged_segments" not in current:
        current["merged_segments"] = [current["id"]]

    current["merged_segments"].append(next_seg["id"])

    next_seg["merged"] = True

    return True


def is_sentence_complete(text):
    text = text.strip()
    if not text:
        return False
    return text[-1] in ['.', '!', '?']


def correct_transcript_segments(input_file, output_file=None, start_timestamp=None):
    if output_file is None:
        base_dir = os.path.dirname(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(base_dir, f"{base_name}_corrected.json")

    with open(input_file, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    segments = transcript.get('segments', [])
    print(f"Loaded {len(segments)} segments for processing.")

    max_merges = len(segments)

    corrected_segments = []
    for segment in segments:
        corrected_segments.append(segment.copy())

    i = 0
    merge_count = 0

    while i < len(corrected_segments) and merge_count < max_merges:
        if corrected_segments[i].get("merged", False):
            i += 1
            continue

        if not is_sentence_complete(corrected_segments[i]["text"]):
            next_idx = i + 1
            while next_idx < len(corrected_segments) and corrected_segments[next_idx].get("merged", False):
                next_idx += 1

            if next_idx < len(corrected_segments):
                print(
                    f"Merging segment {i} (ID: {corrected_segments[i]['id']}) with segment {next_idx} (ID: {corrected_segments[next_idx]['id']})"
                )

                merge_result = merge_segments(corrected_segments, i, next_idx)

                if merge_result:
                    merge_count += 1

                if merge_count >= max_merges:
                    print("WARNING! Merging limit reached, stopping the process.")
                    i += 1
            else:
                corrected_segments[i]["text"] = corrected_segments[i]["text"].strip() + "."
                print(f"Added a period to segment {i}")
                i += 1
        else:
            i += 1

    transcript['segments'] = corrected_segments

    full_text = ""
    for segment in corrected_segments:
        if not segment.get("merged", False):
            full_text += " " + segment["text"].strip()

    transcript['text'] = full_text.strip()

    if start_timestamp is not None:
        for segment in transcript['segments']:
            if segment['id'] == 0 and not segment.get("merged", False):
                old_start = segment["start"]
                segment["start"] = start_timestamp
                print(f"Corrected start time for first segment from {old_start} to {start_timestamp}")
                break

        if 'words' in transcript and transcript['words']:
            first_word = transcript['words'][0]
            old_word_start = first_word["start"]
            first_word["start"] = start_timestamp
            print(f"Corrected start time for first word from {old_word_start} to {start_timestamp}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    print(f"Correction complete. Result saved to {output_file}.")

    return output_file
