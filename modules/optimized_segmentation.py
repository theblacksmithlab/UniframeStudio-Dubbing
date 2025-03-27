import re
import json
import os

def optimize_transcription_segments(transcription_file, output_file=None):
    with open(transcription_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if output_file is None:
        base_name = os.path.splitext(transcription_file)[0]
        output_file = f"{base_name}_optimized.json"

    result = {
        "text": data["text"],
        "segments": []
    }

    words = data.get("words", [])

    new_segment_id = 0

    sentence_pattern = re.compile(r'([^.!?]+[.!?]+)(?:\s|$)')

    for segment in data["segments"]:
        segment_text = segment["text"].strip()
        segment_start = segment["start"]
        segment_end = segment["end"]

        if not segment_text:
            continue

        segment_words = []
        for word in words:
            if segment_start <= word["start"] <= segment_end:
                segment_words.append(word)

        if len(segment_words) < 2:
            result["segments"].append({
                "id": new_segment_id,
                "start": segment_start,
                "end": segment_end,
                "text": segment_text
            })
            new_segment_id += 1
            continue

        sentences = sentence_pattern.findall(segment_text)

        if not sentences:
            sentences = [segment_text]

        words_per_sentence = []
        for sentence in sentences:
            clean_sentence = re.sub(r'[^\w\s]', '', sentence)
            word_count = len(clean_sentence.split())
            words_per_sentence.append(word_count)

        start_idx = 0
        for i, word_count in enumerate(words_per_sentence):
            if i == len(words_per_sentence) - 1:
                end_idx = len(segment_words)
            else:
                end_idx = min(start_idx + word_count, len(segment_words))

            sentence_words = segment_words[start_idx:end_idx]

            if not sentence_words:
                continue

            sentence_start = sentence_words[0]["start"]
            sentence_end = sentence_words[-1]["end"]

            new_segment = {
                "id": new_segment_id,
                "start": sentence_start,
                "end": sentence_end,
                "text": sentences[i].strip()
            }

            result["segments"].append(new_segment)
            new_segment_id += 1

            start_idx = end_idx

        if start_idx < len(segment_words):
            remaining_words = segment_words[start_idx:]

            remaining_start = remaining_words[0]["start"]
            remaining_end = remaining_words[-1]["end"]

            result["segments"].append({
                "id": new_segment_id,
                "start": remaining_start,
                "end": remaining_end,
                "text": segment_text.split(sentences[-1])[-1].strip() if sentences else segment_text
            })
            new_segment_id += 1

    result["segments"].sort(key=lambda x: x["start"])

    for i, segment in enumerate(result["segments"]):
        segment["id"] = i

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Optimized transcription saved to: {output_file}")
    print(f"Original segments: {len(data['segments'])}, Optimized segments: {len(result['segments'])}")

    check_segment_overlaps(output_file)

    return output_file


def check_segment_overlaps(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = data.get('segments', [])
    if not segments:
        print("File doesn't contain any segments to check.")
        return True

    overlaps_found = False

    for i in range(len(segments) - 1):
        current_segment = segments[i]
        next_segment = segments[i + 1]

        if current_segment["end"] > next_segment["start"]:
            print(f"WARNING! Segment overlap detected:")
            print(f"  Segment {current_segment['id']} (end: {current_segment['end']}) overlaps")
            print(f"  Segment {next_segment['id']} (start: {next_segment['start']})")
            print(f"  Overlap: {current_segment['end'] - next_segment['start']:.2f} seconds")
            overlaps_found = True

    if not overlaps_found:
        print("Check completed: No segment overlaps detected.")

    return not overlaps_found
