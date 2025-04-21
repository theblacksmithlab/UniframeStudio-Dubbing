import json
import os
import re


def split_into_sentences(text):
    # Protect dots in abbreviations and domains
    text = re.sub(r'(\b\w\.\w\.)', lambda m: m.group(0).replace('.', '<DOT>'), text)
    text = re.sub(r'(\b[\w-]+\.[a-zA-Z]{2,})', lambda m: m.group(0).replace('.', '<DOT>'), text)

    sentences = []
    current = ""
    i = 0

    while i < len(text):
        current += text[i]

        # Check for end of sentence
        if text[i] in ['.', '!', '?'] and current.strip():
            # Not a decimal number
            if not re.search(r'\d\.$', current.strip()):
                # Check for space and capital letter after punctuation
                if i + 2 < len(text) and text[i + 1].isspace() and text[i + 2].isupper():
                    sentences.append(current.strip())
                    current = ""
                # Or end of text
                elif i + 1 >= len(text):
                    sentences.append(current.strip())
                    current = ""

        i += 1

    # Add the last sentence if any
    if current.strip():
        sentences.append(current.strip())

    # Restore protected dots
    sentences = [s.replace('<DOT>', '.') for s in sentences]

    return sentences


def find_word_in_words_list(target_word, words_list, start_time, end_time, position="first"):
    target_word = target_word.lower().strip()

    # Remove punctuation for matching
    clean_target = re.sub(r'[^\w\s]', '', target_word)

    # Find all occurrences of the word within the time range
    candidates = []
    for word_info in words_list:
        word = word_info.get("word", "").lower().strip()
        word_time = word_info.get("start", 0) if position == "first" else word_info.get("end", 0)

        # Remove punctuation for matching
        clean_word = re.sub(r'[^\w\s]', '', word)

        # Check if words match and time is within range
        if clean_word == clean_target and start_time <= word_time <= end_time:
            candidates.append(word_info)

    if not candidates:
        return None

    # For first word, return the one with earliest start time
    # For last word, return the one with latest end time
    if position == "first":
        return min(candidates, key=lambda x: x.get("start", 0))
    else:
        return max(candidates, key=lambda x: x.get("end", 0))


def get_sentence_timestamps(sentence, segment, words_list):
    sentence = sentence.strip()
    if not sentence:
        return None

    # Get first and last words of the sentence
    first_word_match = re.search(r'\b(\w+)', sentence)
    last_word_match = re.search(r'\b(\w+)\W*$', sentence)

    if not first_word_match or not last_word_match:
        return None

    first_word = first_word_match.group(1).lower()
    last_word = last_word_match.group(1).lower()

    segment_start = segment.get("start", 0)
    segment_end = segment.get("end", 0)

    # Find the first word in the words list
    first_word_info = find_word_in_words_list(
        first_word, words_list, segment_start, segment_end, "first"
    )

    # Find the last word in the words list
    last_word_info = find_word_in_words_list(
        last_word, words_list, segment_start, segment_end, "last"
    )

    if not first_word_info or not last_word_info:
        # Fallback: use segment timestamps if word detection fails
        return {
            "start": segment_start,
            "end": segment_end
        }

    return {
        "start": first_word_info.get("start", segment_start),
        "end": last_word_info.get("end", segment_end)
    }


def optimize_transcription_segments(transcription_file, output_file=None, min_segment_length=60):
    with open(transcription_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if output_file is None:
        base_name = os.path.splitext(transcription_file)[0]
        output_file = f"{base_name}_optimized.json"

    segments = data.get("segments", [])
    words_list = data.get("words", [])

    print(f"Optimizing {len(segments)} segments...")

    # STEP 1: Split segments into sentences with proper timestamps
    raw_segments = []

    for segment in segments:
        if "merged" in segment and segment["merged"]:
            continue  # Skip merged segments

        segment_text = segment.get("text", "").strip()
        sentences = split_into_sentences(segment_text)

        print(f"Segment {segment.get('id')}: Found {len(sentences)} sentences")

        if len(sentences) <= 1:
            # If there's only one sentence, keep the segment as is
            raw_segments.append({
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": segment_text
            })
        else:
            # Process multiple sentences in the segment
            for sentence in sentences:
                # Get accurate timestamps for the sentence
                timestamps = get_sentence_timestamps(sentence, segment, words_list)

                if not timestamps:
                    print(f"Warning: Could not find timestamps for sentence: {sentence[:30]}...")
                    continue

                # Validate timestamps (ensure start < end)
                if timestamps["start"] > timestamps["end"]:
                    print(f"Warning: Invalid timestamps for sentence: {sentence[:30]}...")
                    timestamps["start"], timestamps["end"] = timestamps["end"], timestamps["start"]

                raw_segments.append({
                    "start": timestamps["start"],
                    "end": timestamps["end"],
                    "text": sentence
                })

    # Sort raw segments by start time
    raw_segments.sort(key=lambda x: x["start"])

    # STEP 2: Merge short segments to meet minimum length requirement
    merged_segments = []
    current_text = ""
    current_start = None
    current_end = None

    for segment in raw_segments:
        text = segment["text"]

        # If we don't have current segment or the current is already long enough
        if not current_text or len(current_text) >= min_segment_length:
            # Save the previous segment if it exists
            if current_text:
                merged_segments.append({
                    "start": current_start,
                    "end": current_end,
                    "text": current_text
                })

            # Start new segment
            current_text = text
            current_start = segment["start"]
            current_end = segment["end"]
        else:
            # Current segment is too short, append this segment to it
            current_text += " " + text
            current_end = segment["end"]

    # Add the last segment if any
    if current_text:
        merged_segments.append({
            "start": current_start,
            "end": current_end,
            "text": current_text
        })

    print(f"After merging short segments: {len(merged_segments)} segments")

    # STEP 3: Assign IDs and fix any overlapping segments
    new_segments = []
    for i, segment in enumerate(merged_segments):
        new_segments.append({
            "id": i,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    # Fix overlapping segments
    for i in range(len(new_segments) - 1):
        if new_segments[i]["end"] > new_segments[i + 1]["start"]:
            midpoint = (new_segments[i]["end"] + new_segments[i + 1]["start"]) / 2
            new_segments[i]["end"] = midpoint
            new_segments[i + 1]["start"] = midpoint
            print(f"Fixed overlap between segments {i} and {i + 1}")

    # Check for minimum segment duration
    MIN_DURATION = 0.5  # seconds
    for i, segment in enumerate(new_segments):
        duration = segment["end"] - segment["start"]
        if duration < MIN_DURATION:
            print(f"Warning: Segment {i} duration is too short ({duration:.2f}s): {segment['text'][:30]}...")

            # Try to extend by borrowing time from adjacent segments
            if i > 0 and i < len(new_segments) - 1:
                # Get time from both previous and next
                borrow_time = (MIN_DURATION - duration) / 2
                new_segments[i - 1]["end"] -= borrow_time
                new_segments[i + 1]["start"] += borrow_time
                new_segments[i]["start"] -= borrow_time
                new_segments[i]["end"] += borrow_time
            elif i > 0:
                # Get time from previous
                new_segments[i - 1]["end"] -= (MIN_DURATION - duration)
                new_segments[i]["start"] -= (MIN_DURATION - duration)
            elif i < len(new_segments) - 1:
                # Get time from next
                new_segments[i + 1]["start"] += (MIN_DURATION - duration)
                new_segments[i]["end"] += (MIN_DURATION - duration)

    # Prepare the result
    result = {
        "text": data.get("text", ""),
        "segments": new_segments
    }

    # Save the result
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Optimization complete! {len(new_segments)} segments created.")
    print(f"Result saved to: {output_file}")

    return output_file