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


def find_exact_sequence(sentence, words_list, segment_start, segment_end):
    # Extract words from the sentence
    sentence_words = [w.lower() for w in re.findall(r'\b(\w+)\b', sentence)]
    if not sentence_words:
        return {"start": segment_start, "end": segment_end}

    # Filter words_list to only include words in the time range
    filtered_words = [w for w in words_list if segment_start <= w.get("start", 0) <= segment_end]

    # Clean words in the list for comparison
    clean_words = []
    for word_info in filtered_words:
        word = word_info.get("word", "").lower()
        clean_word = re.sub(r'[^\w\s]', '', word)
        clean_words.append(clean_word)

    # We'll use a sliding window to find the best match
    best_match_start_idx = -1
    best_match_end_idx = -1
    best_match_score = -1

    # Try each position in the filtered words as a potential starting point
    for start_idx in range(len(filtered_words) - len(sentence_words) + 1):
        match_score = 0

        # Compare words from this starting point with sentence words
        for i in range(min(len(sentence_words), len(filtered_words) - start_idx)):
            clean_sentence_word = re.sub(r'[^\w\s]', '', sentence_words[i])
            if not clean_sentence_word:  # Skip empty words
                continue

            if clean_words[start_idx + i] == clean_sentence_word:
                match_score += 1

        # Calculate match percentage
        match_percentage = match_score / len(sentence_words)

        if match_percentage > best_match_score:
            best_match_score = match_percentage
            best_match_start_idx = start_idx
            best_match_end_idx = start_idx + min(len(sentence_words), len(filtered_words) - start_idx) - 1

    # If match score is too low, fall back to segment timestamps
    if best_match_score < 0.5:  # Require at least 50% match
        return {"start": segment_start, "end": segment_end}

    # Get timestamps from the best match
    start_time = filtered_words[best_match_start_idx].get("start", segment_start)
    end_time = filtered_words[best_match_end_idx].get("end", segment_end)

    return {"start": start_time, "end": end_time}


def get_sentence_timestamps(sentence, segment, words_list):
    sentence = sentence.strip()
    if not sentence:
        return None

    segment_start = segment.get("start", 0)
    segment_end = segment.get("end", 0)

    # Find the exact sequence match
    timestamps = find_exact_sequence(sentence, words_list, segment_start, segment_end)

    # # Add a small buffer to avoid exact overlaps
    # buffer = 0.01  # 10ms buffer

    # Ensure timestamps are valid
    if timestamps["start"] >= timestamps["end"]:
        print(f"WARNING: Invalid timestamps, using segment with buffer")
        return {
            "start": segment_start,
            "end": segment_end
        }

    # Ensure minimum duration
    min_duration = 0.5  # Half second minimum
    if timestamps["end"] - timestamps["start"] < min_duration:
        timestamps["end"] = timestamps["start"] + min_duration

    # Ensure timestamps are within segment bounds
    timestamps["start"] = max(timestamps["start"], segment_start)
    timestamps["end"] = min(timestamps["end"], segment_end)

    return timestamps


def optimize_transcription_segments(transcription_file, output_file=None, min_segment_length=60):
    with open(transcription_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if output_file is None:
        base_name = os.path.splitext(transcription_file)[0]
        output_file = f"{base_name}_optimized.json"

    segments = data.get("segments", [])
    words_list = data.get("words", [])

    print(f"Optimizing {len(segments)} transcription segments...")

    # STEP 1: Split segments into sentences with proper timestamps
    raw_segments = []

    for segment in segments:
        if "merged" in segment and segment["merged"]:
            continue  # Skip merged segments

        segment_text = segment.get("text", "").strip()
        sentences = split_into_sentences(segment_text)

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

    # Prepare the result
    result = {
        "text": data.get("text", ""),
        "segments": new_segments
    }

    # Save the result
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Transcription optimization complete! {len(new_segments)} segments created.")
    print(f"Result saved to: {output_file}")

    return output_file
