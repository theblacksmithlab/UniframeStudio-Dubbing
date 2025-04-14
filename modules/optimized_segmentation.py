import json
import os
import re


def optimize_transcription_segments(transcription_file, output_file=None, min_segment_length=60):
    with open(transcription_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if output_file is None:
        base_name = os.path.splitext(transcription_file)[0]
        output_file = f"{base_name}_optimized.json"

    full_text = data["text"]
    original_segments = data.get("segments", [])
    words_with_times = data.get("words", [])

    original_segments.sort(key=lambda x: x["start"])

    segment_time_map = {}
    for segment in original_segments:
        segment_time_map[segment["text"]] = {
            "start": segment["start"],
            "end": segment["end"]
        }

    word_time_index = {}
    sorted_words = sorted(words_with_times, key=lambda x: x["start"])

    for word_info in sorted_words:
        word_text = word_info["word"].lower()
        if word_text not in word_time_index:
            word_time_index[word_text] = []
        word_time_index[word_text].append({
            "start": word_info["start"],
            "end": word_info["end"]
        })

    sentences = split_into_sentences(full_text)
    print(f"Total sentences found: {len(sentences)}")

    segment_sentence_map = {}
    for segment in original_segments:
        segment_sentences = split_into_sentences(segment["text"])
        for s in segment_sentences:
            if s not in segment_sentence_map:
                segment_sentence_map[s] = []
            segment_sentence_map[s].append({
                "start": segment["start"],
                "end": segment["end"],
                "original_segment": segment
            })

    sentence_times = []

    timeline = []
    for segment in original_segments:
        timeline.append({"time": segment["start"], "type": "start", "segment": segment})
        timeline.append({"time": segment["end"], "type": "end", "segment": segment})

    timeline.sort(key=lambda x: x["time"])

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        times = None

        if sentence in segment_time_map:
            times = segment_time_map[sentence]
        elif sentence in segment_sentence_map:
            segment_times = segment_sentence_map[sentence][0]
            times = {
                "start": segment_times["start"],
                "end": segment_times["end"]
            }
        else:
            times = find_sentence_times_by_words(sentence, word_time_index, sorted_words, timeline)

        if not times:
            times = find_segment_times_for_sentence(sentence, original_segments)

        if times:
            if times["start"] > times["end"]:
                print(f"Warning: Invalid timestamps for sentence: {sentence[:50]}...")
                print(f"  start: {times['start']}, end: {times['end']}")
                times["start"], times["end"] = min(times["start"], times["end"]), max(times["start"], times["end"])

            sentence_times.append({
                "text": sentence,
                "start": times["start"],
                "end": times["end"]
            })
        else:
            print(f"Warning: Could not find timestamps for sentence: {sentence[:50]}...")

    for i in range(len(sentence_times)):
        if sentence_times[i]["start"] > sentence_times[i]["end"]:
            print(f"Fixing inverted labels for: {sentence_times[i]['text'][:30]}...")
            sentence_times[i]["start"], sentence_times[i]["end"] = sentence_times[i]["end"], sentence_times[i]["start"]

    sentence_times.sort(key=lambda x: x["start"])

    for i in range(1, len(sentence_times)):
        if abs(sentence_times[i]["start"] - sentence_times[i - 1]["start"]) < 0.01:
            sentence_times[i]["start"] = sentence_times[i - 1]["end"] + 0.01

    new_segments = []
    current_text = ""
    current_start = None
    current_end = None

    for sent_info in sentence_times:
        sentence = sent_info["text"]

        if not current_text or len(current_text) >= min_segment_length:
            if current_text:
                new_segments.append({
                    "id": len(new_segments),
                    "start": current_start,
                    "end": current_end,
                    "text": current_text
                })

            current_text = sentence
            current_start = sent_info["start"]
            current_end = sent_info["end"]
        else:
            current_text += " " + sentence
            current_end = sent_info["end"]

    if current_text:
        new_segments.append({
            "id": len(new_segments),
            "start": current_start,
            "end": current_end,
            "text": current_text
        })

    new_segments.sort(key=lambda x: x["start"])

    for i, segment in enumerate(new_segments):
        segment["id"] = i

    for i in range(len(new_segments) - 1):
        if new_segments[i]["end"] > new_segments[i + 1]["start"]:
            midpoint = (new_segments[i]["end"] + new_segments[i + 1]["start"]) / 2
            new_segments[i]["end"] = midpoint
            new_segments[i + 1]["start"] = midpoint

    for i in range(len(new_segments)):
        if new_segments[i]["start"] > new_segments[i]["end"]:
            print(f"Fixing inverted labels for segment {i}")
            new_segments[i]["start"], new_segments[i]["end"] = new_segments[i]["end"], new_segments[i]["start"]

    result = {
        "text": data["text"],
        "segments": new_segments
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Optimized transcription is saved in file: {output_file}")
    print(f"Total original segments: {len(original_segments)}, New segments: {len(new_segments)}")

    check_segment_overlaps(output_file)
    return output_file


def find_sentence_times_by_words(sentence, word_time_index, words_with_times, timeline=None):
    import re

    sentence = sentence.strip().lower()
    words = re.findall(r'\b\w+\b', sentence)

    if not words:
        return None

    first_word = words[0]
    last_word = words[-1]

    first_word_times = None
    last_word_times = None

    if first_word in word_time_index:
        word_occurrences = sorted(word_time_index[first_word], key=lambda x: x["start"])
        for time_info in word_occurrences:
            context_before = get_word_context(time_info["start"], words_with_times, -5, 0)
            if is_sentence_beginning(context_before):
                first_word_times = time_info
                break

        if not first_word_times and word_occurrences:
            first_word_times = word_occurrences[0]

    if last_word in word_time_index:
        word_occurrences = sorted(word_time_index[last_word], key=lambda x: x["end"])
        for time_info in word_occurrences:
            context_after = get_word_context(time_info["end"], words_with_times, 0, 5)
            if is_sentence_ending(context_after):
                last_word_times = time_info
                break

        if not last_word_times and word_occurrences:
            last_word_times = word_occurrences[-1]

    if first_word_times and last_word_times:
        if first_word_times["start"] <= last_word_times["end"]:
            return {
                "start": first_word_times["start"],
                "end": last_word_times["end"]
            }
        else:
            print(f"Warning: Timestamps overlap for sentence: '{sentence[:30]}...'")
            print(f"  Start: {first_word_times['start']}, End: {last_word_times['end']}")
            return {
                "start": min(first_word_times["start"], last_word_times["end"]),
                "end": max(first_word_times["start"], last_word_times["end"])
            }

    if timeline:
        found_segments = find_segments_containing_words(words, word_time_index, timeline)
        if found_segments:
            return {
                "start": min(s["start"] for s in found_segments),
                "end": max(s["end"] for s in found_segments)
            }

    all_times = []
    for word in words:
        if word in word_time_index:
            for time_info in word_time_index[word]:
                all_times.append(time_info)

    if all_times:
        all_times.sort(key=lambda x: x["start"])
        return {
            "start": all_times[0]["start"],
            "end": all_times[-1]["end"]
        }

    return None


def find_segments_containing_words(words, word_time_index, timeline):
    relevant_times = []

    for word in words:
        if word in word_time_index:
            for time_info in word_time_index[word]:
                containing_segments = []
                current_segments = []

                for event in timeline:
                    if event["type"] == "start":
                        current_segments.append(event["segment"])
                    elif event["type"] == "end" and event["segment"] in current_segments:
                        current_segments.remove(event["segment"])

                    if time_info["start"] >= event["time"] and current_segments:
                        containing_segments = list(current_segments)

                if containing_segments:
                    for segment in containing_segments:
                        relevant_times.append({
                            "start": segment["start"],
                            "end": segment["end"]
                        })

    return relevant_times


def get_word_context(time_point, words_with_times, offset_before, offset_after):
    closest_index = None
    min_diff = float('inf')

    for i, word_info in enumerate(words_with_times):
        if abs(word_info["start"] - time_point) < min_diff:
            min_diff = abs(word_info["start"] - time_point)
            closest_index = i

    if closest_index is None:
        return []

    start_idx = max(0, closest_index + offset_before)
    end_idx = min(len(words_with_times) - 1, closest_index + offset_after)

    return words_with_times[start_idx:end_idx + 1]


def is_sentence_beginning(context_words):
    if not context_words:
        return True

    last_word = context_words[-1]["word"] if context_words else ""
    return last_word.endswith(('.', '!', '?')) or not context_words


def is_sentence_ending(context_words):
    if not context_words:
        return True

    first_word = context_words[0]["word"] if context_words else ""
    return first_word[0].isupper() if first_word else True


def split_into_sentences(text):
    text = re.sub(r'(\b\w\.\w\.)', lambda m: m.group(0).replace('.', '<DOT>'), text)

    sentences = []
    current = ""

    for char in text:
        current += char
        if char in ['.', '!', '?'] and current.strip():
            if not re.search(r'\d\.$', current.strip()):
                sentences.append(current.strip())
                current = ""

    if current.strip():
        sentences.append(current.strip())

    sentences = [s.replace('<DOT>', '.') for s in sentences]

    return sentences


def find_segment_times_for_sentence(sentence, segments):
    sentence = sentence.strip()

    for segment in segments:
        if segment["text"].strip() == sentence:
            return {
                "start": segment["start"],
                "end": segment["end"]
            }

    for segment in segments:
        if sentence in segment["text"]:
            return {
                "start": segment["start"],
                "end": segment["end"]
            }

    containing_segments = []
    for segment in segments:
        clean_segment = re.sub(r'[^\w\s]', ' ', segment["text"].lower())
        clean_sentence = re.sub(r'[^\w\s]', ' ', sentence.lower())

        segment_words = set(clean_segment.split())
        sentence_words = set(clean_sentence.split())

        intersection = segment_words.intersection(sentence_words)
        if len(intersection) > len(sentence_words) / 2:
            containing_segments.append(segment)

    if containing_segments:
        start_time = min(s["start"] for s in containing_segments)
        end_time = max(s["end"] for s in containing_segments)

        return {
            "start": start_time,
            "end": end_time
        }

    for segment in segments:
        sentence_words = set(re.sub(r'[^\w\s]', ' ', sentence.lower()).split())
        segment_words = set(re.sub(r'[^\w\s]', ' ', segment["text"].lower()).split())

        if sentence_words.intersection(segment_words):
            return {
                "start": segment["start"],
                "end": segment["end"]
            }

    return None


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