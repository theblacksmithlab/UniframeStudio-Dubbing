import json
import os
import re
import openai
from utils.ai_utils import load_system_role_for_sentence_boundaries
from utils.logger_config import setup_logger
from utils.logger_config import get_job_logger

logger = setup_logger(name=__name__, log_file="logs/app.log")


def split_into_sentences(text):
    text = re.sub(r'(\b\w\.\w\.)', lambda m: m.group(0).replace('.', '<DOT>'), text)
    text = re.sub(r'(\b[\w-]+\.[a-zA-Z]{2,})', lambda m: m.group(0).replace('.', '<DOT>'), text)

    sentences = []
    current = ""
    i = 0

    while i < len(text):
        current += text[i]

        if text[i] in ['.', '!', '?'] and current.strip():
            if not re.search(r'\d\.$', current.strip()):
                if i + 2 < len(text) and text[i + 1].isspace() and text[i + 2].isupper():
                    sentences.append(current.strip())
                    current = ""
                elif i + 1 >= len(text):
                    sentences.append(current.strip())
                    current = ""

        i += 1

    if current.strip():
        sentences.append(current.strip())

    sentences = [s.replace('<DOT>', '.') for s in sentences]

    return sentences


def filter_words_by_timeframe(words_array, segment_start, segment_end):
    filtered_words = []
    for word_info in words_array:
        word_start = word_info.get("start", 0)
        word_end = word_info.get("end", 0)

        if word_start <= segment_end and word_end >= segment_start:
            filtered_words.append(word_info)

    return filtered_words


def call_llm_for_sentence_boundaries(job_id, sentence, words_array, openai_api_key, model="gpt-4.1-2025-04-14"):
    log = get_job_logger(logger, job_id)

    try:
        words_for_llm = []
        for i, word in enumerate(words_array):
            words_for_llm.append({
                "index": i,
                "start": word.get("start"),
                "end": word.get("end"),
                "word": word.get("word", "").strip()
            })

        system_prompt = load_system_role_for_sentence_boundaries()

        user_prompt = f"""ПРЕДЛОЖЕНИЕ: "{sentence}"
        СЛОВА СЕГМЕНТА ({len(words_for_llm)} слов):
        {json.dumps(words_for_llm, ensure_ascii=False, indent=1)}"""

        log.info(f"Calling LLM for sentence: '{sentence[:50]}...'")

        client = openai.OpenAI(api_key=openai_api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=100,
            response_format={"type": "json_object"}
        )

        llm_response = response.choices[0].message.content.strip()

        result = json.loads(llm_response)

        if "start_time" not in result or "end_time" not in result:
            raise KeyError("Missing start_time or end_time in LLM response")

        log.info(f"LLM found boundaries: {result['start_time']:.2f} - {result['end_time']:.2f}")
        return result

    except json.JSONDecodeError as e:
        log.error(f"CRITICAL: Failed to parse guaranteed JSON response: {e}")
        log.error(f"Raw LLM response: {llm_response if 'llm_response' in locals() else 'N/A'}")
        return None

    except KeyError as e:
        log.error(f"LLM returned valid JSON but missing required fields: {e}")
        log.error(f"LLM response: {llm_response if 'llm_response' in locals() else 'N/A'}")
        return None

    except FileNotFoundError as e:
        log.error(f"System role file not found: {e}")
        return None

    except Exception as e:
        log.error(f"LLM call failed: {e}")
        return None


def get_sentence_timestamps_with_llm(job_id, sentence, segment, words_list, openai_api_key, model="gpt-4.1-2025-04-14"):
    log = get_job_logger(logger, job_id)

    sentence = sentence.strip()
    if not sentence:
        return None

    segment_start = segment.get("start", 0)
    segment_end = segment.get("end", 0)

    log.info(f"Processing sentence with LLM: '{sentence[:50]}...'")

    segment_words = filter_words_by_timeframe(words_list, segment_start, segment_end)

    if not segment_words:
        log.warning(f"No words found in timeframe {segment_start:.2f}-{segment_end:.2f}")
        return None

    log.info(f"Found {len(segment_words)} words in segment timeframe")

    llm_result = call_llm_for_sentence_boundaries(job_id, sentence, segment_words, openai_api_key, model)

    if not llm_result:
        log.warning(f"LLM failed to find boundaries for sentence: {sentence[:30]}...")
        return None

    start_time = llm_result.get("start_time")
    end_time = llm_result.get("end_time")

    if start_time is None or end_time is None:
        log.warning(f"LLM returned null timestamps for sentence: {sentence[:30]}...")
        return None

    if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
        log.warning(f"LLM returned non-numeric timestamps for sentence: {sentence[:30]}...")
        return None

    if start_time >= end_time:
        log.warning(f"LLM returned invalid time order for sentence: {sentence[:30]}...")
        return None

    buffer = 5.0
    if start_time < segment_start - buffer or end_time > segment_end + buffer:
        log.warning(f"LLM returned timestamps outside reasonable range for sentence: {sentence[:30]}...")
        return None

    min_duration = 0.5
    if end_time - start_time < min_duration:
        end_time = start_time + min_duration

    final_start = max(start_time, segment_start)
    final_end = min(end_time, segment_end)

    log.info(f"LLM successfully found timestamps: {final_start:.2f} - {final_end:.2f}")

    return {"start": final_start, "end": final_end}


def validate_sentence_segments(sentence_segments, original_segment, prev_segment=None, next_segment=None):
    if not sentence_segments:
        return False

    for i in range(len(sentence_segments) - 1):
        current_end = sentence_segments[i]["end"]
        next_start = sentence_segments[i + 1]["start"]

        if current_end > next_start:
            return False

    first_start = sentence_segments[0]["start"]
    last_end = sentence_segments[-1]["end"]

    if first_start < original_segment["start"] or last_end > original_segment["end"]:
        return False

    if prev_segment and first_start < prev_segment["end"]:
        return False

    if next_segment and last_end > next_segment["start"]:
        return False

    return True


def process_segment_with_retry(job_id, segment, words_list, openai_api_key, model,
                               prev_segment=None, next_segment=None, max_retries=3):
    log = get_job_logger(logger, job_id)

    segment_text = segment.get("text", "").strip()
    sentences = split_into_sentences(segment_text)


    if len(sentences) <= 1:
        return [segment]

    log.info(f"Attempting to split segment {segment.get('id', 'unknown')} into {len(sentences)} sentences")

    for attempt in range(max_retries):
        log.info(f"Attempt {attempt + 1}/{max_retries} for segment {segment.get('id', 'unknown')}")

        sentence_segments = []
        all_sentences_processed = True

        for sentence in sentences:
            timestamps = get_sentence_timestamps_with_llm(job_id, sentence, segment, words_list,
                                                          openai_api_key, model)

            if timestamps:
                sentence_segments.append({
                    "start": timestamps["start"],
                    "end": timestamps["end"],
                    "text": sentence
                })
            else:
                all_sentences_processed = False
                break

        if not all_sentences_processed:
            log.warning(f"LLM failed to process some sentences in attempt {attempt + 1}")
            continue

        if validate_sentence_segments(sentence_segments, segment, prev_segment, next_segment):
            log.info(
                f"Successfully split segment {segment.get('id', 'unknown')} into {len(sentence_segments)} sentences on attempt {attempt + 1}")
            return sentence_segments
        else:
            log.warning(f"Validation failed for segment {segment.get('id', 'unknown')} on attempt {attempt + 1}")

    log.warning(f"All {max_retries} attempts failed for segment {segment.get('id', 'unknown')}, keeping original")
    return [segment]


def optimize_transcription_segments(transcription_file, job_id, output_file=None,
                                    openai_api_key=None, model="gpt-4.1-2025-04-14"):
    log = get_job_logger(logger, job_id)

    if not openai_api_key:
        raise ValueError("OpenAI API key is required for LLM-powered segmentation but not provided")

    with open(transcription_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if output_file is None:
        base_name = os.path.splitext(transcription_file)[0]
        output_file = f"{base_name}_optimized.json"

    outro_gap_duration = data.get('outro_gap_duration')

    segments = data.get("segments", [])
    words_list = data.get("words", [])

    log.info(f"Optimizing {len(segments)} transcription segments with LLM...")
    log.info(f"Total words available: {len(words_list)}")
    log.info(f"Using model: {model}")

    # STEP 1: Split segments into sentences with LLM-powered precise timestamps + validation
    raw_segments = []

    segments_to_process = [seg for seg in segments if not seg.get("merged", False)]

    log.info(f"Processing {len(segments_to_process)} segments (skipping merged ones)")

    for i, segment in enumerate(segments_to_process):
        prev_segment = raw_segments[-1] if raw_segments else None
        next_segment = segments_to_process[i + 1] if i + 1 < len(segments_to_process) else None

        log.info(f"Processing segment {segment.get('id', 'unknown')}: '{segment.get('text', '')[:50]}...'")

        processed_segments = process_segment_with_retry(
            job_id, segment, words_list, openai_api_key, model,
            prev_segment, next_segment
        )

        raw_segments.extend(processed_segments)

    raw_segments.sort(key=lambda x: x["start"])

    log.info(f"After LLM sentence splitting with validation: {len(raw_segments)} segments")

    # STEP 2: Merge short segments based on word count OR duration and time gaps
    merged_segments = []
    current_text = ""
    current_start = None
    current_end = None

    for segment in raw_segments:
        text = segment["text"]

        if not current_text:
            current_text = text
            current_start = segment["start"]
            current_end = segment["end"]
        else:
            current_word_count = len(current_text.split())
            current_duration = current_end - current_start

            if current_word_count >= 2 and current_duration >= 2.0:
                merged_segments.append({
                    "start": current_start,
                    "end": current_end,
                    "text": current_text
                })

                current_text = text
                current_start = segment["start"]
                current_end = segment["end"]
            else:
                time_gap = segment["start"] - current_end
                max_allowed_gap = 1

                if time_gap <= max_allowed_gap:
                    current_text += " " + text
                    current_end = segment["end"]
                else:
                    merged_segments.append({
                        "start": current_start,
                        "end": current_end,
                        "text": current_text
                    })

                    current_text = text
                    current_start = segment["start"]
                    current_end = segment["end"]

    if current_text:
        merged_segments.append({
            "start": current_start,
            "end": current_end,
            "text": current_text
        })
    log.info(f"After merging short segments: {len(merged_segments)} segments")

    # STEP 3: Assign IDs
    new_segments = []
    for i, segment in enumerate(merged_segments):
        new_segments.append({
            "id": i,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    overlap_count = 0
    for i in range(len(new_segments) - 1):
        current_seg = new_segments[i]
        next_seg = new_segments[i + 1]

        if current_seg["end"] > next_seg["start"]:
            overlap_count += 1
            log.error(f"UNEXPECTED OVERLAP detected between segments {i} and {i + 1}!")
            log.error(f"Segment {i}: {current_seg['end']:.2f}, Segment {i + 1}: {next_seg['start']:.2f}")

    if overlap_count == 0:
        log.info("No overlaps detected - validation system working correctly!")
    else:
        log.error(f"Found {overlap_count} overlaps - validation system needs debugging!")

    result = {
        "text": data.get("text", ""),
        "segments": new_segments
    }

    if outro_gap_duration is not None:
        result['outro_gap_duration'] = outro_gap_duration

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log.info(f"LLM-powered transcription optimization complete! {len(new_segments)} segments created.")

    return output_file
