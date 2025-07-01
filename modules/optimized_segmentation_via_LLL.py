import json
import os
import re
import openai

from utils.ai_utils import load_system_role_for_sentence_boundaries
from utils.logger_config import setup_logger

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


def call_llm_for_sentence_boundaries(sentence, words_array, openai_api_key, model="gpt-4o"):
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

        logger.info(f"Calling LLM for sentence: '{sentence[:50]}...'")
        logger.debug(f"DEBUG request for LLM:\n{user_prompt}")

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
        logger.debug(f"LLM response: {llm_response}")

        result = json.loads(llm_response)

        if "start_time" not in result or "end_time" not in result:
            raise KeyError("Missing start_time or end_time in LLM response")

        logger.info(f"LLM found boundaries: {result['start_time']:.2f} - {result['end_time']:.2f}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"CRITICAL: Failed to parse guaranteed JSON response: {e}")
        logger.error(f"Raw LLM response: {llm_response if 'llm_response' in locals() else 'N/A'}")
        return None

    except KeyError as e:
        logger.error(f"LLM returned valid JSON but missing required fields: {e}")
        logger.error(f"LLM response: {llm_response if 'llm_response' in locals() else 'N/A'}")
        return None

    except FileNotFoundError as e:
        logger.error(f"System role file not found: {e}")
        return None

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


def get_sentence_timestamps_with_llm(sentence, segment, words_list, openai_api_key, model="gpt-4o"):
    sentence = sentence.strip()
    if not sentence:
        return None

    segment_start = segment.get("start", 0)
    segment_end = segment.get("end", 0)

    logger.info(f"Processing sentence with LLM: '{sentence[:50]}...'")

    segment_words = filter_words_by_timeframe(words_list, segment_start, segment_end)

    if not segment_words:
        logger.warning(f"No words found in timeframe {segment_start:.2f}-{segment_end:.2f}")
        return None

    logger.info(f"Found {len(segment_words)} words in segment timeframe")

    llm_result = call_llm_for_sentence_boundaries(sentence, segment_words, openai_api_key, model)

    if not llm_result:
        logger.warning(f"LLM failed to find boundaries for sentence: {sentence[:30]}...")
        return None

    start_time = llm_result.get("start_time")
    end_time = llm_result.get("end_time")

    if start_time is None or end_time is None:
        logger.warning(f"LLM returned null timestamps for sentence: {sentence[:30]}...")
        return None

    if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
        logger.warning(f"LLM returned non-numeric timestamps for sentence: {sentence[:30]}...")
        return None

    if start_time >= end_time:
        logger.warning(f"LLM returned invalid time order for sentence: {sentence[:30]}...")
        return None

    buffer = 5.0
    if start_time < segment_start - buffer or end_time > segment_end + buffer:
        logger.warning(f"LLM returned timestamps outside reasonable range for sentence: {sentence[:30]}...")
        return None

    min_duration = 0.5
    if end_time - start_time < min_duration:
        end_time = start_time + min_duration

    final_start = max(start_time, segment_start)
    final_end = min(end_time, segment_end)

    logger.info(f"LLM successfully found timestamps: {final_start:.2f} - {final_end:.2f}")

    return {"start": final_start, "end": final_end}


def optimize_transcription_segments(transcription_file, output_file=None, min_segment_length=60,
                                    openai_api_key=None, model="gpt-4o-mini"):
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

    logger.info(f"Optimizing {len(segments)} transcription segments with LLM...")
    logger.info(f"Total words available: {len(words_list)}")
    logger.info(f"Using model: {model}")

    # STEP 1: Split segments into sentences with LLM-powered precise timestamps
    raw_segments = []

    for segment in segments:
        if "merged" in segment and segment["merged"]:
            continue

        segment_text = segment.get("text", "").strip()
        sentences = split_into_sentences(segment_text)

        logger.info(
            f"Processing segment {segment.get('id', 'unknown')}: '{segment_text[:50]}...' -> {len(sentences)} sentences")

        if len(sentences) <= 1:
            raw_segments.append({
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": segment_text
            })
        else:
            all_sentences_processed = True
            sentence_segments = []

            for sentence in sentences:
                timestamps = get_sentence_timestamps_with_llm(sentence, segment, words_list,
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

            if all_sentences_processed:
                raw_segments.extend(sentence_segments)
                logger.info(
                    f"Successfully split segment {segment.get('id', 'unknown')} into {len(sentence_segments)} sentences")
            else:
                raw_segments.append({
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment_text
                })
                logger.warning(
                    f"LLM failed to process segment {segment.get('id', 'unknown')}, keeping it whole: {segment_text[:50]}...")

    raw_segments.sort(key=lambda x: x["start"])

    logger.info(f"After LLM sentence splitting: {len(raw_segments)} segments")

    # STEP 2: Merge short segments to meet minimum length requirement
    merged_segments = []
    current_text = ""
    current_start = None
    current_end = None

    for segment in raw_segments:
        text = segment["text"]

        if not current_text or len(current_text) >= min_segment_length:
            if current_text:
                merged_segments.append({
                    "start": current_start,
                    "end": current_end,
                    "text": current_text
                })

            current_text = text
            current_start = segment["start"]
            current_end = segment["end"]
        else:
            current_text += " " + text
            current_end = segment["end"]

    if current_text:
        merged_segments.append({
            "start": current_start,
            "end": current_end,
            "text": current_text
        })

    logger.info(f"After merging short segments: {len(merged_segments)} segments")

    # STEP 3: Assign IDs and fix any overlapping segments
    new_segments = []
    for i, segment in enumerate(merged_segments):
        new_segments.append({
            "id": i,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    for i in range(len(new_segments) - 1):
        current_seg = new_segments[i]
        next_seg = new_segments[i + 1]

        if current_seg["end"] > next_seg["start"]:
            logger.error(f"UNEXPECTED OVERLAP detected between segments {i} and {i + 1}!")
            logger.error(f"Segment {i}: {current_seg['end']:.2f}, Segment {i + 1}: {next_seg['start']:.2f}")
            logger.error(f"This indicates a problem in LLM timestamp detection!")

            overlap_mid = (current_seg["end"] + next_seg["start"]) / 2
            gap = 0.05

            current_seg["end"] = overlap_mid - gap / 2
            next_seg["start"] = overlap_mid + gap / 2

            logger.warning(f"Fixed overlap between segments {i} and {i + 1}: "
                           f"set boundary at {overlap_mid:.2f}s with {gap * 1000:.0f}ms gap")

    result = {
        "text": data.get("text", ""),
        "segments": new_segments
    }

    if outro_gap_duration is not None:
        result['outro_gap_duration'] = outro_gap_duration

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"LLM-powered transcription optimization complete! {len(new_segments)} segments created.")
    logger.info(f"Result saved to: {output_file}")

    return output_file