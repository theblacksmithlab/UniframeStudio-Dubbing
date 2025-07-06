import os
import json
from utils.logger_config import setup_logger
from utils.logger_config import get_job_logger


logger = setup_logger(name=__name__, log_file="logs/app.log")


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


def correct_transcript_segments(input_file, output_file=None, job_id=None):
    if job_id:
        log = get_job_logger(logger, job_id)
    else:
        log = logger

    if output_file is None:
        base_dir = os.path.dirname(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(base_dir, f"{base_name}_corrected.json")

    with open(input_file, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    outro_gap_duration = transcript.get('outro_gap_duration')

    segments = transcript.get('segments', [])

    log.info(f"Loaded {len(segments)} transcription segments for processing.")

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
                # log.info(
                #     f"Merging segment {i} (ID: {corrected_segments[i]['id']}) with segment {next_idx} (ID: {corrected_segments[next_idx]['id']})"
                # )

                merge_result = merge_segments(corrected_segments, i, next_idx)

                if merge_result:
                    merge_count += 1

                if merge_count >= max_merges:
                    log.warning("WARNING! Merging limit reached, stopping the process.")
                    i += 1
            else:
                corrected_segments[i]["text"] = corrected_segments[i]["text"].strip() + "."
                log.info(f"Added a period to segment {i}")
                i += 1
        else:
            i += 1

    transcript['segments'] = corrected_segments

    full_text = ""
    for segment in corrected_segments:
        if not segment.get("merged", False):
            full_text += " " + segment["text"].strip()

    transcript['text'] = full_text.strip()

    if outro_gap_duration is not None:
        transcript['outro_gap_duration'] = outro_gap_duration

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    return output_file
