import json
import os
import re
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


def get_sequential_word_timestamps(sentences, segment, words_list):
    """
    Последовательно распределяет слова из words_list между предложениями
    """
    segment_start = segment.get("start", 0)
    segment_end = segment.get("end", 0)

    # Получаем слова, которые попадают в данный сегмент
    segment_words = []
    for word_info in words_list:
        word_start = word_info.get("start", 0)
        word_end = word_info.get("end", 0)

        # Слово попадает в сегмент, если хотя бы частично пересекается с ним
        if (word_start <= segment_end and word_end >= segment_start):
            segment_words.append(word_info)

    # Сортируем слова по времени начала
    segment_words.sort(key=lambda x: x.get("start", 0))

    if not segment_words:
        logger.warning(f"No words found for segment {segment_start:.2f}-{segment_end:.2f}")
        return []

    logger.info(f"Found {len(segment_words)} words for segment {segment_start:.2f}-{segment_end:.2f}")

    # Подсчитываем примерное количество слов и символов в каждом предложении
    sentence_word_counts = []
    sentence_char_counts = []

    for sentence in sentences:
        # Считаем слова в предложении (приблизительно)
        sentence_words_approx = len(re.findall(r'\b\w+\b', sentence))
        sentence_word_counts.append(sentence_words_approx)
        sentence_char_counts.append(len(sentence))

    total_words_estimated = sum(sentence_word_counts)
    total_chars = sum(sentence_char_counts)

    if total_words_estimated == 0:
        logger.warning("No words estimated in sentences")
        return []

    # Распределяем слова пропорционально
    sentence_timestamps = []
    word_index = 0

    for i, sentence in enumerate(sentences):
        if i == len(sentences) - 1:
            # Последнему предложению отдаем все оставшиеся слова
            sentence_word_count = len(segment_words) - word_index
        else:
            # Комбинированное пропорциональное распределение (символы + слова)
            char_proportion = sentence_char_counts[i] / total_chars
            word_proportion = sentence_word_counts[i] / total_words_estimated
            combined_proportion = (char_proportion + word_proportion) / 2
            sentence_word_count = max(1, int(len(segment_words) * combined_proportion))

        # Убеждаемся, что не выходим за границы
        sentence_word_count = min(sentence_word_count, len(segment_words) - word_index)

        if sentence_word_count <= 0:
            logger.warning(f"No words allocated for sentence: {sentence[:30]}...")
            # Fallback: равномерное распределение времени
            remaining_sentences = len(sentences) - i
            if remaining_sentences > 0:
                sentence_duration = (segment_end - (segment_words[word_index - 1].get("end",
                                                                                      segment_start) if word_index > 0 else segment_start)) / remaining_sentences
                start_time = segment_words[word_index - 1].get("end",
                                                               segment_start) if word_index > 0 else segment_start
                sentence_timestamps.append({
                    "start": start_time,
                    "end": start_time + sentence_duration,
                    "text": sentence
                })
            continue

        # Получаем временные метки для этого предложения
        sentence_words = segment_words[word_index:word_index + sentence_word_count]

        if sentence_words:
            start_time = sentence_words[0].get("start", segment_start)
            end_time = sentence_words[-1].get("end", segment_end)

            # Проверяем минимальную длительность
            min_duration = 0.5
            if end_time - start_time < min_duration:
                end_time = start_time + min_duration

            # Не выходим за границы сегмента
            start_time = max(start_time, segment_start)
            end_time = min(end_time, segment_end)

            sentence_timestamps.append({
                "start": start_time,
                "end": end_time,
                "text": sentence
            })

            logger.info(
                f"Sentence '{sentence[:30]}...' -> {start_time:.2f}-{end_time:.2f} ({sentence_word_count} words)")

        word_index += sentence_word_count

    return sentence_timestamps


def optimize_transcription_segments(transcription_file, output_file=None, min_segment_length=60):
    with open(transcription_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if output_file is None:
        base_name = os.path.splitext(transcription_file)[0]
        output_file = f"{base_name}_optimized.json"

    outro_gap_duration = data.get('outro_gap_duration')

    segments = data.get("segments", [])
    words_list = data.get("words", [])

    logger.info(f"Optimizing {len(segments)} transcription segments...")
    logger.info(f"Total words available: {len(words_list)}")

    # STEP 1: Split segments into sentences with proper timestamps
    raw_segments = []

    for segment in segments:
        if "merged" in segment and segment["merged"]:
            continue

        segment_text = segment.get("text", "").strip()
        sentences = split_into_sentences(segment_text)

        logger.info(f"Processing segment {segment.get('id', 'unknown')}: {len(sentences)} sentences")

        if len(sentences) <= 1:
            # Односложный сегмент, оставляем как есть
            raw_segments.append({
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": segment_text
            })
        else:
            # Многосложный сегмент, разбиваем на предложения
            sentence_timestamps = get_sequential_word_timestamps(sentences, segment, words_list)

            if not sentence_timestamps:
                logger.warning(f"Could not split segment {segment.get('id', 'unknown')}, keeping as is")
                raw_segments.append({
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment_text
                })
            else:
                for timestamp_info in sentence_timestamps:
                    raw_segments.append(timestamp_info)

    # Сортируем по времени начала
    raw_segments.sort(key=lambda x: x["start"])

    logger.info(f"After sentence splitting: {len(raw_segments)} segments")

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

    # Исправляем перекрытия более аккуратно
    for i in range(len(new_segments) - 1):
        current_seg = new_segments[i]
        next_seg = new_segments[i + 1]

        if current_seg["end"] > next_seg["start"]:
            # Находим середину перекрытия
            overlap_mid = (current_seg["end"] + next_seg["start"]) / 2

            # Устанавливаем небольшой зазор (50мс)
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

    logger.info(f"Transcription optimization complete! {len(new_segments)} segments created.")
    logger.info(f"Result saved to: {output_file}")

    return output_file