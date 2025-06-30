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


def extract_words_from_text(text):
    """Извлекает слова из текста, очищая от пунктуации"""
    words = re.findall(r'\b\w+\b', text.lower())
    return words


def normalize_word(word):
    """Нормализует слово для сравнения (убирает пунктуацию, приводит к нижнему регистру)"""
    return re.sub(r'[^\w]', '', word.lower().strip())


def get_sequence_length(sentence_words):
    """Определяет оптимальную длину последовательности для поиска"""
    total_words = len(sentence_words)
    if total_words <= 4:
        return total_words  # Используем все слова для коротких предложений
    elif total_words <= 8:
        return 3  # Для средних предложений
    else:
        return 5  # Для длинных предложений


def find_word_sequence_in_timeframe(target_sequence, words_array, segment_start, segment_end, search_from_start=True):
    # Фильтруем слова только в пределах сегмента
    segment_words = []
    for i, word_info in enumerate(words_array):
        word_start = word_info.get("start", 0)
        word_end = word_info.get("end", 0)

        # Слово попадает в сегмент, если хотя бы частично пересекается с ним
        if word_start <= segment_end and word_end >= segment_start:
            segment_words.append({
                "original_index": i,
                "word": word_info.get("word", ""),
                "start": word_start,
                "end": word_end
            })

    if not segment_words:
        logger.warning(f"No words found in timeframe {segment_start:.2f}-{segment_end:.2f}")
        return None

    logger.info(
        f"Searching for sequence {target_sequence} in {len(segment_words)} words (timeframe {segment_start:.2f}-{segment_end:.2f})")

    # Определяем направление поиска
    search_range = range(len(segment_words)) if search_from_start else range(len(segment_words) - 1, -1, -1)

    for i in search_range:
        # Проверяем, хватает ли слов для полной последовательности
        if search_from_start:
            if i + len(target_sequence) > len(segment_words):
                continue
            check_range = range(len(target_sequence))
        else:
            if i - len(target_sequence) + 1 < 0:
                continue
            check_range = range(len(target_sequence))

        # Проверяем совпадение последовательности
        match_found = True
        for j in check_range:
            if search_from_start:
                word_index = i + j
                target_word = target_sequence[j]
            else:
                word_index = i - len(target_sequence) + 1 + j
                target_word = target_sequence[j]

            current_word = normalize_word(segment_words[word_index]["word"])
            target_normalized = normalize_word(target_word)

            if current_word != target_normalized:
                match_found = False
                break

        if match_found:
            if search_from_start:
                found_index = segment_words[i]["original_index"]
            else:
                found_index = segment_words[i - len(target_sequence) + 1]["original_index"]

            logger.info(f"Found sequence {target_sequence} at word index {found_index}")
            return found_index

    logger.warning(f"Sequence {target_sequence} not found in timeframe")
    return None


def get_sentence_timestamps_precise(sentence, segment, words_list):
    """
    Находит точные временные метки для предложения, используя поиск в пределах сегмента
    """
    sentence = sentence.strip()
    if not sentence:
        return None

    segment_start = segment.get("start", 0)
    segment_end = segment.get("end", 0)

    # Извлекаем слова из предложения
    sentence_words = extract_words_from_text(sentence)

    if not sentence_words:
        logger.warning(f"No words extracted from sentence: {sentence[:30]}...")
        return {"start": segment_start, "end": segment_end}

    logger.info(f"Processing sentence: '{sentence[:50]}...' ({len(sentence_words)} words)")

    # Определяем длину последовательности для поиска
    seq_len = get_sequence_length(sentence_words)

    # Последовательности для поиска
    start_sequence = sentence_words[:seq_len]
    end_sequence = sentence_words[-seq_len:]

    logger.info(f"Searching for start sequence: {start_sequence}")
    logger.info(f"Searching for end sequence: {end_sequence}")

    # Ищем начало предложения
    start_word_index = find_word_sequence_in_timeframe(
        start_sequence, words_list, segment_start, segment_end, search_from_start=True
    )

    if start_word_index is None:
        logger.warning(f"Could not find start sequence for sentence: {sentence[:30]}...")
        return {"start": segment_start, "end": segment_end}

    # Ищем конец предложения (ищем после найденного начала)
    start_word_time = words_list[start_word_index].get("start", segment_start)

    end_word_index = find_word_sequence_in_timeframe(
        end_sequence, words_list, start_word_time, segment_end, search_from_start=False
    )

    if end_word_index is None:
        logger.warning(f"Could not find end sequence for sentence: {sentence[:30]}...")
        # Fallback: используем время начала + минимальную длительность
        return {"start": start_word_time, "end": min(start_word_time + 2.0, segment_end)}

    # Для конечной последовательности берем время конца последнего слова
    end_sequence_last_index = end_word_index + len(end_sequence) - 1
    end_time = words_list[end_sequence_last_index].get("end", segment_end)

    # Проверяем корректность временных меток
    if start_word_time >= end_time:
        logger.warning(f"Invalid timestamps detected, using fallback")
        return {"start": segment_start, "end": segment_end}

    # Убеждаемся в минимальной длительности
    min_duration = 0.5
    if end_time - start_word_time < min_duration:
        end_time = start_word_time + min_duration

    # Не выходим за границы сегмента
    final_start = max(start_word_time, segment_start)
    final_end = min(end_time, segment_end)

    logger.info(f"Found timestamps: {final_start:.2f} - {final_end:.2f}")

    return {"start": final_start, "end": final_end}


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

    # STEP 1: Split segments into sentences with precise timestamps
    raw_segments = []

    for segment in segments:
        if "merged" in segment and segment["merged"]:
            continue

        segment_text = segment.get("text", "").strip()
        sentences = split_into_sentences(segment_text)

        logger.info(
            f"Processing segment {segment.get('id', 'unknown')}: '{segment_text[:50]}...' -> {len(sentences)} sentences")

        if len(sentences) <= 1:
            # Односложный сегмент, оставляем как есть
            raw_segments.append({
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": segment_text
            })
        else:
            # Многосложный сегмент, разбиваем на предложения с точными временными метками
            for sentence in sentences:
                timestamps = get_sentence_timestamps_precise(sentence, segment, words_list)

                if timestamps:
                    raw_segments.append({
                        "start": timestamps["start"],
                        "end": timestamps["end"],
                        "text": sentence
                    })
                else:
                    logger.warning(f"Could not get timestamps for sentence: {sentence[:30]}...")

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

    # Исправляем перекрытия
    for i in range(len(new_segments) - 1):
        current_seg = new_segments[i]
        next_seg = new_segments[i + 1]

        if current_seg["end"] > next_seg["start"]:
            logger.error(f"UNEXPECTED OVERLAP detected between segments {i} and {i + 1}!")
            logger.error(f"Segment {i}: {current_seg['end']:.2f}, Segment {i + 1}: {next_seg['start']:.2f}")
            logger.error(f"This indicates a problem in timestamp detection algorithm!")

            # Находим середину перекрытия и создаем небольшой зазор
            overlap_mid = (current_seg["end"] + next_seg["start"]) / 2
            gap = 0.05  # 50мс зазор

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