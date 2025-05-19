import os
import json
import openai
from utils.ai_utils import load_system_role_for_timestamped_translation
from utils.logger_config import setup_logger


logger = setup_logger(name=__name__, log_file="logs/app.log")


def translate_transcribed_segments(input_file, output_file=None, target_language=None,
                                   model="gpt-4o", openai_api_key=None):
    if not openai_api_key:
        raise ValueError("OpenAI API key is required but not provided by user")

    system_role_template = load_system_role_for_timestamped_translation()

    system_role = system_role_template.format(target_language=target_language)

    if output_file is None:
        base_dir = os.path.dirname(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(base_dir, f"{base_name}_translated.json")

    with open(input_file, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    outro_gap_duration = transcript.get('outro_gap_duration')

    segments = transcript.get('segments', [])
    total_segments = len(segments)

    logger.info(f"Starting translation of {total_segments} segments...")
    logger.info(f"Target language: {target_language}")
    logger.info(f"Using model: {model}")

    client = openai.OpenAI(api_key=openai_api_key)

    for i, segment in enumerate(segments):
        current_text = segment.get("text", "").strip()

        prev_text = segments[i - 1].get("text", "").strip() if i > 0 else ""
        next_text = segments[i + 1].get("text", "").strip() if i < total_segments - 1 else ""

        prompt = (
            f"Previous segment text: {prev_text} (already translated, provided for context only)\n"
            f"Current segment to translate: {current_text}\n"
            f"Next segment text: {next_text} (provided for context only)"
        )

        logger.info(f"Translating segment {i + 1}/{total_segments}: {current_text[:50]}...")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            translated_text = response.choices[0].message.content.strip()

            segment["translated_text"] = translated_text
            segment["initial_translation"] = translated_text

            logger.info(f"-> Translated: {translated_text[:50]}...")

        except Exception as e:
            logger.error(f"Error translating segment {i}: {e}")
            raise ValueError(f"Failed to translate segment {i}: {str(e)}")

    translated_full_text = " ".join([s.get("translated_text", "") for s in segments])
    transcript["translated_text"] = translated_full_text
    transcript["target_language"] = target_language

    if outro_gap_duration is not None:
        transcript['outro_gap_duration'] = outro_gap_duration

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    logger.info(f"Translation complete! Result saved to: {output_file}")

    return output_file
