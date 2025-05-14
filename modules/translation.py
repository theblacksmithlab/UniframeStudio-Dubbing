import os
import json
import openai
from utils.ai_utils import load_system_role_for_timestamped_translation


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

    segments = transcript.get('segments', [])
    total_segments = len(segments)

    print(f"Starting translation of {total_segments} segments...")
    print(f"Target language: {target_language}")
    print(f"Using model: {model}")

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

        print(f"Translating segment {i + 1}/{total_segments}: {current_text[:50]}...")

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

            print(f"-> Translated: {translated_text[:50]}...")

        except Exception as e:
            print(f"Error translating segment {i}: {e}")
            error_message = f"[ERROR] Failed to translate: {str(e)[:100]}"
            segment["translated_text"] = error_message
            segment["initial_translation"] = error_message

    translated_full_text = " ".join([s.get("translated_text", "") for s in segments])
    transcript["translated_text"] = translated_full_text
    transcript["target_language"] = target_language

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    print(f"Translation complete! Result saved to: {output_file}")

    return output_file
