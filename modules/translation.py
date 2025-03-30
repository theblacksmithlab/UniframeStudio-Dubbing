import os
import json
import openai
from utils.ai_utils import load_system_role_for_timestamped_translation


def translate_transcript_segments(input_file, output_file=None):
    system_role = load_system_role_for_timestamped_translation()

    if output_file is None:
        base_dir = os.path.dirname(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(base_dir, f"{base_name}_translated.json")

    with open(input_file, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    segments = transcript.get('segments', [])
    total_segments = len(segments)

    print(f"Starting translation of {total_segments} segments...")

    for i, segment in enumerate(segments):
        current_text = segment.get("text", "").strip()

        prev_text = segments[i - 1].get("text", "").strip() if i > 0 else ""
        next_text = segments[i + 1].get("text", "").strip() if i < total_segments - 1 else ""

        duration = segment.get("end", 0) - segment.get("start", 0)

        prompt = (f"Текст предыдущего сегмента:"
                  f"{prev_text} (уже переведено, нужен только для понимания контекста)\n"
                  f"Текст текущего сегмента для перевода: {current_text}\n"
                  f"Длительность сегмента: {duration:.2f} секунд. Переведите так, чтобы английская версия могла быть естественно произнесена за это время или немного дольше (в пределах 5-10%).\n"
                  f"Текст следующего сегмента:"
                  f"{next_text} (нужен только для понимания контекста)")

        print(f"Translating segment {i + 1}/{total_segments}: {current_text[:50]}...")

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            translated_text = response.choices[0].message.content.strip()

            # translated_text = translated_text + " <end of the sentence />"

            segment["translated_text"] = translated_text

            print(f"  -> Translated: {translated_text[:50]}...")

        except Exception as e:
            print(f"Error translating segment {i}: {e}")
            segment["translated_text"] = f"[ERROR] Failed to translate: {str(e)[:100]}"

    translated_full_text = " ".join([s.get("translated_text", "") for s in segments])
    transcript["translated_text"] = translated_full_text

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    print(f"Translation complete! Result saved to: {output_file}")

    return output_file
