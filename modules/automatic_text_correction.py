import os
import json
import openai
from modules.tts_correction import regenerate_segment
from utils.ai_utils import load_system_role_for_text_correction


def correct_text_through_api(
        current_text,
        original_duration,
        tts_duration,
        mode,
        previous_text="",
        next_text="",
        openai_api_key=None
):

    if not openai_api_key:
        raise ValueError("OpenAI API key is required for text correction but not provided")

    system_role = load_system_role_for_text_correction(mode)

    context_parts = []
    if previous_text:
        context_parts.append(f"Previous segment text (for context): {previous_text}")
    if next_text:
        context_parts.append(f"Next segment text (for context): {next_text}")

    context_prompt = "\n".join(context_parts) + "\n" if context_parts else ""

    prompt = (f"{context_prompt}"
              f"Current segment text to correct: {current_text}\n"
              f"Original duration: {original_duration:.3f} seconds\n"
              f"TTS duration: {tts_duration:.3f} seconds\n")

    print(f"Prompt: {prompt}")

    try:
        client = openai.OpenAI(api_key=openai_api_key)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        corrected_text = response.choices[0].message.content.strip()
        return corrected_text

    except Exception as e:
        print(f"Error correcting text: {e}")
        return current_text


def correct_segment_durations(translation_file, job_id, max_attempts=5, threshold=0.2, voice="onyx", dealer="openai",
                              elevenlabs_api_key=None, openai_api_key=None):
    base_dir = f"jobs/{job_id}/output"
    segments_dir = os.path.join(base_dir, "audio_segments")

    for attempt in range(1, max_attempts + 1):
        print(f"\n=== Attempt {attempt}/{max_attempts} for correcting segment durations ===")

        with open(translation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])
        segments_needing_correction = []

        for segment in segments:
            if "tts_duration" not in segment or "original_duration" not in segment:
                continue

            original_duration = segment["original_duration"]
            tts_duration = segment["tts_duration"]

            diff_ratio = abs(tts_duration - original_duration) / original_duration

            if diff_ratio > threshold:
                segments_needing_correction.append({
                    "id": segment["id"],
                    "segment": segment,
                    "diff_ratio": diff_ratio,
                    "needs_reduction": tts_duration > original_duration
                })

        if not segments_needing_correction:
            print("All segments are within acceptable range. Correction complete!")
            return translation_file

        print(f"Found {len(segments_needing_correction)} segments that need correction")

        for segment_data in segments_needing_correction:
            segment = segment_data["segment"]
            segment_id = segment_data["id"]
            mode = "reduce" if segment_data["needs_reduction"] else "expand"

            print(f"\nProcessing segment {segment_id}...")
            print(f"Original duration: {segment['original_duration']:.3f}s")
            print(f"TTS duration: {segment['tts_duration']:.3f}s")
            print(f"Difference: {segment_data['diff_ratio'] * 100:.1f}%")
            print(f"Action needed: {mode}")

            segment_index = next((i for i, s in enumerate(segments) if s["id"] == segment_id), None)
            if segment_index is None:
                print(f"Warning: Could not find segment {segment_id} in segments list")
                continue

            previous_text = ""
            next_text = ""

            if segment_index > 0:
                previous_text = segments[segment_index - 1].get("translated_text", "").strip()
            if segment_index < len(segments) - 1:
                next_text = segments[segment_index + 1].get("translated_text", "").strip()

            corrected_text = correct_text_through_api(
                segment["translated_text"],
                segment["original_duration"],
                segment["tts_duration"],
                mode,
                previous_text,
                next_text,
                openai_api_key=openai_api_key
            )

            data["segments"][segment_index]["translated_text"] = corrected_text

            with open(translation_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            segment_audio_file = os.path.join(segments_dir, f"segment_{segment_id}.mp3")

            print(f"Regenerating audio for segment {segment_id}...")
            regenerate_segment(
                translation_file,
                job_id,
                segment_id,
                output_audio_file=segment_audio_file,
                voice=voice,
                dealer=dealer,
                elevenlabs_api_key=elevenlabs_api_key,
                openai_api_key=openai_api_key,
            )

            with open(translation_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            segments = data.get("segments", [])

    with open(translation_file, "r", encoding="utf-8") as f:
        final_data = json.load(f)

    segments = final_data.get("segments", [])
    remaining_issues = []

    for segment in segments:
        if "tts_duration" not in segment or "original_duration" not in segment:
            continue

        original_duration = segment["original_duration"]
        tts_duration = segment["tts_duration"]
        diff_ratio = abs(tts_duration - original_duration) / original_duration

        if diff_ratio > threshold:
            remaining_issues.append({
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "original_duration": original_duration,
                "tts_duration": tts_duration,
                "diff_percentage": diff_ratio * 100
            })

    if remaining_issues:
        print(
            f"\nWARNING: After {max_attempts} attempts, {len(remaining_issues)} segments still need manual correction:")
        for issue in remaining_issues:
            print(f"  - Segment {issue['id']}: {issue['start']:.2f}s - {issue['end']:.2f}s")
            print(f"    Original: {issue['original_duration']:.3f}s, TTS: {issue['tts_duration']:.3f}s")
            print(f"    Difference: {issue['diff_percentage']:.1f}%")

    return translation_file