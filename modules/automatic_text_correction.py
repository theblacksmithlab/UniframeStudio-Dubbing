"""
Automatic Text Correction Module
This module adjusts text length to match original audio duration
"""

import os
import json
import openai
from dotenv import load_dotenv
from modules.tts_correction import regenerate_segment
from utils.ai_utils import load_system_role_for_text_correction


def correct_text_through_api(current_text, original_duration, tts_duration, mode, previous_text="", next_text=""):
    """
    Send text to OpenAI API for correction (expansion or reduction)

    Args:
        current_text (str): Text to correct
        original_duration (float): Original duration in seconds
        tts_duration (float): TTS duration in seconds
        mode (str): 'reduce' or 'expand'
        previous_text (str): Text of the previous segment for context
        next_text (str): Text of the next segment for context

    Returns:
        str: Corrected text
    """
    system_role = load_system_role_for_text_correction(mode)

    # Calculate the percentage difference
    percentage_diff = ((tts_duration - original_duration) / original_duration) * 100

    # Construct prompt with details about the duration mismatch and context
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
        response = openai.chat.completions.create(
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
        return current_text  # Return original text if correction fails


def correct_segment_durations(translation_file, max_attempts=5, threshold=0.2, voice="onyx", dealer="openai"):
    """
    Main function to correct segment durations by adjusting text length

    Args:
        translation_file (str): Path to JSON file with segments
        max_attempts (int): Maximum number of correction attempts
        threshold (float): Threshold for duration difference (default 20%)
        voice (str): Voice to use for TTS
        dealer (str): TTS provider ('openai' or 'elevenlabs')

    Returns:
        str: Path to updated JSON file
    """
    load_dotenv()

    base_dir = os.path.dirname(translation_file)
    segments_dir = os.path.join(base_dir, "audio_segments")

    for attempt in range(1, max_attempts + 1):
        print(f"\n=== Attempt {attempt}/{max_attempts} for correcting segment durations ===")

        # Load the current state of the transcription file
        with open(translation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])
        segments_needing_correction = []

        # Identify segments that need correction
        for segment in segments:
            if "tts_duration" not in segment or "original_duration" not in segment:
                continue

            original_duration = segment["original_duration"]
            tts_duration = segment["tts_duration"]

            # Calculate the difference ratio
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

        # Process each segment that needs correction
        for segment_data in segments_needing_correction:
            segment = segment_data["segment"]
            segment_id = segment_data["id"]
            mode = "reduce" if segment_data["needs_reduction"] else "expand"

            print(f"\nProcessing segment {segment_id}...")
            print(f"Original duration: {segment['original_duration']:.3f}s")
            print(f"TTS duration: {segment['tts_duration']:.3f}s")
            print(f"Difference: {segment_data['diff_ratio'] * 100:.1f}%")
            print(f"Action needed: {mode}")

            # Get segment index for context
            segment_index = next((i for i, s in enumerate(segments) if s["id"] == segment_id), None)
            if segment_index is None:
                print(f"Warning: Could not find segment {segment_id} in segments list")
                continue

            # Get context from surrounding segments
            previous_text = ""
            next_text = ""

            if segment_index > 0:
                previous_text = segments[segment_index - 1].get("translated_text", "").strip()
            if segment_index < len(segments) - 1:
                next_text = segments[segment_index + 1].get("translated_text", "").strip()

            # Get corrected text from OpenAI
            corrected_text = correct_text_through_api(
                segment["translated_text"],
                segment["original_duration"],
                segment["tts_duration"],
                mode,
                previous_text,
                next_text
            )

            # Update the segment with corrected text
            data["segments"][segment_index]["translated_text"] = corrected_text

            # Save the updated data to file before regenerating
            with open(translation_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Define output path for the segment audio
            segment_audio_file = os.path.join(segments_dir, f"segment_{segment_id}.mp3")

            # Regenerate the audio segment
            print(f"Regenerating audio for segment {segment_id}...")
            regenerate_segment(
                translation_file,
                segment_id,
                output_audio_file=segment_audio_file,
                voice=voice,
                dealer=dealer
            )

            # Reload the data after regeneration to get updated metrics
            with open(translation_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Update the segments list for the next iteration
            segments = data.get("segments", [])

    # After all attempts, check if any segments still need correction
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