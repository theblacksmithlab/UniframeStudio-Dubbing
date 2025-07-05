import os
import json
import openai
from modules.tts_correction import regenerate_segment
from utils.ai_utils import load_system_role_for_text_correction
from utils.logger_config import setup_logger


logger = setup_logger(name=__name__, log_file="logs/app.log")


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

    logger.debug(f"Prompt: {prompt}")

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

        if not corrected_text:
            logger.warning("API returned empty text, using original")
            return current_text

        return corrected_text


    except Exception as e:
        logger.error(f"Error with OpenAI API: {e}")
        raise ValueError(f"OpenAI API error: {str(e)}")


def correct_segment_durations(translation_file, job_id, max_attempts=5, threshold=0.2, voice="onyx", dealer="openai",
                              elevenlabs_api_key=None, openai_api_key=None):
    if not openai_api_key:
        raise ValueError("OpenAI API key is required for text correction but not provided")

    if dealer.lower() == "elevenlabs" and not elevenlabs_api_key:
        raise ValueError("ElevenLabs API key is required for ElevenLabs TTS but not provided")

    base_dir = f"jobs/{job_id}/output"
    segments_dir = os.path.join(base_dir, "audio_segments")

    for attempt in range(1, max_attempts + 1):
        logger.info(f"\n=== Attempt {attempt}/{max_attempts} for correcting segment durations ===")

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
            logger.info("All segments are within acceptable range. Correction complete!")
            return translation_file

        logger.info(f"Found {len(segments_needing_correction)} segments that need correction")

        for segment_data in segments_needing_correction:
            segment = segment_data["segment"]
            segment_id = segment_data["id"]
            mode = "reduce" if segment_data["needs_reduction"] else "expand"

            logger.info(f"\nProcessing segment {segment_id}...")
            logger.info(f"Original duration: {segment['original_duration']:.3f}s")
            logger.info(f"TTS duration: {segment['tts_duration']:.3f}s")
            logger.info(f"Difference: {segment_data['diff_ratio'] * 100:.1f}%")
            logger.info(f"Action needed: {mode}")

            segment_index = next((i for i, s in enumerate(segments) if s["id"] == segment_id), None)
            if segment_index is None:
                logger.warning(f"Warning: Could not find segment {segment_id} in segments list")
                continue

            # test
            previous_texts = []
            next_texts = []

            for i in range(max(0, segment_index - 2), segment_index):
                if i >= 0:
                    prev_text = segments[i].get("translated_text", "").strip()
                    if prev_text:
                        previous_texts.append(prev_text)

            for i in range(segment_index + 1, min(len(segments), segment_index + 3)):
                if i < len(segments):
                    next_text = segments[i].get("translated_text", "").strip()
                    if next_text:
                        next_texts.append(next_text)

            previous_text = " ".join(previous_texts)
            next_text = " ".join(next_texts)

            try:
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
                # end of the test

            # previous_text = ""
            # next_text = ""

            # if segment_index > 0:
            #     previous_text = segments[segment_index - 1].get("translated_text", "").strip()
            # if segment_index < len(segments) - 1:
            #     next_text = segments[segment_index + 1].get("translated_text", "").strip()

            # try:
            #     corrected_text = correct_text_through_api(
            #         segment["translated_text"],
            #         segment["original_duration"],
            #         segment["tts_duration"],
            #         mode,
            #         previous_text,
            #         next_text,
            #         openai_api_key=openai_api_key
            #     )
            #
            #     data["segments"][segment_index]["translated_text"] = corrected_text

                try:
                    with open(translation_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except IOError as e:
                    logger.warning(f"Error saving corrected text: {e}")
                    continue

                segment_audio_file = os.path.join(segments_dir, f"segment_{segment_id}.mp3")

                if dealer.lower() == "openai":
                    suggested_voice = segment.get("suggested_voice")
                    confidence = segment.get("gender_confidence", 0)

                    if suggested_voice:
                        segment_voice = suggested_voice
                        logger.info(
                            f"Using analyzed voice '{segment_voice}' for segment {segment_id} correction (gender: {segment.get('predicted_gender', 'unknown')}, confidence: {confidence:.2f})")
                    else:
                        segment_voice = voice
                        logger.info(
                            f"Using fallback voice '{voice}' for segment {segment_id} correction (no voice analysis)")
                else:
                    segment_voice = voice
                    logger.info(f"Using ElevenLabs voice '{voice}' for segment {segment_id} correction")

                logger.info(f"Regenerating audio for segment {segment_id} with voice: {segment_voice}")
                result = regenerate_segment(
                    translation_file,
                    job_id,
                    segment_id,
                    output_audio_file=segment_audio_file,
                    voice=segment_voice,
                    dealer=dealer,
                    elevenlabs_api_key=elevenlabs_api_key,
                    openai_api_key=openai_api_key,
                )

                if not result:
                    logger.warning(f"Failed to regenerate segment {segment_id}")
                    continue

                try:
                    with open(translation_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
                    logger.warning(f"Error reading updated translation file: {e}")
                    continue

                segments = data.get("segments", [])

            except Exception as e:
                logger.warning(f"Error processing segment {segment_id}: {e}")
                continue

    try:
        with open(translation_file, "r", encoding="utf-8") as f:
            final_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
        raise ValueError(f"Error reading final translation file: {str(e)}")

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
        logger.info(
            f"\nWARNING: After {max_attempts} attempts, {len(remaining_issues)} segments still need manual correction:")
        for issue in remaining_issues:
            logger.info(f"  - Segment {issue['id']}: {issue['start']:.2f}s - {issue['end']:.2f}s")
            logger.info(f"    Original: {issue['original_duration']:.3f}s, TTS: {issue['tts_duration']:.3f}s")
            logger.info(f"    Difference: {issue['diff_percentage']:.1f}%")

    if dealer.lower() == "openai":
        correction_voice_stats = {}
        total_corrections = len(segments_needing_correction) if 'segments_needing_correction' in locals() else 0

        if total_corrections > 0:
            logger.info("=" * 50)
            logger.info("SEGMENT CORRECTION VOICE STATISTICS")
            logger.info("=" * 50)

            for segment_data in segments_needing_correction:
                segment = segment_data["segment"]
                used_voice = segment.get("suggested_voice", voice)
                correction_voice_stats[used_voice] = correction_voice_stats.get(used_voice, 0) + 1

            for voice_name, count in correction_voice_stats.items():
                percentage = (count / total_corrections) * 100
                logger.info(f"Corrections with voice '{voice_name}': {count}/{total_corrections} ({percentage:.1f}%)")

    return translation_file
