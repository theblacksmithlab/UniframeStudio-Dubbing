import os
import json
import shutil
import openai
from elevenlabs import ElevenLabs
from pydub import AudioSegment
from dotenv import load_dotenv

from modules.tts import speedup_audio_to_target_duration, stretch_audio_to_target_duration


def regenerate_segment(translation_file, segment_id, output_audio_file=None, voice="onyx", dealer="openai"):
    load_dotenv()
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

    # Load translation file
    with open(translation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("Error: There are no segments in the transcription file.")
        return None

    # Find the segment with the specified ID
    target_segment = None
    for segment in segments:
        if segment.get("id") == int(segment_id):
            target_segment = segment
            break

    if not target_segment:
        print(f"Error: Segment with ID {segment_id} not found in the translation file.")
        return None

    text = target_segment.get("translated_text", "").strip()
    if not text:
        print(f"Error: Segment {segment_id} has no translated text.")
        return None

    start_time_ms = int(target_segment["start"] * 1000)
    end_time_ms = int(target_segment["end"] * 1000)
    target_duration_ms = end_time_ms - start_time_ms

    # Define segment neighbors for context
    segment_index = segments.index(target_segment)
    previous_text = ""
    next_text = ""

    if segment_index > 0:
        previous_text = segments[segment_index - 1].get("translated_text", "").strip()

    if segment_index < len(segments) - 1:
        next_text = segments[segment_index + 1].get("translated_text", "").strip()

    # Set output path
    if output_audio_file is None:
        base_dir = os.path.dirname(translation_file)
        base_name = os.path.splitext(os.path.basename(translation_file))[0]
        output_audio_file = os.path.join(base_dir, f"{base_name}_segment_{segment_id}.mp3")

    print(f"Regenerating segment {segment_id}: '{text[:30]}...'")
    print(
        f"  Start: {target_segment['start']}s, End: {target_segment['end']}s, Target Duration: {target_duration_ms}ms")

    # Create temp directory
    temp_dir = "temp_audio_segments"
    os.makedirs(temp_dir, exist_ok=True)

    temp_file = os.path.join(temp_dir, f"segment_{segment_id}.mp3")

    try:
        # Generate TTS - this part is similar to generate_tts_for_segments
        if dealer.lower() == "openai":
            response = openai.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )

            with open(temp_file, "wb") as f:
                f.write(response.content)

        elif dealer.lower() == "elevenlabs":
            elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)

            test_file = os.path.join(temp_dir, f"test_{segment_id}.mp3")

            seed_value = hash(text) % 4294967295

            response = elevenlabs_client.text_to_speech.convert(
                voice_id="ksNuhhaBnNLdMLz6SavZ",
                output_format="mp3_44100_192",
                text=text,
                model_id="eleven_multilingual_v2",
                voice_settings={
                    "similarity_boost": 1,
                    "stability": 0.75,
                    "speed": 1,
                    "use_speaker_boost": False
                },
                previous_text=previous_text,
                next_text=next_text,
                seed=seed_value
            )

            audio_data = b"".join(response)

            with open(test_file, "wb") as f:
                f.write(audio_data)

            test_audio = AudioSegment.from_file(test_file)
            actual_duration_ms = len(test_audio)

            print(f"  Test audio duration: {actual_duration_ms}ms")

            needed_speed = actual_duration_ms / target_duration_ms

            if abs(needed_speed - 1.0) > 0.15:
                speed_value = max(0.92, min(1.08, needed_speed))

                print(f"  Using ElevenLabs speed control: {speed_value:.2f}")

                response = elevenlabs_client.text_to_speech.convert(
                    voice_id="ksNuhhaBnNLdMLz6SavZ",
                    output_format="mp3_44100_192",
                    text=text,
                    model_id="eleven_multilingual_v2",
                    voice_settings={
                        "similarity_boost": 1,
                        "stability": 0.75,
                        "speed": speed_value,
                        "use_speaker_boost": False
                    },
                    previous_text=previous_text,
                    next_text=next_text,
                    seed=seed_value
                )

                audio_data = b"".join(response)

                with open(temp_file, "wb") as f:
                    f.write(audio_data)

                os.remove(test_file)
            else:
                print(f"  Speed adjustment within 15%, using original generation")
                os.rename(test_file, temp_file)
        else:
            raise ValueError(f"Unknown TTS dealer: {dealer}. Supported options: openai, elevenlabs")

        # Process the generated audio
        segment_audio = AudioSegment.from_file(temp_file)
        actual_duration_ms = len(segment_audio)

        print(f"  Actual audio duration: {actual_duration_ms}ms")

        # Apply the same adjustment logic as in generate_tts_for_segments
        if actual_duration_ms > target_duration_ms:
            speed_factor = actual_duration_ms / target_duration_ms
            print(f"  Final adjustment - speeding up by factor: {speed_factor:.2f}x")
            segment_audio = speedup_audio_to_target_duration(segment_audio, speed_factor, temp_dir, segment_id)
        elif target_duration_ms > actual_duration_ms > 0:
            silence_duration_ms = min(200, int(target_duration_ms - actual_duration_ms) // 2)
            stretch_target_ms = target_duration_ms - silence_duration_ms

            if stretch_target_ms > actual_duration_ms:
                print(f"  Final adjustment - stretching to: {stretch_target_ms}ms")
                segment_audio = stretch_audio_to_target_duration(segment_audio, stretch_target_ms, temp_dir, segment_id)

            segment_audio = segment_audio + AudioSegment.silent(duration=silence_duration_ms)
            print(f"  Final segment duration with silence: {len(segment_audio)}ms vs target {target_duration_ms}ms")

        # Export the final audio
        segment_audio.export(output_audio_file, format="mp3")
        print(f"Done! Segment audio saved to {output_audio_file}")

        final_segment_duration = len(segment_audio)
        if abs(final_segment_duration - target_duration_ms) > 100:
            print(
                f"  WARNING! Final segment duration {final_segment_duration}ms differs from target {target_duration_ms}ms")

    except Exception as e:
        print(f"Error processing segment {segment_id}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        shutil.rmtree(temp_dir)

    return output_audio_file


def replace_segment_in_audio(main_audio_file, segment_audio_file, translation_file, segment_id, output_audio_file=None):
    # Set output path if not specified
    if output_audio_file is None:
        base_dir = os.path.dirname(main_audio_file)
        base_name = os.path.splitext(os.path.basename(main_audio_file))[0]
        output_audio_file = os.path.join(base_dir, f"{base_name}_replaced_segment_{segment_id}.mp3")

    # Load the translation data to get segment times
    with open(translation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("Error: There are no segments in the transcription file.")
        return None

    # Find the segment with the specified ID
    target_segment = None
    for segment in segments:
        if segment.get("id") == int(segment_id):
            target_segment = segment
            break

    if not target_segment:
        print(f"Error: Segment with ID {segment_id} not found in the translation file.")
        return None

    # Get segment timestamps
    start_time_ms = int(target_segment["start"] * 1000)
    end_time_ms = int(target_segment["end"] * 1000)
    target_duration_ms = end_time_ms - start_time_ms

    print(f"Replacing segment {segment_id} in main audio")
    print(
        f"  Start: {target_segment['start']}s, End: {target_segment['end']}s, Target Duration: {target_duration_ms}ms")

    # Load the main audio and new segment
    main_audio = AudioSegment.from_file(main_audio_file)
    segment_audio = AudioSegment.from_file(segment_audio_file)

    # Check if the new segment duration needs adjustment
    actual_duration_ms = len(segment_audio)
    print(f"  New segment duration: {actual_duration_ms}ms")

    temp_dir = "temp_audio_segments"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Adjust the segment duration if needed
        if actual_duration_ms > target_duration_ms:
            speed_factor = actual_duration_ms / target_duration_ms
            print(f"  Final adjustment - speeding up by factor: {speed_factor:.2f}x")
            segment_audio = speedup_audio_to_target_duration(segment_audio, speed_factor, temp_dir, segment_id)
        elif target_duration_ms > actual_duration_ms:
            print(f"  Final adjustment - stretching to: {target_duration_ms}ms")
            segment_audio = stretch_audio_to_target_duration(segment_audio, target_duration_ms, temp_dir, segment_id)

        # Verify the exact duration after adjustments
        final_segment_duration = len(segment_audio)
        print(f"  Final adjusted segment duration: {final_segment_duration}ms")

        # Create the new audio file with the replaced segment
        # Take audio before the segment
        before_segment = main_audio[:start_time_ms]

        # Take audio after the segment
        after_segment = main_audio[end_time_ms:]

        # Combine parts
        new_audio = before_segment + segment_audio + after_segment

        # Export the result
        new_audio.export(output_audio_file, format="mp3")
        print(f"Done! New audio with replaced segment saved to {output_audio_file}")

    except Exception as e:
        print(f"Error replacing segment: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

    return output_audio_file