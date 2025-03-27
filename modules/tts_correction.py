import os
import json
import shutil
import openai
from pydub import AudioSegment
from dotenv import load_dotenv

from modules.tts import speedup_audio_to_target_duration, stretch_audio_to_target_duration, make_api_request_with_retry


def regenerate_segment(translation_file, segment_id, output_audio_file=None, voice="onyx", dealer="openai"):
    load_dotenv()
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

    with open(translation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("Error: There are no segments in the transcription file.")
        return None

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

    segment_index = segments.index(target_segment)
    previous_text = ""
    next_text = ""

    if segment_index > 0:
        previous_text = segments[segment_index - 1].get("translated_text", "").strip()

    if segment_index < len(segments) - 1:
        next_text = segments[segment_index + 1].get("translated_text", "").strip()

    if output_audio_file is None:
        base_dir = os.path.dirname(translation_file)
        base_name = os.path.splitext(os.path.basename(translation_file))[0]
        output_audio_file = os.path.join(base_dir, f"{base_name}_segment_{segment_id}.mp3")

    print(f"Regenerating segment {segment_id}: '{text[:30]}...'")
    print(
        f"  Start: {target_segment['start']}s, End: {target_segment['end']}s, Target Duration: {target_duration_ms}ms")

    temp_dir = "temp_audio_segments"
    os.makedirs(temp_dir, exist_ok=True)

    temp_file = os.path.join(temp_dir, f"segment_{segment_id}.mp3")

    try:
        if dealer.lower() == "openai":
            response = openai.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )

            with open(temp_file, "wb") as f:
                f.write(response.content)

        elif dealer.lower() == "elevenlabs":
            test_file = os.path.join(temp_dir, f"test_{segment_id}.mp3")

            seed_value = hash(text) % 4294967295

            request_data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "output_format": "mp3_44100_192",
                "voice_settings": {
                    "similarity_boost": 1,
                    "stability": 0.75,
                    "speed": 1,
                    "use_speaker_boost": False
                },
                "previous_text": previous_text,
                "next_text": next_text,
                "seed": seed_value
            }

            if not hasattr(regenerate_segment, 'segment_request_ids'):
                regenerate_segment.segment_request_ids = {}

            headers = {"xi-api-key": elevenlabs_api_key}

            response = make_api_request_with_retry(
                f"https://api.elevenlabs.io/v1/text-to-speech/ksNuhhaBnNLdMLz6SavZ/stream",
                request_data,
                headers
            )

            current_request_id = response.headers.get("request-id")
            if current_request_id:
                regenerate_segment.segment_request_ids[segment_id] = current_request_id
                print(f"  Got request_id: {current_request_id}")

            with open(test_file, "wb") as f:
                f.write(response.content)

            test_audio = AudioSegment.from_file(test_file)
            actual_duration_ms = len(test_audio)

            print(f"  Test audio duration: {actual_duration_ms}ms")

            needed_speed = actual_duration_ms / target_duration_ms

            if abs(needed_speed - 1.0) > 0.15:
                speed_value = max(0.92, min(1.08, needed_speed))

                print(f"  Using ElevenLabs speed control: {speed_value:.2f}")

                request_data["voice_settings"]["speed"] = speed_value

                response = make_api_request_with_retry(
                    f"https://api.elevenlabs.io/v1/text-to-speech/ksNuhhaBnNLdMLz6SavZ/stream",
                    request_data,
                    headers
                )

                new_request_id = response.headers.get("request-id")
                if new_request_id:
                    regenerate_segment.segment_request_ids[segment_id] = new_request_id
                    print(f"  Updated request_id: {new_request_id}")

                with open(temp_file, "wb") as f:
                    f.write(response.content)

                os.remove(test_file)
            else:
                print(f"  Speed adjustment within 15%, using original generation")
                os.rename(test_file, temp_file)
        else:
            raise ValueError(f"Unknown TTS dealer: {dealer}. Supported options: openai, elevenlabs")

        segment_audio = AudioSegment.from_file(temp_file)
        actual_duration_ms = len(segment_audio)

        print(f"  Actual audio duration: {actual_duration_ms}ms")

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
        if os.path.exists(temp_file):
            os.remove(temp_file)
        shutil.rmtree(temp_dir)

    return output_audio_file
