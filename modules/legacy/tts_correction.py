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
    data["segments"][segment_index]["target_duration"] = target_duration_ms

    previous_text = ""
    next_text = ""

    if segment_index > 0:
        previous_text = segments[segment_index - 1].get("translated_text", "").strip()

    if segment_index < len(segments) - 1:
        next_text = segments[segment_index + 1].get("translated_text", "").strip()

    previous_ids = []
    if segment_index > 0:
        if "request_id" in segments[segment_index - 1]:
            previous_ids.append(segments[segment_index - 1]["request_id"])
        if segment_index > 1 and "request_id" in segments[segment_index - 2]:
            previous_ids.append(segments[segment_index - 2]["request_id"])

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

            # seed_value = hash(text) % 4294967295

            request_data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "output_format": "pcm_24000",
                "voice_settings": {
                    "similarity_boost": 1,
                    "stability": 1,
                    "speed": 0.88,
                    "use_speaker_boost": False
                },
                "previous_text": previous_text,
                "next_text": next_text,
                # "seed": seed_value
            }

            if previous_ids:
                print(f"  Using previous request IDs: {previous_ids}")
                request_data["previous_request_ids"] = previous_ids[:3]
            else:
                print("  No previous request IDs available")

            if not hasattr(regenerate_segment, 'segment_request_ids'):
                regenerate_segment.segment_request_ids = {}

            headers = {"xi-api-key": elevenlabs_api_key}

            response = make_api_request_with_retry(
                f"https://api.elevenlabs.io/v1/text-to-speech/JDgAnGtjhmdCMmtbyRYK/stream",
                request_data,
                headers
            )

            current_request_id = response.headers.get("request-id")
            if current_request_id:
                print(f"  Got request_id: {current_request_id}")
                data["segments"][segment_index]["request_id"] = current_request_id
                with open(translation_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                print("  Warning: No request_id received in response")

            with open(temp_file, "wb") as f:
                f.write(response.content)

            segment_audio = AudioSegment.from_file(temp_file)
            actual_duration_ms = len(segment_audio)

            print(f"  Generated audio duration: {actual_duration_ms}ms")
        else:
            raise ValueError(f"Unknown TTS dealer: {dealer}. Supported options: openai, elevenlabs")

        segment_audio = AudioSegment.from_file(temp_file)
        actual_duration_ms = len(segment_audio)

        print(f"  Actual audio duration: {actual_duration_ms}ms")

        if actual_duration_ms > target_duration_ms:
            speed_factor = actual_duration_ms / target_duration_ms
            print(f"  Final adjustment - speeding up by factor: {speed_factor:.2f}x")
            segment_audio = speedup_audio_to_target_duration(segment_audio, speed_factor, temp_dir, segment_id)

            data["segments"][segment_index]["speed_up"] = round(speed_factor, 4)
            if "stretched" in data["segments"][segment_index]:
                del data["segments"][segment_index]["stretched"]
        elif target_duration_ms > actual_duration_ms > 0:
            if target_duration_ms > actual_duration_ms:
                stretch_factor = target_duration_ms / actual_duration_ms
                print(f"  Final adjustment - stretching to: {target_duration_ms}ms")
                segment_audio = stretch_audio_to_target_duration(segment_audio, target_duration_ms, temp_dir,
                                                                 segment_id)

                data["segments"][segment_index]["stretched"] = round(stretch_factor, 4)
                if "speed_up" in data["segments"][segment_index]:
                    del data["segments"][segment_index]["speed_up"]
            print(f"  Final segment duration with silence: {len(segment_audio)}ms vs target {target_duration_ms}ms")

        segment_audio.export(output_audio_file, format="mp3")
        print(f"Done! Segment audio saved to {output_audio_file}")

        final_segment_duration = len(segment_audio)
        if abs(final_segment_duration - target_duration_ms) > 100:
            print(
                f"  WARNING! Final segment duration {final_segment_duration}ms differs from target {target_duration_ms}ms")

        with open(translation_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

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
