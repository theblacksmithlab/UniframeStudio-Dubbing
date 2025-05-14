import os
import json
import subprocess
import openai
from pydub import AudioSegment
from modules.tts import make_api_request_with_retry
from modules.tts import match_target_amplitude


def regenerate_segment(translation_file, job_id, segment_id, output_audio_file=None, voice="onyx", dealer="openai",
                       elevenlabs_api_key=None, openai_api_key=None):
    if dealer.lower() == "elevenlabs" and not elevenlabs_api_key:
        raise ValueError("ElevenLabs API key is required for ElevenLabs TTS but not provided")

    if dealer.lower() == "openai" and not openai_api_key:
        raise ValueError("OpenAI API key is required for OpenAI TTS but not provided")

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
    original_duration_ms = end_time_ms - start_time_ms
    original_duration_sec = round(target_segment["end"] - target_segment["start"], 6)

    segment_index = segments.index(target_segment)

    data["segments"][segment_index]["original_duration"] = original_duration_sec

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

    base_dir = f"jobs/{job_id}/output"

    if output_audio_file is None:
        segments_dir = os.path.join(base_dir, "audio_segments")
        os.makedirs(segments_dir, exist_ok=True)
        output_audio_file = os.path.join(segments_dir, f"segment_{segment_id}.mp3")

    print(f"Regenerating segment {segment_id}: '{text[:30]}...'")
    print(
        f"Start: {target_segment['start']}s, End: {target_segment['end']}s, Original Duration: {original_duration_sec}s")

    temp_dir = os.path.join(base_dir, "temp_audio_segments")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, f"segment_{segment_id}.mp3")

    try:
        if dealer.lower() == "openai":
            client = openai.OpenAI(api_key=openai_api_key)

            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )

            with open(temp_file, "wb") as f:
                f.write(response.content)

        elif dealer.lower() == "elevenlabs":
            request_data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "output_format": "pcm_24000",
                "voice_settings": {
                    "similarity_boost": 1,
                    "stability": 0.75,
                    "speed": 0.9,
                    "use_speaker_boost": True
                },
                "previous_text": previous_text,
                "next_text": next_text,
            }

            if previous_ids:
                print(f"Using previous request IDs: {previous_ids}")
                request_data["previous_request_ids"] = previous_ids[:3]
            else:
                print("No previous request IDs available")

            headers = {"xi-api-key": elevenlabs_api_key}

            print(f"Using ElevenLabs voice ID: {voice}")
            response = make_api_request_with_retry(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream",
                request_data,
                headers
            )

            current_request_id = response.headers.get("request-id")
            if current_request_id:
                print(f"Got request_id: {current_request_id}")
                data["segments"][segment_index]["request_id"] = current_request_id
            else:
                print("Warning: No request_id received in response")

            with open(temp_file, "wb") as f:
                f.write(response.content)

        else:
            raise ValueError(f"Unknown TTS dealer: {dealer}. Supported options: openai, elevenlabs")

        def get_precise_audio_duration(file_path):
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return float(result.stdout.strip())

        precise_duration = get_precise_audio_duration(temp_file)
        actual_duration_ms = precise_duration * 1000

        print(f"Generated audio duration: {precise_duration:.6f}s ({actual_duration_ms:.2f}ms)")

        data["segments"][segment_index]["tts_duration"] = round(precise_duration, 6)

        speed_ratio = actual_duration_ms / original_duration_ms
        data["segments"][segment_index]["speed_ratio"] = round(speed_ratio, 6)

        print(f"Speed ratio for video adjustment: {speed_ratio:.4f}x")

        diff_ratio = abs(actual_duration_ms - original_duration_ms) / original_duration_ms
        if diff_ratio > 0.2:
            print(f"WARNING! TTS duration differs from original by more than 20% ({diff_ratio:.2%})")

        segment_audio = AudioSegment.from_file(temp_file)
        segment_audio = match_target_amplitude(segment_audio, -16.0)

        segment_audio.export(output_audio_file, format="mp3", bitrate="192k")
        print(f"Done! Segment audio saved to {output_audio_file}")

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

    return output_audio_file
