import os
import json
import shutil
import ssl
import time

import openai
import requests
from pydub import AudioSegment
from dotenv import load_dotenv
import subprocess


def generate_tts_for_segments(translation_file, output_audio_file=None, voice="onyx", dealer="openai", intro=False, outro=False):
    load_dotenv()
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

    if output_audio_file is None:
        base_dir = os.path.dirname(translation_file)
        base_name = os.path.splitext(os.path.basename(translation_file))[0]
        output_audio_file = os.path.join(base_dir, f"{base_name}.mp3")

    os.makedirs(os.path.dirname(output_audio_file), exist_ok=True)

    with open(translation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("Error: There are no segments in the transcription file.")
        return None

    print(f"Uploaded {len(segments)} segments for voice-over using {dealer}")

    segments_dir = os.path.join(os.path.dirname(output_audio_file), "audio_segments")
    os.makedirs(segments_dir, exist_ok=True)

    temp_dir = "temp_audio_segments"
    os.makedirs(temp_dir, exist_ok=True)

    generated_segments = []

    for i, segment in enumerate(segments):
        text = segment.get("translated_text", "").strip()
        if not text:
            print(f"Skipping segment {i + 1}/{len(segments)}: empty text")
            continue

        start_time_ms = int(segment["start"] * 1000)
        end_time_ms = int(segment["end"] * 1000)
        original_duration_ms = end_time_ms - start_time_ms
        original_duration_sec = round(segment["end"] - segment["start"], 6)
        data["segments"][i]["original_duration"] = original_duration_sec

        print(f"Processing segment {i + 1}/{len(segments)}: '{text[:30]}...'")
        print(f"Start: {segment['start']}s, End: {segment['end']}s, Target Duration: {original_duration_ms / 1000}s")

        segment_file = os.path.join(segments_dir, f"segment_{i}.mp3")
        temp_file = os.path.join(temp_dir, f"segment_{i}.mp3")

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
                previous_text = ""
                next_text = ""

                if i > 0:
                    previous_text = segments[i - 1].get("translated_text", "").strip()
                if i < len(segments) - 1:
                    next_text = segments[i + 1].get("translated_text", "").strip()

                if not hasattr(generate_tts_for_segments, 'segment_request_ids'):
                    generate_tts_for_segments.segment_request_ids = {}

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

                if i > 0:
                    previous_ids = []
                    if i - 1 in generate_tts_for_segments.segment_request_ids:
                        previous_ids.append(generate_tts_for_segments.segment_request_ids[i - 1])
                    if i - 2 in generate_tts_for_segments.segment_request_ids:
                        previous_ids.append(generate_tts_for_segments.segment_request_ids[i - 2])
                    if previous_ids:
                        request_data["previous_request_ids"] = previous_ids[:3]
                        print(f"Using previous request IDs: {previous_ids}")
                    else:
                        print("No previous request IDs available")

                headers = {"xi-api-key": elevenlabs_api_key}

                response = make_api_request_with_retry(
                    f"https://api.elevenlabs.io/v1/text-to-speech/JDgAnGtjhmdCMmtbyRYK/stream",
                    request_data,
                    headers
                )

                current_request_id = response.headers.get("request-id")
                if current_request_id:
                    generate_tts_for_segments.segment_request_ids[i] = current_request_id
                    print(f"  Got request_id: {current_request_id}")
                    data["segments"][i]["request_id"] = current_request_id
                else:
                    print("  Warning: No request_id received in response")

                with open(temp_file, "wb") as f:
                    f.write(response.content)

            segment_audio = AudioSegment.from_file(temp_file)

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

            print(f"  Generated audio duration: {actual_duration_ms}ms")
            data["segments"][i]["tts_duration"] = round(actual_duration_ms / 1000, 6)

            speed_ratio = actual_duration_ms / original_duration_ms
            data["segments"][i]["speed_ratio"] = round(speed_ratio, 6)

            print(f"  Generated audio duration: {actual_duration_ms / 1000}s")
            print(f"  Speed ratio for video adjustment: {speed_ratio:.2f}x")

            diff_ratio = abs(actual_duration_ms - original_duration_ms) / original_duration_ms
            if diff_ratio > 0.2:
                print(f"  WARNING! TTS duration differs from target by more than 20% ({diff_ratio:.2%})")

            segment_audio = match_target_amplitude(segment_audio, -16.0)

            segment_audio.export(segment_file, format="mp3", bitrate="192k")
            print(f"  Saved segment to {segment_file}")

            generated_segments.append({
                "id": i,
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms,
                "file": segment_file
            })

            os.remove(temp_file)

        except Exception as e:
            print(f"Error processing segment {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("Assembling final audio file...")
    assemble_audio_file(generated_segments, output_audio_file, intro, outro)

    shutil.rmtree(temp_dir)

    with open(translation_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_audio_file


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def assemble_audio_file(segments, output_file, intro=False, outro=False):
    if not segments:
        print("No segments to assemble")
        return None

    segments.sort(key=lambda x: x["start_time_ms"])

    final_audio = AudioSegment.empty()

    intro_outro_file = "resources/intro_outro.mp3"
    intro_length_ms = 4000

    if intro and os.path.exists(intro_outro_file):
        special_audio = AudioSegment.from_file(intro_outro_file)
        special_audio = match_target_amplitude(special_audio, -16.0)

        if len(special_audio) < intro_length_ms:
            print(f"Warning: intro audio is shorter than {intro_length_ms}ms")
            intro_length_ms = len(special_audio)

        final_audio += special_audio[:intro_length_ms]
        print(f"Added intro at the beginning (first {intro_length_ms / 1000} seconds)")

        has_intro = True
    else:
        has_intro = False
        if intro and not os.path.exists(intro_outro_file):
            print(f"Warning: Intro file {intro_outro_file} not found, skipping intro")

    current_position = intro_length_ms if has_intro else 0

    for segment in segments:
        segment_audio = AudioSegment.from_file(segment["file"])
        segment_audio = match_target_amplitude(segment_audio, -16.0)

        if has_intro and segment == segments[0]:
            # First segment after intro should be placed right after intro
            # (its original position of 4000ms is already accounted for by the intro)
            position = current_position
        else:
            # For segments without intro or non-first segments, use their relative positions
            # If intro is present, we need to account for any shifts in timing
            if has_intro and segment["start_time_ms"] < intro_length_ms:
                # Skip any segment that would overlap with the intro
                continue

            if segment != segments[0]:
                prev_segment = segments[segments.index(segment) - 1]
                position = current_position + (segment["start_time_ms"] - prev_segment["end_time_ms"])
            else:
                position = segment["start_time_ms"]

        if position > len(final_audio):
            silence_needed = position - len(final_audio)
            final_audio += AudioSegment.silent(duration=silence_needed)

        final_audio += segment_audio

        current_position = len(final_audio)

    # Add outro if needed
    if outro and os.path.exists(intro_outro_file):
        special_audio = AudioSegment.from_file(intro_outro_file)
        special_audio = match_target_amplitude(special_audio, -16.0)
        final_audio += special_audio
        print(f"Added outro to the end of the audio file")
    elif outro:
        print(f"Warning: Outro file {intro_outro_file} not found, skipping outro")

    final_audio.export(output_file, format="mp3", bitrate="192k")
    print(f"Final audio assembled and saved to {output_file}")
    print(f"Final audio duration: {len(final_audio) / 1000:.2f} seconds")

    return output_file


def reassemble_audio_file(translation_file, output_audio_file=None, intro=False, outro=False):
    if output_audio_file is None:
        base_dir = os.path.dirname(translation_file)
        base_name = os.path.splitext(os.path.basename(translation_file))[0]
        output_audio_file = os.path.join(base_dir, f"{base_name}_reassembled.mp3")

    with open(translation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("Error: There are no segments in the transcription file.")
        return None

    segments_dir = os.path.join(os.path.dirname(translation_file), "audio_segments")

    if not os.path.exists(segments_dir):
        print(f"Error: Segments directory {segments_dir} not found. Have you generated the segments first?")
        return None

    available_segments = []
    missing_segments = []

    for i, segment in enumerate(segments):
        segment_file = os.path.join(segments_dir, f"segment_{i}.mp3")

        if os.path.exists(segment_file):
            start_time_ms = int(segment["start"] * 1000)
            end_time_ms = int(segment["end"] * 1000)

            available_segments.append({
                "id": i,
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms,
                "file": segment_file
            })
            print(f"Found segment {i} at {segment_file}")
        else:
            missing_segments.append(i)

    if missing_segments:
        print(f"Warning: The following segments are missing: {missing_segments}")
        if not available_segments:
            print("Error: No segments found to assemble.")
            return None
        print("Proceeding with available segments only.")

    print(f"Reassembling audio file from {len(available_segments)} segments...")
    result = assemble_audio_file(available_segments, output_audio_file, intro, outro)

    return result


def make_api_request_with_retry(url, data, headers, max_retries=10, retry_delay=2):
    retries = 0
    last_exception = None

    while retries < max_retries:
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            if response.status_code == 200:
                return response
            else:
                print(f"  API error (attempt {retries + 1}/{max_retries}): {response.status_code} - {response.text}")
        except (requests.RequestException, TimeoutError, ConnectionError, ssl.SSLError) as e:
            last_exception = e
            print(f"  Request failed (attempt {retries + 1}/{max_retries}): {e}")

        retries += 1
        if retries < max_retries:
            sleep_time = retry_delay * retries
            print(f"  Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

    raise Exception(f"Failed after {max_retries} attempts. Last error: {last_exception}")
