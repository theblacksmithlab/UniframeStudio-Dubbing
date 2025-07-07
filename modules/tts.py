import os
import json
import shutil
import ssl
import time
import openai
import requests
from pydub import AudioSegment
import subprocess
from utils.logger_config import setup_logger, get_job_logger

logger = setup_logger(name=__name__, log_file="logs/app.log")


def generate_tts_for_segments(translation_file, job_id, output_audio_file=None, voice="onyx", dealer="openai",
                              elevenlabs_api_key=None, openai_api_key=None):
    log = get_job_logger(logger, job_id)

    if dealer.lower() == "elevenlabs" and not elevenlabs_api_key:
        raise ValueError("ElevenLabs API key is required for ElevenLabs TTS but not provided by user")

    if dealer.lower() == "openai" and not openai_api_key:
        raise ValueError("OpenAI API key is required for OpenAI TTS but not provided by user")

    base_dir = f"jobs/{job_id}/output"

    if output_audio_file is None:
        audio_result_dir = os.path.join(base_dir, "audio_result")
        os.makedirs(audio_result_dir, exist_ok=True)
        output_audio_file = os.path.join(audio_result_dir, "new_audio.mp3")

    os.makedirs(os.path.dirname(output_audio_file), exist_ok=True)

    with open(translation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        log.error("There are no segments in the transcription file.")
        return None

    log.info(f"Uploaded {len(segments)} segments for voice-over using {dealer}")

    segments_dir = os.path.join(base_dir, "audio_segments")
    os.makedirs(segments_dir, exist_ok=True)

    temp_dir = os.path.join(base_dir, "temp_audio_segments")
    os.makedirs(temp_dir, exist_ok=True)

    generated_segments = []

    openai_client = None
    if dealer.lower() == "openai":
        openai_client = openai.OpenAI(api_key=openai_api_key)

    for i, segment in enumerate(segments):
        text = segment.get("translated_text", "").strip()
        if not text:
            log.warning(f"Skipping segment {i + 1}/{len(segments)}: empty text")
            continue

        start_time_ms = int(segment["start"] * 1000)
        end_time_ms = int(segment["end"] * 1000)
        original_duration_ms = end_time_ms - start_time_ms
        original_duration_sec = round(segment["end"] - segment["start"], 6)
        data["segments"][i]["original_duration"] = original_duration_sec

        if dealer.lower() == "openai":
            segment_voice = segment.get("suggested_voice", voice)
            if segment_voice != voice:
                log.info(
                    f"Using dynamic voice '{segment_voice}' for segment {i + 1} (gender: {segment.get('predicted_gender', 'unknown')})")
        else:
            segment_voice = voice

        log.info(f"Processing segment {i + 1}/{len(segments)}: '{text[:30]}...'")
        log.info(
            f"Start: {segment['start']}s, End: {segment['end']}s, Target Duration: {original_duration_ms / 1000}s")
        log.info(f"Voice: {segment_voice}")

        segment_file = os.path.join(segments_dir, f"segment_{i}.mp3")
        temp_file = os.path.join(temp_dir, f"segment_{i}.mp3")

        try:
            if dealer.lower() == "openai":
                generate_openai_tts_with_retry(openai_client, text, segment_voice, temp_file, job_id)

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
                        log.info(f"Using previous request IDs: {previous_ids}")
                    else:
                        log.info("No previous request IDs available")

                headers = {"xi-api-key": elevenlabs_api_key}

                response = make_api_request_with_retry(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream",
                    request_data,
                    headers,
                    job_id=job_id
                )

                current_request_id = response.headers.get("request-id")
                if current_request_id:
                    generate_tts_for_segments.segment_request_ids[i] = current_request_id
                    log.info(f"Got request_id: {current_request_id}")
                    data["segments"][i]["request_id"] = current_request_id
                else:
                    log.info("Warning: No request_id received in response")

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

                if result.returncode != 0:
                    log.error(f"Error getting precise audio duration while generating tts: {result.stderr.strip()}")
                    return 0.0

                return float(result.stdout.strip())

            precise_duration = get_precise_audio_duration(temp_file)
            actual_duration_ms = precise_duration * 1000

            log.info(f"Generated audio duration: {actual_duration_ms}ms")
            data["segments"][i]["tts_duration"] = round(actual_duration_ms / 1000, 6)

            diff_ratio = abs(actual_duration_ms - original_duration_ms) / original_duration_ms
            if diff_ratio > 0.20:
                log.warning(f"ATTENTION! TTS duration differs from target by more than 20% ({diff_ratio:.2%})")

            data["segments"][i]["speed_ratio"] = round(diff_ratio, 2)

            segment_audio = match_target_amplitude(segment_audio, -16.0)

            segment_audio.export(segment_file, format="mp3", bitrate="320k")

            generated_segments.append({
                "id": i,
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms,
                "file": segment_file
            })

            if os.path.exists(temp_file):
                os.remove(temp_file)


        except Exception as e:
            log.error(f"Error processing segment {i}: {e}")
            import traceback
            traceback.print_exc()
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except (OSError, PermissionError):
                    pass
            continue

    if not generated_segments:
        log.error("No segments were successfully processed")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None

    try:
        log.info("Assembling tts audio-file...")
        assemble_audio_file(generated_segments, output_audio_file, data, job_id=job_id)
    except Exception as e:
        log.error(f"Error assembling tts audio-file: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None

    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        log.warning(f"Failed to clean up temporary directory: {e}")

    try:
        with open(translation_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Error saving updated data to file: {e}")
        return None

    if dealer.lower() == "openai":
        voice_usage_stats = {}
        gender_stats = {}

        for segment in segments:
            used_voice = segment.get("suggested_voice", voice)
            predicted_gender = segment.get("predicted_gender", "unknown")

            voice_usage_stats[used_voice] = voice_usage_stats.get(used_voice, 0) + 1
            gender_stats[predicted_gender] = gender_stats.get(predicted_gender, 0) + 1

        log.info("=" * 50)
        log.info("OPENAI TTS VOICE USAGE STATISTICS")
        log.info("=" * 50)

        total_segments = len([s for s in segments if s.get("translated_text", "").strip()])

        for voice_name, count in voice_usage_stats.items():
            percentage = (count / total_segments) * 100 if total_segments > 0 else 0
            log.info(f"Voice '{voice_name}': {count} segments ({percentage:.1f}%)")

        log.info("Gender distribution:")
        for gender, count in gender_stats.items():
            percentage = (count / total_segments) * 100 if total_segments > 0 else 0
            log.info(f"  {gender}: {count} segments ({percentage:.1f}%)")

        segments_without_analysis = len([s for s in segments if not s.get("suggested_voice")])
        if segments_without_analysis > 0:
            log.warning(
                f"Found {segments_without_analysis} segments without voice analysis - using fallback voice '{voice}'")

    return output_audio_file


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def assemble_audio_file(segments, output_file, json_data=None, job_id=None):
    if job_id:
        log = get_job_logger(logger, job_id)
    else:
        log = logger

    if not segments:
        log.error("No segments to assemble")
        return None

    segments.sort(key=lambda x: x["start_time_ms"])

    final_audio = AudioSegment.empty()
    current_position = 0

    for segment in segments:
        segment_audio = AudioSegment.from_file(segment["file"])
        segment_audio = match_target_amplitude(segment_audio, -16.0)

        if segment == segments[0]:
            position = segment["start_time_ms"]
        else:
            prev_segment = segments[segments.index(segment) - 1]
            position = current_position + (segment["start_time_ms"] - prev_segment["end_time_ms"])

        if position > len(final_audio):
            silence_needed = position - len(final_audio)
            final_audio += AudioSegment.silent(duration=silence_needed)

        final_audio += segment_audio
        current_position = len(final_audio)

    if json_data and "outro_gap_duration" in json_data:
        outro_gap_duration_ms = json_data["outro_gap_duration"] * 1000
        if outro_gap_duration_ms > 0:
            final_audio += AudioSegment.silent(duration=outro_gap_duration_ms)
            log.info(f"Added {outro_gap_duration_ms:.0f}ms outro gap from original")

    final_audio.export(output_file, format="mp3", bitrate="320k")
    log.info(f"Audio-file assembled and saved to {output_file}")
    log.info(f"Audio-file duration: {len(final_audio) / 1000:.2f} seconds")

    stereo_output_file = os.path.splitext(output_file)[0] + "_stereo.mp3"

    try:
        if os.path.exists(output_file):
            cmd = [
                'ffmpeg',
                '-i', output_file,
                '-filter_complex', '[0]pan=stereo|c0=c0|c1=c0',
                '-y',
                stereo_output_file
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if os.path.exists(stereo_output_file):
                log.info("Stereo version of audio created successfully")
            else:
                log.warning(f"Failed to create stereo version. Command exit code: {result.returncode}")
        else:
            log.warning("Cannot create stereo version because original file not found")

    except Exception as e:
        log.warning(f"Error creating stereo version: {e}")

    return output_file

def reassemble_audio_file(translation_file, job_id, output_audio_file=None):
    log = get_job_logger(logger, job_id)

    base_dir = f"jobs/{job_id}/output"

    if output_audio_file is None:
        audio_result_dir = os.path.join(base_dir, "audio_result")
        os.makedirs(audio_result_dir, exist_ok=True)
        output_audio_file = os.path.join(audio_result_dir, "new_audio.mp3")

    with open(translation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        log.error("There are no segments in the transcription file")
        return None

    segments_dir = os.path.join(base_dir, "audio_segments")

    if not os.path.exists(segments_dir):
        log.error(f"Segments directory {segments_dir} not found. Have you generated the segments first?")
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
        else:
            missing_segments.append(i)

    if missing_segments:
        log.warning(f"The following segments are missing: {missing_segments}")
        if not available_segments:
            log.error("No segments found to assemble")
            return None
        log.warning("Proceeding with available segments only")

    log.info(f"Reassembling audio file from {len(available_segments)} segments...")

    result = assemble_audio_file(available_segments, output_audio_file, data, job_id=job_id)

    return result


def make_api_request_with_retry(url, data, headers, max_retries=10, retry_delay=2, job_id=None):
    if job_id:
        log = get_job_logger(logger, job_id)
    else:
        log = logger

    retries = 0
    last_exception = None
    last_status_code = None
    last_response_text = None

    while retries < max_retries:
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            if response.status_code == 200:
                return response
            else:
                log.warning(f"API error (attempt {retries + 1}/{max_retries}): {response.status_code} - {response.text}")
                last_status_code = response.status_code
                last_response_text = response.text

                if response.status_code in [400, 401, 403]:  # Bad request, Unauthorized, Forbidden
                    raise ValueError(f"API returned error {response.status_code}: {response.text}")
        except (requests.RequestException, TimeoutError, ConnectionError, ssl.SSLError) as e:
            last_exception = e
            log.warning(f"Request failed (attempt {retries + 1}/{max_retries}): {e}")

        retries += 1
        if retries < max_retries:
            sleep_time = retry_delay * retries
            log.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

    if last_status_code:
        raise ValueError(f"API request failed with status code {last_status_code}: {last_response_text}")
    else:
        raise Exception(f"Failed after {max_retries} attempts. Last error: {last_exception}")


def generate_openai_tts_with_retry(client, text, voice, temp_file, job_id,
                                   instructions=None, max_retries=10, retry_delay=2):
    log = get_job_logger(logger, job_id)

    if instructions is None:
        instructions = "Speak vigorously, 20% faster than normal pace."

    retries = 0
    last_exception = None

    while retries < max_retries:
        try:
            log.info(f"Attempting OpenAI TTS generation (attempt {retries + 1}/{max_retries})")
            log.info("NEW METHOD!")
            with client.audio.speech.with_streaming_response.create(
                    model="gpt-4o-mini-tts",
                    voice=voice,
                    input=text,
                    instructions=instructions,
                    response_format="mp3"
            ) as response:
                response.stream_to_file(temp_file)

            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                log.info("OpenAI TTS generation successful")
                return True
            else:
                raise Exception("Generated audio file is empty or was not created")

        except Exception as e:
            last_exception = e
            log.warning(f"OpenAI TTS failed (attempt {retries + 1}/{max_retries}): {e}")

            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except (OSError, PermissionError) as cleanup_error:
                    log.warning(f"Failed to remove damaged file {temp_file}: {cleanup_error}")

        retries += 1
        if retries < max_retries:
            sleep_time = retry_delay * retries
            log.info(f"Retrying OpenAI TTS in {sleep_time} seconds...")
            time.sleep(sleep_time)

    raise Exception(f"OpenAI TTS failed after {max_retries} attempts. Last error: {last_exception}")


# def generate_openai_tts_with_retry(client, text, voice, temp_file, job_id, max_retries=10, retry_delay=2):
#     log = get_job_logger(logger, job_id)
#
#     retries = 0
#     last_exception = None
#
#     while retries < max_retries:
#         try:
#             log.info(f"Attempting OpenAI TTS generation (attempt {retries + 1}/{max_retries})")
#
#             response = client.audio.speech.create(
#                 model="tts-1",
#                 voice=voice,
#                 input=text,
#                 speed=1.2
#             )
#
#             if hasattr(response, 'content') and response.content:
#                 with open(temp_file, "wb") as f:
#                     f.write(response.content)
#
#                 if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
#                     log.info("OpenAI TTS generation successful")
#                     return True
#                 else:
#                     raise Exception("Generated audio file is empty or was not created")
#             else:
#                 raise Exception("OpenAI API returned empty response")
#
#
#         except Exception as e:
#             last_exception = e
#             log.warning(f"OpenAI TTS failed (attempt {retries + 1}/{max_retries}): {e}")
#             if os.path.exists(temp_file):
#                 try:
#                     os.remove(temp_file)
#                 except (OSError, PermissionError) as cleanup_error:
#                     log.warning(f"Failed to remove damaged file {temp_file}: {cleanup_error}")
#
#         retries += 1
#         if retries < max_retries:
#             sleep_time = retry_delay * retries
#             log.info(f"Retrying OpenAI TTS in {sleep_time} seconds...")
#             time.sleep(sleep_time)
#
#     raise Exception(f"OpenAI TTS failed after {max_retries} attempts. Last error: {last_exception}")
