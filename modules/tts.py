import os
import json
import shutil
import openai
from elevenlabs import ElevenLabs
from pydub import AudioSegment
from dotenv import load_dotenv


def stretch_audio_to_target_duration(segment_audio, target_duration_ms, temp_dir, i):
    import librosa
    import numpy as np
    import soundfile as sf

    actual_duration_ms = len(segment_audio)

    max_stretch_ratio = 1.05
    stretch_ratio = min(max_stretch_ratio, target_duration_ms / actual_duration_ms)

    samples = np.array(segment_audio.get_array_of_samples())
    sample_rate = segment_audio.frame_rate

    if segment_audio.sample_width == 2:  # 16-bit audio
        samples = samples.astype(np.float32) / 32767.0
    elif segment_audio.sample_width == 4:  # 32-bit audio
        samples = samples.astype(np.float32) / 2147483647.0

    stretched_samples = librosa.effects.time_stretch(samples, rate=1.0 / stretch_ratio)

    if segment_audio.sample_width == 2:
        stretched_samples = (stretched_samples * 32767.0).astype(np.int16)
    elif segment_audio.sample_width == 4:
        stretched_samples = (stretched_samples * 2147483647.0).astype(np.int32)

    temp_stretched_file = os.path.join(temp_dir, f"stretched_{i}.wav")
    sf.write(temp_stretched_file, stretched_samples, sample_rate)

    stretched_audio = AudioSegment.from_file(temp_stretched_file)
    os.remove(temp_stretched_file)

    stretched_duration_ms = len(stretched_audio)
    if stretched_duration_ms < target_duration_ms:
        remaining_silence = target_duration_ms - stretched_duration_ms
        stretched_audio += AudioSegment.silent(duration=remaining_silence)

    return stretched_audio

def generate_tts_for_segments(translation_file, output_audio_file=None, voice="onyx"):
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

    print(f"Uploaded {len(segments)} segments for voice-over")

    temp_dir = "temp_audio_segments"
    os.makedirs(temp_dir, exist_ok=True)

    final_duration_ms = int(segments[-1]["end"] * 1000) + 1000  # +1 second for buffer (test if we need)
    final_audio = AudioSegment.silent(duration=final_duration_ms)

    for i, segment in enumerate(segments):
        text = segment.get("translated_text", "").strip()
        if not text:
            print(f"Skipping segment {i + 1}/{len(segments)}: empty text")
            continue

        start_time_ms = int(segment["start"] * 1000)
        end_time_ms = int(segment["end"] * 1000)
        target_duration_ms = end_time_ms - start_time_ms

        print(f"Processing segment {i + 1}/{len(segments)}: {text[:50]}...")

        temp_file = os.path.join(temp_dir, f"segment_{i}.mp3")

        # elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)

        try:
            # For OpenAI API
            response = openai.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )

            # response = elevenlabs_client.text_to_speech.convert(
            #     voice_id="Et117koAJzDg0IZtjcim",
            #     output_format="mp3_44100_128",
            #     text=text,
            #     model_id="eleven_multilingual_v2",
            #     voice_settings= {"similarity_boost": 1, "stability": 1}
            # )

            # audio_data = b"".join(response)

            # with open(temp_file, "wb") as f:
            #     f.write(audio_data)

            with open(temp_file, "wb") as f:
                f.write(response.content)

            segment_audio = AudioSegment.from_file(temp_file)

            actual_duration_ms = len(segment_audio)

            if actual_duration_ms > target_duration_ms:
                speed_factor = actual_duration_ms / target_duration_ms
                segment_audio = segment_audio.speedup(playback_speed=speed_factor)
            elif target_duration_ms > actual_duration_ms > 0:
                segment_audio = stretch_audio_to_target_duration(segment_audio, target_duration_ms, temp_dir, i)

            final_audio = final_audio.overlay(segment_audio, position=start_time_ms)

            os.remove(temp_file)

        except Exception as e:
            print(f"Error processing segment {i}: {e}")
            continue

    final_audio.export(output_audio_file, format="mp3")
    print(f"Done! Audio file saved to {output_audio_file}")

    shutil.rmtree(temp_dir)

    return output_audio_file
