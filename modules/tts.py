import os
import json
import shutil
import openai
from elevenlabs import ElevenLabs
from pydub import AudioSegment
from dotenv import load_dotenv
import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import subprocess


def stretch_audio_to_target_duration(segment_audio, target_duration_ms, temp_dir, i):
    actual_duration_ms = len(segment_audio)
    stretch_ratio = target_duration_ms / actual_duration_ms

    print(f"Segment {i}: Stretching from {actual_duration_ms}ms to target {target_duration_ms}ms")
    print(f"Using stretch ratio: {stretch_ratio:.2f}x")

    input_file = os.path.join(temp_dir, f"input_{i}.wav")
    segment_audio.export(input_file, format="wav")

    output_file = os.path.join(temp_dir, f"stretched_{i}.wav")

    try:
        subprocess.run([
            "rubberband",
            "--time", str(stretch_ratio),
            "--pitch", "1",
            "--formant",
            input_file,
            output_file
        ], check=True, stdout=subprocess.DEVNULL)

        stretched_audio = AudioSegment.from_file(output_file)

        stretched_duration = len(stretched_audio)
        expected_duration = int(actual_duration_ms * stretch_ratio)

        print(f"Segment {i}: After stretching - Expected: ~{expected_duration}ms, Actual: {stretched_duration}ms")

        return stretched_audio

    except (subprocess.SubprocessError, subprocess.CalledProcessError) as e:
        print(f"Error executing Rubberband: {e}")
        print(f"Falling back to librosa for stretching")
    except FileNotFoundError:
        print(f"Rubberband executable not found")
        print(f"Falling back to librosa for stretching")
    except Exception as e:
        print(f"Unexpected error using Rubberband: {e}")
        print(f"Falling back to librosa for stretching")

    try:
        samples = np.array(segment_audio.get_array_of_samples())
        sample_rate = segment_audio.frame_rate

        if segment_audio.sample_width == 2:
            samples = samples.astype(np.float32) / 32767.0
        elif segment_audio.sample_width == 4:
            samples = samples.astype(np.float32) / 2147483647.0

        stretched_samples = librosa.effects.time_stretch(
            samples,
            rate=1.0 / stretch_ratio,
            n_fft=2048,
            hop_length=512
        )

        if segment_audio.sample_width == 2:
            stretched_samples = (stretched_samples * 32767.0).astype(np.int16)
        elif segment_audio.sample_width == 4:
            stretched_samples = (stretched_samples * 2147483647.0).astype(np.int32)

        sf.write(output_file, stretched_samples, sample_rate)

        stretched_audio = AudioSegment.from_file(output_file)

        return stretched_audio
    except Exception as e:
        print(f"Error in librosa fallback: {e}")
        print(f"Returning original audio")
        return segment_audio


    finally:
        for temp_file in [input_file, output_file]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError as e:
                    print(f"Warning: Could not remove temporary file {temp_file}: {e}")


def speedup_audio_to_target_duration(segment_audio, speed_factor, temp_dir, i):
    input_file = os.path.join(temp_dir, f"input_speed_{i}.wav")
    segment_audio.export(input_file, format="wav")

    output_file = os.path.join(temp_dir, f"sped_up_{i}.wav")

    try:
        stretch_ratio = 1.0 / speed_factor

        subprocess.run([
            "rubberband",
            "--time", str(stretch_ratio),
            "--pitch", "1",
            "--formant",
            input_file,
            output_file
        ], check=True, stdout=subprocess.DEVNULL)

        sped_up_audio = AudioSegment.from_file(output_file)
        return sped_up_audio

    except (subprocess.SubprocessError, subprocess.CalledProcessError) as e:
        print(f"Error executing Rubberband for speedup: {e}")
        print("Falling back to librosa method")
    except FileNotFoundError:
        print(f"Rubberband executable not found")
        print("Falling back to librosa method")
    except Exception as e:
        print(f"Unexpected error using Rubberband for speedup: {e}")
        print("Falling back to librosa method")

    # Librosa fallback code
    try:

        y, sr = librosa.load(input_file, sr=None)

        y_sped_up = librosa.effects.time_stretch(y, rate=speed_factor)

        b, a = signal.butter(4, 0.9, 'low')
        y_filtered = signal.filtfilt(b, a, y_sped_up)

        sf.write(output_file, y_filtered, sr)

        sped_up_audio = AudioSegment.from_file(output_file)
        return sped_up_audio

    except Exception as e:
        print(f"Error in librosa speedup fallback: {e}")
        print("Using pydub speedup as last resort")
        return segment_audio.speedup(playback_speed=speed_factor)

    finally:
        for temp_file in [input_file, output_file]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError as e:
                    print(f"Warning: Could not remove temporary file {temp_file}: {e}")


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

    temp_dir = "temp_audio_segments"
    os.makedirs(temp_dir, exist_ok=True)

    final_duration_ms = int(segments[-1]["end"] * 1000)
    final_audio = AudioSegment.silent(duration=final_duration_ms)

    for i, segment in enumerate(segments):
        text = segment.get("translated_text", "").strip()
        if not text:
            print(f"Skipping segment {i + 1}/{len(segments)}: empty text")
            continue

        start_time_ms = int(segment["start"] * 1000)
        end_time_ms = int(segment["end"] * 1000)
        target_duration_ms = end_time_ms - start_time_ms

        print(f"Processing segment {i + 1}/{len(segments)}: '{text[:30]}...'")
        print(f"  Start: {segment['start']}s, End: {segment['end']}s, Target Duration: {target_duration_ms}ms")

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

                elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)

                test_file = os.path.join(temp_dir, f"test_{i}.mp3")

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

            segment_audio = AudioSegment.from_file(temp_file)
            actual_duration_ms = len(segment_audio)

            print(f"  Actual audio duration: {actual_duration_ms}ms")

            if actual_duration_ms > target_duration_ms:
                speed_factor = actual_duration_ms / target_duration_ms
                print(f"  Final adjustment - speeding up by factor: {speed_factor:.2f}x")
                segment_audio = speedup_audio_to_target_duration(segment_audio, speed_factor, temp_dir, i)
            elif target_duration_ms > actual_duration_ms > 0:
                silence_duration_ms = min(200, int(target_duration_ms - actual_duration_ms) // 2)
                stretch_target_ms = target_duration_ms - silence_duration_ms

                if stretch_target_ms > actual_duration_ms:
                    print(f"  Final adjustment - stretching to: {stretch_target_ms}ms")
                    segment_audio = stretch_audio_to_target_duration(segment_audio, stretch_target_ms, temp_dir, i)

                segment_audio = segment_audio + AudioSegment.silent(duration=silence_duration_ms)
                print(f"  Final segment duration with silence: {len(segment_audio)}ms vs target {target_duration_ms}ms")

            final_audio = final_audio.overlay(segment_audio, position=start_time_ms)

            final_segment_duration = len(segment_audio)
            if abs(final_segment_duration - target_duration_ms) > 100:
                print(
                    f"  WARNING: Final segment duration {final_segment_duration}ms differs from target {target_duration_ms}ms")

            os.remove(temp_file)

        except Exception as e:
            print(f"Error processing segment {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    final_audio.export(output_audio_file, format="mp3")
    print(f"Done! Audio file saved to {output_audio_file}")

    intro_outro_file = "resources/intro_outro.wav"
    if (intro or outro) and os.path.exists(intro_outro_file):
        try:
            main_audio = AudioSegment.from_file(output_audio_file)

            special_audio = AudioSegment.from_file(intro_outro_file)
            intro_length_ms = 4000

            special_audio = special_audio[:intro_length_ms]

            combined_audio = main_audio

            if intro:
                combined_audio = special_audio + main_audio[intro_length_ms:]
                print(f"Replaced first {intro_length_ms / 1000} seconds with intro.")

            if outro:
                combined_audio = combined_audio + special_audio
                print(f"Added outro to the end of the audio file.")

            combined_audio.export(output_audio_file, format="mp3")
            print(f"Final audio duration: {len(combined_audio) / 1000:.2f} seconds")
        except Exception as e:
            print(f"Error adding intro/outro: {e}")
            import traceback
            traceback.print_exc()
    elif intro or outro:
        print(f"Warning: Intro/outro file {intro_outro_file} not found, skipping.")

    shutil.rmtree(temp_dir)
    return output_audio_file
