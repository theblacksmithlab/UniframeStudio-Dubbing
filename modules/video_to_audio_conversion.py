import os
import subprocess


def extract_audio(input_video_path, extracted_audio_path=None):
    if extracted_audio_path is None:
        base_name = os.path.splitext(os.path.basename(input_video_path))[0]
        output_dir = os.path.dirname(input_video_path)
        extracted_audio_path = os.path.join(output_dir, f"{base_name}.mp3")

    command = [
        "ffmpeg", "-y",
        "-i", input_video_path,
        "-vn",
        "-codec:a", "libmp3lame",
        "-qscale:a", "2",
        "-ac", "1",
        "-ar", "24000",
        extracted_audio_path
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error during audio extraction: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error executing ffmpeg command: {e}")
        return None

    if not os.path.exists(extracted_audio_path):
        print(f"Error: Audio file was not created at {extracted_audio_path}")
        return None

    file_size_mb = os.path.getsize(extracted_audio_path) / (1024 * 1024)
    print(f"Audio successfully extracted: {extracted_audio_path} (Size: {file_size_mb:.2f} MB)")

    if file_size_mb > 25:
        print(f"WARNING: Audio file is larger than 25MB ({file_size_mb:.2f} MB). "
              f"It will be split into chunks at the transcription step.")

    return extracted_audio_path
