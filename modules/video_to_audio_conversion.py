import os
import subprocess

INPUT_DIR = "input"
AUDIO_OUTPUT_DIR = "audio_input"

os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)


def extract_audio(input_path):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.3gp', '.m4v']

    if os.path.isdir(input_path):
        video_files = [f for f in os.listdir(input_path)
                       if os.path.isfile(os.path.join(input_path, f)) and
                       os.path.splitext(f)[1].lower() in video_extensions]

        if not video_files:
            print(f"No video files found in {input_path}")
            return None

        print(f"Found {len(video_files)} video files to process")
        extracted_files = []

        for video_file in video_files:
            video_path = os.path.join(input_path, video_file)
            print(f"Processing: {video_file}")
            result = _extract_single_audio(video_path)
            if result:
                extracted_files.append(result)

        print(f"Processed {len(extracted_files)} of {len(video_files)} video files successfully.")
        return extracted_files

    elif os.path.isfile(input_path):
        return _extract_single_audio(input_path)

    else:
        print(f"Error: Path {input_path} does not exist.")
        return None


def _extract_single_audio(input_video):
    if not os.path.exists(input_video):
        print(f"Error: Video file {input_video} not found.")
        return None

    base_name = os.path.splitext(os.path.basename(input_video))[0]
    output_audio = os.path.join(AUDIO_OUTPUT_DIR, f"{base_name}.mp3")

    command = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vn",
        "-codec:a", "libmp3lame",
        "-qscale:a", "2",
        "-ac", "1",
        "-ar", "24000",
        output_audio
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error during audio extraction: {result.stderr}")
            return None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None

    file_size_mb = os.path.getsize(output_audio) / (1024 * 1024)
    print(f"Audio successfully extracted: {output_audio} (Size: {file_size_mb:.2f} MB)")

    if file_size_mb > 25:
        print(f"WARNING: Audio file is still larger than 25MB ({file_size_mb:.2f} MB). You may need to split it.")

    return output_audio
