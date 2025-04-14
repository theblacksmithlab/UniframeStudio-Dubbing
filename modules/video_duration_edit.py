import json
import os
import shutil
import subprocess
from tqdm import tqdm

VIDEO_PATH = "../video_input/input.mp4"
SEGMENTS_JSON = "../output/timestamped_transcriptions/input_timestamped_corrected_cleaned_optimized_adjusted_translated.json"
SEGMENTS_DIR = "../video_segments"
FINAL_OUTPUT = "../output_final.mp4"
INTRO_OUTRO_FILE = "../resources/intro_outro_converted.mp4"

os.makedirs(SEGMENTS_DIR, exist_ok=True)


def load_segments(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
        return data.get("segments", [])

# def extract_segment(input_path, start_sec, end_sec, output_path):
#
#     cmd = [
#         "ffmpeg", "-y",
#         "-i", input_path,
#         "-ss", f"{start_sec:.6f}",  # Начальная точка
#         "-to", f"{end_sec:.6f}",  # Конечная точка (не длительность!)
#         "-c:v", "libx264", "-crf", "18",
#         "-an",  # Без аудио
#         output_path
#     ]
#
#     result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     if result.returncode != 0:
#         print(f"  Error extracting segment: {result.stderr.decode('utf-8')[:200]}")
#         return False
#
#     return True

def extract_segment(input_path, start_sec, end_sec, output_path):

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", f"{start_sec:.6f}",
        "-to", f"{end_sec:.6f}",
        "-force_key_frames", f"expr:gte(t,{start_sec})",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-an",
        output_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"  Error extracting segment: {result.stderr.decode('utf-8')[:200]}")
        return False

    return True

def adjust_speed(input_path, output_path, original_duration, target_duration):
    speed_factor = original_duration / target_duration

    print(
        f"  Adjusting speed: original={original_duration:.3f}s, target={target_duration:.3f}s, factor={speed_factor:.3f}")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter:v", f"setpts={1 / speed_factor:.6f}*PTS",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-an",
        output_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"  Error adjusting speed: {result.stderr.decode('utf-8')[:200]}")
        return False

    duration = get_video_duration(output_path)
    error = abs(duration - target_duration)
    print(f"  Result: duration={duration:.3f}s, error={error:.3f}s ({error / target_duration * 100:.1f}%)")

    return error < 0.1


# def concatenate_segments(segment_paths, output_path):
#     print(f"\nConcatenating {len(segment_paths)} segments...")
#
#     if os.path.exists(INTRO_OUTRO_FILE):
#         outro_path = os.path.join(SEGMENTS_DIR, "outro.mp4")
#         shutil.copy(INTRO_OUTRO_FILE, outro_path)
#         segment_paths.append(outro_path)
#         print(f"  Added outro from: {INTRO_OUTRO_FILE}")
#
#     list_file = os.path.join(SEGMENTS_DIR, "segments.txt")
#     with open(list_file, "w") as f:
#         for path in segment_paths:
#             if os.path.exists(path) and os.path.getsize(path) > 0:
#                 f.write(f"file '{os.path.abspath(path)}'\n")
#             else:
#                 print(f"  Warning: Skipping invalid segment {path}")
#
#     cmd = [
#         "ffmpeg", "-y",
#         "-f", "concat",
#         "-safe", "0",
#         "-i", list_file,
#         "-c", "copy",
#         output_path
#     ]
#
#     result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     if result.returncode != 0:
#         print(f"  Error concatenating segments: {result.stderr.decode('utf-8')[:200]}")
#         return False
#
#     print(f"Final video created: {output_path}")
#     return True

def concatenate_segments(segment_paths, output_path):
    print(f"\nConcatenating {len(segment_paths)} segments...")

    if os.path.exists(INTRO_OUTRO_FILE):
        outro_path = os.path.join(SEGMENTS_DIR, "outro.mp4")
        shutil.copy(INTRO_OUTRO_FILE, outro_path)
        segment_paths.append(outro_path)
        print(f"  Added outro from: {INTRO_OUTRO_FILE}")

    list_file = os.path.join(SEGMENTS_DIR, "segments.txt")
    with open(list_file, "w") as f:
        for path in segment_paths:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                f.write(f"file '{os.path.abspath(path)}'\n")
            else:
                print(f"  Warning: Skipping invalid segment {path}")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        output_path
    ]

    # # For ultimate quality
    # cmd = [
    #     "ffmpeg", "-y",
    #     "-f", "concat",
    #     "-safe", "0",
    #     "-i", list_file,
    #     "-c:v", "libx264", "-preset", "veryslow", "-crf", "16",
    #     "-pix_fmt", "yuv420p",
    #     "-movflags", "+faststart",
    #     "-an",
    #     output_path
    # ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"  Error concatenating segments: {result.stderr.decode('utf-8')[:200]}")
        return False

    print(f"Final video created: {output_path}")
    return True

def get_video_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        return float(result.stdout.strip())
    except ValueError:
        print(f"  Error getting duration: {result.stderr}")
        return 0.0

def process_video(json_path=None, convert_25fps=False):
    if json_path:
        global SEGMENTS_JSON
        SEGMENTS_JSON = json_path

    with open(SEGMENTS_JSON, "r") as f:
        data = json.load(f)
    segments = data.get("segments", [])

    print(f"Loaded {len(segments)} segments from JSON")

    output_paths = []
    segment_index = 0

    if segments and segments[0]["start"] > 0:
        intro_path = os.path.join(SEGMENTS_DIR, f"intro_{segment_index:03d}.mp4")
        print(f"\nExtracting intro from 0s to {segments[0]['start']}s")

        if extract_segment(VIDEO_PATH, 0, segments[0]["start"], intro_path):
            output_paths.append(intro_path)
            segment_index += 1

    prev_end = None

    for i, segment in enumerate(tqdm(segments, desc="Processing segments")):
        start = segment["start"]
        end = segment["end"]
        tts_duration = segment.get("tts_duration")

        if prev_end is not None and abs(start - prev_end) > 0.01:
            gap_path = os.path.join(SEGMENTS_DIR, f"gap_{segment_index:03d}.mp4")
            print(f"\nExtracting gap between segments from {prev_end:.3f}s to {start:.3f}s")

            if extract_segment(VIDEO_PATH, prev_end, start, gap_path):
                output_paths.append(gap_path)
                segment_index += 1

        print(f"\nProcessing segment {i} from {start:.3f}s to {end:.3f}s")

        original_segment = os.path.join(SEGMENTS_DIR, f"orig_{segment_index}.mp4")
        if not extract_segment(VIDEO_PATH, start, end, original_segment):
            print(f"  Failed to extract segment {i}")
            continue

        original_duration = get_video_duration(original_segment)
        if original_duration <= 0:
            print(f"  Invalid original duration for segment {i}")
            continue

        final_path = os.path.join(SEGMENTS_DIR, f"segment_{segment_index:03d}.mp4")
        if adjust_speed(original_segment, final_path, original_duration, tts_duration):
            output_paths.append(final_path)
            print(f"  Successfully processed segment {i}")
        else:
            print(f"  Warning: Could not achieve exact target duration for segment {i}")
            output_paths.append(final_path)

        os.remove(original_segment)

        prev_end = end
        segment_index += 1

    concatenate_segments(output_paths, FINAL_OUTPUT)

    if convert_25fps:
        print(f"\nConverting final video to 25 fps...")
        fps25_output = FINAL_OUTPUT.replace(".mp4", "_25fps.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-i", FINAL_OUTPUT,
            "-filter:v", "fps=fps=25",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            fps25_output
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"  Error converting to 25 fps: {result.stderr.decode('utf-8')[:200]}")
            return FINAL_OUTPUT

        print(f"Successfully converted to 25 fps: {fps25_output}")
        return fps25_output

    return FINAL_OUTPUT

if __name__ == "__main__":
    process_video()