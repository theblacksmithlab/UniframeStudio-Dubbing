import json
import os
import shutil
import subprocess
from tqdm import tqdm

VIDEO_PATH = "./video_input/input.mp4"
SEGMENTS_JSON = "./output/timestamped_transcriptions/input_timestamped_corrected_cleaned_optimized_adjusted_translated.json"
SEGMENTS_DIR = "./video_segments"
FINAL_OUTPUT = "./output_final.mp4"
INTRO_OUTRO_FILE = "./resources/intro_outro_converted.mp4"

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


def change_segment_duration(input_path, output_path, target_duration):
    """
    Принудительно устанавливает длительность видео с максимальной точностью
    """
    print(f"  Forcing exact duration: target={target_duration:.6f}s")

    # Получаем длительность исходного видео
    probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", input_path]
    result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    original_duration = float(result.stdout.strip())

    # Получаем информацию о кадрах
    fps_cmd = ["ffprobe", "-v", "error", "-select_streams", "v",
               "-show_entries", "stream=r_frame_rate,nb_frames", "-of", "json", input_path]
    result = subprocess.run(fps_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)

    fps_str = info["streams"][0]["r_frame_rate"]
    if '/' in fps_str:
        num, den = map(int, fps_str.split('/'))
        fps = num / den
    else:
        fps = float(fps_str)

    # Рассчитываем точное количество кадров для целевой длительности
    target_frames = int(round(target_duration * fps))
    output_fps = target_frames / target_duration

    print(f"  Input: {original_duration:.6f}s at {fps:.2f}fps")
    print(f"  Output: {target_duration:.6f}s with {target_frames} frames at {output_fps:.6f}fps")

    # Временный файл с извлеченными кадрами
    temp_dir = f"{output_path}.frames"
    os.makedirs(temp_dir, exist_ok=True)

    # Извлекаем все кадры
    extract_cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vsync", "0",
        f"{temp_dir}/frame_%05d.png"
    ]

    subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Определяем, сколько кадров извлечено
    frames = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])

    if not frames:
        print("  Error: No frames extracted!")
        return False

    print(f"  Extracted {len(frames)} frames")

    # Определяем, нужно ли добавить или удалить кадры
    if len(frames) > target_frames:
        # Нужно удалить кадры
        to_remove = len(frames) - target_frames
        if to_remove > 0:
            print(f"  Removing {to_remove} frames to match target duration")
            # Равномерно удаляем кадры
            keep_ratio = target_frames / len(frames)
            to_keep = [frames[int(i / keep_ratio)] for i in range(target_frames)]

            # Удаляем неиспользуемые кадры
            for frame in frames:
                if frame not in to_keep:
                    os.remove(os.path.join(temp_dir, frame))

    elif len(frames) < target_frames:
        # Нужно добавить кадры (дублировать)
        to_add = target_frames - len(frames)
        if to_add > 0:
            print(f"  Adding {to_add} frames to match target duration")
            # Равномерно дублируем кадры
            step = len(frames) / to_add

            for i in range(to_add):
                frame_idx = min(int(i * step), len(frames) - 1)
                src = os.path.join(temp_dir, frames[frame_idx])
                dst = os.path.join(temp_dir, f"dup_{i:05d}.png")
                shutil.copy(src, dst)

    # Создаем видео с точным fps
    concat_cmd = [
        "ffmpeg", "-y",
        "-framerate", f"{output_fps}",
        "-pattern_type", "glob",
        "-i", f"{temp_dir}/*.png",
        "-c:v", "libx264",
        "-preset", "veryslow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",  # Без аудио
        output_path
    ]

    subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Очищаем временную директорию
    shutil.rmtree(temp_dir)

    # Проверяем результат
    check_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", output_path]
    result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.stdout.strip():
        final_duration = float(result.stdout.strip())
        error = abs(final_duration - target_duration)
        print(
            f"  Final result: duration={final_duration:.6f}s, error={error:.6f}s ({error / target_duration * 100:.4f}%)")
        return error < 0.1
    else:
        print("  Error: Could not determine final duration")
        return False

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

    # Проверка всех сегментов перед конкатенацией
    valid_segments = []
    total_expected_duration = 0

    for i, path in enumerate(segment_paths):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            # Проверка длительности и fps
            probe_cmd = ["ffprobe", "-v", "error", "-show_entries",
                         "format=duration:stream=r_frame_rate", "-of", "json", path]
            result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            try:
                info = json.loads(result.stdout)
                duration = float(info["format"]["duration"])
                fps = info["streams"][0]["r_frame_rate"]

                print(f"  Segment {i}: {os.path.basename(path)}, duration={duration:.3f}s, fps={fps}")
                valid_segments.append(path)
                total_expected_duration += duration

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"  Warning: Unable to analyze segment {i}: {e}")
                valid_segments.append(path)
        else:
            print(f"  Warning: Skipping invalid segment {path}")

    print(f"  Total expected duration: {total_expected_duration:.3f} seconds")

    list_file = os.path.join(SEGMENTS_DIR, "segments.txt")
    with open(list_file, "w") as f:
        for path in valid_segments:
            f.write(f"file '{os.path.abspath(path)}'\n")

    # Используем -c copy для сохранения оригинальных потоков без перекодирования
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",  # Копируем потоки без перекодирования
        "-an",  # Без аудио
        output_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Если простая конкатенация не работает, пробуем с перекодированием
    if result.returncode != 0:
        print(f"  Simple concatenation failed, trying with re-encoding...")

        cmd_encode = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-vsync", "vfr",  # Важно для сохранения временных меток
            "-an",
            output_path
        ]

        result = subprocess.run(cmd_encode, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            error_message = result.stderr.decode('utf-8')
            print(f"  Error concatenating segments: {error_message[:200]}...")

            # Сохраняем полный лог ошибки для анализа
            error_log = os.path.join(SEGMENTS_DIR, "concat_error.log")
            with open(error_log, "w") as f:
                f.write(error_message)
            print(f"  Full error log saved to: {error_log}")

            return False

    # Проверяем итоговую длительность
    check_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", output_path]
    result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.stdout.strip():
        final_duration = float(result.stdout.strip())
        print(f"Final video created: {output_path}, duration={final_duration:.3f}s")

        # Проверяем отклонение от ожидаемой длительности
        if abs(final_duration - total_expected_duration) > 5:  # Допустимая погрешность 5 секунд
            print(
                f"WARNING: Final duration ({final_duration:.3f}s) significantly differs from expected ({total_expected_duration:.3f}s)")
    else:
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
        if change_segment_duration(original_segment, final_path, tts_duration):
            output_paths.append(final_path)
            print(f"  Successfully processed segment {i}")
        else:
            print(f"  Warning: Failed to process segment {i}")

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