import json


def adjust_segments_timing(file_path, output_file=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = data.get('segments', [])

    ADJUSTMENT_THRESHOLD = 0.5

    adjusted_count = 0

    for i in range(len(segments) - 1):
        current_end = segments[i]['end']
        next_start = segments[i + 1]['start']
        gap = next_start - current_end

        if 0 <= gap < ADJUSTMENT_THRESHOLD:
            old_end = segments[i]['end']
            segments[i]['end'] = next_start
            adjusted_count += 1
            print(f"Adjusted segment {i}: end time {old_end:.3f}s -> {next_start:.3f}s (gap was {gap:.3f}s)")
        else:
            print(f"Segment {i}: keeping gap of {gap:.3f}s")

    print(f"Timing adjustment complete. {adjusted_count} segments adjusted.")

    if output_file is None:
        output_file = file_path.replace('.json', '_adjusted.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_file
