import json


def adjust_segments_timing(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = data.get('segments', [])

    MAX_GAP = 1.5
    WARNING_GAP = 4.0

    for i in range(len(segments) - 1):
        current_end = segments[i]['end']
        next_start = segments[i + 1]['start']
        gap = next_start - current_end

        if next_start - current_end < MAX_GAP:
            segments[i]['end'] = next_start
        elif gap <= WARNING_GAP:
            print(f"Segment {i}: Regular intentional gap detected: {gap:.2f}s")
        else:
            print(f"WARNING! Segment {i}: Unusually large gap detected: {gap:.2f}s between segments")

    output_file = file_path.replace('.json', '_adjusted.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_file
