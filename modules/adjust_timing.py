import json


def adjust_segments_timing(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = data.get('segments', [])

    MAX_GAP = 1.5

    for i in range(len(segments) - 1):
        current_end = segments[i]['end']
        next_start = segments[i + 1]['start']

        if next_start - current_end < MAX_GAP:
            segments[i]['end'] = next_start

    output_file = file_path.replace('.json', '_adjusted.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_file