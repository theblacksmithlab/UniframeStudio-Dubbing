#!/usr/bin/env python3
"""
Рабочий скрипт для speaker diarization на основе официальной документации
"""

import json
import torch
import torchaudio
import warnings
from pathlib import Path
import argparse

# Отключаем предупреждения для чистого вывода
warnings.filterwarnings("ignore")


def test_speaker_diarization(audio_path: str, token: str, output_path: str = None, num_speakers: int = None):
    """
    Тестирует speaker diarization используя официальный API

    Args:
        audio_path: путь к аудио файлу
        token: HuggingFace токен
        output_path: путь для сохранения результата
        num_speakers: количество спикеров (если известно)
    """
    try:
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.utils.hook import ProgressHook

        print(f"🎵 Обрабатываю аудио: {audio_path}")

        # Проверяем наличие файла
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")

        # Загружаем модель
        print("📥 Загружаю модель speaker-diarization-3.1...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )

        # Отправляем на GPU если доступен
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        print(f"🔧 Используется устройство: {device}")

        # Предзагружаем аудио для быстрой обработки
        print("🔄 Загружаю аудио в память...")
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_data = {"waveform": waveform, "sample_rate": sample_rate}

        # Готовим параметры для diarization
        diarization_params = {}
        if num_speakers:
            diarization_params["num_speakers"] = num_speakers
            print(f"🎙️ Ожидаемое количество спикеров: {num_speakers}")

        # Запускаем diarization с мониторингом прогресса
        print("🔄 Анализирую спикеров...")
        with ProgressHook() as hook:
            diarization = pipeline(audio_data, hook=hook, **diarization_params)

        # Конвертируем результат в удобный формат
        speakers_timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers_timeline.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,
                "duration": float(turn.end - turn.start)
            })

        # Сортируем по времени
        speakers_timeline.sort(key=lambda x: x["start"])

        # Подготавливаем результат
        speakers_found = sorted(list(set([s["speaker"] for s in speakers_timeline])))
        total_duration = max([s["end"] for s in speakers_timeline]) if speakers_timeline else 0

        result = {
            "audio_file": audio_path,
            "model": "pyannote/speaker-diarization-3.1",
            "total_duration": total_duration,
            "speakers_count": len(speakers_found),
            "speakers_found": speakers_found,
            "total_segments": len(speakers_timeline),
            "timeline": speakers_timeline
        }

        # Выводим результат
        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТЫ SPEAKER DIARIZATION")
        print("=" * 60)
        print(f"Файл: {audio_path}")
        print(f"Модель: pyannote/speaker-diarization-3.1")
        print(f"Общая продолжительность: {total_duration:.2f} сек")
        print(f"Количество спикеров: {len(speakers_found)}")
        print(f"Найденные спикеры: {', '.join(speakers_found)}")
        print(f"Всего сегментов: {len(speakers_timeline)}")

        # Статистика по спикерам
        print(f"\n📈 Статистика по спикерам:")
        for speaker in speakers_found:
            speaker_segments = [s for s in speakers_timeline if s['speaker'] == speaker]
            speaker_time = sum([s['duration'] for s in speaker_segments])
            percentage = (speaker_time / total_duration) * 100 if total_duration > 0 else 0
            print(f"  {speaker}: {speaker_time:.1f}s ({percentage:.1f}%) в {len(speaker_segments)} сегментах")

        # Показываем временную разметку
        print(f"\n📋 Временная разметка (первые 20 сегментов):")
        for i, segment in enumerate(speakers_timeline[:20]):
            print(f"  {i + 1:2d}. {segment['start']:7.2f}s - {segment['end']:7.2f}s "
                  f"({segment['duration']:5.2f}s) | {segment['speaker']}")

        if len(speakers_timeline) > 20:
            print(f"  ... и ещё {len(speakers_timeline) - 20} сегментов")

        # Сохраняем результат
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 JSON результат сохранён в: {output_path}")

        # Также сохраняем в RTTM формате (стандарт для diarization)
        rttm_path = Path(audio_path).with_suffix('.rttm')
        with open(rttm_path, "w") as rttm:
            diarization.write_rttm(rttm)
        print(f"📄 RTTM результат сохранён в: {rttm_path}")

        return result

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Speaker Diarization с официальным API")
    parser.add_argument("audio_path", help="Путь к аудио файлу")
    parser.add_argument("--token", required=True, help="HuggingFace токен")
    parser.add_argument("-o", "--output", help="Путь для сохранения JSON результата")
    parser.add_argument("--speakers", type=int, help="Ожидаемое количество спикеров")

    args = parser.parse_args()

    # Тестируем diarization
    result = test_speaker_diarization(
        audio_path=args.audio_path,
        token=args.token,
        output_path=args.output,
        num_speakers=args.speakers
    )

    if result:
        print("\n✅ Анализ завершён успешно!")
        print(f"📊 Найдено {result['speakers_count']} спикеров в {result['total_segments']} сегментах")
        print("\n🔗 Теперь можете интегрировать это в ваш python-dubbing-service")
        print("💡 Результат содержит точные временные метки для каждого спикера")
    else:
        print("\n❌ Анализ завершён с ошибкой")


if __name__ == "__main__":
    main()