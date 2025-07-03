#!/usr/bin/env python3
"""
Альтернативный скрипт с моделью pyannote/speaker-diarization (без версии)
"""

import json
import torch
import torchaudio
import warnings
from pathlib import Path
import argparse

# Отключаем предупреждения
warnings.filterwarnings("ignore")


def test_speaker_diarization_alternative(audio_path: str, token: str, output_path: str = None):
    """
    Тестирует speaker diarization с базовой моделью
    """
    try:
        from pyannote.audio import Pipeline

        print(f"🎵 Обрабатываю аудио: {audio_path}")

        # Проверяем наличие файла
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")

        # Пробуем загрузить базовую модель (которая у вас уже работала)
        print("📥 Загружаю модель pyannote/speaker-diarization...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=token
        )

        print("✅ Модель загружена успешно!")

        # Отправляем на GPU если доступен
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        print(f"🔧 Используется устройство: {device}")

        # Запускаем анализ
        print("🔄 Анализирую спикеров... (это может занять несколько минут)")
        diarization = pipeline(audio_path)

        print("✅ Анализ завершён! Обрабатываю результаты...")

        # Конвертируем результат
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
            "model": "pyannote/speaker-diarization",
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
        print(f"Модель: pyannote/speaker-diarization")
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

        # Показываем первые сегменты
        print(f"\n📋 Первые 15 сегментов:")
        for i, segment in enumerate(speakers_timeline[:15]):
            print(f"  {i + 1:2d}. {segment['start']:7.2f}s - {segment['end']:7.2f}s "
                  f"({segment['duration']:5.2f}s) | {segment['speaker']}")

        if len(speakers_timeline) > 15:
            print(f"  ... и ещё {len(speakers_timeline) - 15} сегментов")

        # Сохраняем результат
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Результат сохранён в: {output_path}")

        return result

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Speaker Diarization (альтернативная модель)")
    parser.add_argument("audio_path", help="Путь к аудио файлу")
    parser.add_argument("--token", required=True, help="HuggingFace токен")
    parser.add_argument("-o", "--output", help="Путь для сохранения JSON результата")

    args = parser.parse_args()

    # Тестируем diarization
    result = test_speaker_diarization_alternative(
        audio_path=args.audio_path,
        token=args.token,
        output_path=args.output
    )

    if result:
        print("\n✅ Анализ завершён успешно!")
        print(f"📊 Найдено {result['speakers_count']} спикеров в {result['total_segments']} сегментах")
        print("\n🔗 Результат готов для интеграции в ваш python-dubbing-service")
    else:
        print("\n❌ Анализ завершён с ошибкой")


if __name__ == "__main__":
    main()