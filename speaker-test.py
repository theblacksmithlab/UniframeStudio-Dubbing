#!/usr/bin/env python3
"""
Исправленный скрипт для тестирования speaker diarization
"""

import json
import sys
from pathlib import Path
import argparse


def test_diarization(audio_path: str, output_path: str = None, hf_token: str = None):
    """
    Тестирует speaker diarization на аудио файле
    """
    try:
        from pyannote.audio import Pipeline
        import torch

        print(f"🎵 Обрабатываю аудио: {audio_path}")

        # Проверяем наличие файла
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")

        # Список моделей для попытки загрузки
        models_to_try = [
            "pyannote/speaker-diarization-3.1",
            "pyannote/speaker-diarization",
            "pyannote/speaker-diarization-3.0"
        ]

        pipeline = None
        successful_model = None

        for model_name in models_to_try:
            try:
                print(f"📥 Пробую загрузить модель: {model_name}")
                pipeline = Pipeline.from_pretrained(
                    model_name,
                    use_auth_token=hf_token
                )
                successful_model = model_name
                print(f"✅ Модель {model_name} загружена успешно!")
                break
            except Exception as e:
                print(f"❌ Не удалось загрузить {model_name}: {str(e)[:100]}...")
                continue

        if pipeline is None:
            print("\n🔑 Все модели требуют авторизации или недоступны!")
            print("1. Создайте токен: https://huggingface.co/settings/tokens")
            print("2. Примите условия: https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("3. Попробуйте: python speaker-test.py audio.wav --token YOUR_TOKEN")
            return None

        # Устанавливаем GPU если доступен
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = pipeline.to(device)
        print(f"🔧 Используется устройство: {device}")
        print(f"🚀 Используется модель: {successful_model}")

        # Обрабатываем аудио
        print("🔄 Анализирую спикеров... (это может занять время)")
        diarization = pipeline(audio_path)

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
        result = {
            "model_used": successful_model,
            "total_duration": max([s["end"] for s in speakers_timeline]) if speakers_timeline else 0,
            "speakers_count": len(set([s["speaker"] for s in speakers_timeline])),
            "speakers_found": sorted(list(set([s["speaker"] for s in speakers_timeline]))),
            "timeline": speakers_timeline
        }

        # Выводим результат
        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА СПИКЕРОВ")
        print("=" * 60)
        print(f"Модель: {successful_model}")
        print(f"Общая продолжительность: {result['total_duration']:.2f} сек")
        print(f"Количество спикеров: {result['speakers_count']}")
        print(f"Найденные спикеры: {', '.join(result['speakers_found'])}")
        print(f"Всего сегментов: {len(speakers_timeline)}")
        print("\n📋 Временная разметка (первые 15 сегментов):")

        for i, segment in enumerate(speakers_timeline[:15]):
            print(f"  {i + 1:2d}. {segment['start']:7.2f}s - {segment['end']:7.2f}s "
                  f"({segment['duration']:5.2f}s) | {segment['speaker']}")

        if len(speakers_timeline) > 15:
            print(f"  ... и ещё {len(speakers_timeline) - 15} сегментов")

        # Статистика по спикерам
        print(f"\n📈 Статистика по спикерам:")
        for speaker in result['speakers_found']:
            speaker_segments = [s for s in speakers_timeline if s['speaker'] == speaker]
            total_time = sum([s['duration'] for s in speaker_segments])
            percentage = (total_time / result['total_duration']) * 100
            print(f"  {speaker}: {total_time:.1f}s ({percentage:.1f}%) в {len(speaker_segments)} сегментах")

        # Сохраняем результат в файл
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Результат сохранён в: {output_path}")

        return result

    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("Установите зависимости: pip install pyannote.audio torch torchaudio")
        return None

    except Exception as e:
        print(f"❌ Ошибка обработки: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Тест speaker diarization (исправленная версия)")
    parser.add_argument("audio_path", help="Путь к аудио файлу")
    parser.add_argument("-o", "--output", help="Путь для сохранения JSON результата")
    parser.add_argument("--token", help="HuggingFace токен для авторизации")

    args = parser.parse_args()

    # Тестируем diarization
    result = test_diarization(args.audio_path, args.output, args.token)

    if result:
        print("\n✅ Тест завершён успешно!")
        print("\n🔗 Теперь можно интегрировать это в ваш python-dubbing-service")
        print("💡 Результат содержит временные метки для каждого спикера")
    else:
        print("\n❌ Тест завершён с ошибкой")


if __name__ == "__main__":
    main()