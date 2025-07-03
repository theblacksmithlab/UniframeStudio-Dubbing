#!/usr/bin/env python3
"""
Базовый скрипт для тестирования speaker diarization
Использует PyAnnote.audio для определения спикеров в аудио файле
"""

import json
import sys
from pathlib import Path
import argparse


def install_requirements():
    """Установка необходимых зависимостей"""
    import subprocess

    packages = [
        "pyannote.audio",
        "torch",
        "torchaudio"
    ]

    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package} уже установлен")
        except ImportError:
            print(f"Устанавливаю {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def test_diarization(audio_path: str, output_path: str = None, hf_token: str = None):
    """
    Тестирует speaker diarization на аудио файле

    Args:
        audio_path: путь к аудио файлу
        output_path: путь для сохранения результата (опционально)
        hf_token: HuggingFace токен для доступа к моделям
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
        for model_name in models_to_try:
            try:
                print(f"📥 Пробую загрузить модель: {model_name}")
                pipeline = Pipeline.from_pretrained(
                    model_name,
                    use_auth_token=hf_token
                )
                print(f"✅ Модель {model_name} загружена успешно!")
                break
            except Exception as e:
                print(f"❌ Не удалось загрузить {model_name}: {e}")
                continue

        if pipeline is None:
            print("\n🔑 Все модели требуют авторизации!")
            print("Создайте токен на https://huggingface.co/settings/tokens")
            print("И примите условия на https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("Затем запустите: python speaker-test.py audio.wav --token YOUR_TOKEN")
            return None

        # Устанавливаем GPU если доступен
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = pipeline.to(device)
        print(f"🔧 Используется устройство: {device}")

        # Обрабатываем аудио
        print("🔄 Анализирую спикеров...")
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
            "total_duration": max([s["end"] for s in speakers_timeline]) if speakers_timeline else 0,
            "speakers_count": len(set([s["speaker"] for s in speakers_timeline])),
            "speakers_found": list(set([s["speaker"] for s in speakers_timeline])),
            "timeline": speakers_timeline
        }

        # Выводим результат
        print("\n" + "=" * 50)
        print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА СПИКЕРОВ")
        print("=" * 50)
        print(f"Общая продолжительность: {result['total_duration']:.2f} сек")
        print(f"Количество спикеров: {result['speakers_count']}")
        print(f"Найденные спикеры: {', '.join(result['speakers_found'])}")
        print("\n📋 Временная разметка:")

        for i, segment in enumerate(speakers_timeline[:10]):  # Показываем первые 10
            print(f"  {i + 1:2d}. {segment['start']:6.2f}s - {segment['end']:6.2f}s "
                  f"({segment['duration']:5.2f}s) | {segment['speaker']}")

        if len(speakers_timeline) > 10:
            print(f"  ... и ещё {len(speakers_timeline) - 10} сегментов")

        # Сохраняем результат в файл
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Результат сохранён в: {output_path}")

        return result

    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("Попробуйте установить зависимости:")
        print("pip install pyannote.audio torch torchaudio")
        return None

    except Exception as e:
        print(f"❌ Ошибка обработки: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Тест speaker diarization")
    parser.add_argument("audio_path", help="Путь к аудио файлу")
    parser.add_argument("-o", "--output", help="Путь для сохранения JSON результата")
    parser.add_argument("--token", help="HuggingFace токен для авторизации")
    parser.add_argument("--install", action="store_true", help="Установить зависимости")

    args = parser.parse_args()

    if args.install:
        install_requirements()
        return

    # Тестируем diarization
    result = test_diarization(args.audio_path, args.output, args.token)

    if result:
        print("\n✅ Тест завершён успешно!")
        print("\nТеперь можно интегрировать это в ваш python-dubbing-service")
    else:
        print("\n❌ Тест завершён с ошибкой")


if __name__ == "__main__":
    main()