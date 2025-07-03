import json
import sys
from pathlib import Path
import argparse


def install_requirements():
    """Установка необходимых зависимостей"""
    import subprocess

    packages = [
        "pyannote.audio",
    ]

    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package} уже установлен")
        except ImportError:
            print(f"Устанавливаю {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def test_diarization(audio_path: str, output_path: str = None):
    """
    Тестирует speaker diarization на аудио файле

    Args:
        audio_path: путь к аудио файлу
        output_path: путь для сохранения результата (опционально)
    """
    try:
        from pyannote.audio import Pipeline
        import torch

        print(f"🎵 Обрабатываю аудио: {audio_path}")

        # Проверяем наличие файла
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")

        # Загружаем предобученную модель
        print("📥 Загружаю модель speaker diarization...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=None  # Может потребоваться HuggingFace токен
        )

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
    parser.add_argument("--install", action="store_true", help="Установить зависимости")

    args = parser.parse_args()

    if args.install:
        install_requirements()
        return

    # Тестируем diarization
    result = test_diarization(args.audio_path, args.output)

    if result:
        print("\n✅ Тест завершён успешно!")
        print("\nТеперь можно интегрировать это в ваш python-dubbing-service")
    else:
        print("\n❌ Тест завершён с ошибкой")


if __name__ == "__main__":
    main()