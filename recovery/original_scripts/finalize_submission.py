"""Validate a completed run and export full backup plus browser-sized GitHub files."""
import argparse
import json
import re
import zipfile
from pathlib import Path
from common import ROOT, read_jsonl
from review_scores import summarize_review


def validate_completed_run():
    required = ["outputs_type1.txt", "outputs_type2.txt", "RESULTS.md", "REVIEW.md", "artifacts/model_lock.json", "artifacts/train_type1.json", "artifacts/train_type2.json", "artifacts/adapted_test.jsonl", "adapters/type1/adapter_model.safetensors", "adapters/type2/adapter_model.safetensors"]
    for relative in required:
        if not (ROOT / relative).is_file():
            raise ValueError(f"Не найден {relative}: выполните соответствующий этап перед упаковкой")
    report = (ROOT / "RESULTS.md").read_text(encoding="utf-8")
    if "[ЗАПОЛНИТЬ" in report:
        raise ValueError("В RESULTS.md остались [ЗАПОЛНИТЬ]. Отредактируйте отчёт и загрузите его обратно в Colab")
    for style in ("type1", "type2"):
        text = (ROOT / f"outputs_{style}.txt").read_text(encoding="utf-8")
        chunks = re.split(r"^===== id=(\d+) =====\s*$", text, flags=re.MULTILINE)
        ids = [int(value) for value in chunks[1::2]]
        bodies = chunks[2::2]
        if ids != [1, 2, 3, 4, 5] or len(bodies) != 5 or any(not body.strip() for body in bodies):
            raise ValueError(f"outputs_{style}.txt: нужны пять непустых постов id=1…5")
    rows = read_jsonl(ROOT / "artifacts/adapted_test.jsonl")
    if len(rows) != 10 or {(r["style"], r["id"]) for r in rows} != {(s, i) for s in ("type1", "type2") for i in range(1, 6)}:
        raise ValueError("В artifacts/adapted_test.jsonl должны быть десять уникальных результатов")
    lock = json.loads((ROOT / "artifacts/model_lock.json").read_text(encoding="utf-8"))
    for style in ("type1", "type2"):
        metadata = json.loads((ROOT / f"artifacts/train_{style}.json").read_text(encoding="utf-8"))
        if metadata["model_lock"] != lock:
            raise ValueError("Адаптеры обучались на разных моделях или ревизиях")
    summarize_review()
    return lock


def completed_readme(lock):
    return f'''# Адаптация новостей под два Telegram-стиля

Модель: `{lock["model_id"]}`. Ревизия: `{lock["revision"]}`. Используются два QLoRA-адаптера, обучение через Transformers/PEFT в Google Colab с GPU. Данные разделены на 20 train и 5 validation-примеров для каждого стиля; пять финальных исходников не участвуют в обучении.

Результаты: `outputs_type1.txt`, `outputs_type2.txt`, `RESULTS.md`. Ручная оценка трёх режимов — `REVIEW.md` и `REVIEW_SUMMARY.md`. Точные параметры, промпты, версии и сырые генерации — в `artifacts/`. Разбор vLLM — в `PRODUCTION.md`, полная команда — `serve.sh`. Фактический статус GPU-развёртывания указан в RESULTS.md.

## Воспроизведение

Скачайте репозиторий через Code → Download ZIP. Откройте `style_task_colab.ipynb` в Google Colab, выберите GPU и загрузите ZIP в первую ячейку. Затем выполняйте ячейки по порядку: установка → подготовка и проверка данных → baseline → prompt-only → обучение type1/type2 → validation → ручная оценка → финальные генерации → отчёт.

Подробная инструкция для Windows и браузера: `GUIDE_RU.md`.

## Веса

В полном локальном архиве `submission.zip` сохранены обученные адаптеры. В архиве для браузерной загрузки на GitHub `github_upload.zip` бинарные веса исключены; код, конфигурации и данные позволяют их воспроизвести. При предоставлении проверяющему отдельного архива адаптеров добавьте ссылку или информацию о приложении здесь.

Полные базовые веса Qwen скачиваются по указанной ревизии и в репозиторий не включаются. Промежуточные checkpoints и Python-окружения исключены из экспорта.
'''


def export_github(destination):
    count = 0
    excluded = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(part in {".git", "__pycache__", ".ipynb_checkpoints", "hf_cache"} or part.startswith(".venv") or part.startswith("checkpoints_") for part in relative.parts):
                continue
            if path.suffix in {".zip", ".pyc", ".log"}:
                continue
            if path.suffix in {".safetensors", ".bin", ".pt", ".pth"} or path.stat().st_size > 24 * 1024**2:
                excluded.append(str(relative))
                continue
            archive.write(path, relative)
            count += 1
        archive.writestr("GITHUB_EXPORT.txt", "Бинарные веса и слишком большие файлы исключены:\n" + "\n".join(excluded) + "\n")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/content")
    args = parser.parse_args()
    lock = validate_completed_run()
    destination = Path(args.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (ROOT / "README.md").write_text(completed_readme(lock), encoding="utf-8")
    import subprocess
    import sys
    subprocess.run([sys.executable, str(ROOT / "package_submission.py"), "--output", str(destination / "submission.zip")], check=True)
    count = export_github(destination / "github_upload.zip")
    print(f"Готово: {destination / 'submission.zip'} — полный проект с адаптерами")
    print(f"Готово: {destination / 'github_upload.zip'} — {count} файлов для GitHub через браузер")
    print("Проверено заполнение оценок, отсутствие placeholders и число постов. Смысловая проверка фактов остаётся человеческой.")


if __name__ == "__main__":
    main()

