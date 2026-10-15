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
    path = ROOT / 'README.md'
    if path.exists():
        return path.read_text(encoding='utf-8')
    return '# Адаптация новостей под два Telegram-стиля\n\nМодель: `Qwen/Qwen2.5-1.5B-Instruct`, ревизия `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Два независимых QLoRA-адаптера обучены через Transformers/PEFT на Tesla T4 в Google Colab. Для каждого стиля выделено 20 train- и 5 validation-пар; пять финальных исходников не участвовали в обучении. Гиперпараметры и чат-шаблон зафиксированы в `RESULTS.md` и `artifacts/train_type*.json`.\n\n## Результаты\n\n- `outputs_type1.txt` и `outputs_type2.txt`: по пять исходных финальных генераций.\n- `RESULTS.md`: архитектура, три сравнения baseline/prompt-only/LoRA, оценки, ограничения, время и план vLLM для GPU с 24 GB.\n- `REVIEW.md`, `REVIEW_SUMMARY.md`: оценки 30 validation-ответов. Средняя структура улучшилась с 1.6/5 до 4.3/5, стиль — с 1.3/5 до 2.5/5. Фактические искажения остались в 4 из 10 ответов каждого режима; улучшение достоверности не подтверждено.\n- В десяти финальных постах отмечены три существенные фактические ошибки и китайский лид в одном ответе. Текущие результаты требуют редакторской проверки; подробно разобраны в `RESULTS.md`.\n\nПосле сброса Colab часть текстовых записей восстановлена из сохранённых REVIEW и TXT; неизвестная телеметрия оставлена `null`. Происхождение записей описано в `RECOVERY.md`. Веса, параметры обучения и исходные baseline/prompt-only-журналы сохранены.\n\n## Воспроизведение\n\nРаботайте в отдельной копии проекта: подготовка данных, генерация и составление отчёта записывают файлы с результатами.\n\n1. Скачайте репозиторий через **Code → Download ZIP**.\n2. Откройте `style_task_colab.ipynb` в Google Colab, выберите среду с NVIDIA GPU и загрузите ZIP в первую ячейку.\n3. Выполняйте ячейки по порядку: установка → подготовка и проверка данных → baseline → prompt-only → обучение двух адаптеров → validation → оценка → test → отчёт.\n\n`executed_colab.ipynb` содержит сохранённый ход исходной работы, в том числе промежуточные ошибки загрузки и восстановления. Финальные файлы проверены отдельно при упаковке. Инструкция для Windows и браузера — `GUIDE_RU.md`; сводка проверки комплекта — `PACKAGING_CHECK.md`.\n\n## Обученные веса\n\nПолный **`submission.zip`** содержит оба исходных `adapter_model.safetensors` и весь проект в папке `llm-style-task/`. Этот архив предназначен для приложения к **GitHub Release** либо отдельной передачи проверяющему. После скачивания распакуйте его; файлы адаптеров находятся в `llm-style-task/adapters/type1/` и `llm-style-task/adapters/type2/`.\n\nВ **`github_upload.zip`** бинарные веса исключены для загрузки через браузер. Его распакованное содержимое загружается в корень репозитория. Базовые веса Qwen скачиваются по закреплённой ревизии и в архивы не входят. Промежуточные checkpoints и Python-окружения исключены из экспорта.\n\n## vLLM\n\nПлан, расчёт памяти и проверка под нагрузкой — `PRODUCTION.md`; команда с двумя LoRA — `serve.sh`. Запускать в отдельной Linux-среде с зависимостями из `requirements-serve.txt`. Для расчёта выбраны 15 активных последовательностей и 4096 токенов суммарно на промпт и ответ. vLLM-сервер и нагрузочный тест на GPU с 24 GB в рамках этого эксперимента не запускались.\n'



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

