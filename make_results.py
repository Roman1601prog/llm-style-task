"""Build RESULTS.md from real run artifacts; leave human conclusions explicit."""
import json
from common import ROOT, read_jsonl


def main():
    summary_path = ROOT / "REVIEW_SUMMARY.md"
    if not summary_path.exists():
        raise ValueError("Сначала заполните REVIEW.md и запустите review_scores.py")
    summary = summary_path.read_text(encoding="utf-8")
    production = (ROOT / "PRODUCTION.md").read_text(encoding="utf-8")
    production = production.split("## Проверено ли развёртывание")[0] + "## Первичные источники\n" + production.split("## Первичные источники", 1)[1]
    configs = {s: json.loads((ROOT / f"artifacts/train_{s}.json").read_text(encoding="utf-8")) for s in ["type1", "type2"]}
    data_report = json.loads((ROOT / "data/data_report.json").read_text(encoding="utf-8"))
    modes = ["baseline", "prompt-only", "adapted"]
    val = {m: {(r["style"], r["id"]): r for r in read_jsonl(ROOT / f"artifacts/{m}_validation.jsonl")} for m in modes}
    final = read_jsonl(ROOT / "artifacts/adapted_test.jsonl")
    if len(final) != 10 or any(len([r for r in final if r['style'] == s]) != 5 for s in configs):
        raise ValueError("В итоговой генерации должно быть 5 текстов на каждый стиль")
    md = ["# Результаты адаптации новостных постов", "", "## Архитектура", "", "Базовая модель: " + configs['type1']['model_lock']['model_id'] + ". Выбраны два отдельных QLoRA-адаптера: первый для короткой общественно-экономической подачи с допустимой иронией, второй для городского новостного формата. Обучение выполняется с замороженной NF4-базой, double quantization и assistant-only loss. Система сохраняет одинаковые ограничения на факты при обучении и инференсе. Веса модели не сливаются с адаптерами.", "", "Небольшая инструктивная модель ограничивает расходы GPU и время эксперимента. Два адаптера позволяют переключать стиль на уровне запроса и обслуживать оба на одной базе. По 20 примеров на стиль используются для обучения и по 5 для validation; пять финальных тестовых текстов в обучении не используются. Такая выборка позволяет проверить работоспособность подхода, но не даёт надёжной оценки качества на широком распределении новостей.", "", f"Документированных правок targets: {len(data_report['changes'])}. Исходные файлы сохранены в data/raw; основания правок — в data/target_overrides.json. Статус проверки DATA_REVIEW.md: **[ЗАПОЛНИТЬ]**.", "", "## Гиперпараметры и окружение", "", "| Параметр | type1 | type2 |", "|---|---|---|"]
    fields = [
        ("GPU", lambda c:c['gpu']), ("Compute dtype", lambda c:c['compute_dtype']),
        ("Epochs", lambda c:c['training_args']['num_train_epochs']), ("Learning rate", lambda c:c['training_args']['learning_rate']),
        ("Batch size", lambda c:c['training_args']['per_device_train_batch_size']), ("Gradient accumulation", lambda c:c['training_args']['gradient_accumulation_steps']),
        ("Rank", lambda c:c['lora']['r']), ("Alpha", lambda c:c['lora']['lora_alpha']), ("Dropout", lambda c:c['lora']['lora_dropout']),
        ("Max observed train tokens", lambda c:c['max_train_tokens']), ("Validation loss", lambda c:c['validation_metrics']['eval_loss']),
        ("Peak allocated GiB", lambda c:round(c['peak_allocated_gib'],3)), ("Process seconds", lambda c:round(c['total_seconds'],1)),
    ]
    for label, get in fields:
        md.append(f"| {label} | {get(configs['type1'])} | {get(configs['type2'])} |")
    md.extend(["", "Зафиксированная ревизия базы: `" + configs['type1']['model_lock']['revision'] + "`. Точные версии библиотек, полный chat_template, target_modules и параметры сохранены в artifacts/train_type1.json и artifacts/train_type2.json.", "", "Промпт задаётся функцией common.messages_for. При инференсе используется tokenizer.apply_chat_template с add_generation_prompt=True. Только токены ответа assistant и его завершение участвуют в loss; system, user и padding маскированы -100. Точная подпись первого стиля добавляется детерминированно при оформлении и не оценивается как приобретённый моделью навык.", "", "## Сравнение до и после", "", "Три примера ниже выбраны по фиксированному правилу из validation: первые два type1 и первый type2. Наивный baseline, сильный промпт без адаптера и сильный промпт с адаптером используют одну ревизию базы, одинаковое NF4-инференс-окружение и greedy decoding. Дополнительная колонка prompt-only помогает отличить эффект LoRA от эффекта промпта. Приводятся тела ответов без добавленной кодом подписи."])
    keys1 = [k for k in val['adapted'] if k[0] == 'type1']
    keys2 = [k for k in val['adapted'] if k[0] == 'type2']
    for index, key in enumerate(keys1[:2] + keys2[:1], 1):
        md.extend(["", f"### Пример {index}: {key[0]}, {key[1]}", "", "Исходник:", "", val['adapted'][key]['source']])
        for mode in modes:
            md.extend(["", f"**{mode}**", "", val[mode][key]['raw_text']])
        md.extend(["", "Вывод о фактах, стиле и структуре: **[ЗАПОЛНИТЬ]**."])
    md.extend(["", "## Общая оценка", "", "Заполнить по REVIEW.md: число ответов с существенными фактическими искажениями для каждого режима; средние оценки стиля, структуры и русского языка; наблюдаемый эффект LoRA относительно prompt-only. Если адаптер не улучшил качество, это нужно написать прямо. Validation loss сам по себе не подтверждает верность фактов или редакционный стиль.", "", "## Финальные генерации", "", "В outputs_type1.txt и outputs_type2.txt записано по пять результатов открытой модели. Финальные посты восстановлены из сохранённых TXT в artifacts/adapted_test.jsonl; исходные сырые test-ответы, диагностические подсказки и замеры инференса утрачены. Подробности восстановления — RECOVERY.md. Статус проверки всех десяти постов: **[ЗАПОЛНИТЬ]**.", "", "## Время и сложности", "", "В таблице приведено измеренное время процессов обучения. Полное затраченное человеком время: **[ЗАПОЛНИТЬ]**. Реальные сложности и способы решения: **[ЗАПОЛНИТЬ]**.", "", "## Инференс в продакшене", "", "Статус реального vLLM-развёртывания и стресс-теста: **[ЗАПОЛНИТЬ: план либо выполненный тест с GPU/версией/результатами]**.", "", (ROOT / "PRODUCTION.md").read_text(encoding="utf-8")])
    # Remove the starter-status paragraph; the actual deployment status is filled above.
    md[-1] = production
    md.extend(["", "## Восстановление после сброса Colab", "", (ROOT / "RECOVERY.md").read_text(encoding="utf-8")])
    instruction = "Заполнить по REVIEW.md: число ответов с существенными фактическими искажениями для каждого режима; средние оценки стиля, структуры и русского языка; наблюдаемый эффект LoRA относительно prompt-only. Если адаптер не улучшил качество, это нужно написать прямо. Validation loss сам по себе не подтверждает верность фактов или редакционный стиль."
    md[md.index(instruction)] = summary + "\n\nНаблюдаемый вклад LoRA относительно prompt-only и ограничения: **[ЗАПОЛНИТЬ]**."
    md.extend(["", "### Полная команда запуска", "", "```bash", (ROOT / "serve.sh").read_text(encoding="utf-8").strip(), "```"])
    (ROOT / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Создан RESULTS.md. Заполните отмеченные человеческие выводы и фактическое время перед сдачей.")


if __name__ == "__main__":
    main()

