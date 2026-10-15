"""Create the human-review worksheet from actual generated validation texts."""
from common import ROOT, read_jsonl


def main():
    modes = ["baseline", "prompt-only", "adapted"]
    rows = {mode: {(r["style"], r["id"]): r for r in read_jsonl(ROOT / f"artifacts/{mode}_validation.jsonl")} for mode in modes}
    if not all(set(rows[m]) == set(rows[modes[0]]) for m in modes):
        raise ValueError("В режимах отличаются проверочные id")
    md = ["# Ручная оценка validation", "", "Факты: 0/1 (есть существенное искажение / нет обнаруженного искажения). Стиль, структура и русский язык: 1–5. Заполните оценки самостоятельно, читая исходник.", "", "| Стиль | ID | Режим | Факты 0/1 | Стиль 1–5 | Структура 1–5 | Язык 1–5 |", "|---|---|---|---|---|---|---|"]
    for key in rows[modes[0]]:
        for mode in modes:
            md.append(f"| {key[0]} | {key[1]} | {mode} |  |  |  |  |")
    for key in rows[modes[0]]:
        source = rows[modes[0]][key]["source"]
        md.extend(["", f"## {key[0]} / {key[1]}", "", "Исходник:", "", source])
        for mode in modes:
            record = rows[mode][key]
            md.extend(["", f"### {mode}", "", record["raw_text"], "", "Подсказки: " + str(record["warnings"])])
    (ROOT / "REVIEW.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Создан REVIEW.md: оценки не заполнены автоматически.")


if __name__ == "__main__":
    main()

