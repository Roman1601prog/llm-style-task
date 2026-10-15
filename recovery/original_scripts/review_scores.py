"""Validate human scores and summarize them without inventing assessments."""
from common import ROOT, read_jsonl, write_json

MODES = ("baseline", "prompt-only", "adapted")


def parse_scores(text, expected_keys):
    records = []
    seen = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if not line.lstrip().startswith("|") or not fields or fields[0] not in {"type1", "type2"}:
            continue
        if len(fields) != 7:
            raise ValueError(f"REVIEW.md, строка {line_number}: оставьте 7 столбцов исходной таблицы")
        style, row_id, mode = fields[:3]
        key = (style, row_id, mode)
        if key not in expected_keys or key in seen:
            raise ValueError(f"REVIEW.md, строка {line_number}: неизвестная или повторная запись {key}")
        try:
            facts, style_score, structure, language = map(int, fields[3:])
        except ValueError as exc:
            raise ValueError(f"REVIEW.md, строка {line_number}: заполните все четыре оценки целыми числами") from exc
        if facts not in (0, 1) or any(not 1 <= score <= 5 for score in (style_score, structure, language)):
            raise ValueError(f"REVIEW.md, строка {line_number}: факты 0/1, остальные оценки 1–5")
        records.append({"style": style, "id": row_id, "mode": mode, "facts": facts, "style_score": style_score, "structure": structure, "language": language})
        seen.add(key)
    if seen != expected_keys:
        raise ValueError(f"REVIEW.md: отсутствуют оценки для {len(expected_keys - seen)} записей")
    return records


def summarize_review():
    expected = set()
    for mode in MODES:
        rows = read_jsonl(ROOT / f"artifacts/{mode}_validation.jsonl")
        if len(rows) != 10:
            raise ValueError(f"{mode}: ожидалось 10 validation-ответов")
        expected.update((r["style"], str(r["id"]), mode) for r in rows)
    if len(expected) != 30:
        raise ValueError("В validation есть повторные идентификаторы")
    records = parse_scores((ROOT / "REVIEW.md").read_text(encoding="utf-8"), expected)
    summary = []
    for mode in MODES:
        rows = [row for row in records if row["mode"] == mode]
        summary.append({"mode": mode, "count": len(rows), "factual_errors": sum(row["facts"] == 0 for row in rows), "mean_style": round(sum(row["style_score"] for row in rows) / len(rows), 2), "mean_structure": round(sum(row["structure"] for row in rows) / len(rows), 2), "mean_language": round(sum(row["language"] for row in rows) / len(rows), 2)})
    lines = ["# Сводная ручная оценка", "", "Оценки поставлены человеком относительно предоставленных исходников. Скрипт только проверяет заполнение и считает сводные значения. Выборка — 10 validation-случаев на режим.", "", "| Режим | Ответов | Существенных искажений | Средний стиль | Средняя структура | Средний язык |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['mode']} | {row['count']} | {row['factual_errors']} | {row['mean_style']} | {row['mean_structure']} | {row['mean_language']} |")
    text = "\n".join(lines) + "\n"
    (ROOT / "REVIEW_SUMMARY.md").write_text(text, encoding="utf-8")
    write_json(ROOT / "artifacts/review_scores.json", {"scores": records, "summary": summary})
    return text


if __name__ == "__main__":
    print(summarize_review())

