"""Inspect supplied pairs, apply explicit target overrides, split reproducibly."""
import json
import random
from common import ROOT, read_jsonl, write_jsonl, write_json, text_hash, diagnose, strip_channel


def main():
    override_path = ROOT / "data/target_overrides.json"
    overrides = json.loads(override_path.read_text(encoding="utf-8"))
    seen_overrides = set()
    test_rows = read_jsonl(ROOT / "data/raw/test_inputs.jsonl")
    if len(test_rows) != 5 or len({r["id"] for r in test_rows}) != 5:
        raise ValueError("Ожидалось 5 уникальных тестовых id")
    for r in test_rows:
        if not isinstance(r.get("text"), str) or not r["text"].strip():
            raise ValueError("Пустой или неверный тестовый исходник")
    test_sources = {text_hash(r["text"].strip()) for r in test_rows}
    report = {"seed": 42, "test_count": len(test_rows), "styles": {}, "changes": []}
    review = ["# Проверка обучающих пар", "", "Сверьте смысл каждой пары; цифровые предупреждения — только подсказки.", ""]
    for style in ("type1", "type2"):
        raw = read_jsonl(ROOT / f"data/raw/train_{style}.jsonl")
        if len(raw) != 25:
            raise ValueError(f"{style}: ожидалось 25 строк")
        rows = []
        hashes = set()
        for index, row in enumerate(raw, 1):
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("input", "output", "instruction")):
                raise ValueError(f"{style}, строка {index}: неверная схема")
            row_id = f"{style}-{index:03d}"
            source = row["input"].strip()
            digest = text_hash(source)
            if digest in hashes or digest in test_sources:
                raise ValueError(f"{row_id}: дубль или утечка тестового текста")
            hashes.add(digest)
            output = row["output"].strip()
            if row_id in overrides:
                change = overrides[row_id]
                if change["source_sha256"] != digest:
                    raise ValueError(f"{row_id}: правка относится к другому исходнику")
                output = change["target"].strip()
                seen_overrides.add(row_id)
                report["changes"].append({"id": row_id, "reason": change["reason"]})
            target = strip_channel(output)
            item = {"id": row_id, "style": style, "input": source, "target": target, "source_sha256": digest}
            rows.append(item)
            review.extend([f"## {row_id}", "", "Исходник:", "", source, "", "Цель после документированных правок:", "", target, "", "Подсказки: " + str(diagnose(style, source, target)), "", "Ручная проверка: [ ] выполнена", ""])
        order = list(range(len(rows)))
        random.Random(42).shuffle(order)
        validation_ids = set(order[:5])
        train = [r for i, r in enumerate(rows) if i not in validation_ids]
        val = [r for i, r in enumerate(rows) if i in validation_ids]
        write_jsonl(ROOT / f"data/processed/{style}_train.jsonl", train)
        write_jsonl(ROOT / f"data/processed/{style}_val.jsonl", val)
        report["styles"][style] = {"train_count": len(train), "val_count": len(val), "train_ids": [r["id"] for r in train], "val_ids": [r["id"] for r in val]}
    unused = set(overrides) - seen_overrides
    if unused:
        raise ValueError(f"Неизвестные id в overrides: {unused}")
    write_json(ROOT / "data/data_report.json", report)
    (ROOT / "DATA_REVIEW.md").write_text("\n".join(review), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

