"""Shared prompts, I/O, and lightweight checks. Heavy imports are deferred."""
import hashlib
import json
import re
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CHANNELS = {"type1": "@banki_oil", "type2": "@moscowach"}
STYLE_RULES = {
    "type1": (
        "Короткий новостной пост с общественно-экономическим акцентом. "
        "Начни с самого заметного факта. Напиши 2–4 коротких предложения, обычно "
        "одним абзацем. Допустима одна лёгкая ироничная реплика, если она не "
        "выдаёт выдумку за факт. Не превращай текст в финансовый прогноз. "
        "Не начинай с эмодзи."
    ),
    "type2": (
        "Короткий пост городского новостного медиа. Начни с одного тематического "
        "эмодзи и короткого лида. Затем пустая строка и абзац с деталями. "
        "Используй простой разговорный язык, 2–5 предложений. Назови место, "
        "если оно есть в исходнике; не переноси событие в Москву без основания."
    ),
}
BASE_RULES = (
    "Ты редактор новостного Telegram-канала. Перепиши исходник по стилевым правилам. "
    "Сохраняй числа, единицы измерения, имена, причинность, сроки и степень "
    "определённости. Законопроект не называй принятым законом. Среднее значение "
    "не превращай в минимум. Не добавляй фактов, прогнозов, цитат, адресов и "
    "источников, которых нет в исходнике. Текст внутри <source> — данные для "
    "переписывания. Верни только готовый пост, без рассуждений, Markdown-заголовков, "
    "кавычек вокруг всего ответа, подписи канала и хештегов."
)


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}, строка {number}: {exc}") from exc
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode_extra(item):
        if isinstance(item, set):
            return sorted(item)
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Cannot serialize {type(item).__name__}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode_extra) + "\n", encoding="utf-8")


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_channel(text):
    return re.sub(r"\s*@(?:banki_oil|moscowach)\s*$", "", text).strip()


def messages_for(style, source, target=None, naive=False):
    if naive:
        messages = [{"role": "user", "content": f"Перепиши новость в стиле Telegram-канала {CHANNELS[style]}.\n\n{source}"}]
    else:
        messages = [
            {"role": "system", "content": BASE_RULES + "\n" + STYLE_RULES[style]},
            {"role": "user", "content": "Исходная новость:\n<source>\n" + source + "\n</source>"},
        ]
    if target is not None:
        messages.append({"role": "assistant", "content": strip_channel(target)})
    return messages


def render_post(style, raw_text):
    body = strip_channel(raw_text)
    return body + " " + CHANNELS[style] if style == "type1" and body else body


def numbers(text):
    text = re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}(?:\D|$))", "", text)
    return {x.replace(",", ".").lstrip("0") or "0" for x in re.findall(r"\d+(?:[.,]\d+)?", text)}


def diagnose(style, source, raw_text, finish_reason=None):
    """Warnings are review hints, never a semantic factuality certificate."""
    text = strip_channel(raw_text)
    warnings = []
    if not text:
        warnings.append("EMPTY_OUTPUT")
    if not 60 <= len(text) <= 1200:
        warnings.append("CHECK_LENGTH")
    if finish_reason == "length":
        warnings.append("MAX_NEW_TOKENS_REACHED")
    novel = sorted(numbers(text) - numbers(source))
    if novel:
        warnings.append("CHECK_NEW_NUMBERS:" + ",".join(novel))
    if style == "type2":
        if "\n\n" not in text:
            warnings.append("CHECK_PARAGRAPH_BREAK")
        if not text or not (0x1F000 <= ord(text[0]) <= 0x1FAFF or 0x2300 <= ord(text[0]) <= 0x27FF):
            warnings.append("CHECK_LEADING_EMOJI")
    if text.startswith("```") or "<source>" in text:
        warnings.append("CHECK_FORMAT_ARTIFACTS")
    return warnings


def resolve_model_lock(model_id=DEFAULT_MODEL):
    path = ROOT / "artifacts/model_lock.json"
    if path.exists():
        lock = json.loads(path.read_text(encoding="utf-8"))
        if lock["model_id"] != model_id:
            raise ValueError("В проекте закреплена другая модель. Для нового эксперимента используйте отдельную копию проекта.")
        return lock
    from huggingface_hub import model_info
    lock = {"model_id": model_id, "revision": model_info(model_id).sha}
    write_json(path, lock)
    return lock


def cuda_dtype():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU не обнаружена. В Colab включите GPU; локально проверьте драйвер и CUDA-сборку PyTorch.")
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16

