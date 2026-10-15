"""Run naive baseline, strong prompt control, or both trained adapters."""
import argparse
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import PeftModel
from common import ROOT, DEFAULT_MODEL, cuda_dtype, diagnose, messages_for, read_jsonl, render_post, resolve_model_lock, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "prompt-only", "adapted"], required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=4096)
    args = parser.parse_args()
    set_seed(42)
    dtype = cuda_dtype()
    lock = resolve_model_lock(args.model)
    tokenizer = AutoTokenizer.from_pretrained(lock["model_id"], revision=lock["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(lock["model_id"], revision=lock["revision"], quantization_config=quant, torch_dtype=dtype, device_map={"": 0}, attn_implementation="sdpa")
    if args.mode == "adapted":
        for style in ("type1", "type2"):
            metadata = json.loads((ROOT / f"artifacts/train_{style}.json").read_text(encoding="utf-8"))
            if metadata["model_lock"] != lock:
                raise ValueError(f"{style}: адаптер обучался на другой ревизии базовой модели")
        model = PeftModel.from_pretrained(model, ROOT / "adapters/type1", adapter_name="type1", is_trainable=False)
        model.load_adapter(ROOT / "adapters/type2", adapter_name="type2", is_trainable=False)
    model.eval()
    model.config.use_cache = True
    records = []
    for style in ["type1", "type2"]:
        if args.mode == "adapted":
            model.set_adapter(style)
        if args.split == "test":
            rows = [{"id": r["id"], "input": r["text"]} for r in read_jsonl(ROOT / "data/raw/test_inputs.jsonl")]
        else:
            rows = read_jsonl(ROOT / f"data/processed/{style}_val.jsonl")
        style_posts = []
        for row in rows:
            messages = messages_for(style, row["input"], naive=args.mode == "baseline")
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
            prompt_tokens = inputs["input_ids"].shape[-1]
            if prompt_tokens + args.max_new_tokens > args.max_context:
                raise ValueError(f"{style}/{row['id']}: prompt={prompt_tokens}, output budget={args.max_new_tokens} > context={args.max_context}")
            before = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.05)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - before
            generated_ids = output[0, prompt_tokens:].tolist()
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            finish_reason = "stop" if generated_ids and generated_ids[-1] == tokenizer.eos_token_id else "length"
            post = render_post(style, raw_text)
            record = {"id": row["id"], "style": style, "mode": args.mode, "split": args.split, "source": row["input"], "reference": row.get("target"), "raw_text": raw_text, "text": post, "messages": messages, "chat_template": tokenizer.chat_template, "finish_reason": finish_reason, "warnings": diagnose(style, row["input"], raw_text, finish_reason), "prompt_tokens": prompt_tokens, "generated_tokens": len(generated_ids), "generation_seconds": seconds, "model_lock": lock, "generation_config": {"do_sample": False, "max_new_tokens": args.max_new_tokens, "repetition_penalty": 1.05, "compute_dtype": str(dtype), "quantization": "NF4"}}
            records.append(record)
            style_posts.append(f"===== id={row['id']} =====\n\n{post}\n")
            print(f"{style}/{row['id']}: {len(generated_ids)} токенов; подсказки={record['warnings']}")
        if args.mode == "adapted" and args.split == "test":
            (ROOT / f"outputs_{style}.txt").write_text("\n".join(style_posts), encoding="utf-8")
    path = ROOT / f"artifacts/{args.mode}_{args.split}.jsonl"
    write_jsonl(path, records)
    print(f"Сохранено: {path}")


if __name__ == "__main__":
    main()

