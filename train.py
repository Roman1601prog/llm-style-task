"""QLoRA SFT with an explicit assistant-only loss mask; one style per run."""
import argparse
import importlib.metadata
import json
import time
from datetime import datetime, timezone
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from common import ROOT, DEFAULT_MODEL, cuda_dtype, messages_for, read_jsonl, resolve_model_lock, write_json


class Pairs(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.items = []
        self.lengths = []
        for row in rows:
            prefix = tokenizer.apply_chat_template(messages_for(row["style"], row["input"]), tokenize=True, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(messages_for(row["style"], row["input"], row["target"]), tokenize=True, add_generation_prompt=False)
            if full[:len(prefix)] != prefix:
                raise ValueError(f"{row['id']}: префикс чат-шаблона не совпал")
            if len(full) > max_length:
                raise ValueError(f"{row['id']}: {len(full)} токенов > {max_length}. Увеличьте --max-length; исходник и ответ не обрезаются.")
            if len(full) <= len(prefix):
                raise ValueError("Пустая область assistant loss")
            self.items.append({"input_ids": full, "attention_mask": [1] * len(full), "labels": [-100] * len(prefix) + full[len(prefix):]})
            self.lengths.append(len(full))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        size = max(len(e["input_ids"]) for e in examples)
        result = {}
        for key, pad in (("input_ids", self.pad_id), ("attention_mask", 0), ("labels", -100)):
            result[key] = torch.tensor([e[key] + [pad] * (size - len(e[key])) for e in examples], dtype=torch.long)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=["type1", "type2"], required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--grad-acc", type=int, default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    set_seed(42)
    dtype = cuda_dtype()
    lock = resolve_model_lock(args.model)
    tokenizer = AutoTokenizer.from_pretrained(lock["model_id"], revision=lock["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_rows = read_jsonl(ROOT / f"data/processed/{args.style}_train.jsonl")
    val_rows = read_jsonl(ROOT / f"data/processed/{args.style}_val.jsonl")
    train_data = Pairs(train_rows, tokenizer, args.max_length)
    val_data = Pairs(val_rows, tokenizer, args.max_length)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(lock["model_id"], revision=lock["revision"], quantization_config=quant, torch_dtype=dtype, device_map={"": 0}, attn_implementation="sdpa")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(ROOT / f"artifacts/checkpoints_{args.style}"),
        num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_acc,
        warmup_ratio=0.1, lr_scheduler_type="cosine", weight_decay=0.01,
        max_grad_norm=1.0, optim="adamw_torch",
        fp16=dtype == torch.float16, bf16=dtype == torch.bfloat16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=1, report_to="none", seed=42, data_seed=42,
        dataloader_num_workers=0, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data, data_collator=Collator(tokenizer.pad_token_id), processing_class=tokenizer)
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    val_metrics = trainer.evaluate()
    adapter_dir = ROOT / f"adapters/{args.style}"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    versions = {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface-hub"]}
    config = {
        "style": args.style, "model_lock": lock, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": versions, "gpu": torch.cuda.get_device_name(0), "compute_dtype": str(dtype),
        "quantization": "NF4, double quantization", "lora": lora.to_dict(),
        "training_args": training_args.to_dict(), "train_count": len(train_data), "val_count": len(val_data),
        "max_train_tokens": max(train_data.lengths), "max_val_tokens": max(val_data.lengths),
        "chat_template": tokenizer.chat_template, "prompt_example": messages_for(args.style, "<нейтральная новость>"),
        "loss_mask": "system/user/padding=-100; only assistant tokens and its terminator contribute to loss",
        "metrics": result.metrics, "validation_metrics": val_metrics,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "total_seconds": time.perf_counter() - started,
    }
    write_json(ROOT / f"artifacts/train_{args.style}.json", config)
    write_json(ROOT / f"artifacts/log_{args.style}.json", trainer.state.log_history)
    print(f"Сохранён адаптер: {adapter_dir}")


if __name__ == "__main__":
    main()

