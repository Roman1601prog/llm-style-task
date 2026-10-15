"""A bounded synthetic stress test. Run separately from the vLLM server."""
import argparse
import asyncio
import json
import math
import time
import httpx
from transformers import AutoTokenizer
from common import ROOT, messages_for


def near_limit_messages(tokenizer, style, target_tokens):
    # This is a synthetic memory test, not a measure of editorial quality.
    unit = "В Москве обсуждают развитие городского транспорта. Детали проекта уточняются. "
    low, high = 0, target_tokens * 2
    best = messages_for(style, "")
    while low <= high:
        middle = (low + high) // 2
        candidate = messages_for(style, unit * middle)
        count = len(tokenizer.apply_chat_template(candidate, tokenize=True, add_generation_prompt=True))
        if count <= target_tokens:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    count = len(tokenizer.apply_chat_template(best, tokenize=True, add_generation_prompt=True))
    return best, count


async def run(args):
    lock = json.loads((ROOT / "artifacts/model_lock.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(lock["model_id"], revision=lock["revision"])
    prompts = {s: near_limit_messages(tokenizer, s, args.context - args.output_tokens) for s in ("type1", "type2")}
    if any(count + args.output_tokens > args.context for _, count in prompts.values()):
        raise ValueError("Контекст превышен")
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=300.0, limits=httpx.Limits(max_connections=args.concurrency)) as client:
        async def one(index):
            style = "type1" if index % 2 == 0 else "type2"
            messages, count = prompts[style]
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.post(args.url.rstrip("/") + "/v1/chat/completions", json={"model": style, "messages": messages, "max_tokens": args.output_tokens, "temperature": 0, "stream": False, "ignore_eos": True})
                    response.raise_for_status()
                    body = response.json()
                    return {"id": index, "style": style, "ok": True, "seconds": time.perf_counter() - started, "requested_prompt_tokens": count, "usage": body.get("usage"), "finish_reason": body["choices"][0]["finish_reason"]}
                except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
                    return {"id": index, "style": style, "ok": False, "seconds": time.perf_counter() - started, "error": str(exc)}
        results = await asyncio.gather(*(one(i) for i in range(args.requests)))
    durations = sorted(r["seconds"] for r in results if r["ok"])
    summary = {"concurrency": args.concurrency, "requests": args.requests, "errors": sum(not r["ok"] for r in results), "context_limit": args.context, "output_tokens": args.output_tokens, "p95_seconds": durations[max(0, math.ceil(len(durations) * .95) - 1)] if durations else None, "notes": "Whole-response wall time excluding local semaphore wait; TTFT is not measured. ignore_eos is only for synthetic stress testing. Monitor GPU and vLLM logs separately.", "results": results}
    path = ROOT / f"artifacts/load_test_c{args.concurrency}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
    if summary["errors"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--requests", type=int, default=45)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.concurrency < 1 or args.requests < 1 or not 0 < args.output_tokens < args.context:
        parser.error("Проверьте concurrency, requests, context и output-tokens")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

