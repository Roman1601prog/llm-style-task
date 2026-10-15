#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Run only in a separate vLLM environment, after both adapters are trained.
readarray -t STYLE_MODEL_LOCK < <(python - <<'PY'
import json
lock=json.load(open('artifacts/model_lock.json',encoding='utf-8'))
print(lock['model_id'])
print(lock['revision'])
PY
)
python -m vllm.entrypoints.openai.api_server \
  --model "${STYLE_MODEL_LOCK[0]}" \
  --revision "${STYLE_MODEL_LOCK[1]}" \
  --tokenizer-revision "${STYLE_MODEL_LOCK[1]}" \
  --served-model-name news-base \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype half \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 15 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill \
  --enable-lora \
  --lora-modules "type1=$PWD/adapters/type1" "type2=$PWD/adapters/type2" \
  --max-loras 2 \
  --max-cpu-loras 2 \
  --max-lora-rank 16 \
  --enforce-eager

