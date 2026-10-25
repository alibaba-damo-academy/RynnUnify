#!/usr/bin/env bash
# Real training followed by evaluation of the resulting checkpoint.
set -euo pipefail
: "${REASONER_MODEL:?Set REASONER_MODEL to the reasoner checkpoint}"
: "${TRAIN_DATA:?Set TRAIN_DATA to training JSONL}"
: "${EVAL_DATA:?Set EVAL_DATA to held-out JSONL}"
: "${RUN_DIR:?Set RUN_DIR to a new output directory}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -e "$RUN_DIR" ]]; then
  echo "RUN_DIR already exists; choose a new directory." >&2
  exit 1
fi
python "$ROOT/scripts/check_data.py" "$TRAIN_DATA"
python "$ROOT/scripts/check_data.py" "$EVAL_DATA"
common=(--device "${DEVICE:-cuda}" --dtype "${DTYPE:-bfloat16}")
if [[ -n "${GENERATOR_URL:-}" ]]; then common+=(--generator-url "$GENERATOR_URL"); fi
if [[ -n "${JUDGE_URL:-}" || -n "${JUDGE_MODEL:-}" ]]; then
  : "${JUDGE_URL:?Set both JUDGE_URL and JUDGE_MODEL}"
  : "${JUDGE_MODEL:?Set both JUDGE_URL and JUDGE_MODEL}"
  common+=(--judge-url "$JUDGE_URL" --judge-model "$JUDGE_MODEL")
fi
python -m rynn_rl.train --model "$REASONER_MODEL" --data "$TRAIN_DATA" \
  --output "$RUN_DIR/train" "${common[@]}" \
  --batch-size "${BATCH_SIZE:-64}" --epochs "${EPOCHS:-1}" \
  --learning-rate "${LEARNING_RATE:-1e-6}" --concurrency "${CONCURRENCY:-4}" \
  --max-context "${MAX_CONTEXT:-32768}" --max-new-tokens "${MAX_NEW_TOKENS:-2048}"
python -m rynn_rl.evaluate --model "$RUN_DIR/train/final" --data "$EVAL_DATA" \
  --output "$RUN_DIR/eval" --mode "${EVAL_MODE:-learned}" "${common[@]}"
