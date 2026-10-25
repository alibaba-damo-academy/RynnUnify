#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Interleaved visual-planning rollout with scene memory and cross-chunk continuation.
#
#   CHECKPOINT_PATH=/path/to/hf_export TEST_DATA=/path/to/episodes.jsonl ./run_interleave.sh
#
# Every value below is an env var; unset ones keep the default shown. The recipe block
# (context frames / cross-chunk V2V / duration prediction) must match how the checkpoint
# was trained — changing it without retraining silently changes the input distribution.
#
#   DRY_RUN=1 ./run_interleave.sh          # print the resolved command, run nothing
#   NPROC_PER_NODE=4 ./run_interleave.sh   # fewer GPUs
#   GT_TEXT_AND_LENGTH=1 ./run_interleave.sh  # teacher-forced upper bound (oracle text+length)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# interleave_vp lives here; the framework itself comes from the user's install.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${COSMOS_FRAMEWORK_DIR:-}" ]]; then
  # Needed when the framework is not pip-installed: --config-file resolves relative to it.
  cd "$COSMOS_FRAMEWORK_DIR"
  export PYTHONPATH="$PWD:$PYTHONPATH"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Runtime env the multi-rank rollout depends on (not machine-specific).
# expandable_segments cuts fragmentation while the VAE decodes a whole chunk; the NCCL
# heartbeat has to outlast the final barrier, because data-parallel ranks roll unrelated
# episodes and the fastest one can otherwise wait on the slowest for longer than the default.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"

# ── required ────────────────────────────────────────────────────────────────
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
TEST_DATA="${TEST_DATA:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$HERE/outputs/$(date +%Y%m%d_%H%M%S)}"

# ── trained inference settings ───────────────────────
RESOLUTION="${RESOLUTION:-480}"          # main-video tier
SHIFT="${SHIFT:-5}"                       # 256:3 / 480:5 / 720:10
NUM_STEPS="${NUM_STEPS:-50}"              # UniPC sampling steps
GUIDANCE="${GUIDANCE:-5.0}"              # classifier-free guidance
FPS="${FPS:-24}"
NUM_FRAMES="${NUM_FRAMES:-81}"            # default chunk length when nothing forces it
CONTEXT_FRAMES="${CONTEXT_FRAMES:-4}"     # anchor + 3 rolling keyframes
CONTEXT_FILL="${CONTEXT_FILL:-1}"         # spread the slots over all history, always K items
ANCHOR_RES="${ANCHOR_RES:-720}"           # anchor tier paired with 480p generation
CROSS_CHUNK_V2V="${CROSS_CHUNK_V2V:-1}"   # continue from the previous generated chunk
CROSS_CHUNK_LATENTS="${CROSS_CHUNK_LATENTS:-4}"   # k -> 4k-3 = 13 pixel frames of prefix
USE_PRED_DURATION="${USE_PRED_DURATION:-1}"       # chunk length from the predicted duration
REASONER_PROMPT_FORMAT="${REASONER_PROMPT_FORMAT:-tags}"
PRED_DURATION_OFFSET="${PRED_DURATION_OFFSET:-0.5}"  # tuning bias; set 0 for the raw prediction
GENERATOR_WORLD_MODELING="${GENERATOR_WORLD_MODELING:-1}"  # <world_modeling> -> caption temporal_caption

# ── sampling / scheduling ───────────────────────────────────────────────────
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-8}"               # shuffle episode order
MAX_PLAN_STEPS="${MAX_PLAN_STEPS:-15}"
HISTORY_WINDOW="${HISTORY_WINDOW:-3}"
# Reasoner input geometry expected by the trained checkpoint. `${VAR-default}` (no
# colon) is deliberate: REASONER_INPUT_SIZE=  explicitly disables the resize and falls back
# to the --resolution tier's bucket geometry.
REASONER_INPUT_SIZE="${REASONER_INPUT_SIZE-1920x1080}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
REASONER_DO_SAMPLE="${REASONER_DO_SAMPLE:-0}"
REASONER_TEMPERATURE="${REASONER_TEMPERATURE:-0.2}"
REASONER_TOP_P="${REASONER_TOP_P:-0.9}"

# ── ablation switches (off by default) ──────────────────────────────────────
USE_GT_PLAN="${USE_GT_PLAN:-0}"           # execute GT text, keep predicted length
GT_TEXT_AND_LENGTH="${GT_TEXT_AND_LENGTH:-0}"   # execute GT text AND GT length

# ── parallelism ─────────────────────────────────────────────────────────────
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-50013}"
DATA_PARALLEL="${DATA_PARALLEL:-1}"       # 1 = one full replica per rank, episodes sharded

# ── advanced: checkpoint plumbing ───────────────────────────────────────────
EXPERIMENT="${EXPERIMENT:-}"              # Hydra experiment name, for DCP checkpoints
EXPERIMENT_OVERRIDE="${EXPERIMENT_OVERRIDE:-}"   # pipe-separated, e.g. model.config.tokenizer.vae_path=/abs/Wan2.2_VAE.pth
CONFIG_FILE="${CONFIG_FILE:-}"
VLM_PROCESSOR_PATH="${VLM_PROCESSOR_PATH:-}"     # HF dir with the multimodal Qwen3-VL processor

[[ -n "$CHECKPOINT_PATH" ]] || { echo "set CHECKPOINT_PATH" >&2; exit 2; }
[[ -n "$TEST_DATA" ]] || { echo "set TEST_DATA" >&2; exit 2; }

add_flag() {   # append "$2" when the env var named by $1 is truthy (never returns non-zero)
  if [[ "${!1:-0}" =~ ^[1YyTt] ]]; then args+=("$2"); fi
}

args=(--test-data "$TEST_DATA" --output-dir "$OUTPUT_DIR" --checkpoint-path "$CHECKPOINT_PATH"
      --resolution "$RESOLUTION" --shift "$SHIFT" --num-steps "$NUM_STEPS" --guidance "$GUIDANCE"
      --fps "$FPS" --num-frames "$NUM_FRAMES" --seed "$SEED" --history-window "$HISTORY_WINDOW"
      --max-new-tokens "$MAX_NEW_TOKENS" --reasoner-prompt-format "$REASONER_PROMPT_FORMAT")
if [[ -n "$REASONER_TOP_P" ]]; then args+=(--top-p "$REASONER_TOP_P"); fi
if [[ -n "$DATA_SEED" ]]; then args+=(--data-seed "$DATA_SEED"); fi
if [[ -n "$MAX_PLAN_STEPS" ]]; then args+=(--max-plan-steps "$MAX_PLAN_STEPS"); fi
if [[ -n "$REASONER_INPUT_SIZE" ]]; then args+=(--reasoner-input-size "$REASONER_INPUT_SIZE"); fi
# Temperature only matters while sampling; greedy decoding leaves it unused, so pass it only
# when sampling is enabled.
if [[ "$REASONER_DO_SAMPLE" =~ ^[1YyTt] ]]; then args+=(--temperature "$REASONER_TEMPERATURE"); fi

# Recipe switches: pass each mechanism's dependent flags only when the switch is on, so
# turning one off degrades gracefully instead of tripping the CLI contract.
if (( CONTEXT_FRAMES > 0 )); then
  args+=(--context-frames "$CONTEXT_FRAMES")
  if [[ -n "$ANCHOR_RES" ]]; then args+=(--anchor-resolution "$ANCHOR_RES"); fi
  if [[ "$CONTEXT_FILL" =~ ^[1YyTt] ]]; then args+=(--context-fill); fi
fi
if [[ "$CROSS_CHUNK_V2V" =~ ^[1YyTt] ]]; then
  args+=(--cross-chunk-v2v --cross-chunk-latents "$CROSS_CHUNK_LATENTS")
fi
if [[ "$USE_PRED_DURATION" =~ ^[1YyTt] ]]; then
  args+=(--use-pred-duration)
  if [[ "$REASONER_PROMPT_FORMAT" == "tags" && "$PRED_DURATION_OFFSET" != "0" ]]; then
    args+=(--pred-duration-offset "$PRED_DURATION_OFFSET")
  fi
fi
add_flag GENERATOR_WORLD_MODELING --generator-world-modeling
add_flag USE_GT_PLAN --use-gt-plan
add_flag GT_TEXT_AND_LENGTH --gt-text-and-length
add_flag REASONER_DO_SAMPLE --do-sample
if [[ -n "$EXPERIMENT" ]]; then args+=(--experiment "$EXPERIMENT"); fi
if [[ -n "$CONFIG_FILE" ]]; then args+=(--config-file "$CONFIG_FILE"); fi
if [[ -n "$VLM_PROCESSOR_PATH" ]]; then args+=(--vlm-processor-path "$VLM_PROCESSOR_PATH"); fi
if [[ -n "$EXPERIMENT_OVERRIDE" ]]; then
  IFS='|' read -ra _ovr <<<"$EXPERIMENT_OVERRIDE"
  for item in "${_ovr[@]}"; do if [[ -n "$item" ]]; then args+=(--experiment-override "$item"); fi; done
fi
if [[ "$DATA_PARALLEL" =~ ^[1YyTt] ]]; then
  args+=(--data-parallel --dp-shard-size "${DP_SHARD_SIZE:-1}" --dp-replicate-size "${DP_REPLICATE_SIZE:-0}"
         --cp-size "${CP_SIZE:-1}" --cfgp-size "${CFGP_SIZE:-1}")
fi

echo "[run_interleave] checkpoint = $CHECKPOINT_PATH"
echo "[run_interleave] episodes   = $TEST_DATA"
echo "[run_interleave] output     = $OUTPUT_DIR"
echo "[run_interleave] recipe     = ctx${CONTEXT_FRAMES}(fill=${CONTEXT_FILL},anchor=${ANCHOR_RES:-tier}) "
echo "                   cross-v2v=${CROSS_CHUNK_V2V}/k${CROSS_CHUNK_LATENTS} "
echo "                   duration=${USE_PRED_DURATION}(+${PRED_DURATION_OFFSET}s) prompt=${REASONER_PROMPT_FORMAT} world_modeling=${GENERATOR_WORLD_MODELING}"
echo "[run_interleave] sampling   = ${RESOLUTION}p steps=${NUM_STEPS} guidance=${GUIDANCE} shift=${SHIFT} seed=${SEED}"

if [[ "${DRY_RUN:-0}" =~ ^[1YyTt] ]]; then
  if (( NPROC_PER_NODE > 1 )); then
    runner=("$PYTHON_BIN" -m torch.distributed.run --nnodes="$NNODES" --node_rank="$NODE_RANK"
            --nproc_per_node="$NPROC_PER_NODE" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
  else
    runner=("$PYTHON_BIN")
  fi
  echo "[run_interleave] DRY_RUN: would execute"
  printf ' %q' "${runner[@]}" -m interleave_vp.eval_interleave_plan; printf '\n'
  printf ' %q\n' "${args[@]}"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
if (( NPROC_PER_NODE > 1 )); then
  "$PYTHON_BIN" -m torch.distributed.run --nnodes="$NNODES" --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC_PER_NODE" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    -m interleave_vp.eval_interleave_plan "${args[@]}"
else
  "$PYTHON_BIN" -m interleave_vp.eval_interleave_plan "${args[@]}"
fi

echo "[run_interleave] done -> $OUTPUT_DIR/summary.json"
echo "[run_interleave] per-episode continuous video: $OUTPUT_DIR/episode_*/continuous_visual_plan.mp4"
