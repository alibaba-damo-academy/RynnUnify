# Installation and training recipe

## Environment

Use Python 3.10+ and a fresh virtual environment. For actual visual training, use a Linux CUDA machine and install PyTorch matching your CUDA environment first. Install drivers separately; start the included model servers following [Deployment](deployment.md).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,eval,serve,dev]'
pytest -q
```

The portable trainer is synchronous and single-device; trajectory gradients accumulate to one update per batch. It keeps an actor and frozen reference model on the selected device. For a large visual reasoner, memory must accommodate both models, optimizer states, and the longest two-turn sequence. Start with batch size 1 and a short real-data run.

## Inputs

1. A compatible Hugging Face visual reasoner checkpoint with a native multimodal chat template, support for assistant continuation, and separate single-token D/G symbols. `AutoModelForImageTextToText` and `AutoProcessor` load the checkpoint without `trust_remote_code`.
2. Training JSONL in the [documented schema](data.md), with local observed media.
3. A frozen generator started with the included server or implementing the [HTTP contract](services.md).
4. An OpenAI-compatible judge service if Text or Mess tasks are present. Binary text judging and the five-level Mess rubric are included.

Environment variables `GENERATOR_API_KEY` and `JUDGE_API_KEY` are optional authentication headers. Set them locally; credentials are not written to run configuration. No default company service, account, dataset, or checkpoint is needed.

## Small real-data run

```bash
rynn-rl-train \
  --model /path/to/reasoner \
  --data /path/to/train.jsonl \
  --output runs/initial-training \
  --generator-url http://127.0.0.1:18900/generate_video \
  --judge-url http://127.0.0.1:8001/v1 \
  --judge-model your-judge-model \
  --batch-size 1 --max-steps 1 \
  --device cuda --dtype bfloat16
```

A routable question produces eight trajectories even with batch size 1. This command incurs real generation and judge work on your services; it is not run automatically by tests. Check the saved responses and `videos/`, then run a larger batch. Outputs must be a new directory; existing runs are not overwritten.

## Parameters

| Parameter | Default / behavior |
|---|---|
| Questions per update | `--batch-size 64` |
| Trajectories per question | 4D + 4G; native multi-image choice questions use 8D |
| Learning rate | `--learning-rate 1e-6` |
| Epochs | `--epochs 1` |
| Updates per epoch | `ceil(N/Q)`; cyclic padding for the final batch |
| GSPO clipping | Lower 0.0003 / upper 0.0004 |
| Routing temperature / loss weight | 0.1 / 0.01 |
| KL / entropy coefficients | 0.001 / 0.001 |
| Checkpoint interval | `--save-every 20`, plus final export |
| Validation | Run the evaluation CLI on saved checkpoints |
| Sampling | Temperature 1, top-p 1, top-k 0 |
| Context / turn budget | `--max-context 32768` / `--max-new-tokens 2048` |
| Optimizer | AdamW, weight decay 0; gradient norm clip 1 |
| Precision | `--dtype bfloat16`; float32 and float16 also accepted |
| Judge | Configurable model, temperature 0, maximum 512 output tokens |
| Judge timeout / attempts | 120 seconds / 3 attempts on the same answer |
| Generator timeout / attempts | 7,200 seconds / 3 attempts on the same request |
| Generator settings | Resolution 480, 24 fps, 25 steps, guidance scale 5 |
| Tool concurrency | `--concurrency 4` |

The training CLI uses a single-device Hugging Face backend. Bounded concurrent tool requests overlap external service work; local model inference is sequential. Configure the question batch and token budget for the available memory. Loss coefficients are defined in `rynn_rl/selective_loss.py`; service parameters are in `rynn_rl/services.py`.

## Outputs, export and continuation

- `config.json`: runtime arguments and loss coefficients.
- `metrics.jsonl`: per-step loss terms, reward, routing eligibility and gradient norm.
- `rollout-*.json`: route, responses, validity, reward and generated-video paths.
- `videos/`: verified generated videos.
- `checkpoint-*/` and `final/`: standard Hugging Face model and processor exports.

Load an export with `--model runs/your-run/final` for evaluation or a new training stage. This starts a new optimizer and reference policy; it is **not an exact mid-run resume**. Optimizer/RNG/data-cursor recovery is not included. Do not upload run outputs or benchmark media by default; `.gitignore` excludes the standard output directories.
