# RynnUnify Reinforcement Learning

Train visual reasoners with **Branch-Conditional Policy Optimization (BCPO)** and evaluate learned generation-assisted reasoning.

[Project overview](../README.md) · [Deployment](docs/deployment.md) · [Training parameters](docs/training.md) · [Data format](docs/data.md)

## Installation

Use Linux, Python 3.10+, and a CUDA-compatible PyTorch installation. Run the following commands from the `RL/` component directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,eval,serve]'
```

The trainer keeps the actor and frozen reference on one device. Allocate separate devices to the generator and judge, and set the batch size and context budget to fit available memory.

## Deployment

Start the generator and judge in separate terminals with the environment activated.

**Generator — Cosmos3 Diffusers directory.** Install the Cosmos3-compatible Diffusers revision listed in the [deployment guide](docs/deployment.md), then run:

```bash
CUDA_VISIBLE_DEVICES=1 rynn-rl-serve-generator \
  --backend cosmos3 --model /path/to/cosmos3-diffusers \
  --device cuda --dtype bfloat16 --port 18900
```

This profile supports single-image continuation. For first/last-frame generation, use a compatible Wan Diffusers checkpoint with `--backend diffusers --enable-fl2v`. Native checkpoints require their runtime adapter; configuration is described in the [deployment guide](docs/deployment.md).

**Judge — required for `text` and `mess` tasks:**

```bash
CUDA_VISIBLE_DEVICES=2 rynn-rl-serve-judge \
  --model /path/to/judge --served-model-name judge \
  --device cuda --dtype bfloat16 --port 8001
```

Existing compatible services can also be used; see the [HTTP interfaces](docs/services.md).

## Training

Prepare training and evaluation JSONL files, with one question per line. Media paths are relative to the JSONL file:

```json
{"id":"sample-001","question":"What is the final speed?","reference":"20 cm/s","task":"phys","videos":["media/observed.mp4"],"native":false}
```

See [Data format](docs/data.md) for supported tasks. The reasoner checkpoint needs a multimodal chat template, video input support, and distinct single-token `D` / `G` route symbols.

Run training followed by evaluation of the exported checkpoint:

```bash
export CUDA_VISIBLE_DEVICES=0
export REASONER_MODEL=/path/to/reasoner
export TRAIN_DATA=/path/to/train.jsonl
export EVAL_DATA=/path/to/eval.jsonl
export RUN_DIR=runs/bcpo
export GENERATOR_URL=http://127.0.0.1:18900/generate_video
export JUDGE_URL=http://127.0.0.1:8001/v1
export JUDGE_MODEL=judge
export BATCH_SIZE=64
export EPOCHS=1
bash scripts/train_and_evaluate.sh
```

`BATCH_SIZE` counts questions; each question produces eight trajectories. Set `MAX_CONTEXT` and `MAX_NEW_TOKENS` to adjust token budgets. For tasks without a judge, omit `JUDGE_URL` and `JUDGE_MODEL`.

Use a new `RUN_DIR` for each run. The script checks both datasets, saves the trained checkpoint to `train/final/`, and writes evaluation results to `eval/`. Training also saves metrics, trajectories and generated videos. See [Training parameters](docs/training.md) for individual training commands and settings.

## Evaluation

Evaluate a saved checkpoint independently:

```bash
CUDA_VISIBLE_DEVICES=0 rynn-rl-evaluate \
  --model runs/bcpo/train/final \
  --data /path/to/eval.jsonl \
  --output runs/eval-learned \
  --mode learned \
  --generator-url http://127.0.0.1:18900/generate_video \
  --judge-url http://127.0.0.1:8001/v1 --judge-model judge \
  --device cuda --dtype bfloat16
```

Choose `--mode learned`, `direct`, or `generate` for learned routing, direct reasoning, or generation-assisted reasoning. Use a new output directory for each evaluation. Results include `predictions.jsonl` and `summary.json`, with per-question predictions, mean task reward and generation rate.

[Evaluation details](docs/evaluation.md) · [Reward definitions](docs/rewards.md)

---

[Apache License 2.0](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
