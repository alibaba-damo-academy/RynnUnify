# Deployment: model services → BCPO → evaluation

Use Linux with CUDA and Python 3.10+. Install a CUDA-compatible PyTorch build before the package. Allocate GPUs separately to the reasoner, generator and judge. The included trainer uses one device for the actor, frozen reference, gradients and optimizer; choose a checkpoint and context budget that fit that device. The included servers each load one model per process. This is not a distributed 128-GPU launcher.

## Install

Run from the `RL/` component directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,eval,serve]'
```

Cosmos3 requires a Diffusers build containing `Cosmos3OmniPipeline`. The following source revision was resolved when preparing this package; model inference on GPUs has not been validated here:

```bash
python -m pip install 'diffusers @ git+https://github.com/huggingface/diffusers.git@1ddaf8f4e7c33afaa7c6c2c3be9e56f2c2a634c8'
```

Obtain model weights under their own licenses. Keep checkpoints and datasets outside the repository. The reasoner must support native multimodal chat templates, assistant continuation and distinct single-token `D` / `G` route symbols. Its processor must accept local video paths. A generic instruction model is not automatically a compatible trained reasoner.

## Start a generator

Choose one profile in a dedicated terminal, after activating the environment.

### Cosmos3 in Diffusers layout

```bash
CUDA_VISIBLE_DEVICES=1 rynn-rl-serve-generator \
  --backend cosmos3 --model /path/to/cosmos3-diffusers \
  --device cuda --dtype bfloat16 --port 18900
```

The local directory must contain the Diffusers pipeline configuration and components. The built-in Cosmos3 profile supports image-to-video, including conditioning on the last observed video frame. It does not claim first/last-frame interpolation support. Datasets containing two conditioning images can produce `fl2v` requests and require a compatible backend; do not run those datasets with the i2v-only profile.

### Diffusers Wan image-to-video / first-last-frame model

```bash
CUDA_VISIBLE_DEVICES=1 rynn-rl-serve-generator \
  --backend diffusers --model /path/to/wan-flf2v-diffusers \
  --enable-fl2v --device cuda --dtype bfloat16 --port 18900
```

This profile explicitly supports `WanImageToVideoPipeline`. Enable `fl2v` only for a checkpoint trained for first/last-frame conditioning. For an i2v checkpoint, omit `--enable-fl2v`. This is an alternative generator, not a claim to reproduce Cosmos3 outputs. Other Diffusers pipeline classes need a specific adapter.

### Native Cosmos3 checkpoint runtime

A raw checkpoint is not a complete inference runtime. Install the runtime matching your checkpoint and expose a factory:

```bash
CUDA_VISIBLE_DEVICES=1 rynn-rl-serve-generator \
  --backend native --model /path/to/native-checkpoint-directory \
  --native-factory your_runtime.adapter:create_backend --port 18900
```

`your_runtime.adapter` is supplied by that runtime, not bundled here. Its factory receives `model_path`, `device`, `dtype`, loads the actual model once, and returns an object with:

- `modes`: a set containing `i2v` and, only if supported, `fl2v`.
- `generate(request, images)`: accepts the validated request object and one or two RGB PIL images; returns a nonempty list of RGB PIL video frames. Honor prompt, seed, duration, fps, resolution, inference steps and guidance, or explicitly reject unsupported values.

The server handles authentication, serialized execution, MP4 encoding, checksum and sampled PNG responses. This native adapter interface is implemented; no checkpoint-specific native Cosmos3 loader or format conversion is bundled. Use the built-in Diffusers layout for deployment without supplying a custom runtime.

## Start the judge

Required for Text/Mess tasks; omit for numerical or choice-only data. In another activated terminal:

```bash
CUDA_VISIBLE_DEVICES=2 rynn-rl-serve-judge \
  --model /path/to/text-judge --served-model-name judge \
  --device cuda --dtype bfloat16 --port 8001
```

The server loads `AutoModelForCausalLM` and a tokenizer with a chat template. It requests non-thinking mode where the template supports it, uses greedy decoding, and reports EOS versus length truncation. Use a judge that follows the included verdict formats within 512 tokens. Requests are serialized; existing OpenAI-compatible inference servers can be substituted for higher throughput. Generic remote-code checkpoints and specialized quantized runtimes are not automatically supported.

Both services bind to loopback by default. Use `--host` for a private network deployment. Set `GENERATOR_API_KEY` / `JUDGE_API_KEY` in the corresponding server and training shells to enable bearer authentication. Credentials are read from the environment. Keep the services on trusted hosts; TLS and network ingress are deployment responsibilities.

Check readiness (only returns ready after weights are loaded):

```bash
curl --fail http://127.0.0.1:18900/health
curl --fail http://127.0.0.1:8001/health
```

## Train and evaluate

In a third activated terminal, set your model and data locations:

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
export MAX_CONTEXT=32768
export MAX_NEW_TOKENS=2048
bash scripts/train_and_evaluate.sh
```

Use a fresh `RUN_DIR`. Training saves the model under `train/final/`; evaluation loads that export and writes `eval/predictions.jsonl` and `eval/summary.json`. Generator and judge processes stay running for reuse; stop them with Ctrl-C in their own terminals when finished. The script does not download data, overwrite runs or resume optimizer state.

The generator returns the actual number of frames and duration. Temporal VAE alignment may round the requested frame count upward. No future ground truth enters the generation prompt. See [Data](data.md), [Training](training.md), [Rewards](rewards.md) and [Service contract](services.md).

## Validation coverage

```bash
python -m pip install -e '.[dev,train,serve]'
pytest -q
ruff check rynn_rl tests
python -m build
```

Tests exercise gradient updates, routing masks, scoring, HTTP contracts, authentication, real MP4 encoding/decoding, and checkpoint export/reload with a small local test model. Production reasoner, generator and judge weights have not been run on GPUs in this release preparation. Run your actual checkpoints and inspect trajectories before a long training job. Passing these tests establishes software behavior, not model quality or GPU memory fit.

Upstream interfaces: [Cosmos3](https://huggingface.co/docs/diffusers/main/api/pipelines/cosmos3) and [Wan](https://huggingface.co/docs/diffusers/main/api/pipelines/wan).
