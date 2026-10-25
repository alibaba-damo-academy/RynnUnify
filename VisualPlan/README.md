# RynnUnify Visual Planning

[Project overview](../README.md) · [BCPO](../RL/README.md)

Interleave **Reasoner** predictions with **Generator** rollouts: predict a subtask, future scene description and duration, generate a video chunk, then continue from its final frame. Scene memory and cross-chunk conditioning preserve context across steps.

## Requirements

- Linux, Python 3.10+, and a CUDA-compatible PyTorch environment.
- A compatible `cosmos-framework` installation providing `OmniMoTModel`, the video VAE and checkpoint loaders, with its dependencies installed (including torchvision, transformers, OpenCV, Pillow and einops).
- A complete compatible checkpoint with a **visual reasoner in its `und` branch**. The framework and weights are external dependencies; a generic Cosmos checkpoint is not guaranteed to work.

Run the commands below from `VisualPlan/`. The launcher adds `interleave_vp` to `PYTHONPATH`; no separate package installation is needed. If the framework is a source checkout, set `COSMOS_FRAMEWORK_DIR` to its absolute path. If the checkpoint lacks a multimodal processor, set `VLM_PROCESSOR_PATH` to a compatible processor directory.

## Prepare data

Use one episode per JSONL line; see [`examples/bench_episodes.example.jsonl`](examples/bench_episodes.example.jsonl). It is a **format template**: supply your own images and update the paths before inference. No input media is bundled.

```json
{
  "id": "example-001",
  "high_level_task": "Place the cup on the tray.",
  "sub_plan": [
    {"plan_index": 0, "instruction": "Pick up the cup.", "first_frame_path": "media/step_000.jpg"},
    {"plan_index": 1, "instruction": "Place the cup on the tray.", "first_frame_path": "media/step_001.jpg"}
  ]
}
```

Save each episode as a single line. All fields shown are required; media paths are relative to the JSONL file or absolute. The first step supplies the initial observation; subsequent observations come from generated video. The number of labelled steps bounds the rollout, which can stop early when the Reasoner predicts completion.

Optional fields per step: `frame_count` and `fps` for duration evaluation, and `clip_video_path` for ground-truth video comparison. `GT_TEXT_AND_LENGTH=1` requires `frame_count` on every step. Multi-step episodes are needed to exercise cross-chunk continuation.

## Run

```bash
CHECKPOINT_PATH=/path/to/compatible/hf_export \
TEST_DATA=/path/to/episodes.jsonl \
OUTPUT_DIR="$PWD/outputs/demo" \
NPROC_PER_NODE=1 bash run_interleave.sh
```

Add `DRY_RUN=1` to print the command without loading models. Increase `NPROC_PER_NODE` to distribute episodes across GPUs; the default is eight processes, with one full model replica per rank. Each GPU must fit the model.

The launcher defaults enable 480p video at 24 fps, four scene-memory items with a 720p anchor, cross-chunk continuation, and predicted durations. Keep these settings aligned with checkpoint training.

| Setting | Default | Purpose |
| --- | --- | --- |
| `NPROC_PER_NODE` | `8` | Worker processes / GPU replicas |
| `MAX_PLAN_STEPS` | `15` | Maximum evaluated steps per episode |
| `NUM_STEPS` | `50` | Diffusion sampling steps |
| `SEED` | `42` | Generation seed |
| `PRED_DURATION_OFFSET` | `0.5` | Seconds added to predicted duration; `0` uses the raw prediction |
| `VLM_PROCESSOR_PATH` | unset | Compatible multimodal processor, if needed |

For DCP checkpoints, supply `EXPERIMENT` and, if needed, `CONFIG_FILE` / `EXPERIMENT_OVERRIDE` for the external framework. For generator-only evaluation, set `USE_GT_PLAN=1` (labelled instructions) or `GT_TEXT_AND_LENGTH=1 USE_PRED_DURATION=0` (labelled instructions and durations).

All CLI options are defined in [`interleave_vp/cli_args.py`](interleave_vp/cli_args.py).

## Outputs

`OUTPUT_DIR` contains the run-level `summary.json` and one directory per episode:

- `continuous_visual_plan.mp4` and its JSON manifest: stitched rollout.
- `summary.json`: per-episode predictions and run configuration.
- `step_*/`: generated clips, observations, predicted/labelled plans and comparison records. Ground-truth comparison videos are written when reference clips are provided.

To re-stitch an existing rollout (requires OpenCV):

```bash
python -m interleave_vp.merge_interleave_video --episode-dir /path/to/episode_0000
# For an entire run, parallel episode merging defaults to 8 workers:
python -m interleave_vp.merge_interleave_video --input-dir /path/to/run --workers 8
```

The merge removes the repeated cross-chunk V2V prefix before appending each continuation.

## Code and checks

[`interleave_vp/eval_interleave_plan.py`](interleave_vp/eval_interleave_plan.py) runs the loop; `plan_reasoner.py` and `plan_generator.py` implement the two model interfaces. Other modules handle scene context, continuity, duration, parsing and video stitching.

```bash
python3 tests/run_tests.py
```

The CPU suite covers data handling, launch arguments, code structure, skip reporting and video stitching. Runtime import checks need PyTorch and the other runtime dependencies; video checks need OpenCV. Unavailable dependencies are reported as skipped, separately from passed checks. GPU inference requires the external framework and checkpoint.

## License

Existing source files, launcher and tests retain their OpenMDW-1.1 notices; see [LICENSE](LICENSE). Model weights and user-supplied data follow their own terms.
