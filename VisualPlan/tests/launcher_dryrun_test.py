# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The launcher's resolved command line must parse through the real parser.

This is the strongest check available without a GPU: it runs ``run_interleave.sh`` in
``DRY_RUN`` mode, takes the exact argument vector it would hand to the evaluator, and feeds
it to :func:`interleave_vp.cli_args.build_parser`. Any typo, removed flag, or recipe default
that no longer validates shows up here rather than on the cluster.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interleave_vp.cli_args import build_parser, validate_run_args  # noqa: E402

MODULE = "interleave_vp.eval_interleave_plan"


def dry_run_args(tmp: Path) -> list[str]:
    env = {
        "CHECKPOINT_PATH": str(tmp / "hf_export"),
        "TEST_DATA": str(ROOT / "examples" / "bench_episodes.example.jsonl"),
        "OUTPUT_DIR": str(tmp / "out"),
        "DRY_RUN": "1",
        "PATH": "/usr/bin:/bin",
    }
    import os

    proc = subprocess.run(
        ["bash", str(ROOT / "run_interleave.sh")],
        capture_output=True, text=True, cwd=str(ROOT), env={**os.environ, **env}, check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DRY_RUN: would execute" in proc.stdout, proc.stdout
    lines = [ln.lstrip() for ln in proc.stdout.splitlines() if ln.startswith(" ")]
    tokens: list[str] = []
    for line in lines:
        tokens.extend(line.split())
    assert MODULE in tokens, tokens
    return tokens[tokens.index(MODULE) + 1 :]


def test_launcher_dry_run_parses() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td:
        args = dry_run_args(Path(td))
        parsed = build_parser().parse_args(args)
        validate_run_args(build_parser(), parsed)
        assert parsed.checkpoint_path.endswith("hf_export")
        assert parsed.resolution == "480" and parsed.num_steps == 50
        assert parsed.fps == 24 and parsed.history_window == 3 and parsed.max_new_tokens == 512
        assert parsed.guidance == 5.0 and parsed.shift == 5.0
        assert parsed.context_frames == 4 and parsed.context_fill is True
        assert parsed.anchor_resolution == "720"
        assert parsed.cross_chunk_v2v is True and parsed.cross_chunk_latents == 4
        assert parsed.use_pred_duration is True and parsed.reasoner_prompt_format == "tags"
        assert parsed.pred_duration_offset == 0.5 and parsed.generator_world_modeling is True
        assert parsed.data_parallel is True and parsed.max_plan_steps == 15
        assert parsed.reasoner_input_size == (1920, 1080)
        assert parsed.do_sample is False and parsed.temperature == 1.0  # greedy: no --temperature
        assert parsed.top_p == 0.9


def test_launcher_switches_toggle_flags() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td:
        base = {
            "CHECKPOINT_PATH": str(Path(td) / "ckpt"),
            "TEST_DATA": str(ROOT / "examples" / "bench_episodes.example.jsonl"),
            "OUTPUT_DIR": str(Path(td) / "out"),
            "DRY_RUN": "1",
        }
        import os

        def run(**overrides) -> list[str]:
            env = {**os.environ, **base, **overrides}
            proc = subprocess.run(["bash", str(ROOT / "run_interleave.sh")], capture_output=True,
                                  text=True, cwd=str(ROOT), env=env, check=False)
            assert proc.returncode == 0, proc.stderr
            tokens = [w for line in proc.stdout.splitlines() if line.startswith(" ") for w in line.split()]
            return tokens[tokens.index(MODULE) + 1 :]

        off = run(USE_PRED_DURATION="0", CROSS_CHUNK_V2V="0", CONTEXT_FILL="0",
                  GENERATOR_WORLD_MODELING="0", USE_GT_PLAN="1")
        assert "--cross-chunk-v2v" not in off and "--context-fill" not in off
        assert "--use-pred-duration" not in off and "--generator-world-modeling" not in off
        assert "--use-gt-plan" in off
        # Turning the switches off must not leave dependent flags behind that the CLI
        # contract then refuses (e.g. --pred-duration-offset without --use-pred-duration).
        assert "--pred-duration-offset" not in off and "--cross-chunk-latents" not in off
        parsed = build_parser().parse_args(off)
        validate_run_args(build_parser(), parsed)

        teacher = run(GT_TEXT_AND_LENGTH="1", USE_PRED_DURATION="0")
        assert "--gt-text-and-length" in teacher
        assert build_parser().parse_args(teacher).gt_text_and_length is True

        # Teacher forcing plus predicted duration is the one combination that cannot exist.
        bad = run(GT_TEXT_AND_LENGTH="1", USE_PRED_DURATION="1")
        try:
            validate_run_args(build_parser(), build_parser().parse_args(bad))
        except SystemExit:
            pass
        else:
            raise AssertionError("predicted duration + GT text-and-length must be refused")
