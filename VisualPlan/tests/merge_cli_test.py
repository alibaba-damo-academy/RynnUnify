"""Regression coverage for single-episode and batch video-merge entry points."""

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import SkipTest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _merge_module():
    try:
        from interleave_vp import merge_interleave_video
    except ModuleNotFoundError as exc:
        if exc.name != "cv2":
            raise
        raise SkipTest("OpenCV is not installed") from exc
    return merge_interleave_video


def test_single_episode_cli_works_without_workers_and_removes_v2v_prefix():
    merge = _merge_module()
    import numpy as np

    with TemporaryDirectory() as td:
        episode = Path(td) / "episode_000"
        steps = []
        for index, length in enumerate((5, 9)):
            path = episode / f"step_{index:03d}" / "visual_plan.mp4"
            path.parent.mkdir(parents=True)
            writer = merge.cv2.VideoWriter(
                str(path), merge.cv2.VideoWriter_fourcc(*"mp4v"), 24, (32, 32)
            )
            assert writer.isOpened(), "MP4 writer unavailable"
            try:
                for frame in range(length):
                    writer.write(np.full((32, 32, 3), index * 80 + frame * 5, dtype=np.uint8))
            finally:
                writer.release()
            step = {"step": index, "video_path": str(path)}
            if index:
                step["cross_chunk_v2v"] = {
                    "prefix_source": "generated_prev_tail", "prefix_pixel_frames": 5
                }
            steps.append(step)
        (episode / "summary.json").write_text(json.dumps({"steps": steps, "config": {"fps": 24}}))
        proc = subprocess.run(
            [sys.executable, "-m", "interleave_vp.merge_interleave_video", "--episode-dir", str(episode)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        manifest = json.loads((episode / merge.MANIFEST_NAME).read_text())
        assert manifest == json.loads(proc.stdout)
        cap = merge.cv2.VideoCapture(str(episode / merge.DEFAULT_OUTPUT_NAME))
        count = 0
        try:
            while cap.read()[0]:
                count += 1
        finally:
            cap.release()
        assert count == 9, f"expected 5 + (9 - 5) frames, got {count}"


def test_batch_worker_defaults_and_invalid_combinations():
    merge = _merge_module()
    for options, expected_workers in (([], 8), (["--workers", "1"], 1)):
        with patch.object(merge, "merge_run_directory", return_value={"episodes_failed": 0}) as run:
            with redirect_stdout(io.StringIO()):
                merge.main(["--input-dir", "unused", *options])
            run.assert_called_once_with("unused", workers=expected_workers)
    # Keep existing explicit single-worker invocation compatible.
    with patch.object(merge, "merge_episode_directory", return_value={}) as single:
        with redirect_stdout(io.StringIO()):
            merge.main(["--episode-dir", "unused", "--workers", "1"])
        single.assert_called_once_with("unused", None)
    for options, message in (
        (["--episode-dir", "unused", "--workers", "8"], "--workers is only valid"),
        (["--input-dir", "unused", "--output", "unused.mp4"], "--output can only"),
    ):
        try:
            merge.main(options)
        except SystemExit as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError(f"invalid combination accepted: {options}")
    try:
        merge.main(["--input-dir", "unused", "--workers", "0"])
    except ValueError as exc:
        assert "workers must be positive" in str(exc)
    else:
        raise AssertionError("zero workers must not fall back to the default")
