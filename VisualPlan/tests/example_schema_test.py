"""The public JSONL template must be consumable by the actual episode loader."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interleave_vp.episodes import normalize_episode, resolve_record_paths  # noqa: E402


def test_example_records_load_with_relative_media_paths():
    source = ROOT / "examples" / "bench_episodes.example.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    assert records, "the data-format example is empty"
    for record in records:
        episode = resolve_record_paths(normalize_episode(record), source.parent)
        for raw, step in zip(record["sub_plan"], episode["sub_plan"]):
            path = Path(raw["first_frame_path"])
            assert not path.is_absolute(), "the portable template must use relative media paths"
            assert ".." not in path.parts
            assert Path(step["first_frame_path"]) == source.parent / path
