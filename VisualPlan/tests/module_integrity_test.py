# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check runtime-module syntax, name references, and imports with framework stubs.

The import check needs the actual third-party runtime dependencies, but no GPU
or model checkpoint. Framework stubs validate imports, not model execution.
"""

import ast
import builtins
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import SkipTest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "interleave_vp"
sys.path.insert(0, str(ROOT))

def test_every_module_compiles_and_imports_are_intra_package():
    for path in sorted(PKG.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("custom."):
                raise AssertionError(f"{path.name} still imports {node.module}")


def test_package_imports_against_stubbed_framework():
    """Import the GPU modules with the framework replaced by stubs.

    Proves module-level wiring (imports, constants, decorators) is intact on any machine;
    it does not run the model, so a pass here is not a GPU smoke test.
    """
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise SkipTest("torch is not installed") from exc

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        stub = Path(td) / "cosmos_framework"
        files = {
            "cosmos_framework/__init__.py": "",
            "cosmos_framework/inference/__init__.py": "",
            "cosmos_framework/inference/common/__init__.py": "",
            "cosmos_framework/inference/common/init.py": "def init_script(*a, **k):\n    return None\n",
            "cosmos_framework/inference/structured_caption.py": (
                "import json\n\n\ndef caption_json_to_prompt(c):\n    return json.dumps(c, ensure_ascii=False)\n"
            ),
            "cosmos_framework/data/__init__.py": "",
            "cosmos_framework/data/vfm/__init__.py": "",
            "cosmos_framework/data/vfm/local_datasets/__init__.py": "",
            "cosmos_framework/data/vfm/local_datasets/helper.py": (
                'def get_aspect_ratio(w, h):\n    return "16,9" if w / h >= 1.55 else "1,1"\n'
            ),
            "cosmos_framework/data/vfm/sequence_packing.py": (
                "class SequencePlan:\n    def __init__(self, **kwargs):\n        self.__dict__.update(kwargs)\n"
            ),
            "cosmos_framework/data/vfm/utils.py": (
                "VIDEO_RES_SIZE_INFO = {\n"
                '    "256": {"1,1": (256, 256), "16,9": (320, 192)},\n'
                '    "480": {"1,1": (640, 640), "16,9": (832, 480)},\n'
                '    "704": {"1,1": (960, 960), "16,9": (1280, 704)},\n'
                '    "720": {"1,1": (960, 960), "16,9": (1280, 720)},\n}\n'
            ),
            "cosmos_framework/tools/__init__.py": "",
            "cosmos_framework/tools/visualize/__init__.py": "",
            "cosmos_framework/tools/visualize/video.py": "def save_img_or_video(*a, **k):\n    return None\n",
            "cosmos_framework/utils/__init__.py": "",
            "cosmos_framework/utils/log.py": (
                "def info(*a, **k):\n    return None\n\n"
                "def warning(*a, **k):\n    return None\n\n"
                "def error(*a, **k):\n    return None\n"
            ),
        }
        for rel, content in files.items():
            target = stub.parent / "site-packages" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)

        code = textwrap.dedent(
            """
            import interleave_vp.cli_args, interleave_vp.episodes, interleave_vp.text_parsing
            import interleave_vp.context_fill, interleave_vp.duration_label, interleave_vp.cross_chunk_v2v
            import interleave_vp.utils
            import interleave_vp.merge_interleave_video
            import interleave_vp.plan_generator, interleave_vp.plan_reasoner
            import interleave_vp.eval_interleave_plan as entry
            assert hasattr(entry, "main") and hasattr(entry, "evaluate_episode")
            assert hasattr(entry, "build_parser")
            print("IMPORTS_OK")
            """
        )
        import os

        existing = os.environ.get("PYTHONPATH", "")
        search = f"{stub.parent / 'site-packages'}:{ROOT}" + (f":{existing}" if existing else "")
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": search, "HF_HUB_OFFLINE": "1"},
        )
        assert "IMPORTS_OK" in proc.stdout, proc.stdout + proc.stderr


def test_no_undefined_names_in_any_module():
    """Every loaded name must be imported, defined, or a builtin.

    A trim deletes a helper or an import and leaves the call site behind; that crash only
    appears on the GPU box at the step that reaches it. This catches it statically.
    """
    for path in sorted(PKG.glob("*.py")):
        tree = ast.parse(path.read_text())
        defined: set[str] = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                defined.add(node.id)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, ast.alias) and node.name != "*":
                defined.add((node.asname or node.name).split(".")[0])
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
        loads = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        missing = sorted(loads - defined)
        assert not missing, f"{path.name}: names used but never imported/defined: {missing}"


def test_entry_reuses_the_pure_layers():
    """The rollout must not re-implement the contract that the tests cover."""
    entry = (PKG / "eval_interleave_plan.py").read_text()
    for helper in ("snap_frames_4n1", "combined_timeline", "prefix_skip_reason", "select_history_units",
                   "build_fill_plan", "parse_interleaved_tag_prediction", "resolve_generation_frames"):
        assert helper in entry, f"{helper} should come from the tested module, not a copy"
    for defined in ("def snap_frames_4n1", "def select_history_units", "def combined_timeline"):
        assert defined not in entry, f"{defined} duplicates a pure module inside the entry point"
