#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run the CPU test suite without pytest.

    python3 tests/run_tests.py            # also: pytest tests/ -q  (pytest is optional)

Pure-logic and static checks need no GPU, torch, or checkpoint. Runtime import
and video checks run when their dependencies are available; otherwise they are
reported separately as skipped.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from unittest import SkipTest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)


def main() -> int:
    modules = sorted(p.name[:-3] for p in HERE.glob("*_test.py"))
    passed, skipped, failed = 0, [], []
    for name in modules:
        module = importlib.import_module(name)
        for attr in sorted(vars(module)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(module, attr)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
            except SkipTest as exc:
                skipped.append(f"{name}.{attr}")
                print(f"SKIP {name}.{attr}: {exc}")
            except Exception:  # noqa: BLE001
                failed.append(f"{name}.{attr}")
                print(f"FAIL {name}.{attr}")
                traceback.print_exc()
        print(f"ok  {name}" if not any(f.startswith(name + ".") for f in failed) else f"FAILED {name}")
    print(f"\n{passed} passed, {len(skipped)} skipped, {len(failed)} failed across {len(modules)} modules")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
