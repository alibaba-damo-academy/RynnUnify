# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Memory-conditioned interleaved Reasoner/Generator inference.

Layering matters here:

* :mod:`interleave_vp.duration_label`, :mod:`context_fill`, :mod:`cross_chunk_v2v`,
  :mod:`text_parsing`, :mod:`episodes`, :mod:`cli_args` — pure Python, no torch. They own
  the frame-grid arithmetic, the context selection rule, prediction parsing, the episode
  schema and the CLI contract, so all of it is testable without a GPU or a checkpoint.
* :mod:`interleave_vp.eval_interleave_plan`, :mod:`plan_generator`, :mod:`plan_reasoner` —
  the rollout and the two engines on top of ``OmniMoTModel`` (needs the framework install).
"""

__version__ = "1.0.0"
