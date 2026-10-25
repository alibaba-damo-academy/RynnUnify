# Contributing

Install `.[dev,train,serve]`, then run `pytest -q` and `ruff check rynn_rl tests`.
Keep regression fixtures inside `tests/`. New generator adapters must preserve all
conditioning images and report unsupported modes explicitly. Infrastructure failures
must raise errors rather than become task rewards. Keep datasets, model weights,
service credentials, absolute local paths and generated run artifacts out of commits.

Describe model-dependent validation separately from CPU contract tests. Changes to
losses or reward normalization should include a numerical regression test.
