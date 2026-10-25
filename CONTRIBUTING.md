# Contributing

Keep visual-planning changes in `VisualPlan/` and reinforcement-learning changes in `BCPO/`. Each component owns its runtime dependencies and tests. Commands below start from the repository root unless noted otherwise.

## Development

Run visual-planning CPU checks from the repository root:

```bash
python3 VisualPlan/tests/run_tests.py
```

Run BCPO checks in its environment:

```bash
cd BCPO
python -m pip install -e '.[dev,train,serve]'
python -m pytest -q
ruff check rynn_rl tests
python -m build
```

These checks validate code contracts and packaging. Model quality, full-scale training, and benchmark results require separate GPU validation.


## Contribution guidelines

Describe CPU checks separately from GPU/model validation. Changes to losses or reward normalization should include numerical regression coverage; generator adapters must preserve conditioning inputs and report unsupported modes explicitly.

Keep credentials, host-specific paths, checkpoints, private datasets, and generated experiment artifacts out of commits. Preserve upstream copyright and license notices. Update the root and component READMEs when changing release scope or quick-start behavior.
