# Contributing

Install the development extras in an isolated environment. Run `python -m pytest -q`, `ruff check src tests scripts`, and `ruff format --check src tests scripts`. Keep scientific checks independent of the implementation: analytical targets, conservation laws, solver duality and held-out rollouts are preferable to tests that restate a loop.

Every experiment must declare its environment version, objective, information access, training budget, seeds, validation rule and final test resets. Preserve every fitted seed, failed comparison and parameter shift. Avoid tuning on test results. Record numeric evidence so figures can be regenerated. Distinguish owned algorithms from maintained library baselines and adaptations from paper reproductions.

Use English for authored code, prose and notebooks. Do not commit raw course slides, other students' work, credentials, environments or large private checkpoints. Keep original files intact in the ignored archive. Preserve existing run directories; use a new output path for changed protocols. Do not silently combine partial runs with reference evidence.
