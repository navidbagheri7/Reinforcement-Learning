# Reproducing the studies

The measured reference uses Python 3.10.20, NumPy 1.26.4, SciPy 1.15.3, PyTorch 2.7.1+cu128, Gymnasium 1.2.3 and Stable-Baselines3 2.8.0. Three NVIDIA L40S cards run the owned wireless networks in float32; standard control and tabular studies run on CPU. Run records include actual package versions, source hashes, complete configurations and elapsed seconds. Dependencies are isolated; no shared global package upgrades are required.

## Full reference

From the repository root after installing all extras:

```bash
decision-lab planning
decision-lab lectures
for seed in 0 1 2; do
  decision-lab tabular --seed "$seed"
  decision-lab neural --seed "$seed" --device cuda
  decision-lab offline --seed "$seed" --device cuda
  decision-lab approximation --seed "$seed"
  decision-lab control --seed "$seed"
done
decision-lab report
python scripts/audit_checkpoints.py
python scripts/execute_notebooks.py
python scripts/validate_release.py
```

The full studies take substantially longer than a smoke check. Use independent processes for different fitting seeds if resources allow, assigning each GPU with `CUDA_VISIBLE_DEVICES`. Do not oversubscribe CPU BLAS; the local reference sets `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. Rendering can use `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy`.

| Configuration | Reference budget |
|---|---|
| planning.json | Two channel profiles, finite horizon64, discount .99, four energy budgets and priority/service-floor extensions |
| tabular.json | Four algorithms × five tasks × three seeds; interactions/task: wireless300k, FrozenLake120k, Cliff120k, Taxi300k, maze180k |
| neural.json | Four recipes × three seeds; 200k transitions/fit, batch128, target1000, training every4, warmup2000 |
| offline.json | Three coverage profiles × four recipes × three seeds; 80k logged transitions/profile/seed, 12k updates/method |
| approximation.json | Two trace presets × three seeds; 3000 official MountainCar episodes/fit |
| control.json | Six task/method presets × three seeds; full budgets/network sizes recorded in the file |
| lectures.json | 100 independent bandit instances/task, 30 prediction seeds, 30k expected counterexample updates |

Use a new output directory for changed or incomplete protocols. Reference output directories are scientific records, not scratch paths. This project does not promise exact resumed training from policy checkpoints. Smoke runs verify execution and math paths with tiny budgets and must never replace reference results.

## Portable evidence and private artifacts

No external dataset is needed for RL: the wireless/maze/bandit/prediction data are generated and official environments are installed packages. Numeric episode arrays, logged synthetic wireless batches, validation histories, figures and finite-state policy exports are public. Large control and neural training weights remain ignored under local `results/reference/**/checkpoints/`. The small linear MountainCar policy weights are exported publicly after the checkpoint audit.

A fresh checkout can reproduce plots, validate public evidence, run exported finite-state policies, and rerun notebooks. `audit_checkpoints.py` additionally needs the locally fitted checkpoints and verifies newly loaded held-out rollouts, including control shifts. It does not modify the final test selection. Raw course slides and original incomplete files remain privately archived.

## Packaging checks

```bash
python -m build
python -m pip install dist/sequential_decision_lab-1.0.0-py3-none-any.whl
```

Use a separate environment for an installed-wheel test. Run tests from a directory outside the checkout with `PYTHONPATH` cleared. The source archive includes authored notebooks and measured numeric evidence, but excludes original coursework, private models, environments and smoke outputs. Release validation requires the full reference set and eleven executed notebooks.
