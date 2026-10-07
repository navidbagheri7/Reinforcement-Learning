# Reproduction and validation tools

Run these from the repository root after installing the documented extras.
The [reproduction guide](../docs/reproduction.md) gives full fitting commands;
opening an explorer or a lesson does not launch those full training budgets.

| Tools | Purpose | Requires private training artifacts? |
|---|---|---|
| `audit_checkpoints.py`, `audit_simulations.py`, `audit_modern.py` | Bind selected fitted models to public deterministic policies/actors and independently repeat held-out episodes | Yes; fit or restore the declared local checkpoints first |
| `play_simulation.py`, `play_modern.py` | Deploy published compact actors in official simulators; modern replay preserves episode memory | No |
| `validate_release.py` | Recompute numeric statistics, source bindings, finite policy contracts and current lesson/explorer evidence | No |
| `validate_simulations.py`, `validate_modern.py` | Independently replay every exported simulation actor and verify fixed test GIFs, selection and episode statistics | No |
| `simulation_baselines.py` | Evaluate fixed random actions and known-model BFS on official navigation tasks | No; BFS has model access and is labeled separately |
| `simulation_analysis.py`, `simulation_gallery.py`, `diagnostics.py` | Build numerical comparisons, interpret navigation failures and expose critic diagnostics | No |
| `create_explorer.py`, `create_simulation_dashboard.py` | Embed current measured evidence into offline HTML, retaining every fitting seed | No |
| `check_explorer_browser.py`, `check_simulation_dashboard.py` | Exercise every preset in Chromium, check mobile layouts and live Bellman calculations | No; optional Playwright/browser installation |
| `execute_notebooks.py` | Execute all fifteen lessons in separate kernels using the current interpreter | No; complete public reference evidence and all extras |
| `review_components.py` | Parse public source/configuration/notebooks and check English content, local links and publication boundaries | No |
| `check_distribution.py` | Inspect built wheel/archive members and source/evidence identities | No |
| `check_installed_distribution.py` | Install the wheel outside the checkout, run all tests and bundled smoke commands, then execute and validate the extracted archive | No |

The `create_notebooks.py` and `create_simulation_notebooks.py` authoring helpers
build their original lesson subsets and overwrite those notebook files. The
additional quantile/memory lessons are authored directly as public notebooks.
For normal use, execute the existing lessons with `execute_notebooks.py` rather
than rebuilding their sources. If editing through an authoring helper, format
and lint the resulting notebook cells and then rerun the full execution record.

`train_simulations.py` is a convenience fitting entry point; the installed
`decision-lab` CLI also covers the modern extension. `render_navigation.py`
and `render_rollout.py` are renderers for declared policies and fixed reset
seeds. Rendered full scenes do not imply a partial-view actor sees the map.
Use `MUJOCO_GL=egl` for headless MuJoCo, and single-thread BLAS for concurrent
fits. Smoke outputs must stay separate from public reference records.

Validation reports bind their own scope. Portable deployment does not reproduce
training optimizer/environment state. Independent episode intervals do not
replace variability across fitting seeds, and concurrent runtime is not an
algorithm throughput benchmark. Failed fits and predetermined unsuccessful
rollouts remain part of the public comparison.
