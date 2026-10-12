"""Add two reproducible English simulation walkthroughs without changing old ones."""

from pathlib import Path

import nbformat as nbf

SETUP = """from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path.cwd()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
import torch
torch.set_num_threads(1)
def read(path): return json.loads((ROOT / path).read_text())
"""


def write(name, title, introduction, sections):
    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
    notebook.cells = [
        nbf.v4.new_markdown_cell(f"# {title}\n\n{introduction}"),
        nbf.v4.new_code_cell(SETUP),
    ]
    for heading, explanation, source in sections:
        notebook.cells += [
            nbf.v4.new_markdown_cell(f"## {heading}\n\n{explanation}"),
            nbf.v4.new_code_cell(source),
        ]
    nbf.write(notebook, Path("notebooks") / (name + ".ipynb"))


write(
    "12_official_navigation",
    "Official mazes: navigation, keys and reward sparsity",
    "MiniGrid Four Rooms randomizes the agent, goal and wall openings. Door Key requires object pickup and unlocking before reaching the goal. Our declared profile exposes the full symbolic map and pose; policies do not consume RGB pixels. Sparse and potential-shaped PPO use identical observations and budgets. A known-model BFS planner supplies a feasibility reference. Final scores always use the official reward.",
    [
        (
            "All fitting seeds",
            "Success rate is the fraction of 100 held-out layouts reaching the goal. Keep every fitted seed, rather than showing only the best animation.",
            """rows = []
paths = [p for family in ["simulations","geometry","curriculum"] for p in (ROOT / "results/reference" / family).glob("MiniGrid*/*/metrics.json")]
for path in sorted(paths):
    row = json.loads(path.read_text())
    rows.append({k: row[k] for k in ['env','profile','seed','success_fraction','mean_length','selected_step']} | {'test_return': row['test']['mean']})
frame = pd.DataFrame(rows)
assert len(frame) == 18
frame""",
        ),
        (
            "Potential differences telescope",
            "Training adds F(s,s') = gamma Phi(s') - Phi(s). The goal terminal potential is zero. Time-limit truncations retain the next potential for value bootstrapping. The distance heuristic supplies map-derived training information; it never enters policy observations. Finite-horizon and discount assumptions matter when interpreting policy-invariance theory.",
            """from decision_lab.simulation_envs import make_environment
from decision_lab.navigation_planner import Planner
env = make_environment('MiniGrid-DoorKey-6x6-v0', shaping=True)
obs, _ = env.reset(seed=300000)
planner = Planner(env)
phi0 = env.previous_potential
shaping_sum, official_return, t = 0., 0., 0
while True:
    action, _ = planner.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    shaping_sum += .99**t * info['shaping_reward']
    official_return += info['official_reward']; t += 1
    if terminated or truncated: break
assert terminated and not truncated
assert np.isclose(shaping_sum, -phi0 + .99**t * env.previous_potential)
env.close()
print('Official return:', official_return, 'discounted shaping sum:', shaping_sum)""",
        ),
        (
            "Portable actor deployment",
            "Load numeric actor weights without an SB3 pickle, and replay the predetermined first three test layouts. Exact episode agreement is checked, including failure cases.",
            """from decision_lab.simulation_deployment import load_actor
from decision_lab.simulations import evaluate
stem = 'MiniGrid-DoorKey-6x6-v0_PPO_potential_0'
actor = load_actor(ROOT / 'assets/actors' / (stem + '.npz'))
fresh = evaluate(actor, actor.metadata['case'], np.arange(3) + 300000)
evidence = np.load(ROOT / 'assets/source_data/simulations' / (stem + '.npz'))
for key in ['returns','lengths','terminated','truncated']:
    np.testing.assert_array_equal(fresh[key], evidence[key][:3])
pd.DataFrame({k: fresh[k] for k in ['seeds','returns','lengths','terminated','truncated']})""",
        ),
        (
            "Learning variability and model access",
            "Each curve is one independently fitted policy; all validation rewards are unshaped. BFS receives the known transition model and is reported separately from sampled-interaction learning. View the fixed-episode GIFs in the repository gallery.",
            """fig, axes = plt.subplots(1,2,figsize=(12,4))
for ax, task in zip(axes, frame.env.unique()):
    for profile in ['sparse','potential','potential_geometry','potential_curriculum']:
        if profile in ['potential_geometry','potential_curriculum'] and task != 'MiniGrid-FourRooms-v0': continue
        for seed in range(3):
            family = {'potential_geometry':'geometry','potential_curriculum':'curriculum'}.get(profile,'simulations')
            h = read(f'results/reference/{family}/{task}_PPO_{profile}/{seed}/validation_history.json')
            ax.plot([r['step'] for r in h], [r['mean_return'] for r in h], alpha=.6, label=f'{profile} seed {seed}')
    ax.set(title=task,xlabel='Interactions',ylabel='Official validation return'); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
pd.DataFrame(read('docs/simulation_baselines.json')['rows']).query("method == 'BFS'")[['env','method','success_fraction','mean_length']]""",
        ),
    ],
)

write(
    "13_mujoco_locomotion",
    "MuJoCo locomotion: PPO versus SAC",
    "Official HalfCheetah-v5 supplies a 17-dimensional proprioceptive observation and six continuous joint torques. The simulator generates interaction data; no external dataset or pretrained policy is used. PPO and SAC receive the same interaction budget. Their update counts and wall times differ, so this is an interaction-matched comparison, not equal compute.",
    [
        (
            "Three independent fits",
            "Test summaries average 100 fixed resets per selected policy. Fitting-seed SD measures training variation and differs from a conditional episode bootstrap interval.",
            """rows = []
for path in sorted((ROOT / 'results/reference/simulations').glob('HalfCheetah*/*/metrics.json')):
    m = json.loads(path.read_text())
    rows.append(dict(algorithm=m['algorithm'],seed=m['seed'],test_return=m['test']['mean'],selected_step=m['selected_step'],actual_steps=m['actual_steps']))
frame = pd.DataFrame(rows)
assert len(frame) == 6 and frame.actual_steps.nunique() == 1
frame.groupby('algorithm').test_return.agg(['mean','std','min','max'])""",
        ),
        (
            "Validation-only selection",
            "The checkpoint with highest mean validation return is selected; ties retain the first checkpoint. Test resets never select training settings or a visually appealing episode.",
            """fig, ax = plt.subplots(figsize=(8,4))
for row in frame.itertuples():
    h = read(f'results/reference/simulations/HalfCheetah-v5_{row.algorithm}_official/{row.seed}/validation_history.json')
    chosen = max(h, key=lambda r:r['mean_return'])
    assert chosen['step'] == row.selected_step
    ax.plot([r['step'] for r in h], [r['mean_return'] for r in h], label=f'{row.algorithm} seed {row.seed}',alpha=.7)
ax.set(xlabel='Interactions',ylabel='Official validation return'); ax.legend(); plt.show()""",
        ),
        (
            "Deploy an exported continuous actor",
            "The actor contains float32 network weights, action bounds and an explicit architecture. Critics, optimizer states and replay buffers are unnecessary for deterministic deployment.",
            """from decision_lab.simulation_deployment import load_actor
from decision_lab.simulations import evaluate
stem = 'HalfCheetah-v5_SAC_official_0'
actor = load_actor(ROOT / 'assets/actors' / (stem + '.npz'))
fresh = evaluate(actor, actor.metadata['case'], [300000])
evidence = np.load(ROOT / 'assets/source_data/simulations' / (stem + '.npz'))
fixed_actions = np.asarray([actor.predict(obs)[0] for obs in evidence['first_observations']])
np.testing.assert_allclose(fixed_actions, evidence['first_actions'], rtol=2e-6, atol=5e-6)
np.testing.assert_array_equal(fresh['first_observations'][0], evidence['first_observations'][0])
for key in ['lengths', 'terminated', 'truncated']:
    np.testing.assert_array_equal(fresh[key], evidence[key][:1])
assert np.isfinite(fresh['returns']).all()
print('Recorded first test return:', evidence['returns'][0])
print('Fresh physics replay return:', fresh['returns'][0], 'episode length:', fresh['lengths'][0])
print('Actor dimensions:', actor.metadata['dimensions'])""",
        ),
        (
            "Numerical evidence behind the animation",
            "Both algorithms' GIFs replay fitting seed 0, test reset 300000, including the full episode. A high reward does not certify a human-like gait. Inspect the visual behavior as well as the aggregate return.",
            """for algorithm in ['PPO','SAC']:
    gif = read(f'assets/rollouts/HalfCheetah-v5_{algorithm}_official_0.json')
    e = np.load(ROOT / f'assets/source_data/simulations/HalfCheetah-v5_{algorithm}_official_0.npz')
    assert gif['test_seed'] == 300000 and gif['return_value'] == float(e['returns'][0])
    print(algorithm, gif['return_value'], gif['length'], gif['gif_sha256'])""",
        ),
    ],
)
