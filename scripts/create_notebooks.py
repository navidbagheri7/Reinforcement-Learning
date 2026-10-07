"""Create authored walkthrough sources; execution is a separate release check."""

from pathlib import Path

import nbformat as nbf

SETUP = """from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path.cwd()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
assert (ROOT / 'src/decision_lab').is_dir()
sys.path.insert(0, str(ROOT / 'src'))
def read(path): return json.loads((ROOT / path).read_text())
"""

STUDIES = [
    (
        "01_wireless_process",
        "A coherent wireless process",
        "Two users harvest energy and transmit packets that expire after one slot. The analytical model and simulator share the outcome function; we explicitly separate the original ambiguous channel interpretations. No learning algorithm receives the transition table.",
        [
            (
                "Source accounting",
                'from decision_lab.wireless import WirelessModel, WirelessEnv\nmodel = WirelessModel()\nstate = model.index[(1,2,2,1,2,2)]\nserved, expired, reward, valid = model.service(state, 3)\nassert np.array_equal(served, [1,1]) and reward == 0\nserved, expired, reward, valid = model.service(state, 0)\nassert np.array_equal(expired, [1,1]) and reward == -2\nprint("states", len(model.states), "legal state/action pairs", model.legal.sum())',
            ),
            (
                "Sampled conservation",
                'env = WirelessEnv(model)\nobs, info = env.reset(seed=23)\nrng = np.random.default_rng(23)\nfor _ in range(1000):\n    action = int(rng.choice(np.flatnonzero(info["action_mask"])))\n    obs, reward, term, trunc, info = env.step(action)\n    assert info["offered"] == info["served"] + info["expired"]\n    assert reward == -info["expired"]\n    if trunc: obs, info = env.reset()\nenv.close()\nprint("1000 conservation checks passed")',
            ),
            (
                "Two declared interpretations",
                'planning = read("results/reference/planning/metrics.json")\npd.DataFrame(planning["rows"])[["profile","states","normalized_optimal_loss","average_packet_loss"]]',
            ),
        ],
        "planning",
    ),
    (
        "02_exact_planning",
        "Bellman operators and certified planning",
        "Discounted value iteration is a contraction; its Bellman residual yields a value error bound. Policy evaluation is a linear solve. Primal Bellman and dual occupancy linear programs supply independent optimal-value checks. Finite-horizon control instead uses a time-indexed policy.",
        [
            (
                "A one-state analytical oracle",
                'from decision_lab.mdp import FiniteMDP, value_iteration, policy_iteration, bellman_lp, finite_horizon\nmdp = FiniteMDP(np.ones((1,1,1)), np.ones((1,1)))\nvi = value_iteration(mdp, .9)\npi = policy_iteration(mdp, .9)\nlp = bellman_lp(mdp, .9)\nassert abs(vi["value"][0]-10) <= vi["error_bound"]+1e-10\nassert np.allclose(pi["value"], lp["value"])\nvh, ph = finite_horizon(mdp, 5)\nassert vh[0,0] == 5\nprint(vi["value"], vi["error_bound"])',
            ),
            (
                "Main project certificates",
                'pd.DataFrame(read("results/reference/planning/metrics.json")["rows"])[["profile","VI_iterations","PI_iterations","VI_value_error_bound","VI_PI_max_value_difference","discounted_primal_dual_gap"]]',
            ),
            (
                "Rebuild the residual plot",
                'data = np.load(ROOT / "assets/source_data/planning.npz")\nfor profile in ["per_user","global"]: plt.semilogy(data[f"{profile}_VI_residual"],label=profile)\nplt.xlabel("VI sweep"); plt.ylabel("Bellman residual"); plt.legend(); plt.show()',
            ),
        ],
        "planning",
    ),
    (
        "03_resource_constraints",
        "Energy, priorities and fairness",
        "Occupancy flow turns discounted resource-constrained control into a linear program. Budgets use normalized discounted costs. A service-floor extension is an explicit new requirement, rather than an undocumented change to the original reward.",
        [
            (
                "A randomized constrained optimum",
                'from decision_lab.mdp import FiniteMDP, discounted_occupancy\nmdp = FiniteMDP(np.ones((1,2,1)), np.array([[0.,1.]]))\nresult = discounted_occupancy(mdp,.9,costs=np.array([[[0.,1.]]]),budgets=[.25])\nassert np.allclose(result["policy"], [[.75,.25]],atol=1e-8)\nprint("policy",result["policy"],"cost",result["normalized_costs"],"shadow",result["budget_shadow_prices"])',
            ),
            (
                "Declared physical tradeoffs",
                'pareto = pd.DataFrame(read("results/reference/planning/metrics.json")["pareto"])\nassert pareto.flow_residual.max()<1e-8\npareto.head(8)',
            ),
            (
                "Priorities change the objective",
                'from IPython.display import Image, display\ndisplay(Image(filename=str(ROOT / "assets/figures/constrained_planning.png")))',
            ),
        ],
        "planning",
    ),
    (
        "04_tabular_control",
        "Four sampled control methods on five tasks",
        "Q-learning, SARSA, Expected SARSA and Double Q receive sampled transitions. Artificial time limits keep continuation value; true termination suppresses it. Model-aware optimality gaps are computed only after fitting, separately from finite-rollout returns.",
        [
            (
                "Expected behavior and a time-limit oracle",
                'from decision_lab.tabular import epsilon_distribution, bootstrap_target\np = epsilon_distribution(np.array([1.,3.,999.]),np.array([True,True,False]),.2)\nassert np.allclose(p,[.1,.9,0])\nassert bootstrap_target(1.,10.,False,.9)==10.\nassert bootstrap_target(1.,10.,True,.9)==1.\nprint("behavior probabilities",p)',
            ),
            (
                "Every fitted seed",
                'fits = pd.read_csv(ROOT / "docs/individual_fits.csv")\ntabular = fits[fits.task.isin(["Wireless","FrozenLake-v1","CliffWalking-v1","Taxi-v3","Maze"])]\nassert len(tabular)==60\ntabular.groupby(["task","method"]).test_return.agg(["mean","std","min"])',
            ),
            (
                "Fresh published policy deployment",
                'from decision_lab.deployment import load_policy\nfrom decision_lab.tabular import rollout_policy\nrecord, make, policy = load_policy(ROOT / "assets/policies/tabular_Maze_q_learning_0.json")\nevidence = rollout_policy(make,policy,np.arange(3)+300000)\nprint(evidence["returns"],evidence["lengths"])',
            ),
        ],
        "tabular",
    ),
    (
        "05_neural_control",
        "Owned DQN and distributional targets",
        "The owned networks use physical state features, legal masks, frozen targets and validation-only selection. Double selection, dueling values, prioritized replay and quantile regression are distinct extensions. Quantile crossing and failed seeds are retained.",
        [
            (
                "Double DQN target separation",
                'import torch\nfrom decision_lab.neural import targets, quantile_huber\nr=torch.tensor([1.]); term=torch.tensor([0.]); online=torch.tensor([[2.,1.]]); target=torch.tensor([[1.,9.]])\nlegal=torch.tensor([[True,True]])\nordinary=targets(r,term,online,target,legal,.9,False)\ndouble=targets(r,term,online,target,legal,.9,True)\nassert torch.allclose(ordinary,torch.tensor([9.1])) and torch.allclose(double,torch.tensor([1.9]))\nprint("ordinary",ordinary,"double",double)',
            ),
            (
                "A quantile-loss hand oracle",
                "loss=quantile_huber(torch.tensor([[0.,2.]]),torch.tensor([[1.,3.]]))\nassert torch.allclose(loss,torch.tensor([.3125]))\nprint(loss)",
            ),
            (
                "Complete neural evidence",
                'fits=pd.read_csv(ROOT / "docs/individual_fits.csv")\nfits[fits.task=="Wireless online"][["method","seed","test_return","exact_normalized_gap","shift_return"]]',
            ),
        ],
        "neural",
    ),
    (
        "06_offline_learning",
        "Logged coverage, CQL and discrete IQL",
        "Four methods share a fixed batch per coverage/seed. No deployment score chooses an offline checkpoint. The deliberately biased behavior prefers user 1 despite user 2 priority three; this is a weighted-objective extension. An action mask does not remove distribution shift within legal actions.",
        [
            (
                "CQL pushes down relative unseen scores",
                "import torch\nfrom decision_lab.offline import conservative_penalty, expectile_loss\nq=torch.tensor([[0.,1.,999.]],requires_grad=True)\nloss=conservative_penalty(q,torch.tensor([0]),torch.tensor([[True,True,False]])).sum()\nloss.backward()\nassert torch.allclose(q.grad,torch.tensor([[-.7310586,.7310586,0.]]),atol=1e-6)\nprint(q.grad)",
            ),
            (
                "Expectile stationarity",
                'v=torch.tensor([1.5],requires_grad=True)\nexpectile_loss(v,torch.tensor([0.,2.]),.75).mean().backward()\nassert abs(float(v.grad))<1e-7\nprint("equal-weight [0,2] expectile at tau .75 is 1.5")',
            ),
            (
                "Measured coverage and deployment gap",
                'from IPython.display import Image, display\ndisplay(Image(filename=str(ROOT / "assets/figures/offline_coverage.png")))\nrecords=[]\nfor p in (ROOT / "results/reference/offline").rglob("metrics.json"):\n    m=json.loads(p.read_text()); records.append({k:m[k] for k in ["recipe","seed","behavior_epsilon","visited_legal_fraction","unsupported_deployment_occupancy","normalized_exact_policy_gap"]})\npd.DataFrame(records).groupby(["behavior_epsilon","recipe"]).mean(numeric_only=True)',
            ),
        ],
        "offline",
    ),
]

STUDIES.extend(
    [
        (
            "07_standard_control",
            "PPO, DQN and SAC on official control tasks",
            "Maintained Stable-Baselines3 implementations provide practical baselines on LunarLander, CartPole and Pendulum. The repository owns the experiment protocol, selection and audits; it does not claim ownership of the library algorithms. Budgets differ across presets, and every fitting seed remains visible.",
            [
                (
                    "Held-out fitting variability",
                    'table=pd.read_csv(ROOT / "docs/comparison.csv")\ntable[table.task.isin(["LunarLander-v3","CartPole-v1","Pendulum-v1"])][["task","method","n_fits","mean_return","fit_sd","worst_fit_return","mean_shift_return"]]',
                ),
                (
                    "Independent episode evidence",
                    'data=np.load(ROOT / "assets/source_data/control/LunarLander-v3_PPO_0.npz")\nassert np.array_equal(data["test_seeds"],np.arange(100)+300000)\nprint("test mean",data["test_returns"].mean(),"fraction >=200",(data["test_returns"]>=200).mean())\nprint("shift mean",data["shift_returns"].mean())\nplt.hist(data["test_returns"],bins=20,alpha=.65,label="official task")\nplt.hist(data["shift_returns"],bins=20,alpha=.65,label="parameter shift")\nplt.xlabel("Episode return");plt.ylabel("Episodes");plt.legend();plt.show()',
                ),
                (
                    "Termination is distinct from success",
                    'print("terminated",data["test_terminated"].mean(),"time limit",data["test_truncated"].mean())\nassert len(data["test_first_observations"])==32\nprint("Fixed first test reset observations:",data["test_first_observations"][:2])',
                ),
            ],
            "control",
        ),
        (
            "08_bandits",
            "Exploration and adaptation on drifting bandits",
            "Ten Gaussian arms share means and arm-specific reward noise across methods. Sample averages, constant updates, UCB, discounted UCB, gradient preferences and a stationary Gaussian Thompson posterior reveal different responses to drift. Pseudo-regret uses known means only for evaluation.",
            [
                (
                    "Measured instance variability",
                    'metrics=read("results/reference/lectures/metrics.json")["bandits"]\nrows=[{"task":r["task"],"method":r["method"],"cumulative_regret":r["cumulative_pseudo_regret"]["mean"],"last500_reward":r["last500_reward"]["mean"]} for r in metrics["rows"]]\npd.DataFrame(rows)',
                ),
                (
                    "Rebuild regret from individual instances",
                    'data=np.load(ROOT / "assets/source_data/lectures/bandits.npz")\nfor name in ["epsilon_average","epsilon_constant","ucb","discounted_ucb","gradient","thompson"]:\n    regret=data[f"drifting_{name}_regret"]\n    assert np.all(regret>=0)\n    plt.plot(regret.cumsum(1).mean(0),label=name)\nplt.xlabel("Interaction");plt.ylabel("Cumulative pseudo-regret");plt.legend();plt.show()',
                ),
                (
                    "A misspecified posterior is a measured limitation",
                    'for task in ["stationary","drifting"]:\n    rewards=data[f"{task}_thompson_rewards"]\n    print(task,"last500 mean",rewards[:,-500:].mean(),"independent instances",len(rewards))',
                ),
            ],
            "lectures",
        ),
        (
            "09_prediction",
            "MC, TD and multi-step prediction against known values",
            "An unbiased random walk has nineteen interior states and terminal rewards -1 and +1. Its exact undiscounted value is linear. Methods share each sampled trajectory, while analytical values are used only to measure error. TD(lambda) uses accumulating traces; Monte Carlo updates every visit.",
            [
                (
                    "Analytical Bellman check",
                    'true=np.linspace(-1.,1.,21)\nassert np.allclose(true[1:-1],.5*(true[:-2]+true[2:]))\nplt.plot(np.arange(1,20),true[1:-1],"o-");plt.xlabel("Interior state");plt.ylabel("True value");plt.show()',
                ),
                (
                    "Every prediction seed",
                    'data=np.load(ROOT / "assets/source_data/lectures/prediction.npz")\nfor name in data.files:\n    rms=data[name]\n    assert rms.shape==(30,300)\n    print(name,"final mean",rms[:,-1].mean(),"fit SD",rms[:,-1].std(ddof=1))\n    plt.plot(rms.mean(0),label=name)\nplt.xlabel("Training episode");plt.ylabel("Known-value RMSE");plt.legend();plt.show()',
                ),
                (
                    "Preset step sizes are not method rankings",
                    'read("results/reference/lectures/metrics.json")["prediction"]["protocol"]',
                ),
            ],
            "lectures",
        ),
        (
            "10_linear_approximation",
            "Tile-coded SARSA and replacing traces",
            "Official MountainCar needs momentum-building behavior. Eight tilings of normalized position and velocity produce sparse linear action values. Lambda zero and .9 compare trace propagation under a fixed episode budget; actual interaction counts differ and are recorded.",
            [
                (
                    "Sparse features and action separation",
                    'from decision_lab.approximation import Tiles\ntiles=Tiles()\nstate=np.array([-.5,0.])\nindices=[tiles.indices(state,a) for a in range(3)]\nassert all(len(np.unique(g))==8 for g in indices)\nassert not np.intersect1d(indices[0],indices[1]).size\nweights=np.zeros(tiles.size);weights[indices[2]]=1\nassert np.array_equal(tiles.q(weights,state),[0,0,8])\nprint("features",tiles.size,"active per action",len(indices[0]))',
                ),
                (
                    "Validation selection and held-out success",
                    'rows=[]\nfor p in (ROOT / "results/reference/approximation").rglob("metrics.json"):\n    for r in json.loads(p.read_text())["rows"]:\n        rows.append({"lambda":r["lambda"],"seed":r["seed"],"steps":r["training_steps"],"test_return":r["test"]["mean"],"success_fraction":r["success_fraction"]})\npd.DataFrame(rows)',
                ),
                (
                    "Raw episode evidence",
                    'data=np.load(ROOT / "assets/source_data/approximation/lambda0.9_0.npz")\nassert np.array_equal(data["seeds"],np.arange(100)+300000)\nprint("success",data["success"].mean(),"mean steps",data["lengths"].mean())\nplt.hist(data["lengths"],bins=15);plt.xlabel("Held-out episode length");plt.ylabel("Episodes");plt.show()',
                ),
            ],
            "approximation",
        ),
        (
            "11_off_policy_instability",
            "Why semigradient TD can diverge",
            "A seven-state Baird-style example isolates the deadly combination of approximation, bootstrapping and off-policy learning. This is an expected-operator illustration, not a sampled-data contest. Feature covariance is singular; reported errors depend on represented values, not arbitrary weight directions.",
            [
                (
                    "Inspect the unstable operator",
                    'from decision_lab.lecture_studies import baird_operators\nphi,c,a=baird_operators()\neigen=np.linalg.eigvals(a)\nassert np.linalg.matrix_rank(c)==7 and eigen.real.min()<0\nprint("operator eigenvalues",eigen)\nprint("covariance rank",np.linalg.matrix_rank(c))',
                ),
                (
                    "Unclipped recorded trajectories",
                    'data=np.load(ROOT / "assets/source_data/lectures/baird.npz")\nfor name in data.files:\n    curve=data[name]\n    assert np.isfinite(curve).all()\n    plt.semilogy(curve[:,0],np.maximum(curve[:,1],1e-16),label=name)\nplt.xlabel("Expected update");plt.ylabel("Value RMS (zero shown at 1e-16)");plt.legend();plt.show()',
                ),
                (
                    "Distinguish GTD2 from the known-model reference",
                    'print(read("results/reference/lectures/metrics.json")["baird"]["protocol"])\nprint("TD growth",data["semigradient_td"][-1,1]/data["semigradient_td"][0,1])\nassert data["gtd2"][-1,1]<data["gtd2"][0,1]',
                ),
            ],
            "lectures",
        ),
    ]
)


def main():
    directory = Path("notebooks")
    directory.mkdir(exist_ok=True)
    for stem, title, intro, sections, command in STUDIES:
        nb = nbf.v4.new_notebook()
        cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}"), nbf.v4.new_code_cell(SETUP)]
        for heading, code in sections:
            cells.extend([nbf.v4.new_markdown_cell(f"## {heading}"), nbf.v4.new_code_cell(code)])
        cells.append(
            nbf.v4.new_markdown_cell(
                f"Full fits: `decision-lab {command} --seed 0` (repeat seeds 1 and 2 where applicable). See [protocol](../docs/protocol.md), [results](../docs/results.md) and [references](../docs/references.md). These cells are numerical checks and evidence readers, not replacement training budgets."
            )
        )
        nb["cells"] = cells
        nb["metadata"] = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        }
        nbf.write(nb, directory / f"{stem}.ipynb")


if __name__ == "__main__":
    main()
