# Executed learning path

All walkthroughs are in English and retain their executed outputs. Open in numeric order: coherent wireless process; exact planning; resource constraints; tabular control; neural targets; logged-data learning; standard control; bandits; prediction; linear approximation; off-policy instability.

The notebooks run small independent numerical checks and read complete reference artifacts. They do not launch full fitting jobs. Install the notebook and deep extras before running all notebooks. From the repository root, execute `python scripts/execute_notebooks.py`. Full experiment commands are in the README and `docs/reproduction.md`.

Recorded figures and tables do not replace the underlying arrays: every comparison links back to source data, fitted seeds and run records. Core-only users can read all notebook outputs; neural-loss cells require PyTorch to execute.

12. [Official navigation](12_official_navigation.ipynb): full-map observations, reward sparsity, key/door manipulation, potential telescoping, BFS model access and deployed actors.
13. [MuJoCo locomotion](13_mujoco_locomotion.ipynb): PPO/SAC budgets, validation selection, training variation and portable continuous actors.

The added walkthroughs require the `simulations` extra. Opening them reads measured evidence and replays short fixed test episodes; it does not launch full training.
