"""Run one fitting seed and simulation group with explicit reference configurations."""

import argparse
import json

from decision_lab.simulations import train_case

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--group", choices=["navigation", "locomotion"], required=True)
args = parser.parse_args()
config = json.load(open("configs/simulations.json"))
cases = [
    c for c in config["cases"] if c["env"].startswith("MiniGrid") == (args.group == "navigation")
]
# Run the dense-reward protocols first; ordering does not affect isolated seeded fits.
cases.sort(key=lambda c: (c["profile"] != "potential", c["algorithm"] != "SAC", c["env"]))
for case in cases:
    train_case(config, case, args.seed)
