"""Embed measured evidence in a local, dependency-free interactive HTML gallery."""

import argparse
import json
from pathlib import Path

from decision_lab.artifacts import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    rows = []
    for family in ["simulations", "geometry", "curriculum", "modern"]:
        for path in sorted(Path("results/reference", family).glob("*/*/metrics.json")):
            m = json.loads(path.read_text())
            record = json.loads(path.with_name("run.json").read_text())
            assert record["status"] == "complete"
            stem = f"{m['env']}_{m['algorithm']}_{m['profile']}_{m['seed']}"
            episode = json.loads(Path("assets/rollouts", stem + ".json").read_text())
            rows.append(
                dict(
                    env=m["env"],
                    algorithm=m["algorithm"],
                    profile=m["profile"],
                    key=f"{m['algorithm']}_{m['profile']}",
                    seed=m["seed"],
                    test=m["test"],
                    success=m["success_fraction"],
                    history=m["validation_history"],
                    gif="../assets/rollouts/" + stem + ".gif",
                    episode=episode,
                    metrics="../" + str(path),
                )
            )
    assert args.allow_partial or len(rows) == 42
    baselines = json.loads(Path("docs/simulation_baselines.json").read_text())["rows"]
    encoded = json.dumps(dict(fits=rows, baselines=baselines), allow_nan=False).replace(
        "<", "\\u003c"
    )
    template = Path("scripts/templates/simulation_dashboard.html").read_text()
    output = Path("docs/simulation_dashboard.html")
    output.write_text(template.replace("__EVIDENCE__", encoded))
    write_json(
        "docs/simulation_dashboard.json",
        dict(
            status="passed" if len(rows) == 42 else "partial",
            fits=len(rows),
            html_sha256=sha256(output),
            template_sha256=sha256("scripts/templates/simulation_dashboard.html"),
            generator_sha256=sha256(__file__),
        ),
    )
    print("Embedded", len(rows), "fits in", output)


if __name__ == "__main__":
    main()
