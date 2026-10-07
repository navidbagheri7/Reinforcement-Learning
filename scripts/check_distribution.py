"""Inspect release distributions for portable evidence and private-file exclusions."""

import argparse
import json
import tarfile
import zipfile
from pathlib import Path

from decision_lab.artifacts import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", default="dist", help="Directory with exactly one wheel and source archive"
    )
    parser.add_argument("--output", default=".local/distribution_audit.json")
    args = parser.parse_args()
    directory = Path(args.directory)
    archives, wheels = list(directory.glob("*.tar.gz")), list(directory.glob("*.whl"))
    assert len(archives) == len(wheels) == 1, "Use a clean distribution directory"
    forbidden = {".local", "originals", "checkpoints", "smoke", "__pycache__", ".git"}
    with tarfile.open(archives[0]) as archive:
        members = archive.getmembers()
        names = set()
        for member in members:
            path = Path(member.name)
            assert not path.is_absolute() and ".." not in path.parts
            assert not member.issym() and not member.islnk()
            assert not (forbidden & set(path.parts)), member.name
            assert not member.name.endswith((".pt", ".zip", ".log", ".pyc")), member.name
            if member.isfile():
                relative = Path(*path.parts[1:])
                names.add(str(relative))
                local = Path(relative)
                if (
                    local.exists()
                    and local.is_file()
                    and relative.parts[0]
                    in [
                        "src",
                        "assets",
                        "notebooks",
                        "results",
                        "docs",
                        "configs",
                        "scripts",
                        "tests",
                    ]
                ):
                    assert archive.extractfile(member).read() == local.read_bytes(), member.name
        assert "docs/simulation_dashboard.html" in names
        assert "scripts/templates/simulation_dashboard.html" in names
        assert (
            len([n for n in names if n.startswith("assets/actors/") and n.endswith(".npz")]) == 24
        )
        assert (
            len([n for n in names if n.startswith("assets/modern_actors/") and n.endswith(".npz")])
            == 18
        )
        assert (
            len([n for n in names if n.startswith("assets/rollouts/") and n.endswith(".gif")]) == 48
        )
        assert len([n for n in names if n.startswith("notebooks/") and n.endswith(".ipynb")]) == 15
        assert (
            len(
                [n for n in names if n.startswith("results/reference/") and n.endswith("/run.json")]
            )
            == 116
        )
    with zipfile.ZipFile(wheels[0]) as wheel:
        package_files = [name for name in wheel.namelist() if name.startswith("decision_lab/")]
        for name in package_files:
            assert wheel.read(name) == (Path("src") / name).read_bytes(), name
        configs = [
            name for name in package_files if "/default_configs/" in name and name.endswith(".json")
        ]
        assert len(configs) == 11
    result = dict(
        status="passed",
        sdist_sha256=sha256(archives[0]),
        wheel_sha256=sha256(wheels[0]),
        source_members=len(names),
        packaged_configs=len(configs),
        public_actors=42,
        rollout_gifs=48,
        executed_notebooks=15,
        reference_protocols=116,
        private_paths_excluded=True,
        distribution_bytes_match_checkout=True,
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
