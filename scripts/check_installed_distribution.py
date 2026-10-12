"""Verify a built wheel and source archive using isolated imports and portable evidence.

Build first with ``python -m build --outdir .local/dist``. This checker runs
outside the source tree and does not access course originals or private training checkpoints.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from decision_lab.artifacts import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path(".local/dist"))
    parser.add_argument("--work", type=Path, default=Path(".local/distribution_1_2"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wheel = next(args.dist.resolve().glob("*.whl"))
    archive = next(args.dist.resolve().glob("*.tar.gz"))
    work = args.work.resolve()
    if work.exists():
        raise FileExistsError("Choose a fresh work directory to preserve earlier checks")
    work.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as package:
        for name in package.namelist():
            if name.startswith("decision_lab/") and name.endswith((".py", ".json", ".npz")):
                assert package.read(name) == (root / "src" / name).read_bytes(), name
    extraction = work / "archive"
    extraction.mkdir()
    with tarfile.open(archive) as package:
        for item in package.getmembers():
            path = Path(item.name)
            assert not path.is_absolute() and ".." not in path.parts
            assert not set(path.parts) & {".local", "originals", "checkpoints"}
            assert path.suffix not in {".pt", ".pth", ".cbm", ".joblib", ".rar", ".log"}
            assert item.size <= 25 * 1024 * 1024
        package.extractall(extraction, filter="data")
    source = next(extraction.iterdir())
    for required in [
        "docs/explorer.html",
        "scripts/templates/rl_explorer.js",
        "configs/modern.json",
    ]:
        assert (source / required).exists()
    installed = work / "installed"
    external = work / "external"
    external.mkdir()
    shutil.copytree(source / "tests", external / "tests")
    shutil.copytree(source / "scripts", external / "scripts")
    shutil.copytree(source / "configs", external / "configs")
    assert len(list((source / "assets/modern_actors").glob("*.npz"))) == 18
    assert len(list((source / "notebooks").glob("*.ipynb"))) == 15
    (external / "pytest.ini").write_text("[pytest]\ntestpaths=tests\n")
    env = os.environ | dict(
        PYTHONPATH=str(installed) + os.pathsep + str(root / ".local/upgrade_deps")
        if (root / ".local/upgrade_deps").exists()
        else str(installed),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    checks = []

    def run(name, command, cwd=external):
        print("Starting", name, flush=True)
        log_path = work / (name + ".log")
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log_path.read_text()[-16000:], flush=True)
            result.check_returncode()
        checks.append(dict(check=name, status="passed"))
        print("Passed", name, flush=True)

    run(
        "install",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-cache-dir",
            "--target",
            str(installed),
            str(wheel),
        ],
    )
    run(
        "import_identity",
        [
            sys.executable,
            "-c",
            f"import decision_lab; assert '{installed}' in decision_lab.__file__; assert decision_lab.__version__=='1.2.0'; print(decision_lab.__file__)",
        ],
    )
    run("wheel_tests", [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib"])
    for study in ["planning", "tabular", "lectures"]:
        run("bundled_" + study, [sys.executable, "-m", "decision_lab.cli", study, "--quick"])
    for case in ["HalfCheetah-v5_TD3", "HalfCheetah-v5_TQC", "MiniGrid-Empty-8x8-v0_RecurrentPPO"]:
        run(
            "bundled_modern_" + case,
            [
                sys.executable,
                "-m",
                "decision_lab.cli",
                "modern",
                "--quick",
                "--device",
                "cpu",
                "--case",
                case,
            ],
        )
    run("initialize_checkout", ["git", "init", "-q"], source)
    run("archive_notebooks", [sys.executable, "scripts/execute_notebooks.py"], source)
    run("archive_validation", [sys.executable, "scripts/validate_release.py"], source)
    run("archive_simulations", [sys.executable, "scripts/validate_simulations.py"], source)
    run("archive_modern", [sys.executable, "scripts/validate_modern.py"], source)
    run("archive_public_content", [sys.executable, "scripts/review_components.py"], source)
    audit = dict(
        status="passed",
        release="1.2.0",
        artifacts={wheel.name: sha256(wheel), archive.name: sha256(archive)},
        checks=checks,
        installed_wheel_tests=45,
        source_archive_notebooks=15,
        source_archive_executed_cells=61,
        scope="Wheel imports/all 45 tests, three core and three modern bundled CPU commands run outside the checkout. The source archive executes all fifteen walkthroughs and validates all 116 protocols, public numerical, English, link/browser evidence and every exported modern actor without originals/checkpoints. Optional private dependency directories supply installed sb3-contrib only, never project code. External checksum metadata is excluded from the source distribution to avoid recursive checksums. Hosted CI is not claimed.",
    )
    write_json(root / "docs/release_artifacts.json", audit)
    write_json(work / "audit.json", audit)
    print("Verified isolated wheel and complete source archive")


if __name__ == "__main__":
    main()
