import json

import pytest

from decision_lab.protocols import completed_batch


def test_completed_offline_rerun_preserves_dataset(tmp_path):
    config = {"recipes": ["bc", "cql"], "behavior_epsilon": [0.1]}
    assert not completed_batch("offline", config, 0, "cpu", False, tmp_path)
    for recipe in config["recipes"]:
        directory = tmp_path / "epsilon0.1" / recipe / "seed0"
        directory.mkdir(parents=True)
        identity = config | {
            "seed": 0,
            "behavior_epsilon": 0.1,
            "recipe": recipe,
            "device": "cpu",
            "quick": False,
        }
        (directory / "run.json").write_text(json.dumps({"status": "complete", "config": identity}))
    assert completed_batch("offline", config, 0, "cpu", False, tmp_path)
    with pytest.raises(ValueError, match="different"):
        completed_batch("offline", config, 0, "cuda", False, tmp_path)
    (tmp_path / "epsilon0.1/cql/seed0/run.json").unlink()
    with pytest.raises(ValueError, match="partially"):
        completed_batch("offline", config, 0, "cpu", False, tmp_path)
