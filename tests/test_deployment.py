import json

import numpy as np
import pytest

from decision_lab.deployment import load_policy
from decision_lab.wireless import WirelessModel


def test_schema_and_legal_action_contract(tmp_path):
    model = WirelessModel()
    record = {
        "schema_version": 1,
        "env": "Wireless",
        "wireless": {},
        "policy": np.zeros(len(model.states), int).tolist(),
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(record))
    _, make, policy = load_policy(path)
    env = make()
    _, info = env.reset(seed=12)
    assert info["action_mask"][policy[info["state"]]]
    env.close()
    record["policy"][0] = 3
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Illegal"):
        load_policy(path)
    record["schema_version"] = 2
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="schema"):
        load_policy(path)
