import json
from importlib.resources import files


def test_installed_default_protocols_are_complete():
    root = files("decision_lab") / "default_configs"
    expected = {
        "planning",
        "tabular",
        "neural",
        "offline",
        "control",
        "approximation",
        "lectures",
        "simulations",
        "geometry",
        "curriculum",
    }
    assert {
        item.name.removesuffix(".json") for item in root.iterdir() if item.name.endswith(".json")
    } == expected
    control = json.loads((root / "control.json").read_text())
    assert len(control["cases"]) == 6 and control["seeds"] == [0, 1, 2]
    offline = json.loads((root / "offline.json").read_text())
    assert offline["behavior_epsilon"] == [0.05, 0.4, 1.0] and offline["updates"] == 12000
