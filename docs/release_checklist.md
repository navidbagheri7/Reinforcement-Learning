# Release validation — 1.1.0

The completed reference contains98 protocols and156 fitted policies:60 tabular,12 owned neural,36 offline,18 maintained-library control,6 tile-coded and24 new MiniGrid/MuJoCo policies. All configurations use the recorded fitting seeds and selection rules. Source bindings match the frozen fitting modules. The eight new presets each use three fitting seeds, 1,048,576 training interactions and100 held-out test episodes per fit.

| Check | Local result |
|---|---|
| Numerical/unit suite | 34 tests passed from an installed wheel outside the checkout |
| Reference completion | All98 protocol records complete; no smoke records in reference |
| Checkpoint deployment | 108 finite exports,24 original continuous-observation checkpoints and24 new portable neural actors audited |
| Fresh finite rollouts | Three independent test resets reproduced per exported policy |
| Fresh control rollouts | Three test and three shifted resets reproduced where shifts exist |
| Fresh simulation rollouts | All24 portable actors reproduced three complete test episodes each without private checkpoints |
| Notebooks | Thirteen English walkthroughs; all54 code cells executed in fresh matching kernels |
| Numeric evidence |222 NPZ artifacts checked for finite, pickle-free arrays |
| Scientific figures |37 PNG/PDF pairs regenerated from numeric records |
| Rendered illustrations | All30 GIFs bound to the predetermined first test episode; return, length and file digest verified |
| Interactive dashboard | All eight presets, three images/curves per preset, desktop and390px mobile tested in Chromium; no JavaScript errors |
| Packaging | Wheel includes10 default configs; source archive contains98 protocols,24 portable actors,30 GIFs and13 notebooks |
| Public exclusions | Raw coursework, environments, private checkpoints, caches and superseded runs excluded |

[validation.json](validation.json), [checkpoint_audit.json](checkpoint_audit.json), [policy_audit.json](policy_audit.json), [simulation_validation.json](simulation_validation.json), [simulation_actor_audit.json](simulation_actor_audit.json), [simulation_browser_validation.json](simulation_browser_validation.json), [notebook_execution.json](notebook_execution.json) and [experiment_index.json](experiment_index.json) provide the portable records. Large training weights remain available locally; compact new actors are public. Conditional episode intervals and fitting-seed variability are kept separate. The maze policies use the declared full symbolic map. Failed seeds and weak ablations remain in the comparison and gallery.

Distribution checks extract the source archive into a fresh directory, inspect every member for private paths, run the installed-wheel test suite outside the checkout, validate the archive's source/evidence bindings, and execute all notebooks again from the extracted tree. Detailed local packaging logs are kept under ignored `.local/`; GitHub CI is supplied but is not claimed to have run remotely.

The owner publishes this local release. Follow [publication.md](publication.md); no GitHub remote or push is created by the rebuild. Code is MIT licensed and privately preserved original materials are excluded from that license grant and public distribution.
