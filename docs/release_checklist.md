# Release validation — 1.0.0

The completed reference contains74 protocols and132 fitted policies:60 tabular,12 owned neural,36 offline,18 maintained-library control and6 tile-coded policies. All configurations use the recorded fitting seeds and selection rules. Source bindings match the frozen fitting modules.

| Check | Local result |
|---|---|
| Numerical/unit suite | 24 tests passed from an installed wheel outside the checkout |
| Reference completion | All74 protocol records complete; no smoke records in reference |
| Checkpoint deployment | 108 finite exports and24 continuous-observation checkpoints audited |
| Fresh finite rollouts | Three independent test resets reproduced per exported policy |
| Fresh control rollouts | Three test and three shifted resets reproduced where shifts exist |
| Notebooks | Eleven English walkthroughs; all44 code cells executed in fresh matching kernels |
| Numeric evidence |169 NPZ artifacts checked for finite, pickle-free arrays |
| Scientific figures |31 PNG/PDF pairs regenerated from numeric records |
| Rendered illustration | Fixed first LunarLander test reset; rendered return exactly matches numeric evidence |
| Packaging | Wheel includes default configs; installed CLI works outside the checkout |
| Public exclusions | Raw coursework, environments, private checkpoints, caches and superseded runs excluded |

[validation.json](validation.json), [checkpoint_audit.json](checkpoint_audit.json), [policy_audit.json](policy_audit.json), [notebook_execution.json](notebook_execution.json) and [experiment_index.json](experiment_index.json) provide the portable records. Large weights remain available locally. Conditional episode intervals and fitting-seed variability are kept separate.

Distribution checks extract the source archive into a fresh directory, inspect every member for private paths, run the installed-wheel test suite outside the checkout, validate the archive's source/evidence bindings, and execute all notebooks again from the extracted tree. Detailed local packaging logs are kept under ignored `.local/`; GitHub CI is supplied but is not claimed to have run remotely.

The owner publishes this local release. Follow [publication.md](publication.md); no GitHub remote or push is created by the rebuild. Code is MIT licensed and privately preserved original materials are excluded from that license grant and public distribution.
