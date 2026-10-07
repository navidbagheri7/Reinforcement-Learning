# Release validation — 1.2.0

The completed reference contains 116 protocols and 174 fitted policies. It retains
the original 156 policies and adds eighteen full-budget TD3/TQC and local-view
PPO/RecurrentPPO policies. Each new preset uses three fitting seeds, 1,048,576
interactions, twelve fixed validation episodes and one hundred final test resets.
All fitting source bindings and declared actual budgets are checked.

| Check | Local evidence |
|---|---|
| Numerical/unit suite | 45 tests; the isolated-wheel run is recorded separately |
| Reference completion | All 116 protocol records complete; smoke records excluded |
| Finite deployment | 108 legal, schema-checked action-table exports |
| Simulation deployment | 24 original and 18 modern portable neural actors |
| Selected modern model audit | All 100 test episodes per actor exactly reproduced, plus eighteen independent three-episode commands |
| Fresh public replay | Separate validators run three full episodes per original/modern actor without private checkpoints |
| Notebooks | Fifteen English walkthroughs, all 61 code cells executed in fresh matching-interpreter kernels and bound by SHA256 |
| Numeric evidence | 258 finite, pickle-free NPZ artifacts |
| Scientific figures | 39 PNG/PDF pairs generated from numerical records |
| Rendered illustrations | All 48 GIFs bound to the predetermined first test episode, return, length and file digest |
| Concept explorer | Eight concept areas, 193 measured presets, 24 independent live Bellman checks and 390px mobile checks |
| Simulation dashboard | Fourteen presets, three actors/images/validation curves each, desktop and mobile browser checks |
| Distributions | Eleven bundled default configurations; complete public actors, protocols, notebooks and demos |
| Publication boundary | Original coursework, private checkpoints, environments, caches and interrupted/smoke runs excluded |

[validation.json](validation.json), [checkpoint_audit.json](checkpoint_audit.json),
[policy_audit.json](policy_audit.json),
[simulation_validation.json](simulation_validation.json),
[simulation_actor_audit.json](simulation_actor_audit.json),
[modern_validation.json](modern_validation.json),
[modern_actor_audit.json](modern_actor_audit.json),
[explorer_browser_validation.json](explorer_browser_validation.json),
[simulation_browser_validation.json](simulation_browser_validation.json),
[notebook_execution.json](notebook_execution.json),
[component_review.json](component_review.json) and
[experiment_index.json](experiment_index.json) record distinct verification scopes.

The original full-map maze policies receive stronger information than local-view
policies. Rendering exposes the global scene to readers, not to a partial-view
actor. Recurrent memory resets at each episode and persists inside the episode.
Official Empty fixes its layout/start/goal; repeated deterministic test resets
are not unseen-layout generalization. DoorKey is randomized. Three fitting seeds
and repeated validation still limit rankings. Failed seeds and predetermined
unsuccessful rollouts remain in the evidence. QR/TQC quantiles are return
distributions, not epistemic confidence intervals. Concurrent runtimes are not
hardware-normalized algorithm throughput.

`check_installed_distribution.py` installs the built wheel outside the checkout,
runs all tests and three core/three modern bundled CPU smoke commands, then
executes every notebook and validates the extracted source archive without
private datasets/checkpoints. The generated `docs/release_artifacts.json` records
artifact hashes and results; it is external to the archive to avoid recursive
checksums. Detailed logs remain under ignored `.local/`. Supplied GitHub CI is
not claimed to have run remotely. No exact training-resume guarantee is made.

The owner publishes this local release using [publication.md](publication.md).
No GitHub remote, push or live deployment is created. Code is MIT licensed;
privately preserved original materials are excluded from the license grant and
public distribution.
