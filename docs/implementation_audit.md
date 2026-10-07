# Scientific implementation review

The public package separates owned wireless learning algorithms, exact known-model
solvers and maintained official-environment baselines. Original materials are
privately preserved with digests. Historical fit source remains bound to its
recorded files; extensions add records rather than replacing weak results.

| Component | Scientific contract and review |
|---|---|
| mdp, wireless, planning | A shared outcome operator defines simulation and analytical kernels; inventory/energy conservation, row stochasticity, model-access distinctions, Bellman bounds and occupancy primal/dual residuals are checked |
| tabular, toytext | Legal masks, retained stochastic transitions, time-limit bootstrap and genuine terminal resets; Q/SARSA/Expected SARSA/Double Q are paired against the same tasks and test protocols |
| networks, replay, deep | Owned target construction and target freezing, dueling centering over legal actions, prioritized replay weighting, double selection and quantile Huber/crossing diagnostics; quantiles are not epistemic intervals |
| offline | Shared logged datasets and support profiles, training-only updates, discrete CQL/IQL formulations and masks; exact model evaluation clarifies poor coverage and conservative extraction failures |
| control, simulation studies | Maintained library versions, explicit dynamics/observation/reward wrappers, fixed validation/test seeds and checkpoint selection; parameter/device/budget differences remain visible |
| geometry, curriculum | Full-map symbolic observations, geodesic potential telescoping, declared layout information and curriculum stages; official sparse test rewards prevent comparing shaped returns with baseline returns |
| modern | Maintained TD3/TQC and partial-view PPO/RecurrentPPO; no global map/pose/carry/time in policy input; actual completed interactions, first-best validation selection and independent held-out deployment |
| approximation | Tile coding, accumulating traces, epsilon-greedy bootstraps and true-terminal/time-limit distinctions on MountainCar |
| bandits, prediction, counterexamples | Instance-level variation, actual pseudo-regret, analytical prediction targets and unclipped off-policy semigradient divergence; lecture instance counts are distinct from fitting seeds |
| deployment and actors | Explicit finite policies or architecture-checked float32 weights, source identities and deterministic inference; recurrent hidden/cell state reset/persistence and continuous-action squashing agree with maintained policies |
| reporting and explorers | All fitted seeds retained, episode uncertainty separated from fitting variation, actual measured histories/GIFs, explicit renderer versus observation access, independent live known-model numerical oracles |
| tests and release tools | Numerical invariant oracles, fresh policy commands, current notebook outputs, source/numeric/actor/media identity, English/local-link checks and installed-wheel/extracted-archive verification |

The [public inventory](component_review.json) records parseable component identities
and local links. The [main evidence validator](validation.json),
[simulation validator](simulation_validation.json), [modern validator](modern_validation.json)
and browser records have separate scopes. Private-checkpoint audits are available
locally and have their own recorded identities; fresh-public replay does not
require those private objects.

The extension review corrected portable TQC squashing/unscaling to preserve the
maintained CPU actor's float32 operations and verified recurrent memory explicitly.
The final optimizer update is evaluated before selection ends. Known-model maze
controls calculate Bellman iterates; they never pretend to replay learned-policy
snapshots. Pareto curves use distinct labeled series, and lecture comparison rows
show their independent instance counts. Superseded smoke and incomplete runs stay
private, with execution failures preserved rather than silently called successful.

The v1.2 deployment review also identified float32 categorical ties: a raw
logit maximum can differ from SB3's normalized-probability maximum. Both portable
PPO implementations now preserve the maintained policy's normalization and tie
rule. Three independent library-policy regression cases cover full-map PPO,
local-view PPO and recurrent PPO. Fitting modules and numerical reference records
are unchanged; full held-out deployment audits check the selected actors after
the correction.

Those regression cases also run after a caller selects a float64 global default.
Portable actors now construct their declared float32 CPU layers explicitly,
without changing the caller's precision setting. This preserves the model's
inference contract when combining it with double-precision scientific code.
The geometry actor also casts the frozen feature transform's output to its
declared float32 precision. Its regression case compares against the maintained
policy under both caller defaults; the fitting feature code remains unchanged.
