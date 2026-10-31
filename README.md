# TraversalMap

Implementation of **Traversal-order-constrained defect mapping for autonomous sewer inspection**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

This implementation operates on pipe-coordinate observations and uncertainty. It implements the sparse soft-association/order mechanism and incremental landmark fusion, using supplied pose estimates. Original surveyed traversals, Sewer-ML front-end weights and experiment logs are not bundled. Generated examples are functional inputs, not evidence for the paper's performance table.

## Data

A pipe episode contains traversal direction and ordered observations with axial/angle estimates, roll, variances and descriptors. The first traversal initializes landmarks. truth_id and independent reference landmarks are used only by scoring.

Axial coordinates use meters, angle and roll use radians, and variances use squared units. `direction` is +1 for forward and −1 for reverse traversal within a single branch-free segment. Roll is added before cyclic comparison; `variance_roll` is included in angle uncertainty. Landmarks are not initialized from `truth`. `independent_observations=true` permits independent Gaussian coordinate fusion; otherwise covariance intersection avoids treating shared odometry as independent evidence.

Each case declares `order_weight`, `cardinality_weight`, `null_cost`, `top_l`, `kappa`, and `map_length_meters`. Local scores include appearance squared distance and covariance-normalized axial/cyclic-angle errors. Candidate hypotheses are restricted to the top L within `axial_gate_sigma` standard deviations (default 5), plus a null/new-landmark hypothesis. The order penalty uses consecutive semantic observations and their direction. `delta_s_variance` can supply the relative-motion variance; otherwise the sum of the two observation variances is the declared independence approximation. The soft objective adds an entropy term with coefficient 0.1 and uses L-BFGS-B with an analytic gradient. These are implementation conventions; they are not recovered original experiment settings.

The bundled generator writes synthetic finite-model scenarios as JSONL. Fix the seed and keep the generated input for paired comparisons.

```bash
python -m traversalmap prepare --count 200 --seed 17 --output data/generated.jsonl
```

## Execution and evaluation

```bash
python -m traversalmap run --data data/generated.jsonl --output runs/main
```

## Comparisons and ablations

```bash
python -m traversalmap suite --data data/generated.jsonl --output runs/comparison
```

Use `--methods` to select: `traversalmap`, `local_only`, `no_cyclic_angle`, `shuffled_order`, `hard_order`, `fixed_map`.

Additional controls are `fixed_order_tolerance`, `no_soft_hypotheses`, `appearance_only`, `position_only`, and `generic_smoothness`. `shuffled_order` permutes only the observation-pair sequence used by the order factor (deterministic `order_seed`, default 17), while keeping each observation's local scores and final index fixed. It does not reverse a landmark-label axis. `fixed_order_tolerance_m` defaults to 0.5 m; `hard_order` applies a 10000 penalty to inversions beyond tolerance. `no_soft_hypotheses` uses greedy local assignment; `fixed_map` freezes coordinate fusion after the anchor traversal. The frozen core has no optional loop-audit term, matching the paper's λ_l=0 core setting.

## Outputs

`runs/main/traversalmap/records.jsonl` stores per-case results; `metrics.json` stores aggregate measurements. Comparison outputs use one directory per method.

```bash
python -m traversalmap plot --inputs runs/comparison/traversalmap --output figures
```

Plotting writes `figures/summary.csv` and metric PNGs. Statistical intervals can be computed from a retained record file:

```bash
python -m traversalmap.statistics --records runs/main/traversalmap/records.jsonl --output results/intervals.json
```

## Method and metric coverage

| Paper quantity | Code entry | Configuration/output |
|---|---|---|
| Cyclic pipe coordinates and roll uncertainty | `core.wrap`, `core.make_landmark` | `angle`, `roll`, `variance_*` |
| Sparse local hypotheses, null identity and cardinality | `core.optimize` | `top_l`, `axial_gate_sigma`, `null_cost`, `cardinality_weight` |
| Direction/uncertainty-conditioned order factor | `core.optimize` | `order_weight`, `kappa`, `delta_s_variance` |
| Incremental landmark update with unknown-correlation handling | `core.update_landmark`, `core.run_case` | `independent_observations` |
| Physical-identity association F1 and precision/recall | `core.score` | `association_*`, including same-identity pair counts |
| Unique-count, duplicates/misses/crossings and localization | `core.score` | `unique_count_error`, `*_landmark_rate`, `axial_rmse`, `cyclic_angle_mae_radians` |
| Map records/km, measured latency/memory, plots and cluster intervals | `cli.execute`, `cli.plot`, `statistics` | `map_records_per_km`, `latency_ms`, `rss_mib` |

Association F1 is the pairwise identity-clustering F1 over all observations in a pipe episode. True positives are unordered observation pairs sharing both predicted and reference identity. Precision divides by all predicted same-identity pairs, recall by all reference same-identity pairs; this penalizes merged and split physical defects. `observation_identity_accuracy` is reported separately and is not called F1. If both reference and predicted same-identity pair sets are empty, F1 is 1; if only one is empty, F1 is 0. Episode summaries average episode metrics; pair counts allow a separately declared pooled F1.

For coordinate error and duplicate/missed rates, the evaluator uses one-to-one Hungarian matching with cost `abs(axial_error_m)+abs(wrapped_angle_error_rad)` and the explicit `evaluation_matching_gate` (default 2 in those combined units). This is a declared evaluation convention where the manuscript does not supply a numeric gate; calibrate/freeze it before test. Angular MAE is in radians; convert to degrees for the paper table. Reference identities and coordinates never enter association. Group all traversals of a physical pipe together before making the paper's 60/15/10/15 train/validation/calibration/test partitions. This geometry-level estimator has no learned component or training split; front-end fitting and threshold selection occur upstream. The statistics command bootstraps declared clusters rather than the full pipe/defect hierarchy.

## Functional validation and remaining integrations

```bash
python verify.py --output runs/functional-check
```

The check runs the existing four covariance/cyclic-fusion regressions, reads/writes two generated forward/reverse episodes, executes all eleven variants, checks serialization/replay and truth-label isolation, verifies a hand-counted merged-identity example (precision 0.4, recall 1, F1 4/7, observation accuracy 0.4), checks cyclic wrap, and renders comparison plots. Environment: Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, Matplotlib 3.10.0, CPU with one BLAS thread.

Joint pose/landmark optimization, pipe/SLAM front ends, JCBB/probabilistic-semantic/data-association-free SLAM baselines, the optional loop audit and class/severity maintenance export require external integration. Loop acceptance/rejection, work-list precision/recall, edit operations and severity agreement cannot be evaluated from the current geometry/identity schema. The present adapter validates the order-conditioned association mechanism; original survey data, pose/reference logs and the missing integrations are still needed for full paper reproduction.
