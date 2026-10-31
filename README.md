# TraversalMap

Implementation of **Traversal-order-constrained defect mapping for autonomous sewer inspection**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

This implementation operates on pipe-coordinate observations and uncertainty. It implements sparse soft-association/order, incremental fusion, a joint pose/landmark state block, and auditable maintenance records. Supplied pose estimates initialize uncertain variables rather than being treated as fixed ground truth. Original surveyed traversals, Sewer-ML front-end weights and experiment logs are not bundled. Generated examples are functional inputs, not evidence for the paper's performance table.

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

Additional controls are `fixed_order_tolerance`, `no_soft_hypotheses`, `appearance_only`, `position_only`, and `generic_smoothness`, and `no_joint_optimization`. `shuffled_order` permutes only the observation-pair sequence used by the order factor (deterministic `order_seed`, default 17), while keeping each observation's local scores and final index fixed. It does not reverse a landmark-label axis. `fixed_order_tolerance_m` defaults to 0.5 m; `hard_order` applies a 10000 penalty to inversions beyond tolerance. `no_soft_hypotheses` uses greedy local assignment; `fixed_map` freezes coordinate fusion after the anchor traversal. The frozen core has no optional loop-audit term, matching the paper's λ_l=0 core setting.

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
| Incremental update and joint pose/landmark state block | `core.update_landmark`, `joint.PoseLandmarkProblem`, `joint.optimize_joint` | `independent_observations`, `joint_optimization`, `pose_prior_covariance`, optional odometry |
| Class/severity distributions, intervals, source frames, review and worklist | `maintenance.build_records`, `maintenance.export_records` | `class_labels`, `severity_levels`, `maintenance`; JSON/CSV outputs |
| Work-list precision/recall, additions/deletions/updates and severity agreement | `maintenance.score_maintenance` | `worklist_*`, `maintenance_*`, `severity_*` |
| Physical-identity association F1 and precision/recall | `core.score` | `association_*`, including same-identity pair counts |
| Unique-count, duplicates/misses/crossings and localization | `core.score` | `unique_count_error`, `*_landmark_rate`, `axial_rmse`, `cyclic_angle_mae_radians` |
| Map records/km, measured latency/memory, plots and cluster intervals | `cli.execute`, `cli.plot`, `statistics` | `map_records_per_km`, `latency_ms`, `rss_mib` |

Association F1 is the pairwise identity-clustering F1 over all observations in a pipe episode. True positives are unordered observation pairs sharing both predicted and reference identity. Precision divides by all predicted same-identity pairs, recall by all reference same-identity pairs; this penalizes merged and split physical defects. `observation_identity_accuracy` is reported separately and is not called F1. If both reference and predicted same-identity pair sets are empty, F1 is 1; if only one is empty, F1 is 0. Episode summaries average episode metrics; pair counts allow a separately declared pooled F1.

For coordinate error and duplicate/missed rates, the evaluator uses one-to-one Hungarian matching with cost `abs(axial_error_m)+abs(wrapped_angle_error_rad)` and the explicit `evaluation_matching_gate` (default 2 in those combined units). This is a declared evaluation convention where the manuscript does not supply a numeric gate; calibrate/freeze it before test. Angular MAE is in radians; convert to degrees for the paper table. Reference identities and coordinates never enter association. Group all traversals of a physical pipe together before making the paper's 60/15/10/15 train/validation/calibration/test partitions. This geometry/semantic-likelihood back end has no learned component or training split; front-end fitting and threshold selection occur upstream. The statistics command bootstraps declared clusters rather than the full pipe/defect hierarchy.

## Functional validation and remaining integrations

```bash
python verify.py --output runs/functional-check
```

The check runs the existing four covariance/cyclic-fusion regressions, reads/writes two generated forward/reverse episodes, executes all twelve variants, checks serialization/replay and truth-label isolation, verifies a hand-counted merged-identity example (precision 0.4, recall 1, F1 4/7, observation accuracy 0.4), checks cyclic wrap, and renders comparison plots. Environment: Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, Matplotlib 3.10.0, CPU with one BLAS thread.

Pipe/SLAM front ends, JCBB/probabilistic-semantic/data-association-free SLAM baseline implementations and the optional loop audit remain external. Their loop acceptance/rejection metrics are not fabricated. The joint state block and class/severity maintenance/export/metrics are now executable; original survey data, pose/reference logs and the external algorithms are still needed for full paper reproduction.

## Joint pose/landmark optimization

The existing association path first creates sparse soft A and incremental map initialization. `joint.optimize_joint` then jointly optimizes every semantic frame's `(pose_s, roll)` and every landmark's `(s, angle)`, conditional on those retained A weights. This is one state-block solve of the paper objective, rather than a claim of global minimization over discrete identities. Its whitened residuals are absolute pose priors, optional measured odometry, `sqrt(A) * [pose_s + axial_offset - landmark_s, wrap(image_angle + roll - landmark_angle)]`, and the existing weighted direction/order terms. The entropy/cardinality terms are constant in this state block. An analytic Jacobian is used with SciPy least squares; convergence failure raises an error. The optional loop weight remains zero.

The input contract is compatible with existing `s`/`angle`/`roll` observations:

- `frame_id` shares one pose variable across detections from one traversal frame. Omitted IDs use observation indices.
- `pose_s` is the proposed camera axial position; `axial_offset` is the measured displacement to the defect. If one is omitted it is inferred from `s = pose_s + axial_offset`; if both are absent, `pose_s=s` and offset=0 are the documented development convention. Shared frames must have identical pose priors.
- Pose prior variances are `pose_variance_s` and `pose_variance_roll`; defaults are `variance_s` and positive `variance_roll`. An absent/zero roll variance uses `default_pose_roll_variance=1e-4 rad²`.
- Conditional measurement variances are `measurement_variance_s` and `measurement_variance_angle`, or an SPD 2x2 `measurement_covariance`. Legacy defaults are `variance_s`/`variance_angle`. Real data should supply the separate components when the legacy variance already propagates pose uncertainty.
- A complete SPD `pose_prior_covariance` may encode correlated priors, ordered by first occurrence of `(traversal,frame_id)` with `(s,roll)` pairs. Otherwise diagonal priors are used. Absolute priors fix the gauge; no reference truth enters optimization. Residual centerline/rho dimensions are held fixed by the local cylindrical pipe model.
- Optional traversal `odometry` entries contain `from`, `to` frame IDs, `delta_s`, `delta_roll`, and either a 2x2 `covariance` or `variance_s`/`variance_roll`. Supply actual relative measurements; the code does not invent independent odometry from the same pose priors.

Gaussian marginal covariance is obtained from the joint information matrix, not the old fused landmark as an additional prior. When `independent_observations=false`, unknown-correlation marginal floors retain at least the input pose and incremental-CI map variances; this conservative floor is an implementation convention, not a general covariance-intersection guarantee for the nonlinear graph. Pose means, 2x2 pose/map covariance, state-block dimensions, initial/final cost, gradient norm, changes and solver convergence are retained. `joint_probability_cutoff=1e-8` and `joint_max_evaluations=100` are numerical conventions. The state block is enabled by default; `joint_optimization=false`, `fixed_map`, or `no_joint_optimization` preserve the supplied/incremental coordinate control.

## Maintenance input, export and scoring

Provide `class_labels` and ordered `severity_levels` (for example `[0,1,2,3]`) plus observation `class_probabilities`/`severity_probabilities`. Omitted label names use zero-based IDs inferred from observation vectors; reference truth is never used to choose labels. `quality` defaults to 1. Accepted observations pool their normalized likelihoods with weights `association_probability * quality`. This is a linear opinion pool: repeated identical likelihoods do not become artificially certain. Missing semantic evidence stays null and sets a review flag.

The `maintenance` policy may declare `actionable_classes`, `minimum_severity`, `action_probability_threshold` (default .5), confidence cutoffs (default .5), `minimum_source_frames` (default 2), and axial/clock standard-deviation review cutoffs (default .5m/pi/4rad). Default actionable classes include all configured classes; default minimum severity is the second ordered level. Action probability is the product of actionable-class and above-threshold severity marginal mass, an explicit interface convention where the paper does not define a decision policy.

Every map record has a stable episode/map ID plus landmark index, 95% axial interval, wrapped clock interval with `wraps`/`full_circle` markers, class/severity distributions, source frames, association probabilities, covariance and review reasons. Clock zero is vertically up, with positive angles interpreted clockwise; `clock_hours` maps zero to 12. JSON preserves the complete map, actionable worklist and review list; the CSV contains actionable records and their audit fields. Inference runs without `truth` still export these records and measured runtime/map size, while omitting reference-based accuracy metrics.

```bash
python -m traversalmap run --data data/annotated-or-unlabelled.jsonl --output runs/main
python -m traversalmap export --records runs/main/traversalmap/records.jsonl --output results/maintenance
```

`run`/`suite` automatically write `maintenance/case-000000.json` and `case-000000-worklist.csv` inside each method directory. Indexed filenames avoid interpreting episode IDs as paths. The standalone export command reads saved results without refitting.

For evaluation, reference landmarks additionally supply `class` and `severity`; the existing one-to-one geometric matching is reused. Worklist precision/recall count matched actionable physical defects, not frame detections. Edit operations are worklist additions for missed actionable defects, deletions for false records, and one update per matched actionable record with incorrect class or severity. Severity agreement is exact ordinal-label agreement on matched annotated records; ordinal MAE and class agreement are also emitted. Incomplete reference maintenance annotations leave worklist metrics null.

The extended CPU verification passed a hand-derived joint problem: priors 0/2 and one landmark yield poses .5/1.5, map s=1, objective 2→.5, pose axial variances .75/.75 and map variance 1; analytic Jacobians match central differences at 1e-8. A measured odometry edge produces poses .7/1.3 as derived. A quality-weighted semantic example gives class [.6,.4] and severity [0,1/3,2/3]. A deliberately imperfect maintenance list gives precision 1/3, recall 1/2, four record edits and severity agreement 1/3. JSON/CSV readback preserves fields and recomputed metrics, including a clock interval crossing pi. `batch_optimization_seconds` records the actual state solve separately from complete graph-update latency. These are generated functionality checks, not manuscript survey results.
