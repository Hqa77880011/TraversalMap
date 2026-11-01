# TraversalMap

Implementation of **Traversal-order-constrained defect mapping for autonomous sewer inspection**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Pipe-coordinate observations and uncertainty drive alternating sparse association and joint pose/map optimization, semantic likelihoods, association baselines, optional measured-loop checks and maintenance exports. Pose estimates initialize uncertain variables.

## Data

A pipe episode contains traversal direction and ordered observations with axial/angle estimates, roll, variances and descriptors. The first traversal initializes landmarks. truth_id and independent reference landmarks are used only by scoring.

Axial coordinates use meters, angle and roll use radians, and variances use squared units. `direction` is +1 for forward and −1 for reverse traversal within a single branch-free segment. Roll is added before cyclic comparison; `variance_roll` is included in angle uncertainty. Landmarks are not initialized from `truth`. `independent_observations=true` permits independent Gaussian coordinate fusion; otherwise covariance intersection avoids treating shared odometry as independent evidence.

Each case declares `order_weight`, `cardinality_weight`, `null_cost`, `top_l`, `kappa`, and `map_length_meters`. Local scores include appearance squared distance, covariance-normalized axial/cyclic-angle errors and `-class_weight*log(p_observation @ p_landmark)` when both class likelihoods exist (`class_weight=1`). This marginalizes the uncertain landmark category. Candidate hypotheses are restricted to the top L within `axial_gate_sigma` standard deviations (default 5), plus a null/new-landmark hypothesis. The order penalty uses consecutive semantic observations and their direction. `delta_s_variance` can supply the relative-motion variance; otherwise the sum of the two observation variances is the declared independence approximation. The soft objective adds entropy with coefficient 0.1 and uses L-BFGS-B with an analytic gradient.

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

Additional controls are `fixed_order_tolerance`, `no_soft_hypotheses`, `appearance_only`, `position_only`, `generic_smoothness`, `no_joint_optimization`, `no_alternation` and `no_class_likelihood`. `shuffled_order` permutes only the observation-pair sequence used by the order factor (`order_seed=17`) while preserving local scores and output indices. `fixed_order_tolerance_m` defaults to 0.5 m; `hard_order` applies a 10000 penalty to inversions beyond tolerance. `no_soft_hypotheses` uses greedy local assignment; `fixed_map` freezes coordinate fusion. The default core has loop auditing disabled (λ_l=0); the optional executable extension is described below.

## Outputs

`runs/main/traversalmap/records.jsonl` stores per-case results; `metrics.json` stores aggregate measurements. Comparison outputs use one directory per method.

```bash
python -m traversalmap plot --inputs runs/comparison/traversalmap --output figures
```

Plotting writes `figures/summary.csv` and metric PNGs. Statistical intervals can be computed from a retained record file:

```bash
python -m traversalmap.statistics --records runs/main/traversalmap/records.jsonl --output results/intervals.json
```

## Components and metrics

| Component / metric | Code entry | Configuration/output |
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

For coordinate error and duplicate/missed rates, the evaluator uses one-to-one Hungarian matching with cost `abs(axial_error_m)+abs(wrapped_angle_error_rad)` and the explicit `evaluation_matching_gate` (default 2 in those combined units). Freeze the matching gate on calibration inputs. Angular MAE is in radians. Reference identities and coordinates never enter association. Keep every traversal of one physical pipe in the same train/validation/calibration/test group. This geometry/semantic-likelihood back end has no learned component or training split; front-end fitting and threshold selection occur upstream. The statistics command bootstraps declared clusters rather than the full pipe/defect hierarchy.

## Checks

```bash
python verify.py --output runs/functional-check
```

Generated fixtures check covariance fusion, grouped traversals, truth isolation, metric cases, persistence and plots. Focused checks cover association, learning and pixel tracking.

## Joint pose/landmark optimization

Association first initializes sparse A and the map. `joint.optimize_joint` solves all semantic-frame `(pose_s,roll)` and landmark `(s,angle)` variables, then corrected projections and full pose/map covariance recompute A, appearance prototypes and class likelihoods before another state solve. `alternating_iterations=3` is the default cap and `alternating_tolerance=1e-4` stops stable A; `no_alternation` performs one state solve. Pose priors remain the original priors across rounds; repeated frames do not become independent measurements. Innovation covariance after a state solve is H C Hᵀ+R including pose/map cross terms. The whitened state residuals are pose priors, optional measured odometry, sqrt(A)-weighted axial/cyclic observations and order terms. Entropy/cardinality/class constants affect the A step rather than its state Jacobian. An analytic Jacobian and SciPy least squares are used. The solver returns a local alternating solution. Each round's actual cost/change and cap are retained.

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

The `maintenance` policy may declare `actionable_classes`, `minimum_severity`, `action_probability_threshold` (default .5), confidence cutoffs (default .5), `minimum_source_frames` (default 2), and axial/clock standard-deviation review cutoffs (default .5m/pi/4rad). Default actionable classes include all configured classes; default minimum severity is the second ordered level. Action probability is the product of actionable-class and above-threshold severity marginal mass, the maintenance decision rule.

Every map record has a stable episode/map ID plus landmark index, 95% axial interval, wrapped clock interval with `wraps`/`full_circle` markers, class/severity distributions, source frames, association probabilities, covariance and review reasons. Clock zero is vertically up, with positive angles interpreted clockwise; `clock_hours` maps zero to 12. JSON preserves the complete map, actionable worklist and review list; the CSV contains actionable records and their audit fields. Inference runs without `truth` still export these records and measured runtime/map size, while omitting reference-based accuracy metrics.

```bash
python -m traversalmap run --data data/annotated-or-unlabelled.jsonl --output runs/main
python -m traversalmap export --records runs/main/traversalmap/records.jsonl --output results/maintenance
```

`run`/`suite` automatically write `maintenance/case-000000.json` and `case-000000-worklist.csv` inside each method directory. Indexed filenames avoid interpreting episode IDs as paths. The standalone export command reads saved results without refitting.

For evaluation, reference landmarks additionally supply `class` and `severity`; the existing one-to-one geometric matching is reused. Worklist precision/recall count matched actionable physical defects, not frame detections. Edit operations are worklist additions for missed actionable defects, deletions for false records, and one update per matched actionable record with incorrect class or severity. Severity agreement is exact ordinal-label agreement on matched annotated records; ordinal MAE and class agreement are also emitted. Incomplete reference maintenance annotations leave worklist metrics null.

`batch_optimization_seconds` measures state solving separately from complete graph-update latency.

## Native baseline definitions and sources

```bash
python -m traversalmap suite --data data/episodes.jsonl --methods traversalmap jcbb probabilistic_semantic kslam no_alternation no_class_likelihood --output runs/native-comparison
python verify_native.py --output runs/native-methods
```

`association.jcbb` implements the maximum-cardinality one-to-one interpretation-tree branch-and-bound and increasing-dimensional joint chi-square tests of [Neira and Tardós (2001)](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/n/00976019.pdf). Individual compatibility only proposes candidates; a jointly incompatible set is rejected. Shared pose and pose/map cross-covariance from the state block enter the joint innovation matrix. External callers may supply off-diagonal projected-observation error through `association_cross_covariance`. Default confidence is 0.99. The independent block-diagonal case remains a valid special case, not the only implemented test.

Both association baselines enforce one-to-one assignments within a shared `frame_id`; successive frames may observe the same physical landmark. Without `frame_id`, each observation is its own view. The reported hard IDs for the probabilistic baseline come from the highest-likelihood complete hypothesis, preserving that constraint; its continuous update uses whole-hypothesis marginals rather than independently rounding each row.

`association.ranked_assignments` implements Murty partitioning with constrained linear assignments. `ranked_marginals` normalizes likelihoods over complete hypotheses to obtain the [Michael et al. (2022)](https://arxiv.org/html/2202.12802v1) marginals. `probabilistic_semantic` combines Gaussian geometry and semantic likelihood, private null/birth columns and a default 200 hypotheses (`k_best_assignments`). The continuous M-step uses those marginal weights and excludes the proposed order factor. The [GPL-3 reference implementation](https://github.com/EladMichael/probabilisticSemSlam) retains its own license; this module contains a mathematical implementation. A conservative remaining-hypothesis count provides the paper's marginal error bound in the public `ranked_marginals` API; a fully enumerated graph has bound zero.

`kslam.run` follows [Zhang et al. (2023), Algorithms 1/2](https://arxiv.org/html/2302.13264v3): world projection, k-means++/Lloyd association and continuous SLAM alternate inside each K; multi-resolution integer gridding searches landmark count using SLAM SSE+`landmark_penalty*K`. It does not receive the reference count. Measurements and landmarks retain a three-dimensional Euclidean objective; the pipe pose is constrained to axial translation and roll, a cylindrical-pose specialization of the general SE(3) formulation. Local points may be supplied explicitly as `local_point`; otherwise measured axial offset and known `pipe_radius_meters` (default .5 m) produce them. `kslam_measurement_sigma_meters=.1`, `landmark_penalty=12`, `kslam_iterations=5`, `kslam_grid_points=11` and seed 17 are defaults for this cylindrical state model. No class/order terms are injected into its geometric objective. It exports the tested K values, SSE, penalized objective, poses, 3-D centers and common map/maintenance fields.

## Native class/severity training

```bash
python -m pip install -e '.[learning]'
python -m traversalmap.semantic train --train data/crops-train.jsonl --validation data/crops-validation.jsonl --output runs/semantic
python -m traversalmap.semantic predict --manifest data/crops-test.jsonl --checkpoint runs/semantic/model.pt --output data/semantic-evidence.jsonl
python -m traversalmap.semantic attach --episodes data/geometry-test.jsonl --evidence data/semantic-evidence.jsonl --output data/episodes-with-semantics.jsonl
python verify_semantic.py --output runs/native-semantics
```

Crop manifests are JSONL with `id,image`, optional predicted `mask`, and supervised `class,severity,group` fields. Paths resolve relative to the manifest. `group` is the physical pipe/segment; training and validation groups must be disjoint. A supplied segmentation mask is input evidence from the frozen shared front end, not a test reference mask; absent masks use the supplied bounding-box crop. The native four-channel RGB+mask CNN uses masked pooling and class/severity cross-entropy, returning class likelihoods, ordered-severity likelihoods and an L2-normalized association descriptor. Image size 64, hidden width 32, Adam lr .001, seed 13, a 150-epoch cap and patience 15 configure the shared front end. Training label order is derived only from training data or explicit configuration. Predictions do not read class/severity reference labels. Attachment matches observation `id`, or `episode_id:traversal_index:observation_index`, and requires consistent label ordering. All methods should consume the same frozen exported likelihoods and descriptors.

`verify_semantic.py` checks generated PNG/mask training, inference-label isolation, learned evidence export and mapping. On Windows conda, use one OMP/MKL/OPENBLAS thread with `MKL_THREADING_LAYER=SEQUENTIAL`.

## Optional native measured-loop audit

Set `enable_loop_audit=true` and supply `loop_proposals`. A proposal declares `id`, `from_traversal/from_frame`, `to_traversal/to_frame`, measured `delta_s/delta_roll`, an SPD 2×2 `covariance`, and a one-to-one list of `landmark_pairs` in initial map-index space. All original-index references are remapped after an accepted merge. The audit independently checks descriptor distance (default gate .5), covariance-normalized axial/cyclic compatibility (9.21), then solves the candidate graph with the actual measured loop edge. Before/after robust pose costs use the same graph including that proposed edge. Acceptance requires a positive cost reduction (default 1e-6) and no confident direction-order inversions after alignment; rejected candidates do not modify the live map. Accepted identities merge their A columns and raw observations are optimized once, not fused a second time as an extra prior. Source frames and maintenance evidence follow the merged assignments.

Optional `reference_valid` is read only by the evaluator for loop false acceptance/rejection; it cannot affect acceptance. The default loop-free core remains λ_l=0. The semantic check includes an improving measured loop, a cyclicly inconsistent proposal, independent accept/reject assertions, and a test that flipping reference validity leaves both decisions unchanged. Supply calibrated loop edges and independent references for loop-error metrics.


## ORB-SLAM3 pixel front end (B2)

The B2 entry executes the real [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4) C++ System/Tracking/LocalMapping/LoopClosing, original ORB extraction/matching and bundled DBoW2/g2o with the full original `ORBvoc.txt` vocabulary. The GPL-3 native runner/build patch and license live under `native/orbslam3/`; Pangolin/GUI display is disabled, while SLAM and map workers remain original. Exact local modifications are recorded in `UPSTREAM.txt`.

```bash
# In an independent native source directory (the pinned revision includes the vocabulary):
git init runs/orb-native/upstream
git -C runs/orb-native/upstream remote add origin https://github.com/UZ-SLAMLab/ORB_SLAM3.git
git -C runs/orb-native/upstream fetch --depth 1 --filter=blob:none origin 4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4
git -C runs/orb-native/upstream sparse-checkout init --cone
git -C runs/orb-native/upstream sparse-checkout set include src Thirdparty Vocabulary
git -C runs/orb-native/upstream checkout --detach FETCH_HEAD
# Linux: provide a private prefix with OpenCV4 development/runtime, Boost serialization,
# Eigen3 and OpenSSL development files. No system package installation is required.
python native/orbslam3/extract_debs.py --debs runs/orb-native/debs --prefix runs/orb-native/prefix
python native/orbslam3/build.py --source runs/orb-native/upstream --prefix runs/orb-native/prefix --output runs/orb-native/build --cache ~/.cache/traversalmap-orbslam3 --jobs 2
```

The extractor accepts downloaded Ubuntu `.deb` files without installing them; Python `zstandard` is required for `.tar.zst`. On Ubuntu 26.04 the tested packages are `libopencv-{core,imgproc,calib3d,features2d,flann}-dev`, their `410` runtimes, `libboost1.90-dev`, `libboost-serialization1.90-dev`/`libboost-serialization1.90.0`, `libeigen3-dev`, `libssl-dev`, `libopenblas0-pthread`, `libgfortran5`, `libquadmath0`, `libtbb12`, `libtbbmalloc2` and `libtbbbind-2-5`; use `apt-get download` inside the private `debs` directory. Existing system runtime dependencies may also be needed according to `ldd`. The default is one compiler job; `--jobs 2` parallelizes ordinary units while Optimizer.cc remains exclusive. For WSL repositories on `/mnt/d`, a Linux-native source/header/object cache avoids metadata/random-write stalls; `--headers` can point to an existing read-only Boost/Eigen include prefix. Completed native outputs remain in `--output`.

```bash
python -m pip install -e '.[frontends]'
tar -xzf runs/orb-native/upstream/Vocabulary/ORBvoc.txt.tar.gz -C runs/orb-native
python -m traversalmap.orbslam3 --frames data/camera-frames.jsonl --executable runs/orb-native/build/orbslam3_runner --vocabulary runs/orb-native/ORBvoc.txt --settings data/camera.yaml --mode mono --libraries runs/orb-native/prefix/usr/lib/x86_64-linux-gnu --episodes data/semantic-observations.jsonl --calibration data/pipe-calibration.json --output runs/orb-b2
python verify_orbslam3.py --executable runs/orb-native/build/orbslam3_runner --vocabulary runs/orb-native/upstream/Vocabulary/ORBvoc.txt.tar.gz --libraries runs/orb-native/prefix/usr/lib/x86_64-linux-gnu --output runs/orb-functional
```

On Windows add `--wsl-distro Ubuntu-D` to the orchestration/check commands; the build command itself runs inside that WSL distribution. Camera-frame JSONL contains `id,episode_id,traversal_index,frame_id,timestamp,image` and `right_image` for stereo, with paths relative to the manifest. Actual pixels are converted to 8-bit PGM and passed with original ORB camera/feature YAML calibration to the native process. Each episode has a separate native map and strictly increasing timestamps. Outputs retain every frame's original tracking state (`2=OK`), failure flag/map ID, actual tracked-map-point count, per-frame processing time and camera-to-world SE(3). The native CSV also exports actual tracked MapPoint XYZ snapshots with map ID/point ID/timestamp and original keyframe TUM trajectories. It never fills lost poses from reference data. Disconnected map gauges require explicit independent `world_to_pipe_by_map` transforms before fusion; they are not silently mixed.

Independent `pipe-calibration.json` supplies rigid `world_to_pipe` (pipe axis is +z), positive `pose_variance_s,pose_variance_roll`, and `metric_scale` for monocular scale ambiguity. Calibration must come from camera/pipe geometry or independent measurements, not test survey fitting. Stereo uses metric baseline; monocular does not invent metric scale. The optional mapping stage requires raw observation `axial_offset`, applies only tracked original poses, adds calibrated pose axial variance to conditional measurement variance (an independence convention), replaces roll variance with the calibrated value, retains failed-observation counts and original traversal indices, then uses fixed native poses with local post-hoc defect fusion. It excludes TraversalMap's alternating pose refinement/order/loop audit from B2. JSON map records and maintenance exports retain semantic evidence, conditional reference scores, tracking failure fraction, front-end P50/P95/rate and actual post-hoc runtime. Front-end timing excludes vocabulary startup and deliberate camera pacing.

The functional fixture renders a textured 3-D scene, runs native tracking, checks trajectories/MapPoints and exports calibrated semantic observations.
