# TraversalMap

Implementation of **Traversal-order-constrained defect mapping for autonomous sewer inspection**.

Combines sparse local appearance/pipe-coordinate association, cyclic clock angles, uncertainty-adaptive direction-conditioned order factors and soft cardinality regularization.

## Install

Use Python 3.11 or newer.

```bash
git clone https://github.com/Hqa77880011/TraversalMap.git
cd TraversalMap
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install.

```bash
python -m pip install -e .
```

## Data preparation

A pipe episode contains traversal direction and ordered observations with axial/angle estimates, roll, variances and descriptors. The first traversal initializes landmarks. truth_id and independent reference landmarks are used only by scoring.

```bash
python -m traversalmap prepare --count 200 --seed 17 --output data/workload.jsonl
```

The generator creates finite workloads from the formal model. Each JSONL row is one scenario. `--count` controls corpus size and `--seed` controls generation. Imported cases use the same schema. Generate calibration and test scenarios independently.

## Execute and compare

```bash
python -m traversalmap run --data data/workload.jsonl --output runs/main
python -m traversalmap suite --data data/workload.jsonl --output runs/comparison
```

Select methods with `--methods`. traversalmap, local_only, no_cyclic_angle, shuffled_order, hard_order and fixed_map.

## Evaluation and visualization

Unique-count error, observation association F1, duplicate/missed landmarks, crossed identities, axial RMSE and cyclic angle MAE after one-to-one reference matching, map records/km and optimization cost.

Each method writes `records.jsonl` and `metrics.json` with means, sample standard deviations, usable counts and P50/P95/P99. Rates are fractions. Undefined conditional metrics remain unavailable. Method timers exclude reference scoring. RSS includes the Python runtime; serialized sizes measure JSON bytes.

```bash
python -m traversalmap plot --inputs runs/comparison/traversalmap --output figures
```

Pass multiple method directories to compare them. Plotting writes `summary.csv` and metric PNGs.

## Implementation conventions

The fixed frontend provides pose/roll posteriors. The numerical optimizer solves soft association over top-L candidate landmarks; null assignments create new records. Matched observations now update landmark axial/cyclic means, variances and descriptor averages. Corrected angle variance includes optional `variance_roll`. Set `independent_observations: true` only for independent measurement errors: the finite generator declares this convention. Unknown/shared pose correlation uses a common covariance-intersection weight rather than halving uncertainty. Angle fusion unwraps around the prior and wraps the result, preserving the clock-angle boundary. `fixed_map` disables fusion while retaining the same association/order factors. New segments must be initialized at branches or geometry discontinuities.


## Uncertainty and paired comparisons

```bash
python -m traversalmap.statistics --records runs/main/traversalmap/records.jsonl --output results/intervals.json
```

Supply --reference with a matched record file for paired differences. Resampling uses independent case IDs, or explicit cluster IDs when provided. Repeated observations from one source unit must share cluster. Zero-event bounds refer to an event in an independent cluster.

## Fixed reproduction setting

Six-defect straight-segment map, three forward/reverse traversals, top-L=4; independent coordinate fusion, cyclic angles and fixed-map ablation.

The checked-in entry point fixes data seeds, methods, budgets and output paths. Inspect its commands without starting any computation:

```bash
python reproduce.py --plan
```

Remove `--plan` to execute the registered pipeline in a fresh output directory. The entry point calls the existing implementation; it does not insert paper results. Installation uses the package instructions above.
