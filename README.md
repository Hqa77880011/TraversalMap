# TraversalMap

Implementation of **Traversal-order-constrained defect mapping for autonomous sewer inspection**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data

A pipe episode contains traversal direction and ordered observations with axial/angle estimates, roll, variances and descriptors. The first traversal initializes landmarks. truth_id and independent reference landmarks are used only by scoring.

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

## Outputs

`runs/main/traversalmap/records.jsonl` stores per-case results; `metrics.json` stores aggregate measurements. Comparison outputs use one directory per method.

```bash
python -m traversalmap plot --inputs runs/comparison/traversalmap --output figures
```

Plotting writes `figures/summary.csv` and metric PNGs. Statistical intervals can be computed from a retained record file:

```bash
python -m traversalmap.statistics --records runs/main/traversalmap/records.jsonl --output results/intervals.json
```
