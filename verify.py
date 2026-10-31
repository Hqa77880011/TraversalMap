"""Finite pipe episodes and analytic identity examples; no survey results."""

import argparse
import copy
import csv
import json
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from traversalmap import core
from traversalmap.cli import execute, plot, read_jsonl, write_jsonl
from traversalmap.joint import PoseLandmarkProblem, optimize_joint
from traversalmap.maintenance import build_records, export_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    tests = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if not result.wasSuccessful():
        raise AssertionError("Existing covariance/cyclic-fusion checks failed")
    cases = list(core.generate(random.Random(17), 2))
    write_jsonl(output / "episodes.jsonl", cases)
    loaded = read_jsonl(output / "episodes.jsonl")
    methods = {}
    for method in core.METHODS:
        rows = execute(loaded, method, output / method)
        assert rows == read_jsonl(output / method / "records.jsonl")
        methods[method] = len(rows)
        altered = copy.deepcopy(loaded[0])
        altered["truth"] = []
        for traversal in altered["traversals"]:
            for observation in traversal["observations"]:
                observation["truth_id"] = "not-an-input"
        assert core.run_case(loaded[0], method) == core.run_case(altered, method)
    landmark = dict(s=0, angle=0, variance_s=0.1, variance_angle=0.1, observations=5)
    case = dict(truth=[dict(id="a", s=0, angle=0), dict(id="b", s=3, angle=0)],
                traversals=[dict(direction=1, observations=[dict(truth_id="a"), dict(truth_id="b")]),
                            dict(direction=1, observations=[dict(truth_id="a"), dict(truth_id="b")]),
                            dict(direction=1, observations=[dict(truth_id="b")])], map_length_meters=10)
    merged = core.score(case, dict(landmarks=[landmark], assignments=[[0, 0], [0, 0], [0]]))
    # Four true same-identity pairs / ten predicted pairs gives F1 = 4/7.
    assert np.isclose(merged["association_precision"], 0.4)
    assert np.isclose(merged["association_recall"], 1)
    assert np.isclose(merged["association_f1"], 4 / 7)
    assert np.isclose(merged["observation_identity_accuracy"], 0.4)
    assert np.isclose(abs(core.wrap(3.13 - (-3.13))), 2 * np.pi - 6.26)
    # Closed-form problem: pose priors 0 and 2, shared landmark, unit axial noise.
    observation = lambda s: dict(s=s, angle=0., roll=0., variance_s=1., variance_angle=.04,
                                variance_roll=.04, descriptor=[1., 0.])
    joint_case = dict(traversals=[dict(direction=1, observations=[observation(0), observation(2)])],
                      independent_observations=True, order_weight=0.)
    seed = [core.make_landmark(observation(0))]
    posterior = [[[1.], [1.]]]
    problem = PoseLandmarkProblem(joint_case, seed, posterior, "traversalmap")
    residual, jacobian = problem.linearize(problem.initial)
    np.testing.assert_allclose(residual, [0, 0, 0, 0, 0, 0, 2, 0], atol=1e-12)
    assert np.linalg.matrix_rank(jacobian) == len(problem.initial)
    numerical = np.column_stack([
        (problem.linearize(problem.initial + np.eye(len(problem.initial))[i] * 1e-6)[0]
         - problem.linearize(problem.initial - np.eye(len(problem.initial))[i] * 1e-6)[0]) / 2e-6
        for i in range(len(problem.initial))])
    np.testing.assert_allclose(jacobian, numerical, atol=1e-8)
    poses, joint_report = optimize_joint(joint_case, seed, posterior, "traversalmap")
    np.testing.assert_allclose([pose["s"] for pose in poses], [.5, 1.5], atol=1e-8)
    np.testing.assert_allclose(seed[0]["s"], 1., atol=1e-8)
    np.testing.assert_allclose([joint_report["initial_cost"], joint_report["final_cost"]], [2., .5], atol=1e-8)
    assert joint_report["pose_change_l2"] > .1 and joint_report["landmark_change_l2"] > .1
    np.testing.assert_allclose([pose["covariance"][0][0] for pose in poses], [.75, .75], atol=1e-8)
    np.testing.assert_allclose(seed[0]["variance_s"], 1., atol=1e-8)
    odometry_case = copy.deepcopy(joint_case)
    odometry_case["traversals"][0]["odometry"] = [dict(
        **{"from": "0", "to": "1"}, delta_s=.5, delta_roll=0., covariance=[[.25, 0], [0, .01]])]
    odometry_seed = [core.make_landmark(observation(0))]
    odometry_poses, odometry_report = optimize_joint(odometry_case, odometry_seed, posterior, "traversalmap")
    np.testing.assert_allclose([pose["s"] for pose in odometry_poses], [.7, 1.3], atol=1e-8)
    assert odometry_report["odometry_factors"] == 1
    # Weighted semantic pooling is not multiplied repeatedly as independent evidence.
    pooled_case = copy.deepcopy(joint_case)
    pooled_case.update(id="pool", class_labels=["crack", "deposit"], severity_levels=[0, 1, 2])
    a, b = pooled_case["traversals"][0]["observations"]
    a.update(class_probabilities=[.8, .2], severity_probabilities=[0, .5, .5], quality=1)
    b.update(class_probabilities=[.2, .8], severity_probabilities=[0, 0, 1], quality=.5)
    pooled = build_records(pooled_case, dict(landmarks=seed, assignments=[[0, 0]], association_posteriors=posterior))[0]
    np.testing.assert_allclose(pooled["class_probabilities"], [.6, .4])
    np.testing.assert_allclose(pooled["severity_probabilities"], [0, 1 / 3, 2 / 3])
    # One actionable match, one missing defect, two false records, one severity edit.
    maintenance_case = dict(id="maintenance-hand", class_labels=["crack", "deposit"], severity_levels=[0, 1, 2],
                            maintenance=dict(actionable_classes=["crack"], minimum_severity=1), map_length_meters=12,
                            truth=[dict(id="a", s=0, angle=3.13, **{"class": "crack"}, severity=2),
                                   dict(id="b", s=3, angle=0, **{"class": "crack"}, severity=2),
                                   dict(id="c", s=6, angle=0, **{"class": "deposit"}, severity=0)],
                            traversals=[dict(direction=1, observations=[])])
    for i, (name, level) in enumerate((("crack", 1), ("deposit", 2), ("crack", 2), ("crack", 2))):
        value = observation(3 * i)
        value.update(angle=3.13 if i == 0 else 0., variance_s=.01, truth_id="abcd"[i], frame_id=str(i),
                     class_probabilities=[float(name == "crack"), float(name == "deposit")],
                     severity_probabilities=[float(j == level) for j in range(3)])
        maintenance_case["traversals"][0]["observations"].append(value)
    maintenance_result = core.run_case(maintenance_case, "fixed_map")
    maintenance_metrics = core.score(maintenance_case, maintenance_result)
    np.testing.assert_allclose([maintenance_metrics["worklist_precision"], maintenance_metrics["worklist_recall"]], [1 / 3, .5])
    assert maintenance_metrics["maintenance_edit_operations"] == 4
    assert [maintenance_metrics[key] for key in ("maintenance_additions", "maintenance_deletions", "maintenance_updates")] == [1, 2, 1]
    np.testing.assert_allclose(maintenance_metrics["severity_agreement"], 1 / 3)
    np.testing.assert_allclose(maintenance_metrics["severity_ordinal_mae"], 1.)
    assert maintenance_result["maintenance_records"][0]["clock_interval95"]["wraps"]
    export_records([dict(id="maintenance-hand", method="fixed_map", result=maintenance_result)], output / "maintenance-hand")
    exported = json.loads((output / "maintenance-hand/case-000000.json").read_text(encoding="utf-8"))
    assert exported["records"] == maintenance_result["maintenance_records"]
    with (output / "maintenance-hand/case-000000-worklist.csv").open(encoding="utf-8", newline="") as stream:
        worklist = list(csv.DictReader(stream))
    assert {row["id"] for row in worklist} == {row["id"] for row in maintenance_result["worklist"]}
    for row in worklist:
        original = next(value for value in maintenance_result["worklist"] if value["id"] == row["id"])
        assert json.loads(row["class_probabilities"]) == original["class_probabilities"]
        assert json.loads(row["source_frames"]) == original["source_frames"]
    restored = copy.deepcopy(maintenance_result)
    restored["maintenance_records"], restored["worklist"] = exported["records"], exported["worklist"]
    assert core.score(maintenance_case, restored) == maintenance_metrics
    unlabelled = copy.deepcopy(maintenance_case)
    unlabelled.pop("truth")
    unlabelled_rows = execute([unlabelled], "traversalmap", output / "unlabelled")
    assert "worklist_precision" not in unlabelled_rows[0]["metrics"]
    plot([output / "traversalmap", output / "local_only"], output / "figures")
    report = {"kind": "synthetic_functional_check", "seed": 17,
              "existing_regressions": result.testsRun, "methods": methods,
              "analytic_pairwise_f1": merged["association_f1"], "truth_isolation": True,
              "joint_closed_form": joint_report, "odometry_pose_s": [pose["s"] for pose in odometry_poses],
              "maintenance_hand": {key: maintenance_metrics[key] for key in ("worklist_precision", "worklist_recall", "maintenance_edit_operations", "severity_agreement")},
              "export_roundtrip": True, "unlabelled_export": True}
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
