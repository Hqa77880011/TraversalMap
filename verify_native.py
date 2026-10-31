"""Hand-check native association/alternation and executable pipe-baseline instances."""

import argparse
import copy
import json
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from traversalmap import core
from traversalmap.association import associate, jcbb, ranked_marginals
from traversalmap.cli import execute, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    tests = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
    if not unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful():
        raise AssertionError("Existing fusion regressions failed")
    marginal, report = ranked_marginals(np.array([[0., 2.], [2., 0.]]), 20)
    exact = 1/(1+np.exp(-4))
    np.testing.assert_allclose(marginal, [[exact, 1-exact], [1-exact, exact]], atol=1e-12)
    assert report["complete"] and report["hypotheses"] == 2 and report["marginal_error_bound"] == 0
    observation = lambda s: dict(s=s, angle=0., roll=0., variance_s=1., variance_angle=1., descriptor=[1.])
    landmarks = [dict(s=s, angle=0., variance_s=0., variance_angle=0., descriptor=np.array([1.])) for s in (0., 100.)]
    inputs = [observation(1.), observation(99.)]
    cross = np.eye(4)
    cross[0, 2] = cross[2, 0] = .99
    independent, _ = jcbb(inputs, landmarks, {})
    correlated, joint = jcbb(inputs, landmarks, dict(association_cross_covariance=cross))
    assert np.sum(independent < 2) == 2 and np.sum(correlated < 2) == 1
    repeated = [dict(observation(.1), frame_id=str(k)) for k in range(2)]
    for backend in ("jcbb", "probabilistic_semantic"):
        ids, _ = associate(repeated, landmarks[:1], dict(null_cost=12.), backend)
        assert ids.tolist() == [0, 0]
        same_frame = [dict(row, frame_id="same") for row in repeated]
        ids, _ = associate(same_frame, landmarks[:1], dict(null_cost=12.), backend)
        assert np.sum(ids == 0) == 1  # joint MAP, not independently rounded marginals
    semantic = dict(top_l=2, null_cost=12., kappa=2., order_weight=0., cardinality_weight=0., class_weight=10.)
    local = observation(0.)
    local["class_probabilities"] = [.01, .99]
    candidates = [dict(landmarks[0], class_probabilities=np.array(value)) for value in ([.99, .01], [.01, .99])]
    selected, probabilities = core.optimize([local], candidates, 1, semantic, "traversalmap")
    _, control = core.optimize([local], candidates, 1, semantic, "no_class_likelihood")
    assert selected[0] == 1 and probabilities[0, 1] > .9 and abs(control[0, 0]-control[0, 1]) < 1e-5
    case = next(core.generate(random.Random(17), 1))
    case["truth"] = case["truth"][:3]
    case["traversals"] = case["traversals"][:2]
    for traversal in case["traversals"]:
        traversal["observations"] = [row for row in traversal["observations"] if int(row["truth_id"]) < 3]
    case.update(kslam_iterations=2, kslam_grid_points=5, alternating_iterations=3)
    write_jsonl(output / "episodes.jsonl", [case])
    loaded = read_jsonl(output / "episodes.jsonl")
    methods = {}
    for method in core.METHODS:
        rows = execute(loaded, method, output / method)
        assert rows == read_jsonl(output / method / "records.jsonl")
        methods[method] = len(rows)
        changed = copy.deepcopy(case)
        changed["truth"] = []
        for traversal in changed["traversals"]:
            for row in traversal["observations"]:
                row["truth_id"] = "unused"
        assert core.run_case(case, method) == core.run_case(changed, method)
    native = core.run_case(case, "traversalmap")
    assert native["joint_optimization"]["alternating_rounds"] >= 2
    for row in native["association_posteriors"]:
        np.testing.assert_allclose(np.asarray(row).sum(1), 1, atol=1e-5)
    fixed = copy.deepcopy(case)
    fixed.update(truth=[], traversals=[], landmark_penalty=12., kslam_iterations=2)
    for r in range(2):
        fixed["traversals"].append(dict(direction=1, observations=[dict(observation(s), pose_s=0.,
            axial_offset=s, frame_id="frame", pose_variance_s=1e-6, pose_variance_roll=1e-6)
            for s in (0., 5.)]))
    exact_map = core.run_case(fixed, "kslam")
    assert exact_map["kslam"]["landmarks"] == 2
    np.testing.assert_allclose(sorted(row["s"] for row in exact_map["landmarks"]), [0, 5], atol=1e-7)
    result = dict(kind="synthetic_functional_check", methods=methods, murty_identity_probability=exact,
                  jcbb_independent_matches=2, jcbb_correlated_matches=1,
                  alternating_rounds=native["joint_optimization"]["alternating_rounds"], exact_kslam_landmarks=2)
    (output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
