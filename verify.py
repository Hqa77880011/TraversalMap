"""Finite pipe episodes and analytic identity examples; no survey results."""

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
from traversalmap.cli import execute, plot, read_jsonl, write_jsonl


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
    plot([output / "traversalmap", output / "local_only"], output / "figures")
    report = {"kind": "synthetic_functional_check", "seed": 17,
              "existing_regressions": result.testsRun, "methods": methods,
              "analytic_pairwise_f1": merged["association_f1"], "truth_isolation": True}
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
