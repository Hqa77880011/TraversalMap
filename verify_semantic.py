"""Image/mask learning and loop-audit counterexamples; functionality only."""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from traversalmap import core, semantic
from traversalmap.cli import write_jsonl
from traversalmap.loops import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(13)
    manifests = []
    for prefix, count in (("train", 6), ("validation", 3)):
        rows = []
        for index in range(count):
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[8:24, 8:24, index % 3] = 160+index*10
            mask = np.zeros((32, 32), dtype=np.uint8); mask[8:24, 8:24] = 255
            name = f"{prefix}-{index}"
            Image.fromarray(image).save(output / (name+".png"))
            Image.fromarray(mask).save(output / (name+"-mask.png"))
            rows.append(dict(id=name, image=name+".png", mask=name+"-mask.png",
                             group=prefix+":pipe", **{"class": "crack" if index % 2 else "deposit"}, severity=index % 3))
        path = output / (prefix+".jsonl")
        write_jsonl(path, rows); manifests.append(path)
    cfg = dict(class_labels=["crack", "deposit"], severity_levels=[0, 1, 2], image_size=32,
               hidden=12, batch_size=3, seed=13, epochs=2, learning_rate=.01)
    data = semantic.Crops(manifests[0], cfg)
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=3)))
    model = semantic.SemanticNetwork(cfg)
    before = {key: value.detach().clone() for key, value in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    loss = semantic.loss(model(batch["x"]), batch)
    assert torch.isfinite(loss)
    loss.backward()
    critical = ["encoder.0.weight", "encoder.2.weight", "class_head.weight", "severity_head.weight"]
    parameters = dict(model.named_parameters())
    for key in critical:
        assert parameters[key].grad is not None and torch.isfinite(parameters[key].grad).all()
        assert parameters[key].grad.abs().sum() > 0, key
    optimizer.step()
    assert all(not torch.equal(before[key], parameters[key]) for key in critical)
    model.eval()
    torch.save(model.state_dict(), output / "roundtrip.pt")
    loaded = semantic.SemanticNetwork(cfg).eval()
    loaded.load_state_dict(torch.load(output / "roundtrip.pt", weights_only=True))
    with torch.no_grad():
        for key, value in model(batch["x"]).items():
            torch.testing.assert_close(value, loaded(batch["x"])[key], rtol=1e-6, atol=1e-7)
    checkpoint = semantic.train(*manifests, output / "model", cfg)
    evidence = semantic.predict(manifests[1], checkpoint)
    assert all(np.isclose(sum(row["class_probabilities"]), 1) and np.isclose(sum(row["severity_probabilities"]), 1) for row in evidence)
    altered, _ = semantic.read_manifest(manifests[1])
    for row in altered:
        row["class"], row["severity"] = "never-read", 999
    write_jsonl(output / "unlabelled.jsonl", altered)
    assert evidence == semantic.predict(output / "unlabelled.jsonl", checkpoint)
    observation = lambda s: dict(s=s, angle=0., roll=0., variance_s=1., variance_angle=.04,
                                variance_roll=.04, descriptor=[1., 0.])
    episode = dict(id="loops", independent_observations=True, order_weight=0., null_cost=12.,
                   cardinality_weight=1., top_l=1, kappa=2., map_length_meters=10,
                   traversals=[dict(direction=1, observations=[observation(0)]),
                               dict(direction=-1, observations=[observation(2)])])
    episode["traversals"][0]["observations"][0]["id"] = evidence[0]["id"]
    episode["traversals"][1]["observations"][0]["id"] = evidence[1]["id"]
    attached = semantic.attach([copy.deepcopy(episode)], evidence)[0]
    assert len(attached["traversals"][0]["observations"][0]["descriptor"]) == 12
    mapped = core.run_case(attached, "traversalmap")
    assert mapped["maintenance_records"][0]["class_probabilities"] is not None
    good = dict(id="measured-loop", from_traversal=0, from_frame="0", to_traversal=1,
                to_frame="0", delta_s=0., delta_roll=0., covariance=[[.25, 0], [0, .01]],
                landmark_pairs=[[0, 0]], reference_valid=True)
    bad = dict(good, id="cyclic-inconsistent", delta_roll=np.pi, reference_valid=False)
    episode.update(enable_loop_audit=True, loop_proposals=[good, bad])
    audited = core.run_case(episode, "traversalmap")
    decisions = audited["loop_audit"]["decisions"]
    assert [row["accepted"] for row in decisions] == [True, False]
    assert decisions[0]["robust_pose_cost_after"] < decisions[0]["robust_pose_cost_before"]
    report = metrics(episode, audited)
    assert report["loop_false_acceptance"] == report["loop_false_rejection"] == 0
    changed = copy.deepcopy(episode)
    for row in changed["loop_proposals"]:
        row["reference_valid"] = not row["reference_valid"]
    assert core.run_case(changed, "traversalmap")["loop_audit"]["decisions"] == decisions
    write_jsonl(output / "learned-evidence.jsonl", evidence)
    write_jsonl(output / "loop-input.jsonl", [episode])
    (output / "loop-result.json").write_text(json.dumps(audited, indent=2), encoding="utf-8")
    result = dict(kind="synthetic_functional_check", initial_loss=float(loss.detach()),
                  gradients_and_updates=critical, roundtrip_rtol=1e-6, class_severity_epochs=2,
                  image_readback=True, label_isolation=True, measured_loop_accepted=True,
                  inconsistent_loop_rejected=True, loop_metrics=report)
    (output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
