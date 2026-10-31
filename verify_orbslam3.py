"""Execute original CPU ORB-SLAM3 on rendered 3-D texture images, without pose input."""

import argparse
import copy
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from traversalmap import core, orbslam3
from traversalmap.cli import write_jsonl


def fixture(root, count=26):
    generator = np.random.default_rng(13)
    positions = generator.uniform([-3.5, -2.5, 3.], [3.5, 2.5, 9.], (2200, 3))
    textures = generator.integers(0, 256, (len(positions), 11, 11), dtype=np.uint8)
    ordering = np.argsort(-positions[:, 2])
    rows = []
    for index in range(count):
        camera = np.array([.015*index, 0., .025*index])
        for side, baseline in (("left", 0.), ("right", .12)):
            image = Image.new("L", (640, 480), 25)
            for j in ordering:
                point = positions[j]-camera-np.array([baseline, 0., 0.])
                x, y = 500*point[:2]/point[2]+[320, 240]
                width = max(3, int(round(500*.13/point[2])))
                left, top = int(round(x-width/2)), int(round(y-width/2))
                if 0 <= left < 640-width and 0 <= top < 480-width:
                    patch = Image.fromarray(textures[j]).resize((width, width), Image.Resampling.BILINEAR)
                    image.paste(patch, (left, top))
            image.save(root / f"{index:04d}-{side}.png")
        rows.append(dict(id=f"pixel-{index}", episode_id="pixel-pipe", traversal_index=int(index >= count//2),
                         frame_id=str(index), timestamp=index/30, image=f"{index:04d}-left.png",
                         right_image=f"{index:04d}-right.png"))
    write_jsonl(root / "frames.jsonl", rows)
    settings = {'Camera.type': '"PinHole"', 'Camera.fx': 500., 'Camera.fy': 500., 'Camera.cx': 320., 'Camera.cy': 240.,
                'Camera.k1': 0., 'Camera.k2': 0., 'Camera.p1': 0., 'Camera.p2': 0., 'Camera.k3': 0.,
                'Camera.bf': 60., 'Camera.fps': 30., 'Camera.RGB': 0, 'ThDepth': 40.,
                'ORBextractor.nFeatures': 1600, 'ORBextractor.scaleFactor': 1.2, 'ORBextractor.nLevels': 8,
                'ORBextractor.iniThFAST': 15, 'ORBextractor.minThFAST': 5}
    (root / "camera.yaml").write_text("%YAML:1.0\n"+"\n".join(f"{key}: {value}" for key, value in settings.items())+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "vocabulary", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--libraries")
    parser.add_argument("--wsl-distro")
    parser.add_argument("--fixture-directory", help="Reuse a separately prepared image fixture")
    parser.add_argument("--modes", nargs="+", choices=["mono", "stereo"], default=["stereo", "mono"])
    args = parser.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    data = Path(args.fixture_directory) if args.fixture_directory else root
    if not args.fixture_directory:
        fixture(data)
    vocabulary = Path(args.vocabulary)
    if vocabulary.name.endswith(".tar.gz"):
        with tarfile.open(vocabulary) as archive:
            member = archive.getmember("ORBvoc.txt")
            with archive.extractfile(member) as source, (root / "ORBvoc.txt").open("wb") as target:
                import shutil
                shutil.copyfileobj(source, target)
        vocabulary = root / "ORBvoc.txt"
    reports = {}
    for mode in args.modes:
        poses = orbslam3.run(data / "frames.jsonl", args.executable, vocabulary, data / "camera.yaml", root / mode,
                            mode, args.libraries, args.wsl_distro)
        tracked = [row for row in poses if not row["failed"]]
        assert len(tracked) >= 3, f"{mode}: too few successfully tracked frames ({len(tracked)})"
        centers = np.array([np.asarray(row["camera_to_world"])[:3, 3] for row in tracked])
        assert np.ptp(centers, axis=0).max() > 1e-3
        assert max(row["tracked_map_points"] for row in tracked) > 20
        for row in tracked:
            matrix = np.asarray(row["camera_to_world"])
            np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
        calibration = dict(world_to_pipe=np.eye(4).tolist(), metric_scale=1., pose_variance_s=.01, pose_variance_roll=.001)
        case = dict(id="pixel-pipe", map_length_meters=10., order_weight=1., cardinality_weight=.1, null_cost=8., top_l=3, kappa=2.,
                    traversals=[dict(direction=1, observations=[]) for _ in range(2)])
        for row in tracked:
            traversal = case["traversals"][row["traversal_index"]]
            if not traversal["observations"]:
                traversal["observations"].append(dict(frame_id=row["frame_id"], s=2., angle=.2, roll=0.,
                    axial_offset=2.-.025*int(row["frame_id"]), variance_s=.01, variance_angle=.001,
                    variance_roll=.001, descriptor=[1., 0.]))
        projected = orbslam3.attach([case], poses, calibration, mode)[0]
        result = core.run_case(projected, "local_only")
        assert len(result["landmarks"]) > 0 and result["maintenance_records"]
        write_jsonl(root / mode / "mapped-episode.jsonl", [projected])
        reports[mode] = dict(input_frames=len(poses), tracked_frames=len(tracked),
            maximum_tracked_map_points=max(row["tracked_map_points"] for row in tracked),
            translation_extent=np.ptp(centers, axis=0).tolist(),
            native_latency_p50_ms=float(np.median([row["frontend_latency_ms"] for row in poses])),
            posthoc_landmarks=len(result["landmarks"]))
    hand = dict(id="hand", traversals=[dict(observations=[dict(frame_id="f", axial_offset=.5, variance_s=.04)])])
    matrix = np.eye(4); matrix[2, 3] = 1.
    pose = dict(episode_id="hand", traversal_index=0, frame_id="f", failed=False, camera_to_world=matrix.tolist())
    cfg = dict(world_to_pipe=np.eye(4).tolist(), metric_scale=2., pose_variance_s=.01, pose_variance_roll=.001)
    expected = orbslam3.attach([hand], [pose], cfg, "mono")[0]
    assert expected["traversals"][0]["observations"][0]["s"] == 2.5
    assert np.isclose(expected["traversals"][0]["observations"][0]["variance_s"], .05)
    altered = copy.deepcopy(hand); altered["truth"] = [dict(s=999)]
    assert expected["traversals"] == orbslam3.attach([altered], [pose], cfg, "mono")[0]["traversals"]
    report = dict(kind="synthetic_functional_check", actual_original_orbslam3=True, modes=reports,
                  hand_independent_metric_projection=2.5, reference_truth_isolation=True,
                  note="Original SLAM worker threads share one CPU affinity; OpenCV/Eigen/BLAS parallelism disabled.")
    (root / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
