"""Run compiled ORB-SLAM3 on pixels and export calibrated pipe poses.

This module orchestrates an external GPL-3 native executable."""

import argparse
import copy
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image


def native_path(path, wsl):
    path = Path(path).resolve()
    if not wsl:
        return str(path)
    value = path.as_posix()
    if len(value) < 3 or value[1:3] != ":/":
        raise ValueError("WSL execution requires local Windows drive paths")
    return "/mnt/"+value[0].lower()+value[2:]


def run(frames, executable, vocabulary, settings, output, mode="mono", libraries=None, wsl=None):
    """Decode real images, execute original SLAM, and retain explicit tracking failures."""
    from .cli import read_jsonl, write_jsonl
    rows, output = read_jsonl(frames), Path(output)
    if mode not in ("mono", "stereo") or not rows:
        raise ValueError("Require nonempty pixel manifest and mono/stereo mode")
    output.mkdir(parents=True, exist_ok=False)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["episode_id"], []).append(row)
    all_poses = []
    launcher = "import os,sys; os.sched_setaffinity(0,{min(os.sched_getaffinity(0))}); os.execvpe(sys.argv[1],sys.argv[1:],os.environ)"
    for sequence, (episode, group) in enumerate(grouped.items()):
        group.sort(key=lambda row: float(row["timestamp"]))
        if np.any(np.diff([float(row["timestamp"]) for row in group]) <= 0):
            raise ValueError("Each native SLAM episode needs strictly increasing timestamps")
        folder = output / f"episode-{sequence:06d}"
        folder.mkdir()
        manifest = []
        for index, row in enumerate(group):
            paths = []
            for key in (("image", "right_image") if mode == "stereo" else ("image",)):
                name = f"{index:06d}-{key}.pgm"
                with Image.open(Path(frames).parent / row[key]) as image:
                    image.convert("L").save(folder / name)
                paths.append(name)
            manifest.append("\t".join([str(row["timestamp"]), *paths]))
        (folder / "frames.tsv").write_bytes(("\n".join(manifest)+"\n").encode("utf-8"))
        arguments = [native_path(executable, wsl), native_path(vocabulary, wsl), native_path(settings, wsl),
                     mode, native_path(folder / "frames.tsv", wsl), native_path(folder / "poses.csv", wsl)]
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        lib = native_path(libraries, wsl) if libraries else ""
        if wsl:
            command = ["wsl", "-d", wsl, "--exec", "env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1"]
            if lib:
                command.append("LD_LIBRARY_PATH="+lib+":"+lib+"/openblas-pthread")
            command += ["python3", "-c", launcher, *arguments]
        else:
            if lib:
                environment["LD_LIBRARY_PATH"] = lib+":"+lib+"/openblas-pthread"
            command = ["python3", "-c", launcher, *arguments]
        with (folder / "native.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        with (folder / "poses.csv").open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        if len(predictions) != len(group):
            raise RuntimeError("Native SLAM did not retain one status per input frame")
        for row, predicted in zip(group, predictions):
            if not np.isclose(float(predicted["timestamp"]), float(row["timestamp"]), rtol=0, atol=1e-6):
                raise RuntimeError("Native output timestamp mismatch")
            matrix = np.array([float(predicted[f"T{i}"]) for i in range(16)]).reshape(4, 4)
            status = int(predicted["tracking_state"])
            all_poses.append(dict(id=row["id"], episode_id=episode, traversal_index=row["traversal_index"],
                frame_id=row["frame_id"], timestamp=float(row["timestamp"]), tracking_state=status,
                frontend_latency_ms=float(predicted["latency_ms"]),
                map_id=int(predicted["map_id"]),
                tracked_map_points=int(predicted["tracked_map_points"]), failed=status != 2,
                camera_to_world=matrix.tolist() if status == 2 else None, source_model="ORB-SLAM3", sensor_mode=mode))
    write_jsonl(output / "poses.jsonl", all_poses)
    return all_poses


def attach(episodes, poses, calibration, mode):
    """Use independent metric/pipe calibration; never fit scale/alignment to truth."""
    if mode == "mono" and "metric_scale" not in calibration:
        raise ValueError("Monocular SLAM requires independent metric_scale calibration")
    scale = float(calibration.get("metric_scale", 1.))
    transform = np.asarray(calibration["world_to_pipe"], dtype=float)
    variance = [calibration["pose_variance_s"], calibration["pose_variance_roll"]]
    if scale <= 0 or transform.shape != (4, 4) or min(variance) <= 0 or not np.isfinite([scale, *variance]).all():
        raise ValueError("Positive scale/pose variances and a 4x4 world_to_pipe are required")
    alignments = [transform]+[np.asarray(value, dtype=float) for value in calibration.get("world_to_pipe_by_map", {}).values()]
    for alignment in alignments:
        if alignment.shape != (4, 4) or not np.isfinite(alignment).all() or not np.allclose(alignment[3], [0, 0, 0, 1]) or not np.allclose(alignment[:3, :3].T @ alignment[:3, :3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(alignment[:3, :3]), 1.):
            raise ValueError("Pipe alignment must be a finite rigid transform")
    lookup = {(str(row["episode_id"]), row["traversal_index"], str(row["frame_id"])): row for row in poses}
    output = copy.deepcopy(episodes)
    for episode in output:
        map_transforms = calibration.get("world_to_pipe_by_map", {})
        active_maps = {row.get("map_id", 0) for row in poses if str(row["episode_id"]) == str(episode["id"]) and not row["failed"]}
        if len(active_maps) > 1 and not all(str(identity) in map_transforms for identity in active_maps):
            raise ValueError("Disconnected native maps require independent world_to_pipe_by_map alignment; cannot mix gauges")
        failures = 0
        for r, traversal in enumerate(episode["traversals"]):
            accepted = []
            for observation in traversal["observations"]:
                row = lookup[(str(episode["id"]), r, str(observation["frame_id"]))]
                if row["failed"]:
                    failures += 1
                    continue
                camera = np.asarray(row["camera_to_world"])
                camera[:3, 3] *= scale
                alignment = np.asarray(map_transforms.get(str(row.get("map_id", 0)), transform))
                pipe = alignment @ camera
                axial, roll = float(pipe[2, 3]), float(np.arctan2(pipe[1, 0], pipe[0, 0]))
                measurement_variance = observation.get("measurement_variance_s", observation["variance_s"])
                observation.update(pose_s=axial, pose_roll=roll, s=axial+observation["axial_offset"], roll=roll,
                    measurement_variance_s=measurement_variance, variance_s=measurement_variance+variance[0],
                    variance_roll=variance[1],
                    pose_variance_s=variance[0], pose_variance_roll=variance[1])
                accepted.append(observation)
            traversal["observations"] = accepted
        episode["pose_frontend_original_traversals"] = [r for r, traversal in enumerate(episode["traversals"]) if traversal["observations"]]
        episode["traversals"] = [traversal for traversal in episode["traversals"] if traversal["observations"]]
        episode.update(pose_frontend="ORB-SLAM3", pose_frontend_failures=failures,
                       joint_optimization=False, enable_loop_audit=False)
    return output


def main():
    from . import core
    from .cli import read_jsonl, write_jsonl
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frames", "executable", "vocabulary", "settings", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--mode", choices=["mono", "stereo"], default="mono")
    parser.add_argument("--libraries")
    parser.add_argument("--wsl-distro")
    parser.add_argument("--episodes")
    parser.add_argument("--calibration")
    args = parser.parse_args()
    if bool(args.episodes) != bool(args.calibration):
        parser.error("Post-hoc defect mapping requires both --episodes and --calibration")
    poses = run(args.frames, args.executable, args.vocabulary, args.settings, args.output,
                args.mode, args.libraries, args.wsl_distro)
    if args.episodes:
        cases = attach(read_jsonl(args.episodes), poses, json.loads(Path(args.calibration).read_text(encoding="utf-8")), args.mode)
        write_jsonl(Path(args.output) / "native-pose-episodes.jsonl", cases)
        records = []
        for case in cases:
            if not any(traversal["observations"] for traversal in case["traversals"]):
                records.append(dict(id=case["id"], failed=True, reason="No tracked semantic observations", native_pose_failures=case["pose_frontend_failures"]))
                continue
            started = time.perf_counter()
            result = core.run_case(case, "local_only")
            mapping_ms = (time.perf_counter()-started)*1000
            front = [row for row in poses if str(row["episode_id"]) == str(case["id"])]
            latency = [row["frontend_latency_ms"] for row in front]
            metrics = core.score(case, result) if case.get("truth") else dict(landmark_count=len(result["landmarks"]))
            metrics.update(frontend_tracking_failure_fraction=float(np.mean([row["failed"] for row in front])),
                           frontend_latency_p50_ms=float(np.median(latency)),
                           frontend_latency_p95_ms=float(np.quantile(latency, .95)),
                           frontend_update_rate_hz=1000/float(np.mean(latency)),
                           posthoc_mapping_ms=mapping_ms, latency_ms=sum(latency)+mapping_ms)
            records.append(dict(id=case["id"], method="orbslam3_posthoc", result=result,
                                metrics=metrics))
        write_jsonl(Path(args.output) / "records.jsonl", records)
        from .maintenance import export_records
        export_records([row for row in records if not row.get("failed")], Path(args.output) / "maintenance")


if __name__ == "__main__":
    main()
