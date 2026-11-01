"""K-SLAM Algorithms 1/2 (Zhang et al., ICRA 2023), cylindrical-pose instance.

The Euclidean 3-D measurement objective uses poses restricted to axial
translation and camera roll around the known pipe axis.
https://arxiv.org/html/2302.13264v3"""

import time

import numpy as np
from scipy.optimize import least_squares
from sklearn.cluster import KMeans

from .joint import PoseLandmarkProblem, wrap
from .maintenance import build_records


def run(case, profile=False):
    start = time.perf_counter()
    observed = [(r, k, value) for r, traversal in enumerate(case["traversals"])
                for k, value in enumerate(traversal["observations"])]
    count = len(observed)
    if not count:
        raise ValueError("K-SLAM requires landmark measurements")
    if case.get("kslam_iterations", 5) < 1:
        raise ValueError("kslam_iterations must be positive")
    radius = case.get("pipe_radius_meters", .5)
    sigma = case.get("kslam_measurement_sigma_meters", .1)
    if radius <= 0 or sigma <= 0:
        raise ValueError("Pipe radius and isotropic Euclidean measurement sigma must be positive")
    scaffold = [dict(s=0., angle=0., variance_s=1., variance_angle=1.)]
    zero = [[[0.] for _ in traversal["observations"]] for traversal in case["traversals"]]
    problem = PoseLandmarkProblem(case, scaffold, zero, "kslam")
    dimensions = problem.pose_dimensions
    local, pose_indices = [], []
    for r, k, observation in observed:
        pose = problem.pose_for_observation[r, k]
        offset = observation.get("axial_offset", observation["s"]-problem.frames[pose]["s"])
        local.append(observation.get("local_point", [offset, radius*np.cos(observation["angle"]),
                                                    radius*np.sin(observation["angle"])]))
        pose_indices.append(pose)
    local, pose_indices = np.asarray(local, dtype=float), np.asarray(pose_indices)

    def project(poses):
        axial, roll = poses.reshape(-1, 2)[pose_indices].T
        result = local.copy()
        result[:, 0] += axial
        result[:, 1] = np.cos(roll)*local[:, 1]-np.sin(roll)*local[:, 2]
        result[:, 2] = np.sin(roll)*local[:, 1]+np.cos(roll)*local[:, 2]
        return result

    def solve(number):
        poses = problem.prior.copy()
        trace = []
        for iteration in range(case.get("kslam_iterations", 5)):
            # Algorithm 1: world projection, k-means++/Lloyd, then a SLAM solve.
            clustering = KMeans(n_clusters=number, init="k-means++", n_init=1,
                                random_state=case.get("seed", 17)+iteration).fit(project(poses))
            assigned = clustering.labels_
            initial = np.r_[poses, clustering.cluster_centers_.ravel()]

            def linearize(state):
                pose, centers = state[:dimensions], state[dimensions:].reshape(number, 3)
                delta = pose-problem.prior
                delta[1::2] = wrap(delta[1::2])
                residual = [problem.pose_whitener @ delta]
                jacobian = [np.c_[problem.pose_whitener, np.zeros((dimensions, 3*number))]]
                for first, second, increment, W in problem.odometry:
                    error = pose[2*second:2*second+2]-pose[2*first:2*first+2]-increment
                    error[1] = wrap(error[1])
                    J = np.zeros((2, len(state)))
                    J[:, 2*first:2*first+2] = -W
                    J[:, 2*second:2*second+2] = W
                    residual.append(W @ error); jacobian.append(J)
                points = project(pose)
                residual.append(((points-centers[assigned])/sigma).ravel())
                J = np.zeros((3*count, len(state)))
                for i, (frame, landmark) in enumerate(zip(pose_indices, assigned)):
                    J[3*i, 2*frame] = 1/sigma
                    J[3*i+1:3*i+3, 2*frame+1] = [-points[i, 2]/sigma, points[i, 1]/sigma]
                    J[3*i:3*i+3, dimensions+3*landmark:dimensions+3*landmark+3] = -np.eye(3)/sigma
                jacobian.append(J)
                return np.concatenate(residual), np.vstack(jacobian)

            fitted = least_squares(lambda state: linearize(state)[0], initial,
                                   jac=lambda state: linearize(state)[1], max_nfev=100,
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9)
            if not fitted.success:
                raise RuntimeError("K-SLAM state solve failed: "+fitted.message)
            poses = fitted.x[:dimensions]
            trace.append(float(2*fitted.cost))
        return dict(number=number, objective=trace[-1]+case.get("landmark_penalty", 12.)*number,
                    slam_sse=trace[-1], trace=trace, assigned=assigned, state=fitted.x,
                    covariance=np.linalg.pinv(fitted.jac.T @ fitted.jac, hermitian=True),
                    optimality=float(fitted.optimality))

    # Algorithm 2 multi-resolution integer gridding; no oracle landmark count.
    low, high, candidates = 1, count, {}
    resolution = max(5, case.get("kslam_grid_points", 11))
    while True:
        grid = sorted(set(int(round(value)) for value in np.linspace(low, high, min(resolution, high-low+1))))
        for number in grid:
            if number not in candidates:
                candidates[number] = solve(number)
        best_index = min(range(len(grid)), key=lambda index: candidates[grid[index]]["objective"])
        if all(b-a <= 1 for a, b in zip(grid, grid[1:])):
            break
        following_low = grid[max(0, best_index-1)]
        following_high = grid[min(len(grid)-1, best_index+1)]
        if (following_low, following_high) == (low, high):
            break
        low, high = following_low, following_high
    best = min(candidates.values(), key=lambda value: value["objective"])
    state, covariance, assigned = best["state"], best["covariance"], best["assigned"]
    centers = state[dimensions:].reshape(-1, 3)
    landmarks = []
    for j, (axial, y, z) in enumerate(centers):
        radial_squared = y*y+z*z
        derivative = np.array([[1, 0, 0], [0, -z/max(radial_squared, 1e-12), y/max(radial_squared, 1e-12)]])
        offset = dimensions+3*j
        block = derivative @ covariance[offset:offset+3, offset:offset+3] @ derivative.T
        if radial_squared < 1e-10:
            block[1, 1] = np.pi**2
        landmarks.append(dict(s=float(axial), angle=float(np.arctan2(z, y)), variance_s=float(block[0, 0]),
                              variance_angle=float(block[1, 1]), covariance=block.tolist(),
                              observations=int(np.sum(assigned == j)), euclidean_center=centers[j].tolist()))
    assignments, posterior, index = [], [], 0
    for traversal in case["traversals"]:
        ids = assigned[index:index+len(traversal["observations"])]
        assignments.append(ids.tolist()); posterior.append(np.eye(len(landmarks))[ids].tolist())
        index += len(ids)
    result = dict(landmarks=landmarks, assignments=assignments, association_posteriors=posterior,
                  new_landmark_hypothesis_probabilities=[[0.]*len(values) for values in assignments],
                  poses=[dict(traversal=frame["traversal"], frame_id=frame["frame_id"], s=float(state[2*i]),
                              roll=float(wrap(state[2*i+1])), covariance=covariance[2*i:2*i+2, 2*i:2*i+2].tolist())
                         for i, frame in enumerate(problem.frames)],
                  joint_optimization=dict(enabled=True, final_cost=best["slam_sse"]/2, optimality=best["optimality"]),
                  kslam=dict(landmarks=best["number"], objective=best["objective"],
                             search=[dict(landmarks=k, objective=v["objective"], slam_sse=v["slam_sse"], trace=v["trace"])
                                     for k, v in sorted(candidates.items())],
                             geometry="3D Euclidean landmarks, cylindrical-pose constraints"))
    if profile:
        result["joint_optimization_seconds"] = time.perf_counter()-start
    result["maintenance_records"] = build_records(case, result)
    result["worklist"] = [record for record in result["maintenance_records"] if record["requires_action"]]
    result["review_list"] = [record["id"] for record in result["maintenance_records"] if record["review_flag"]]
    return result
