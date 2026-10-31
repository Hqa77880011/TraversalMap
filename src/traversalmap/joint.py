"""Joint cylindrical pose/landmark state block with retained soft associations."""

import numpy as np
from scipy.optimize import least_squares


def wrap(value):
    return np.arctan2(np.sin(value), np.cos(value))


def whitener(covariance):
    """Return W with W.T @ W equal to the supplied covariance inverse."""
    covariance = np.asarray(covariance, dtype=float)
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T)):
        raise ValueError("A finite symmetric covariance matrix is required")
    return np.linalg.solve(np.linalg.cholesky(covariance), np.eye(len(covariance)))


class PoseLandmarkProblem:
    """State vector: all (pose_s, roll), followed by all (landmark_s, phi).

    Absolute pose priors fix the pipe-frame gauge. Observation s is decomposed
    into pose_s plus axial_offset; absent explicit pose_s/offset, s is the
    supplied pose proxy and offset is zero. Landmark variables have no fused
    prior, preventing the same observations from entering twice.
    """

    def __init__(self, case, landmarks, posteriors, method):
        self.case = case
        self.landmark_count = len(landmarks)
        self.cyclic = method != "no_cyclic_angle"
        self.frames, self.pose_for_observation = [], {}
        self.factors, self.odometry, self.orders = [], [], []
        pose_variances = []
        frame_maps = []
        for r, traversal in enumerate(case["traversals"]):
            frame_map = {}
            for k, observation in enumerate(traversal["observations"]):
                frame_id = str(observation.get("frame_id", k))
                axial = float(observation.get("pose_s", observation["s"] - observation.get("axial_offset", 0)))
                roll = float(observation["roll"])
                if frame_id not in frame_map:
                    frame_map[frame_id] = len(self.frames)
                    self.frames.append(dict(traversal=r, frame_id=frame_id, s=axial, roll=roll))
                    pose_variances.extend([
                        observation.get("pose_variance_s", observation["variance_s"]),
                        observation.get("pose_variance_roll", observation.get("variance_roll", 0) or case.get("default_pose_roll_variance", 1e-4)),
                    ])
                pose_index = frame_map[frame_id]
                if not np.allclose([self.frames[pose_index]["s"], self.frames[pose_index]["roll"]], [axial, roll]):
                    raise ValueError("Observations sharing a frame must declare the same pose_s and roll prior")
                self.pose_for_observation[r, k] = pose_index
            frame_maps.append(frame_map)
        self.pose_count = len(self.frames)
        self.pose_dimensions = 2 * self.pose_count
        self.prior = np.array([[frame["s"], frame["roll"]] for frame in self.frames]).ravel()
        covariance = case.get("pose_prior_covariance", np.diag(pose_variances))
        self.pose_covariance = np.asarray(covariance, dtype=float)
        if self.pose_covariance.shape != (self.pose_dimensions, self.pose_dimensions):
            raise ValueError("pose_prior_covariance must align with first-seen semantic frames")
        self.pose_whitener = whitener(self.pose_covariance)
        self.initial = np.r_[self.prior, np.array([[value["s"], value["angle"]] for value in landmarks]).ravel()]
        if not np.isfinite(self.initial).all():
            raise ValueError("Initial pose and landmark coordinates must be finite")
        cutoff = case.get("joint_probability_cutoff", 1e-8)
        for r, traversal in enumerate(case["traversals"]):
            for edge in traversal.get("odometry", []):
                first, second = (frame_maps[r][str(edge[key])] for key in ("from", "to"))
                noise = edge["covariance"] if "covariance" in edge else np.diag([edge["variance_s"], edge["variance_roll"]])
                self.odometry.append((first, second, np.array([edge["delta_s"], edge.get("delta_roll", 0)]), whitener(noise)))
            for k, observation in enumerate(traversal["observations"]):
                pose = self.pose_for_observation[r, k]
                offset = observation.get("axial_offset", observation["s"] - self.frames[pose]["s"])
                noise = observation.get("measurement_covariance", np.diag([
                    observation.get("measurement_variance_s", observation["variance_s"]),
                    observation.get("measurement_variance_angle", observation["variance_angle"]),
                ]))
                W = whitener(noise)
                for j, probability in enumerate(posteriors[r][k]):
                    if probability > cutoff:
                        self.factors.append((pose, j, float(offset), float(observation["angle"]), np.sqrt(probability) * W))
            order = np.arange(len(traversal["observations"]))
            if method == "shuffled_order":
                order = np.random.default_rng(case.get("order_seed", 17)).permutation(len(order))
            weight = case.get("order_weight", 0)
            if method in ("local_only", "appearance_only", "position_only", "no_soft_hypotheses"):
                weight = 0
            for first, second in zip(order[:-1], order[1:]):
                a, b = traversal["observations"][first], traversal["observations"][second]
                for j in np.flatnonzero(np.asarray(posteriors[r][first]) > cutoff):
                    for other in np.flatnonzero(np.asarray(posteriors[r][second]) > cutoff):
                        if j == other or weight == 0:
                            continue
                        epsilon = case.get("kappa", 2) * np.sqrt(
                            landmarks[j]["variance_s"] + landmarks[other]["variance_s"]
                            + case.get("delta_s_variance", a["variance_s"] + b["variance_s"]))
                        if method == "fixed_order_tolerance":
                            epsilon = case.get("fixed_order_tolerance_m", .5)
                        factor = np.sqrt(weight * posteriors[r][first][j] * posteriors[r][second][other])
                        self.orders.append((j, other, traversal["direction"], epsilon, factor, method))

    def linearize(self, values):
        """Return whitened residuals and their analytic Jacobian at state values."""
        values = np.asarray(values)
        size = len(self.initial)
        delta = values[:self.pose_dimensions] - self.prior
        delta = delta.copy()
        delta[1::2] = wrap(delta[1::2])
        residuals = [self.pose_whitener @ delta]
        prior_jacobian = np.zeros((self.pose_dimensions, size))
        prior_jacobian[:, :self.pose_dimensions] = self.pose_whitener
        jacobians = [prior_jacobian]
        for first, second, increment, W in self.odometry:
            a, b = 2 * first, 2 * second
            error = values[b:b + 2] - values[a:a + 2] - increment
            error[1] = wrap(error[1])
            jacobian = np.zeros((2, size))
            jacobian[:, a:a + 2] -= W
            jacobian[:, b:b + 2] += W
            residuals.append(W @ error)
            jacobians.append(jacobian)
        for pose, landmark, offset, angle, W in self.factors:
            a, b = 2 * pose, self.pose_dimensions + 2 * landmark
            error = values[a:a + 2] - values[b:b + 2] + [offset, angle]
            if self.cyclic:
                error[1] = wrap(error[1])
            jacobian = np.zeros((2, size))
            jacobian[:, a:a + 2] = W
            jacobian[:, b:b + 2] = -W
            residuals.append(W @ error)
            jacobians.append(jacobian)
        for first, second, direction, epsilon, weight, method in self.orders:
            a, b = self.pose_dimensions + 2 * first, self.pose_dimensions + 2 * second
            jacobian = np.zeros((1, size))
            if method == "generic_smoothness":
                error = values[b] - values[a]
                jacobian[0, a], jacobian[0, b] = -weight, weight
            else:
                error = max(0, direction * (values[a] - values[b]) - epsilon)
                if method == "hard_order":
                    error = 100.0 if error > 0 else 0.0
                elif error > 0:
                    jacobian[0, a], jacobian[0, b] = weight * direction, -weight * direction
            residuals.append(np.array([weight * error]))
            jacobians.append(jacobian)
        return np.concatenate(residuals), np.vstack(jacobians)


def optimize_joint(case, landmarks, posteriors, method):
    """Optimize both state blocks and return conditional Gaussian marginals."""
    problem = PoseLandmarkProblem(case, landmarks, posteriors, method)
    initial_residual, initial_jacobian = problem.linearize(problem.initial)
    fitted = least_squares(lambda value: problem.linearize(value)[0], problem.initial,
                           jac=lambda value: problem.linearize(value)[1],
                           max_nfev=case.get("joint_max_evaluations", 100),
                           ftol=1e-9, xtol=1e-9, gtol=1e-9)
    if not fitted.success:
        raise RuntimeError("Joint pose/landmark optimization failed: " + fitted.message)
    covariance = np.linalg.pinv(fitted.jac.T @ fitted.jac, rcond=1e-12, hermitian=True)
    conservative = not case.get("independent_observations", False)
    poses = []
    for i, frame in enumerate(problem.frames):
        block = covariance[2 * i:2 * i + 2, 2 * i:2 * i + 2].copy()
        if conservative:
            prior = np.diag(problem.pose_covariance)[2 * i:2 * i + 2]
            block += np.diag(np.maximum(prior - np.diag(block), 0))
        poses.append(dict(traversal=frame["traversal"], frame_id=frame["frame_id"],
                          s=float(fitted.x[2 * i]), roll=float(wrap(fitted.x[2 * i + 1])),
                          covariance=block.tolist()))
    for j, landmark in enumerate(landmarks):
        offset = problem.pose_dimensions + 2 * j
        block = covariance[offset:offset + 2, offset:offset + 2].copy()
        if conservative:
            floor = [landmark["variance_s"], landmark["variance_angle"]]
            block += np.diag(np.maximum(np.asarray(floor) - np.diag(block), 0))
        landmark.update(s=float(fitted.x[offset]), angle=float(wrap(fitted.x[offset + 1])),
                        variance_s=float(block[0, 0]), variance_angle=float(block[1, 1]),
                        covariance=block.tolist())
    report = dict(enabled=True, success=bool(fitted.success), pose_variables=problem.pose_dimensions,
                  landmark_variables=2 * len(landmarks), initial_cost=float(initial_residual @ initial_residual / 2),
                  final_cost=float(fitted.cost), initial_gradient_norm=float(np.linalg.norm(initial_jacobian.T @ initial_residual)),
                  pose_change_l2=float(np.linalg.norm(fitted.x[:problem.pose_dimensions] - problem.initial[:problem.pose_dimensions])),
                  landmark_change_l2=float(np.linalg.norm(fitted.x[problem.pose_dimensions:] - problem.initial[problem.pose_dimensions:])),
                  optimality=float(fitted.optimality), evaluations=int(fitted.nfev),
                  pose_prior_residuals=problem.pose_dimensions, odometry_factors=len(problem.odometry),
                  observation_factors=len(problem.factors), order_factors=len(problem.orders),
                  unknown_correlation_marginal_floor=conservative)
    return poses, report
