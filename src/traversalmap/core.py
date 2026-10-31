import numpy as np
import time
from scipy.optimize import linear_sum_assignment, minimize

from .joint import optimize_joint
from .maintenance import build_records, score_maintenance, distribution


METHODS = ["traversalmap", "local_only", "no_cyclic_angle", "shuffled_order", "hard_order", "fixed_map",
           "fixed_order_tolerance", "no_soft_hypotheses", "appearance_only", "position_only",
           "generic_smoothness", "no_joint_optimization", "no_alternation", "no_class_likelihood",
           "jcbb", "probabilistic_semantic", "kslam"]


def wrap(value):
    return np.arctan2(np.sin(value), np.cos(value))


def generate(rng, count):
    for index in range(count):
        n = 6
        truth = [
            {
                "id": str(i),
                "s": i * 1.5 + 1,
                "angle": rng.uniform(-np.pi, np.pi),
                "descriptor": [rng.random() for _ in range(4)],
                "class": "crack" if i % 2 == 0 else "deposit",
                "severity": i % 3 + 1,
            }
            for i in range(n)
        ]
        traversals = []
        for r, direction in enumerate([1, -1, 1]):
            order = list(range(n)) if direction == 1 else list(range(n - 1, -1, -1))
            observations = []
            for k, i in enumerate(order):
                landmark = truth[i]
                observations.append(
                    {
                        "s": landmark["s"] + rng.gauss(0, 0.25),
                        "angle": landmark["angle"] + rng.gauss(0, 0.1),
                        "roll": rng.gauss(0, 0.05),
                        "variance_s": 0.0625,
                        "variance_angle": 0.01,
                        "variance_roll": 0.0025,
                        "descriptor": (
                            np.asarray(landmark["descriptor"])
                            + np.array([rng.gauss(0, 0.1) for _ in range(4)])
                        ).tolist(),
                        "truth_id": landmark["id"],
                        "frame_id": str(k),
                        "source_frame": f"traversal-{r}/frame-{k:04d}",
                        "class_probabilities": [.9, .1] if landmark["class"] == "crack" else [.1, .9],
                        "severity_probabilities": [.9 if level == landmark["severity"] else .1 / 3 for level in range(4)],
                        "quality": .9,
                    }
                )
            traversals.append({"direction": direction, "observations": observations})
        yield {
            "id": str(index),
            "family": f"pipe-{index}",
            "truth": truth,
            "traversals": traversals,
            "order_weight": 2.0,
            "null_cost": 12.0,
            "cardinality_weight": 2.0,
            "top_l": 4,
            "kappa": 2.0,
            "map_length_meters": 12.0,
            "independent_observations": True,
            "class_labels": ["crack", "deposit"],
            "severity_levels": [0, 1, 2, 3],
            "maintenance": {"actionable_classes": ["crack"], "minimum_severity": 2},
        }


def optimize(observations, landmarks, direction, case, method):
    if method in ("jcbb", "probabilistic_semantic"):
        from .association import associate
        return associate(observations, landmarks, case, method)
    n, m = len(observations), len(landmarks)
    local = np.zeros((n, m + 1))
    for k, observation in enumerate(observations):
        for j, landmark in enumerate(landmarks):
            angle = observation["angle"] + observation["roll"] - landmark["angle"]
            if method != "no_cyclic_angle":
                angle = wrap(angle)
            axial = observation["s"] - landmark["s"]
            appearance = np.sum(
                (np.asarray(observation["descriptor"]) - landmark["descriptor"]) ** 2
            )
            class_cost = 0.0
            if method not in ("no_class_likelihood", "appearance_only", "position_only") and "class_probabilities" in observation and "class_probabilities" in landmark:
                classes = list(range(len(observation["class_probabilities"])))
                observed = distribution(observation["class_probabilities"], classes)
                predicted = distribution(landmark["class_probabilities"], classes)
                class_cost = -case.get("class_weight", 1.0) * np.log(max(float(observed @ predicted), 1e-12))
            geometric_cost = (axial**2 / (observation["variance_s"] + landmark["variance_s"])
                              + angle**2 / (observation["variance_angle"] + observation.get("variance_roll", 0)
                                           + landmark["variance_angle"]))
            if "_state_covariance" in case:
                from .association import innovation
                error, covariance = innovation(observations, landmarks, case, [(k, j)])
                if method == "no_cyclic_angle":
                    error[1] = angle
                geometric_cost = float(error @ np.linalg.solve(covariance, error))
            local[k, j] = (
                (0 if method == "position_only" else appearance)
                + class_cost
                + (0 if method == "appearance_only" else
                   geometric_cost)
            )
        local[k, -1] = case["null_cost"]
    active = np.zeros((n, m + 1), dtype=bool)
    for k in range(n):
        active[k, np.argsort(local[k, :-1])[: case["top_l"]]] = True
        if method != "appearance_only":
            gate = case.get("axial_gate_sigma", 5.0)
            for j, landmark in enumerate(landmarks):
                active[k, j] &= abs(observations[k]["s"] - landmark["s"]) <= gate * np.sqrt(
                    observations[k]["variance_s"] + landmark["variance_s"])
        active[k, -1] = True
    order = np.arange(n)
    if method == "shuffled_order":
        order = np.random.default_rng(case.get("order_seed", 17)).permutation(n)
    pairs = list(zip(order[:-1], order[1:]))
    penalty = np.zeros((max(n - 1, 0), m, m))
    for k, (first, second) in enumerate(pairs):
        for j in range(m):
            for other in range(m):
                epsilon = case["kappa"] * np.sqrt(
                    landmarks[j]["variance_s"]
                    + landmarks[other]["variance_s"]
                    + case.get("delta_s_variance", observations[first]["variance_s"]
                               + observations[second]["variance_s"])
                )
                if method == "fixed_order_tolerance":
                    epsilon = case.get("fixed_order_tolerance_m", 0.5)
                difference = direction * (landmarks[other]["s"] - landmarks[j]["s"])
                penalty[k, j, other] = max(0, -difference - epsilon) ** 2
                if method == "generic_smoothness":
                    penalty[k, j, other] = (landmarks[other]["s"] - landmarks[j]["s"]) ** 2
    if method == "hard_order":
        penalty = (penalty > 0).astype(float) * 1e4
    weight = 0 if method in ("local_only", "appearance_only", "position_only", "no_soft_hypotheses") else case["order_weight"]

    def probability(raw):
        logits = raw.reshape(n, m + 1)
        logits = np.where(active, logits, -1e4)
        exponents = np.exp(logits - logits.max(1, keepdims=True))
        return exponents / exponents.sum(1, keepdims=True)

    def objective(raw):
        A = probability(raw)
        energy = np.sum(A * local)
        gradient = local.copy()
        for k, (first, second) in enumerate(pairs):
            energy += weight * (A[first, :-1] @ penalty[k] @ A[second, :-1])
            gradient[first, :-1] += weight * (penalty[k] @ A[second, :-1])
            gradient[second, :-1] += weight * (penalty[k].T @ A[first, :-1])
        excess = np.maximum(A[:, :-1].sum(0) - 1, 0)
        energy += case["cardinality_weight"] * excess.dot(excess)
        gradient[:, :-1] += 2 * case["cardinality_weight"] * excess[None]
        energy += 0.1 * np.sum(A * np.log(np.maximum(A, 1e-12)))
        gradient += 0.1 * (np.log(np.maximum(A, 1e-12)) + 1)
        gradient = A * (gradient - (A * gradient).sum(1, keepdims=True))
        gradient[~active] = 0
        return float(energy), gradient.ravel()

    if method == "no_soft_hypotheses":
        ids = np.where(active, local, np.inf).argmin(1)
        return ids, np.eye(m + 1)[ids]
    fitted = minimize(objective, -np.minimum(local, 10).ravel(), method="L-BFGS-B",
                      jac=True, bounds=[(-30, 30)] * local.size, options={"maxiter": 300})
    if not fitted.success and not np.isfinite(fitted.fun):
        raise RuntimeError("Association objective failed")
    A = probability(fitted.x)
    return A.argmax(1), A


def make_landmark(observation):
    """Initialize a landmark from a roll-corrected, non-reference observation."""
    landmark = dict(s=float(observation["s"]),
                angle=float(wrap(observation["angle"] + observation["roll"])),
                variance_s=float(observation["variance_s"]),
                variance_angle=float(observation["variance_angle"] + observation.get("variance_roll", 0)),
                descriptor=np.asarray(observation["descriptor"], dtype=float), observations=1)
    if "class_probabilities" in observation:
        landmark["class_probabilities"] = distribution(observation["class_probabilities"],
                                                      list(range(len(observation["class_probabilities"]))))
    return landmark


def update_landmark(landmark, observation, independent=False):
    """Fuse axial/cyclic coordinates; unknown correlation uses intersection."""
    observed = make_landmark(observation)
    prior_mean = np.array([landmark["s"], landmark["angle"]])
    measured = np.array([observed["s"], landmark["angle"] + wrap(observed["angle"] - landmark["angle"])])
    prior = np.array([landmark["variance_s"], landmark["variance_angle"]])
    noise = np.array([observed["variance_s"], observed["variance_angle"]])
    if not np.isfinite(np.r_[prior, noise, measured]).all() or np.any(np.r_[prior, noise] <= 0):
        raise ValueError("Finite coordinates and positive axial/angle variances required")
    if independent:
        variance = 1 / (1 / prior + 1 / noise)
        mean = variance * (prior_mean / prior + measured / noise)
    else:
        # One common CI weight preserves the joint diagonal uncertainty contract.
        choices = []
        for weight in np.linspace(0, 1, 101):
            variance = 1 / (weight / prior + (1 - weight) / noise)
            mean = variance * (weight * prior_mean / prior + (1 - weight) * measured / noise)
            choices.append((float(np.log(variance).sum()), mean, variance))
        _, mean, variance = min(choices, key=lambda option: option[0])
    count = landmark["observations"]
    if "class_probabilities" in observed and "class_probabilities" in landmark:
        landmark["class_probabilities"] = (count*landmark["class_probabilities"] + observed["class_probabilities"]) / (count+1)
    landmark.update(s=float(mean[0]), angle=float(wrap(mean[1])),
                    variance_s=float(variance[0]), variance_angle=float(variance[1]),
                    descriptor=(count * landmark["descriptor"] + observed["descriptor"]) / (count + 1),
                    observations=count + 1)


def run_case(case, method, profile=False):
    """Associate traversals, solve the joint state block and build audit records."""
    if method not in METHODS:
        raise ValueError("Unknown traversal mapping method")
    if method == "kslam":
        from .kslam import run
        return run(case, profile)
    traversals = case["traversals"]
    if not traversals or not traversals[0]["observations"]:
        raise ValueError("A nonempty anchor traversal is required")
    landmarks = [make_landmark(o) for o in traversals[0]["observations"]]
    assignments = [list(range(len(landmarks)))]
    soft = [[{j: 1.0} for j in range(len(landmarks))]]
    null_probabilities = [[0.0] * len(landmarks)]
    for traversal in traversals[1:]:
        ids, A = optimize(
            traversal["observations"], landmarks, traversal["direction"], case, method
        )
        current = []
        current_soft = []
        original_count = A.shape[1] - 1
        for k, (observation, index) in enumerate(zip(traversal["observations"], ids)):
            weights = {j: float(probability) for j, probability in enumerate(A[k, :-1]) if probability > 0}
            if index == original_count:
                index = len(landmarks)
                landmarks.append(make_landmark(observation))
                weights[index] = float(A[k, -1])
            elif method != "fixed_map":
                update_landmark(landmarks[index], observation, case.get("independent_observations", False))
            current.append(int(index))
            current_soft.append(weights)
        assignments.append(current)
        soft.append(current_soft)
        null_probabilities.append(A[:, -1].tolist())
    posterior = [[[weights.get(j, 0.0) for j in range(len(landmarks))] for weights in traversal] for traversal in soft]
    joint_enabled = case.get("joint_optimization", True) and method not in ("fixed_map", "no_joint_optimization")
    poses = None
    joint_report = dict(enabled=False)
    joint_seconds = 0.0
    if joint_enabled:
        start = time.perf_counter()
        poses, joint_report, state_covariance = optimize_joint(case, landmarks, posterior, method, return_covariance=True)
        reports = [dict(joint_report)]
        rounds = case.get("alternating_iterations", 3) if method != "no_alternation" else 1
        if rounds < 1:
            raise ValueError("alternating_iterations must be positive")
        for iteration in range(1, rounds):
            pose_lookup = {(pose["traversal"], pose["frame_id"]): (i, pose) for i, pose in enumerate(poses)}
            previous = np.asarray([row for traversal in posterior for row in traversal])
            updated, new_assignments, nulls = [], [], []
            for r, traversal in enumerate(traversals):
                corrected, pose_indices = [], []
                for k, observation in enumerate(traversal["observations"]):
                    pose_index, pose = pose_lookup[r, str(observation.get("frame_id", k))]
                    prior_s = observation.get("pose_s", observation["s"]-observation.get("axial_offset", 0))
                    offset = observation.get("axial_offset", observation["s"]-prior_s)
                    corrected.append(dict(observation, s=pose["s"]+offset, roll=pose["roll"]))
                    pose_indices.append(pose_index)
                association_case = dict(case, _state_covariance=state_covariance,
                                        _pose_indices=pose_indices, _pose_dimensions=2*len(poses))
                ids, probabilities = optimize(corrected, landmarks, traversal["direction"], association_case, method)
                # Births are resolved into persistent variables before the next state solve.
                width = probabilities.shape[1]-1
                rows = []
                for k, index in enumerate(ids):
                    weights = probabilities[k, :-1].tolist()
                    if index == width:
                        index = len(landmarks)
                        landmarks.append(make_landmark(corrected[k]))
                        weights += [0.0]*(index-len(weights)) + [float(probabilities[k, -1])]
                    new_assignments.append((r, k, int(index)))
                    rows.append(weights)
                updated.append(rows)
                nulls.append(probabilities[:, -1].tolist())
            posterior = [[row+[0.0]*(len(landmarks)-len(row)) for row in traversal] for traversal in updated]
            assignments = [[index for rr, _, index in new_assignments if rr == r] for r in range(len(traversals))]
            null_probabilities = nulls
            # Recompute appearance/class prototypes once from current raw evidence and A.
            for j, landmark in enumerate(landmarks):
                descriptors, class_vectors, weights, class_weights = [], [], [], []
                for r, traversal in enumerate(traversals):
                    for k, observation in enumerate(traversal["observations"]):
                        weight = posterior[r][k][j] * observation.get("quality", 1.0)
                        if weight > 0:
                            descriptors.append(np.asarray(observation["descriptor"])); weights.append(weight)
                            if "class_probabilities" in observation:
                                vector = observation["class_probabilities"]
                                class_vectors.append(distribution(vector, list(range(len(vector))))); class_weights.append(weight)
                if weights:
                    landmark["descriptor"] = np.average(descriptors, axis=0, weights=weights)
                if class_weights:
                    landmark["class_probabilities"] = np.average(class_vectors, axis=0, weights=class_weights)
            poses, joint_report, state_covariance = optimize_joint(case, landmarks, posterior, method,
                                                                  initial_poses=poses, return_covariance=True)
            current = np.asarray([row for traversal in posterior for row in traversal])
            delta = float(np.max(np.abs(current-previous))) if current.shape == previous.shape else None
            reports.append(dict(joint_report, posterior_change=delta))
            if delta is not None and delta < case.get("alternating_tolerance", 1e-4):
                break
        joint_report["alternating_rounds"] = len(reports)
        joint_report["rounds"] = reports
        joint_seconds = time.perf_counter() - start
    result = {
        "landmarks": [{key: l[key] for key in ("s", "angle", "variance_s", "variance_angle", "observations")}
                      for l in landmarks],
        "assignments": assignments,
        "association_posteriors": posterior,
        "new_landmark_hypothesis_probabilities": null_probabilities,
        "joint_optimization": joint_report,
    }
    for saved, landmark in zip(result["landmarks"], landmarks):
        if "covariance" in landmark:
            saved["covariance"] = landmark["covariance"]
    if poses is not None:
        result["poses"] = poses
    if profile:
        result["joint_optimization_seconds"] = joint_seconds
    if case.get("enable_loop_audit", False):
        from .loops import apply
        result = apply(case, result, landmarks, method)
    result["maintenance_records"] = build_records(case, result)
    result["worklist"] = [record for record in result["maintenance_records"] if record["requires_action"]]
    result["review_list"] = [record["id"] for record in result["maintenance_records"] if record["review_flag"]]
    return result


def score(case, result):
    landmarks, truth = result["landmarks"], case["truth"]
    cost = np.array(
        [
            [abs(l["s"] - g["s"]) + abs(wrap(l["angle"] - g["angle"])) for g in truth]
            for l in landmarks
        ]
    )
    left, right = linear_sum_assignment(cost)
    matched = {int(a): int(b) for a, b in zip(left, right)
               if cost[a, b] < case.get("evaluation_matching_gate", 2.0)}
    correct = total = crossed = adjacent = 0
    for traversal, assigned in zip(case["traversals"], result["assignments"]):
        for observation, identity in zip(traversal["observations"], assigned):
            correct += (
                identity in matched and truth[matched[identity]]["id"] == observation["truth_id"]
            )
            total += 1
        for a, b in zip(assigned, assigned[1:]):
            crossed += traversal["direction"] * (landmarks[b]["s"] - landmarks[a]["s"]) < 0
            adjacent += 1
    errors_s = [landmarks[a]["s"] - truth[b]["s"] for a, b in matched.items()]
    errors_angle = [
        abs(wrap(landmarks[a]["angle"] - truth[b]["angle"])) for a, b in matched.items()
    ]
    predicted_ids = [identity for assigned in result["assignments"] for identity in assigned]
    truth_ids = [observation["truth_id"] for traversal in case["traversals"]
                 for observation in traversal["observations"]]
    # Pairwise identity F1 penalizes both merged and split physical landmarks.
    contingency = {}
    predicted_counts, truth_counts = {}, {}
    for prediction, reference in zip(predicted_ids, truth_ids):
        contingency[prediction, reference] = contingency.get((prediction, reference), 0) + 1
        predicted_counts[prediction] = predicted_counts.get(prediction, 0) + 1
        truth_counts[reference] = truth_counts.get(reference, 0) + 1
    pairs_count = lambda counts: sum(count * (count - 1) // 2 for count in counts)
    true_positive = pairs_count(contingency.values())
    predicted_pairs, true_pairs = pairs_count(predicted_counts.values()), pairs_count(truth_counts.values())
    precision = true_positive / predicted_pairs if predicted_pairs else float(true_pairs == 0)
    recall = true_positive / true_pairs if true_pairs else float(predicted_pairs == 0)
    report = {
        "unique_count_error": abs(len(landmarks) - len(truth)),
        "association_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "association_precision": precision,
        "association_recall": recall,
        "observation_identity_accuracy": correct / total if total else None,
        "association_true_positive_pairs": true_positive,
        "association_predicted_pairs": predicted_pairs,
        "association_reference_pairs": true_pairs,
        "duplicate_landmark_rate": max(0, len(landmarks) - len(matched)) / max(1, len(landmarks)),
        "missed_landmark_rate": 1 - len(matched) / len(truth),
        "crossed_identity_rate": crossed / adjacent if adjacent else None,
        "axial_rmse": float(np.sqrt(np.mean(np.square(errors_s)))) if errors_s else None,
        "cyclic_angle_mae_radians": float(np.mean(errors_angle)) if errors_angle else None,
        "map_records_per_km": len(landmarks) / (case["map_length_meters"] / 1000),
        "legal_effects": correct,
    }
    report.update(score_maintenance(case, result, matched))
    if "loop_audit" in result:
        from .loops import metrics
        report.update(metrics(case, result))
    joint = result.get("joint_optimization", {})
    for key in ("initial_cost", "final_cost", "initial_gradient_norm", "pose_change_l2", "landmark_change_l2"):
        report["joint_" + key] = joint.get(key)
    return report
