import numpy as np
from scipy.optimize import linear_sum_assignment, minimize


METHODS = ["traversalmap", "local_only", "no_cyclic_angle", "shuffled_order", "hard_order", "fixed_map",
           "fixed_order_tolerance", "no_soft_hypotheses", "appearance_only", "position_only",
           "generic_smoothness"]


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
            }
            for i in range(n)
        ]
        traversals = []
        for direction in [1, -1, 1]:
            order = list(range(n)) if direction == 1 else list(range(n - 1, -1, -1))
            observations = []
            for i in order:
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
        }


def optimize(observations, landmarks, direction, case, method):
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
            local[k, j] = (
                (0 if method == "position_only" else appearance)
                + (0 if method == "appearance_only" else
                   axial**2 / (observation["variance_s"] + landmark["variance_s"])
                   + angle**2 / (observation["variance_angle"] + observation.get("variance_roll", 0)
                                + landmark["variance_angle"]))
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
    return dict(s=float(observation["s"]),
                angle=float(wrap(observation["angle"] + observation["roll"])),
                variance_s=float(observation["variance_s"]),
                variance_angle=float(observation["variance_angle"] + observation.get("variance_roll", 0)),
                descriptor=np.asarray(observation["descriptor"], dtype=float), observations=1)


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
    landmark.update(s=float(mean[0]), angle=float(wrap(mean[1])),
                    variance_s=float(variance[0]), variance_angle=float(variance[1]),
                    descriptor=(count * landmark["descriptor"] + observed["descriptor"]) / (count + 1),
                    observations=count + 1)


def run_case(case, method):
    traversals = case["traversals"]
    if not traversals or not traversals[0]["observations"]:
        raise ValueError("A nonempty anchor traversal is required")
    landmarks = [make_landmark(o) for o in traversals[0]["observations"]]
    assignments = [list(range(len(landmarks)))]
    for traversal in traversals[1:]:
        ids, A = optimize(
            traversal["observations"], landmarks, traversal["direction"], case, method
        )
        current = []
        original_count = A.shape[1] - 1
        for k, (observation, index) in enumerate(zip(traversal["observations"], ids)):
            if index == original_count:
                index = len(landmarks)
                landmarks.append(make_landmark(observation))
            elif method != "fixed_map":
                update_landmark(landmarks[index], observation, case.get("independent_observations", False))
            current.append(int(index))
        assignments.append(current)
    return {
        "landmarks": [{key: l[key] for key in ("s", "angle", "variance_s", "variance_angle", "observations")}
                      for l in landmarks],
        "assignments": assignments,
    }


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
    return {
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
