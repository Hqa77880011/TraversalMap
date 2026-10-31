"""JCBB and ranked-assignment semantic marginals, reimplemented from their papers.

Neira & Tardos, T-RA 2001, doi:10.1109/70.976019.
Michael et al., IROS 2022, arXiv:2202.12802, Eqs. (10)-(14).
No GPL upstream code is copied into this module.
"""

import heapq
import itertools
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.stats import chi2


def innovation(observations, landmarks, case, pairs):
    """Joint residual/covariance including pose-map correlations when available."""
    residual = []
    for k, j in pairs:
        angle = observations[k]["angle"] + observations[k]["roll"] - landmarks[j]["angle"]
        residual.extend([observations[k]["s"]-landmarks[j]["s"], np.arctan2(np.sin(angle), np.cos(angle))])
    size = 2*len(pairs)
    covariance = np.zeros((size, size))
    if "_state_covariance" in case:
        dimension = case["_pose_dimensions"]
        previous = np.asarray(case["_state_covariance"])
        full = np.zeros((dimension+2*len(landmarks), dimension+2*len(landmarks)))
        full[:len(previous), :len(previous)] = previous
        for j in range((len(previous)-dimension)//2, len(landmarks)):
            full[dimension+2*j:dimension+2*j+2, dimension+2*j:dimension+2*j+2] = np.diag(
                [landmarks[j]["variance_s"], landmarks[j]["variance_angle"]])
        H = np.zeros((size, len(full)))
        for index, (k, j) in enumerate(pairs):
            offset = 2*index
            pose = 2*case["_pose_indices"][k]
            landmark = dimension+2*j
            H[offset:offset+2, pose:pose+2] = np.eye(2)
            H[offset:offset+2, landmark:landmark+2] = -np.eye(2)
            observation = observations[k]
            noise = observation.get("measurement_covariance", np.diag([
                observation.get("measurement_variance_s", observation["variance_s"]),
                observation.get("measurement_variance_angle", observation["variance_angle"])]))
            covariance[offset:offset+2, offset:offset+2] = noise
        covariance += H @ full @ H.T
    else:
        for index, (k, j) in enumerate(pairs):
            observation, landmark = observations[k], landmarks[j]
            noise = observation.get("association_covariance", np.diag([
                observation["variance_s"], observation["variance_angle"]+observation.get("variance_roll", 0)]))
            covariance[2*index:2*index+2, 2*index:2*index+2] = np.asarray(noise) + np.asarray(
                landmark.get("covariance", np.diag([landmark["variance_s"], landmark["variance_angle"]])))
        if "association_cross_covariance" in case:
            # Explicit 2N x 2N covariance across projected observation errors.
            cross = np.asarray(case["association_cross_covariance"])
            indices = [axis for k, _ in pairs for axis in (2*k, 2*k+1)]
            selected = cross[np.ix_(indices, indices)].copy()
            for index in range(len(pairs)):
                selected[2*index:2*index+2, 2*index:2*index+2] = 0
            covariance += selected
    covariance = (covariance+covariance.T)/2
    return np.asarray(residual), covariance


def statistic(observations, landmarks, case, pairs):
    if not pairs:
        return 0.0
    residual, covariance = innovation(observations, landmarks, case, pairs)
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), residual)
    return float(whitened @ whitened)


def jcbb(observations, landmarks, case):
    """Maximum-cardinality one-to-one hypothesis under individual/joint chi² tests."""
    n, m = len(observations), len(landmarks)
    confidence = case.get("association_confidence", 0.99)
    candidates = [[j for j in range(m) if statistic(observations, landmarks, case, [(k, j)])
                   <= chi2.ppf(confidence, 2)] for k in range(n)]
    order = sorted(range(n), key=lambda k: len(candidates[k]))
    best, best_cost, visited = [], float("inf"), 0

    def search(depth, pairs, used):
        nonlocal best, best_cost, visited
        visited += 1
        if len(pairs)+n-depth < len(best):
            return
        if depth == n:
            cost = statistic(observations, landmarks, case, pairs)
            if len(pairs) > len(best) or len(pairs) == len(best) and cost < best_cost:
                best, best_cost = list(pairs), cost
            return
        k = order[depth]
        for j in candidates[k]:
            proposed = pairs+[(k, j)]
            if j not in used and statistic(observations, landmarks, case, proposed) <= chi2.ppf(confidence, 2*len(proposed)):
                search(depth+1, proposed, used | {j})
        search(depth+1, pairs, used)

    search(0, [], set())
    ids = np.full(n, m, dtype=int)
    for k, j in best:
        ids[k] = j
    return ids, dict(cardinality=len(best), joint_mahalanobis=best_cost, visited_nodes=visited)


def ranked_assignments(cost, count):
    """Murty partitioning using constrained rectangular linear-sum assignments."""
    cost = np.asarray(cost, dtype=float)
    rows = len(cost)
    counter = itertools.count()

    def solve(fixed, forbidden):
        matrix = cost.copy()
        for row, column in forbidden:
            matrix[row, column] = np.inf
        for row, column in fixed:
            value = matrix[row, column]
            matrix[row] = np.inf
            matrix[:, column] = np.inf
            matrix[row, column] = value
        try:
            left, right = linear_sum_assignment(matrix)
        except ValueError:
            return None
        if len(left) != rows or not np.isfinite(matrix[left, right]).all():
            return None
        assignment = tuple(int(column) for column in right)
        return float(cost[left, right].sum()), assignment

    root = solve((), frozenset())
    if root is None:
        return [], True
    heap = [(root[0], next(counter), (), frozenset(), root[1])]
    result = []
    while heap and len(result) < count:
        value, _, fixed, forbidden, assignment = heapq.heappop(heap)
        result.append((value, assignment))
        for row in range(len(fixed), rows):
            prefix = fixed+tuple((i, assignment[i]) for i in range(len(fixed), row))
            excluded = forbidden | {(row, assignment[row])}
            solution = solve(prefix, excluded)
            if solution is not None:
                heapq.heappush(heap, (solution[0], next(counter), prefix, excluded, solution[1]))
    return result, not heap


def ranked_marginals(cost, count=200, hypothesis_bound=None):
    """Normalize k-best complete-hypothesis likelihoods, not rowwise softmax."""
    ranked, complete = ranked_assignments(cost, count)
    if not ranked:
        raise ValueError("Assignment graph has no feasible hypothesis")
    log_weights = -np.array([value for value, _ in ranked])
    normalizer = logsumexp(log_weights)
    weights = np.exp(log_weights-normalizer)
    marginal = np.zeros_like(cost)
    for weight, (_, assignment) in zip(weights, ranked):
        marginal[np.arange(len(cost)), assignment] += weight
    error_bound = None
    if complete:
        error_bound = 0.0
    elif hypothesis_bound is not None:
        remaining = max(0, hypothesis_bound-len(ranked))
        if remaining:
            log_bound = math.log(remaining)+log_weights[-1]
            error_bound = float(np.exp(log_bound-np.logaddexp(log_bound, normalizer)))
    return marginal, dict(hypotheses=len(ranked), complete=complete, marginal_error_bound=error_bound,
                           map_assignment=list(ranked[0][1]))


def _associate_frame(observations, landmarks, case, method):
    n, m = len(observations), len(landmarks)
    if method == "jcbb":
        ids, _ = jcbb(observations, landmarks, case)
        return ids, np.eye(m+1)[ids]
    cost = np.full((n, m+n), np.inf)
    for k, observation in enumerate(observations):
        for j, landmark in enumerate(landmarks):
            residual, covariance = innovation(observations, landmarks, case, [(k, j)])
            mahalanobis = float(residual @ np.linalg.solve(covariance, residual))
            if mahalanobis > chi2.ppf(case.get("association_confidence", .99), 2):
                continue
            cost[k, j] = .5*(mahalanobis+np.linalg.slogdet(covariance)[1]+2*np.log(2*np.pi))
            if "class_probabilities" in observation and "class_probabilities" in landmark:
                observed = np.asarray(observation["class_probabilities"])
                predicted = np.asarray(landmark["class_probabilities"])
                cost[k, j] -= np.log(max(float(observed @ predicted / (observed.sum()*predicted.sum())), 1e-12))
        cost[k, m+k] = case["null_cost"]
    bound = sum(math.comb(n, matched)*math.perm(m, matched) for matched in range(min(n, m)+1))
    marginal, diagnostics = ranked_marginals(cost, case.get("k_best_assignments", 200), bound)
    probabilities = np.c_[marginal[:, :m], marginal[:, m:].sum(1)]
    ids = np.array([j if j < m else m for j in diagnostics["map_assignment"]])
    return ids, probabilities


def associate(observations, landmarks, case, method):
    """One-to-one association is within a frame, not across repeated views."""
    groups = {}
    for k, observation in enumerate(observations):
        groups.setdefault(str(observation.get("frame_id", k)), []).append(k)
    ids = np.full(len(observations), len(landmarks), dtype=int)
    probability = np.zeros((len(observations), len(landmarks)+1))
    for indices in groups.values():
        frame_case = dict(case)
        if "_pose_indices" in case:
            frame_case["_pose_indices"] = [case["_pose_indices"][i] for i in indices]
        if "association_cross_covariance" in case:
            axes = [axis for k in indices for axis in (2*k, 2*k+1)]
            frame_case["association_cross_covariance"] = np.asarray(case["association_cross_covariance"])[np.ix_(axes, axes)]
        assigned, marginal = _associate_frame([observations[k] for k in indices], landmarks, frame_case, method)
        ids[indices], probability[indices] = assigned, marginal
    return ids, probability
