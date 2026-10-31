"""Optional independent geometry/appearance/order audit of measured loop proposals."""

import copy

import numpy as np

from .joint import PoseLandmarkProblem, optimize_joint, wrap


def robust_pose_cost(problem, poses):
    state = problem.initial.copy()
    lookup = {(row["traversal"], row["frame_id"]): row for row in poses}
    for i, frame in enumerate(problem.frames):
        row = lookup[frame["traversal"], frame["frame_id"]]
        state[2*i:2*i+2] = [row["s"], row["roll"]]
    residual = problem.linearize(state)[0][:problem.pose_dimensions+2*len(problem.odometry)]
    absolute = np.abs(residual)
    return float(np.where(absolute <= 1, .5*residual**2, absolute-.5).sum())


def apply(case, result, landmarks, method):
    """Optimize candidate loop constraints and merge only audited identities."""
    if not case.get("enable_loop_audit", False):
        return result
    if "poses" not in result:
        raise ValueError("Loop audit requires joint pose optimization")
    decisions, accepted_edges = [], []
    original_to_current = np.arange(len(landmarks))
    for proposal in case.get("loop_proposals", []):
        reasons = []
        pairs = [(int(original_to_current[a]), int(original_to_current[b])) for a, b in proposal["landmark_pairs"]]
        if len({a for a, _ in pairs}) != len(pairs) or len({b for _, b in pairs}) != len(pairs):
            reasons.append("nonbijective_identity_proposal")
        poses = {(row["traversal"], row["frame_id"]): row for row in result["poses"]}
        first = poses[proposal["from_traversal"], str(proposal["from_frame"])]
        second = poses[proposal["to_traversal"], str(proposal["to_frame"])]
        correction = np.array([proposal["delta_s"]-(second["s"]-first["s"]),
                               wrap(proposal["delta_roll"]-(second["roll"]-first["roll"]))])
        noise = np.asarray(proposal["covariance"])
        np.linalg.cholesky(noise)
        for a, b in pairs:
            left, right = landmarks[a], landmarks[b]
            distance = float(np.square(np.asarray(left["descriptor"])-right["descriptor"]).sum())
            if distance > case.get("loop_appearance_gate", .5):
                reasons.append("appearance_mismatch")
            error = np.array([left["s"]-right["s"]-correction[0],
                              wrap(left["angle"]-right["angle"]-correction[1])])
            variance = noise+np.diag([left["variance_s"]+right["variance_s"],
                                      left["variance_angle"]+right["variance_angle"]])
            if float(error @ np.linalg.solve(variance, error)) > case.get("loop_geometry_gate", 9.21):
                reasons.append("geometry_or_cyclic_mismatch")
        decision = dict(id=proposal["id"], accepted=False, reasons=sorted(set(reasons)))
        if reasons:
            decisions.append(decision)
            continue
        parent = list(range(len(landmarks)))

        def root(index):
            while parent[index] != index:
                index = parent[index]
            return index

        for a, b in pairs:
            parent[root(b)] = root(a)
        roots = sorted({root(j) for j in range(len(landmarks))})
        mapping = {j: roots.index(root(j)) for j in range(len(landmarks))}
        candidate = copy.deepcopy([landmarks[j] for j in roots])
        posterior = []
        for traversal in result["association_posteriors"]:
            rows = []
            for probabilities in traversal:
                row = np.zeros(len(roots))
                for j, value in enumerate(probabilities):
                    row[mapping[j]] += value
                rows.append(row.tolist())
            posterior.append(rows)
        assignments = [[mapping[j] for j in traversal] for traversal in result["assignments"]]
        edge = {key: proposal[key] for key in ("from_traversal", "from_frame", "to_traversal", "to_frame",
                                               "delta_s", "delta_roll", "covariance")}
        candidate_case = dict(case, _loop_edges=accepted_edges+[edge])
        problem = PoseLandmarkProblem(candidate_case, candidate, posterior, method)
        before = robust_pose_cost(problem, result["poses"])
        fitted_poses, report = optimize_joint(candidate_case, candidate, posterior, method, initial_poses=result["poses"])
        after = robust_pose_cost(problem, fitted_poses)
        inversions = 0
        for traversal, ids in zip(case["traversals"], assignments):
            for a, b in zip(ids, ids[1:]):
                tolerance = case.get("kappa", 2)*np.sqrt(candidate[a]["variance_s"]+candidate[b]["variance_s"])
                inversions += traversal["direction"]*(candidate[b]["s"]-candidate[a]["s"]) < -tolerance
        if after >= before-case.get("loop_minimum_pose_gain", 1e-6):
            decision["reasons"].append("no_robust_pose_gain")
        if inversions > case.get("loop_maximum_order_inversions", 0):
            decision["reasons"].append("direction_order_inversion")
        decision.update(robust_pose_cost_before=before, robust_pose_cost_after=after,
                        confident_order_inversions=int(inversions))
        if not decision["reasons"]:
            decision["accepted"] = True
            landmarks[:] = candidate
            accepted_edges.append(edge)
            original_to_current = np.array([mapping[int(j)] for j in original_to_current])
            result.update(landmarks=[{key: row[key] for key in ("s", "angle", "variance_s", "variance_angle", "observations", "covariance")}
                                     for row in landmarks], poses=fitted_poses, assignments=assignments,
                          association_posteriors=posterior, loop_joint_optimization=report)
        decisions.append(decision)
    result["loop_audit"] = dict(decisions=decisions, accepted=sum(row["accepted"] for row in decisions))
    return result


def metrics(case, result):
    """Reference loop validity is evaluator-only and never enters accept/reject."""
    decisions = result.get("loop_audit", {}).get("decisions", [])
    reference = {row["id"]: row for row in case.get("loop_proposals", [])}
    pairs = [(row["accepted"], reference[row["id"]].get("reference_valid")) for row in decisions]
    positives = [accepted for accepted, valid in pairs if valid is True]
    negatives = [accepted for accepted, valid in pairs if valid is False]
    return dict(loop_false_acceptance=float(np.mean(negatives)) if negatives else None,
                loop_false_rejection=float(np.mean([not value for value in positives])) if positives else None,
                loop_proposals=len(decisions))
