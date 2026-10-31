"""Auditable categorical defect records, work lists and maintenance metrics."""

import csv
import json
from pathlib import Path

import numpy as np


def labels_for(case, key, labels_key):
    """Read configured labels or use zero-based IDs inferred from observation vectors."""
    labels = case.get(labels_key)
    if labels is None:
        for traversal in case["traversals"]:
            for observation in traversal["observations"]:
                if key in observation:
                    labels = list(range(len(observation[key])))
                    break
            if labels is not None:
                break
    labels = [] if labels is None else list(labels)
    if len(set(labels)) != len(labels):
        raise ValueError("Class and ordinal severity labels must be unique")
    return labels


def distribution(value, labels):
    """Normalize a supplied nonnegative likelihood vector without using references."""
    probabilities = np.asarray(value, dtype=float)
    if (probabilities.shape != (len(labels),) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or probabilities.sum() <= 0):
        raise ValueError("Likelihood vectors must be finite, nonnegative and match their labels")
    return probabilities / probabilities.sum()


def action_policy(case, classes, severities):
    """Resolve predeclared actionable classes and an ordinal severity threshold."""
    config = case.get("maintenance", {})
    actionable = config.get("actionable_classes", classes)
    if not set(actionable) <= set(classes):
        raise ValueError("actionable_classes must occur in class_labels")
    minimum = config.get("minimum_severity", severities[min(1, len(severities) - 1)] if severities else None)
    if minimum is not None and minimum not in severities:
        raise ValueError("minimum_severity must occur in the ordered severity_levels")
    threshold = config.get("action_probability_threshold", .5)
    if not 0 <= threshold <= 1:
        raise ValueError("action_probability_threshold must lie in [0,1]")
    return set(actionable), severities.index(minimum) if minimum is not None else None, threshold


def build_records(case, result):
    """Pool accepted semantic evidence and retain identity/provenance/uncertainty."""
    classes = labels_for(case, "class_probabilities", "class_labels")
    severities = labels_for(case, "severity_probabilities", "severity_levels")
    actionable, minimum, threshold = action_policy(case, classes, severities)
    config = case.get("maintenance", {})
    records = []
    for j, landmark in enumerate(result["landmarks"]):
        sources = []
        pooled = {"class": np.zeros(len(classes)), "severity": np.zeros(len(severities))}
        weights = dict.fromkeys(pooled, 0.0)
        for r, (traversal, assigned) in enumerate(zip(case["traversals"], result["assignments"])):
            for k, (observation, identity) in enumerate(zip(traversal["observations"], assigned)):
                if identity != j:
                    continue
                probability = result["association_posteriors"][r][k][j]
                quality = float(observation.get("quality", 1))
                if not np.isfinite(quality) or not 0 <= quality <= 1:
                    raise ValueError("Semantic evidence quality must lie in [0,1]")
                weight = probability * quality
                sources.append(dict(traversal=traversal.get("id", r), frame_id=str(observation.get("frame_id", k)),
                                    observation_index=k, source_frame=observation.get("source_frame", str(observation.get("frame_id", k))),
                                    association_probability=float(probability), quality=quality))
                for name, labels in (("class", classes), ("severity", severities)):
                    key = name + "_probabilities"
                    if key in observation:
                        pooled[name] += weight * distribution(observation[key], labels)
                        weights[name] += weight
        probabilities = {name: (values / weights[name]).tolist() if weights[name] > 0 else None
                         for name, values in pooled.items()}
        class_value = classes[int(np.argmax(probabilities["class"]))] if probabilities["class"] is not None else None
        severity_value = severities[int(np.argmax(probabilities["severity"]))] if probabilities["severity"] is not None else None
        confidence_class = max(probabilities["class"]) if probabilities["class"] is not None else None
        confidence_severity = max(probabilities["severity"]) if probabilities["severity"] is not None else None
        reasons = []
        if confidence_class is None:
            reasons.append("missing_class_evidence")
        elif confidence_class < config.get("minimum_class_confidence", .5):
            reasons.append("ambiguous_class")
        if confidence_severity is None:
            reasons.append("missing_severity_evidence")
        elif confidence_severity < config.get("minimum_severity_confidence", .5):
            reasons.append("ambiguous_severity")
        sigma_s, sigma_angle = np.sqrt([max(0, landmark["variance_s"]), max(0, landmark["variance_angle"])])
        if sigma_s > config.get("review_axial_std_meters", .5):
            reasons.append("axial_uncertainty")
        if sigma_angle > config.get("review_angle_std_radians", np.pi / 4):
            reasons.append("clock_uncertainty")
        if len({(source["traversal"], source["frame_id"]) for source in sources}) < config.get("minimum_source_frames", 2):
            reasons.append("single_source_frame")
        half_width = 1.959963984540054 * sigma_angle
        lower = float(np.arctan2(np.sin(landmark["angle"] - half_width), np.cos(landmark["angle"] - half_width)))
        upper = float(np.arctan2(np.sin(landmark["angle"] + half_width), np.cos(landmark["angle"] + half_width)))
        action_probability = None
        if probabilities["class"] is not None and probabilities["severity"] is not None:
            class_action = sum(p for label, p in zip(classes, probabilities["class"]) if label in actionable)
            severity_action = sum(probabilities["severity"][minimum:])
            action_probability = float(class_action * severity_action)
        records.append(dict(
            id=str(case.get("map_id", case.get("id", "pipe"))) + f":L{j:06d}", landmark_index=j,
            axial_meters=float(landmark["s"]), axial_interval95_meters=[float(landmark["s"] - 1.959963984540054 * sigma_s), float(landmark["s"] + 1.959963984540054 * sigma_s)],
            clock_angle_radians=float(landmark["angle"]),
            clock_hours=float((landmark["angle"] % (2 * np.pi)) * 6 / np.pi) or 12.0,
            clock_interval95=dict(lower_radians=lower, upper_radians=upper, wraps=bool(lower > upper), full_circle=bool(half_width >= np.pi)),
            covariance=landmark.get("covariance", np.diag([landmark["variance_s"], landmark["variance_angle"]]).tolist()),
            class_labels=classes, class_probabilities=probabilities["class"], **{"class": class_value},
            severity_levels=severities, severity_probabilities=probabilities["severity"], severity=severity_value,
            source_frames=sources, review_flag=bool(reasons), review_reasons=reasons,
            action_probability=action_probability, requires_action=action_probability >= threshold if action_probability is not None else None,
        ))
    return records


def score_maintenance(case, result, matched):
    """Score work-list membership, record edits and matched ordinal annotations."""
    keys = ("worklist_precision", "worklist_recall", "worklist_true_positives", "worklist_false_positives",
            "worklist_false_negatives", "maintenance_edit_operations", "maintenance_additions",
            "maintenance_deletions", "maintenance_updates", "severity_agreement", "severity_ordinal_mae", "class_agreement")
    report = dict.fromkeys(keys)
    records = result.get("maintenance_records")
    if records is None:
        return report
    classes = labels_for(case, "class_probabilities", "class_labels")
    severities = labels_for(case, "severity_probabilities", "severity_levels")
    if not classes or not severities:
        return report
    actionable, minimum, _ = action_policy(case, classes, severities)
    truth = case["truth"]
    class_pairs, severity_pairs = [], []
    for prediction, reference in matched.items():
        record, target = records[prediction], truth[reference]
        if record["class"] is not None and "class" in target:
            class_pairs.append(record["class"] == target["class"])
        if record["severity"] is not None and target.get("severity") in severities:
            severity_pairs.append((severities.index(record["severity"]), severities.index(target["severity"])))
    if class_pairs:
        report["class_agreement"] = float(np.mean(class_pairs))
    if severity_pairs:
        report["severity_agreement"] = float(np.mean([a == b for a, b in severity_pairs]))
        report["severity_ordinal_mae"] = float(np.mean([abs(a - b) for a, b in severity_pairs]))
    if not all("class" in target and target.get("severity") in severities for target in truth):
        return report
    required = {i for i, target in enumerate(truth) if target["class"] in actionable and severities.index(target["severity"]) >= minimum}
    predicted = {record["landmark_index"] for record in records if record["requires_action"]}
    true_positive = {j for j in predicted if j in matched and matched[j] in required}
    false_positive = predicted - true_positive
    missing = required - {matched[j] for j in true_positive}
    updates = sum(records[j]["class"] != truth[matched[j]]["class"] or records[j]["severity"] != truth[matched[j]]["severity"] for j in true_positive)
    report.update(worklist_precision=len(true_positive) / len(predicted) if predicted else float(not required),
                  worklist_recall=len(true_positive) / len(required) if required else float(not predicted),
                  worklist_true_positives=len(true_positive), worklist_false_positives=len(false_positive),
                  worklist_false_negatives=len(missing), maintenance_edit_operations=len(false_positive) + len(missing) + updates,
                  maintenance_additions=len(missing), maintenance_deletions=len(false_positive), maintenance_updates=updates)
    return report


def export_records(rows, output):
    """Write complete audit JSON and an actionable CSV per stored pipe episode."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        payload = dict(episode_id=row["id"], method=row["method"], records=row["result"]["maintenance_records"],
                       worklist=row["result"]["worklist"], review_list=row["result"]["review_list"])
        stem = f"case-{index:06d}"
        path = output / (stem + ".json")
        if path.exists():
            raise FileExistsError(str(path))
        path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        fields = ["id", "axial_meters", "axial_interval95_meters", "clock_angle_radians", "clock_interval95", "covariance",
                  "class", "class_labels", "class_probabilities", "severity", "severity_levels", "severity_probabilities",
                  "source_frames", "action_probability", "requires_action", "review_flag", "review_reasons"]
        with (output / (stem + "-worklist.csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in payload["worklist"]:
                writer.writerow({key: json.dumps(record[key], ensure_ascii=False) if isinstance(record[key], (list, dict)) else record[key] for key in fields})
