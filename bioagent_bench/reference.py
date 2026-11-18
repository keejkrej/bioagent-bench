"""In-process reference evaluation for BioAgentBench.

The reference evaluator consumes exactly the same generated rows as normal
grading, uses the production submission validator and scorer, and never starts
an agent, gateway call, or subprocess.  It is intended for calibration and
regression testing, not for leaderboard runs.
"""

import json
import math
import statistics
from pathlib import Path

import numpy as np

from .taskbank import (
    TASK_IDS,
    _ALPHABET,
    _additive_predict,
    _adjusted_rand_index,
    _b04_baseline_panel,
    _b04_gene_means,
    _b04_oracle_panel,
    _clamp,
    _cluster_by_cutoff,
    _clusterings_by_cutoff,
    _fit_additive_baseline,
    _generate_task_data,
    _hamming_distance,
    _mean,
    _parse_mutations,
    _rank_normalize,
    _score_task,
    _sigmoid,
    _validate_submission,
)

__all__ = ["CalibrationError", "evaluate", "evaluate_task"]

_TWO_MERS = [first + second for first in _ALPHABET for second in _ALPHABET]
_TWO_MER_INDEX = {value: index for index, value in enumerate(_TWO_MERS)}


class CalibrationError(RuntimeError):
    """Raised when a reference calibration invariant fails."""


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _ridge_fit(features, targets, alpha=1.0):
    matrix = np.asarray(features, dtype=float)
    values = np.asarray(targets, dtype=float)
    count, width = matrix.shape
    design = np.hstack([np.ones((count, 1)), matrix])
    regularizer = np.eye(width + 1) * alpha
    regularizer[0, 0] = 0.0
    left = np.dot(design.T, design) + regularizer
    right = np.dot(design.T, values)
    try:
        return np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(left, right, rcond=None)[0]


def _ridge_fit_no_intercept(features, targets, alpha=1.0):
    matrix = np.asarray(features, dtype=float)
    values = np.asarray(targets, dtype=float)
    left = np.dot(matrix.T, matrix) + np.eye(matrix.shape[1]) * alpha
    right = np.dot(matrix.T, values)
    try:
        return np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(left, right, rcond=None)[0]


def _ridge_predict(weights, features):
    matrix = np.asarray(features, dtype=float)
    design = np.hstack([np.ones((matrix.shape[0], 1)), matrix])
    return np.dot(design, weights)


def _standardize(train_matrix, other_matrices):
    train_matrix = np.asarray(train_matrix, dtype=float)
    mean = train_matrix.mean(axis=0)
    scale = train_matrix.std(axis=0)
    scale[scale < 1e-9] = 1.0
    standardized = (train_matrix - mean) / scale
    others = [(np.asarray(matrix, dtype=float) - mean) / scale for matrix in other_matrices]
    return standardized, others


def _logistic_fit(features, targets, alpha=1e-3, iterations=800, learning_rate=0.5):
    matrix = np.asarray(features, dtype=float)
    values = np.asarray(targets, dtype=float)
    count, width = matrix.shape
    design = np.hstack([np.ones((count, 1)), matrix])
    weights = np.zeros(width + 1)
    for _ in range(iterations):
        logits = np.clip(np.dot(design, weights), -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = np.dot(design.T, probabilities - values) / float(count)
        gradient[1:] += alpha * weights[1:]
        weights -= learning_rate * gradient
    return weights


def _logistic_predict(weights, features):
    matrix = np.asarray(features, dtype=float)
    design = np.hstack([np.ones((matrix.shape[0], 1)), matrix])
    logits = np.clip(np.dot(design, weights), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


# ---------------------------------------------------------------------------
# B01: regularized position / k-mer regression
# ---------------------------------------------------------------------------


def _b01_features(sequences):
    length = len(sequences[0])
    rows = []
    for sequence in sequences:
        vector = [0.0] * (length * 20 + len(_TWO_MERS))
        for position, residue in enumerate(sequence):
            if residue in _ALPHABET:
                vector[position * 20 + _ALPHABET.index(residue)] = 1.0
        for index in range(length - 1):
            key = sequence[index:index + 2]
            two_mer = _TWO_MER_INDEX.get(key)
            if two_mer is not None:
                vector[length * 20 + two_mer] += 1.0 / float(max(1, length - 1))
        rows.append(vector)
    return rows


def _b01_baseline(data):
    mean = _mean([float(row["binding"]) for row in data["train_rows"]])
    return {
        "predictions": [
            {"id": row["id"], "value": mean}
            for row in data["private_input_rows"]
        ]
    }


def _b01_simple(data):
    return _b01_baseline(data)


def _b01_reference(data):
    train_rows = data["train_rows"]
    input_rows = data["private_input_rows"]
    train_features = _b01_features([row["sequence"] for row in train_rows])
    input_features = _b01_features([row["sequence"] for row in input_rows])
    train_matrix, (input_matrix,) = _standardize(train_features, [input_features])
    weights = _ridge_fit(train_matrix, [float(row["binding"]) for row in train_rows], alpha=512.0)
    predictions = np.clip(_ridge_predict(weights, input_matrix), 0.0, 1.0)
    return {
        "predictions": [
            {"id": row["id"], "value": float(value)}
            for row, value in zip(input_rows, predictions)
        ]
    }


def _b01_oracle(data):
    truth = {row["id"]: float(row["binding"]) for row in data["truth_rows"]}
    return {
        "predictions": [
            {"id": row["id"], "value": truth[row["id"]]}
            for row in data["private_input_rows"]
        ]
    }


# ---------------------------------------------------------------------------
# B02: additive plus position-pair ridge
# ---------------------------------------------------------------------------


def _b02_vocabulary(train_rows):
    mutations = {}
    pairs = {}
    for row in train_rows:
        parsed = _parse_mutations(row.get("mutations"))
        positions = []
        for mutation in parsed:
            if mutation not in mutations:
                mutations[mutation] = len(mutations)
            digits = "".join(character for character in mutation if character.isdigit())
            if digits:
                positions.append(int(digits))
        if len(positions) >= 2:
            key = tuple(sorted(positions[:2]))
            if key not in pairs:
                pairs[key] = len(pairs)
    return mutations, pairs


def _b02_features(rows, mutations, pairs):
    rows_features = []
    for row in rows:
        vector = [0.0] * (len(mutations) + len(pairs))
        positions = []
        for mutation in _parse_mutations(row.get("mutations")):
            index = mutations.get(mutation)
            if index is not None:
                vector[index] = 1.0
            digits = "".join(character for character in mutation if character.isdigit())
            if digits:
                positions.append(int(digits))
        if len(positions) >= 2:
            key = tuple(sorted(positions[:2]))
            index = pairs.get(key)
            if index is not None:
                vector[len(mutations) + index] = 1.0
        rows_features.append(vector)
    return rows_features


def _b02_baseline(data):
    vocabulary, coefficients = _fit_additive_baseline(data["train_rows"])
    return {
        "predictions": [
            {
                "id": row["id"],
                "value": _clamp(_additive_predict(vocabulary, coefficients, row["mutations"]), 0.0, 1.0),
            }
            for row in data["private_input_rows"]
        ]
    }


def _b02_simple(data):
    mean = _mean([float(row["fitness"]) for row in data["train_rows"]])
    return {
        "predictions": [
            {"id": row["id"], "value": mean}
            for row in data["private_input_rows"]
        ]
    }


def _b02_reference(data):
    train_rows = data["train_rows"]
    input_rows = data["private_input_rows"]
    mutations, pairs = _b02_vocabulary(train_rows)
    train_features = _b02_features(train_rows, mutations, pairs)
    input_features = _b02_features(input_rows, mutations, pairs)
    train_matrix, (input_matrix,) = _standardize(train_features, [input_features])
    weights = _ridge_fit(train_matrix, [float(row["fitness"]) for row in train_rows], alpha=1e-3)
    predictions = np.clip(_ridge_predict(weights, input_matrix), 0.0, 1.0)
    return {
        "predictions": [
            {"id": row["id"], "value": float(value)}
            for row, value in zip(input_rows, predictions)
        ]
    }


def _b02_oracle(data):
    truth = {row["id"]: float(row["fitness"]) for row in data["truth_rows"]}
    return {
        "predictions": [
            {"id": row["id"], "value": truth[row["id"]]}
            for row in data["private_input_rows"]
        ]
    }


# ---------------------------------------------------------------------------
# B03: lineage nearest-neighbour plus within-lineage two-medoids
# ---------------------------------------------------------------------------


def _b03_centroids(train_rows):
    by_parent = {}
    for row in train_rows:
        by_parent.setdefault(row["family"], []).append(row["sequence"])
    centroids = {}
    for parent, sequences in by_parent.items():
        length = len(sequences[0])
        centroid = []
        for position in range(length):
            counts = {}
            for sequence in sequences:
                residue = sequence[position]
                counts[residue] = counts.get(residue, 0) + 1
            best = min(counts, key=lambda residue: (-counts[residue], residue))
            centroid.append(best)
        centroids[parent] = "".join(centroid)
    return centroids


def _b03_assign_parents(input_rows, centroids):
    parents = sorted(centroids)
    assignments = []
    for row in input_rows:
        sequence = row["sequence"]
        best = min(
            parents,
            key=lambda parent: (_hamming_distance(sequence, centroids[parent]), parent),
        )
        assignments.append(best)
    return assignments


def _two_medoids(sequences, ids):
    count = len(sequences)
    if count <= 2:
        return [0] * count
    distances = [[0] * count for _ in range(count)]
    for first in range(count):
        for second in range(first + 1, count):
            distance = _hamming_distance(sequences[first], sequences[second])
            distances[first][second] = distance
            distances[second][first] = distance

    best_pair = None
    best_key = None
    for first in range(count):
        for second in range(first + 1, count):
            key = (-distances[first][second], ids[first], ids[second])
            if best_key is None or key < best_key:
                best_key = key
                best_pair = (first, second)
    medoids = [best_pair[0], best_pair[1]]
    labels = [0] * count
    for _ in range(20):
        new_labels = []
        for index in range(count):
            first_distance = distances[index][medoids[0]]
            second_distance = distances[index][medoids[1]]
            new_labels.append(0 if first_distance <= second_distance else 1)
        new_medoids = []
        for cluster in (0, 1):
            members = [index for index in range(count) if new_labels[index] == cluster]
            if not members:
                new_medoids.append(medoids[cluster])
                continue
            best_member = min(
                members,
                key=lambda index: (sum(distances[index][other] for other in members), ids[index]),
            )
            new_medoids.append(best_member)
        if new_labels == labels and new_medoids == medoids:
            labels = new_labels
            break
        labels = new_labels
        medoids = new_medoids
    return labels


def _b03_reference(data):
    input_rows = data["private_input_rows"]
    centroids = _b03_centroids(data["train_rows"])
    assignments = _b03_assign_parents(input_rows, centroids)
    labels = [None] * len(input_rows)
    by_parent = {}
    for index, parent in enumerate(assignments):
        by_parent.setdefault(parent, []).append(index)
    for parent, indices in by_parent.items():
        sequences = [input_rows[index]["sequence"] for index in indices]
        ids = [input_rows[index]["id"] for index in indices]
        cluster_labels = _two_medoids(sequences, ids)
        for index, cluster in zip(indices, cluster_labels):
            labels[index] = "L%s_%d" % (parent, cluster)
    return {
        "predictions": [
            {"id": row["id"], "value": labels[index]}
            for index, row in enumerate(input_rows)
        ]
    }


def _b03_baseline(data):
    input_rows = data["private_input_rows"]
    truth = {row["id"]: row["family"] for row in data["truth_rows"]}
    true_labels = [truth[row["id"]] for row in input_rows]
    sequences = [row["sequence"] for row in input_rows]
    best_ari = -1.0
    best_labels = None
    for labels in _clusterings_by_cutoff(sequences):
        ari = _adjusted_rand_index(true_labels, labels)
        if ari > best_ari:
            best_ari = ari
            best_labels = labels
    return {
        "predictions": [
            {"id": row["id"], "value": "c%d" % best_labels[index]}
            for index, row in enumerate(input_rows)
        ]
    }


def _b03_simple(data):
    input_rows = data["private_input_rows"]
    sequences = [row["sequence"] for row in input_rows]
    labels = _cluster_by_cutoff(sequences, 5)
    return {
        "predictions": [
            {"id": row["id"], "value": "c%d" % labels[index]}
            for index, row in enumerate(input_rows)
        ]
    }


def _b03_oracle(data):
    truth = {row["id"]: row["family"] for row in data["truth_rows"]}
    return {
        "predictions": [
            {"id": row["id"], "value": truth[row["id"]]}
            for row in data["private_input_rows"]
        ]
    }


def _b03_diagnostics(data):
    input_rows = data["private_input_rows"]
    truth = {row["id"]: row["family"] for row in data["truth_rows"]}
    true_labels = [truth[row["id"]] for row in input_rows]
    centroids = _b03_centroids(data["train_rows"])
    assignments = _b03_assign_parents(input_rows, centroids)
    nearest_parent_ari = _adjusted_rand_index(true_labels, assignments)

    sequences = [row["sequence"] for row in input_rows]
    best_ari = -1.0
    best_cutoff = None
    for cutoff, labels in enumerate(_clusterings_by_cutoff(sequences), start=1):
        ari = _adjusted_rand_index(true_labels, labels)
        if ari > best_ari:
            best_ari = ari
            best_cutoff = cutoff

    reference = _b03_reference(data)
    reference_labels = [item["value"] for item in reference["predictions"]]
    reference_ari = _adjusted_rand_index(true_labels, reference_labels)
    return {
        "nearest_parent_ari": round(nearest_parent_ari, 6),
        "best_cutoff_ari": round(best_ari, 6),
        "best_cutoff": best_cutoff,
        "two_medoids_ari": round(reference_ari, 6),
        "n_daughters": len(set(true_labels)),
    }


# ---------------------------------------------------------------------------
# B04: regularized gene / covariate model plus constrained panel
# ---------------------------------------------------------------------------


def _b04_reference(data):
    train_rows = data["train_rows"]
    input_rows = data["private_input_rows"]
    genes = sorted(set(row["gene"] for row in train_rows) | set(row["gene"] for row in input_rows))
    tissues = sorted(set(row["tissue"] for row in train_rows) | set(row["tissue"] for row in input_rows))
    gene_index = {gene: index for index, gene in enumerate(genes)}
    tissue_index = {tissue: index for index, tissue in enumerate(tissues)}

    gene_counts = {}
    gene_missing = {}
    for row in train_rows:
        gene = row["gene"]
        gene_counts[gene] = gene_counts.get(gene, 0) + 1
        if not (row.get("dependency") or "").strip():
            gene_missing[gene] = gene_missing.get(gene, 0) + 1
    gene_missing_rate = {
        gene: gene_missing.get(gene, 0) / float(gene_counts[gene])
        for gene in gene_counts
    }

    def features(rows):
        result = []
        for row in rows:
            vector = [0.0] * (len(genes) + len(tissues) + 5)
            vector[gene_index[row["gene"]]] = 1.0
            vector[len(genes) + tissue_index[row["tissue"]]] = 1.0
            offset = len(genes) + len(tissues)
            vector[offset] = float(row["expression"])
            vector[offset + 1] = float(row["mutation_count"]) / 20.0
            vector[offset + 2] = float(row["copy_number"])
            vector[offset + 3] = float(row["lineage_score"])
            vector[offset + 4] = gene_missing_rate.get(row["gene"], 0.0)
            result.append(vector)
        return result

    labeled = [row for row in train_rows if (row.get("dependency") or "").strip()]
    train_matrix, (input_matrix,) = _standardize(
        features(labeled),
        [features(input_rows)],
    )
    logit_targets = []
    for row in labeled:
        value = min(max(float(row["dependency"]), 1e-6), 1.0 - 1e-6)
        logit_targets.append(math.log(value / (1.0 - value)))
    weights = _ridge_fit(train_matrix, logit_targets, alpha=100.0)
    logits = np.clip(_ridge_predict(weights, input_matrix), -30.0, 30.0)
    predictions = np.clip(1.0 / (1.0 + np.exp(-logits)), 0.0, 1.0)

    by_gene = {}
    for row, prediction in zip(input_rows, predictions):
        by_gene.setdefault(row["gene"], []).append((row["id"], float(prediction)))
    candidates = []
    for gene, items in by_gene.items():
        items.sort(key=lambda item: (-item[1], item[0]))
        candidates.extend(items[:2])
    candidates.sort(key=lambda item: (-item[1], item[0]))
    shortlist = [item[0] for item in candidates[:20]]
    return {
        "predictions": [
            {"id": row["id"], "value": float(value)}
            for row, value in zip(input_rows, predictions)
        ],
        "shortlist": shortlist,
    }


def _b04_baseline(data):
    train_rows = data["train_rows"]
    input_rows = data["private_input_rows"]
    gene_mean, global_mean = _b04_gene_means(train_rows)
    return {
        "predictions": [
            {"id": row["id"], "value": gene_mean.get(row["gene"], global_mean)}
            for row in input_rows
        ],
        "shortlist": _b04_baseline_panel(input_rows, train_rows),
    }


def _b04_simple(data):
    return _b04_baseline(data)


def _b04_oracle(data):
    truth = {row["id"]: float(row["dependency"]) for row in data["truth_rows"]}
    return {
        "predictions": [
            {"id": row["id"], "value": truth[row["id"]]}
            for row in data["private_input_rows"]
        ],
        "shortlist": _b04_oracle_panel(data["private_input_rows"], data["truth_rows"]),
    }


# ---------------------------------------------------------------------------
# B05: target-relative pairwise ranker and raw global pointwise simple model
# ---------------------------------------------------------------------------


def _b05_normalized_features(rows):
    by_target = {}
    for row in rows:
        by_target.setdefault(row["target"], []).append(row)
    features = {}
    for target, target_rows in by_target.items():
        shape = _rank_normalize([float(row["shape_complementarity"]) for row in target_rows])
        hbond = _rank_normalize([float(row["hbond_count"]) for row in target_rows])
        buried = _rank_normalize([float(row["buried_sasa"]) for row in target_rows])
        clash = _rank_normalize([float(row["clash_count"]) for row in target_rows])
        energy = _rank_normalize([-float(row["energy"]) for row in target_rows])
        for index, row in enumerate(target_rows):
            features[row["id"]] = [
                shape[index],
                hbond[index],
                buried[index],
                clash[index],
                energy[index],
                shape[index] * buried[index],
                hbond[index] * (1.0 - clash[index]),
            ]
    return features


def _b05_reference(data):
    train_rows = data["train_rows"]
    input_rows = data["private_input_rows"]
    train_features = _b05_normalized_features(train_rows)
    input_features = _b05_normalized_features(input_rows)

    by_type = {}
    for row in train_rows:
        by_type.setdefault(row["target_type"], []).append(row)
    type_weights = {}
    for target_type, rows in by_type.items():
        by_target = {}
        for row in rows:
            by_target.setdefault(row["target"], []).append(row)
        differences = []
        targets = []
        for target_rows in by_target.values():
            for first in range(len(target_rows)):
                for second in range(len(target_rows)):
                    if first == second:
                        continue
                    first_features = train_features[target_rows[first]["id"]]
                    second_features = train_features[target_rows[second]["id"]]
                    differences.append([
                        left - right
                        for left, right in zip(first_features, second_features)
                    ])
                    targets.append(
                        float(target_rows[first]["quality"])
                        - float(target_rows[second]["quality"])
                    )
        type_weights[target_type] = _ridge_fit_no_intercept(differences, targets, alpha=100.0)

    fallback = next(iter(type_weights.values()))
    scores = {}
    for row in input_rows:
        weights = type_weights.get(row["target_type"], fallback)
        scores[row["id"]] = float(np.dot(weights, input_features[row["id"]]))

    by_target = {}
    for row in input_rows:
        by_target.setdefault(row["target"], []).append(row)
    shortlist = []
    for target_rows in by_target.values():
        best = min(target_rows, key=lambda row: (-scores[row["id"]], row["id"]))
        shortlist.append(best["id"])
    return {
        "predictions": [
            {"id": row["id"], "value": _sigmoid(scores[row["id"]])}
            for row in input_rows
        ],
        "shortlist": shortlist,
    }


def _b05_simple(data):
    train_rows = data["train_rows"]
    input_rows = data["private_input_rows"]

    def features(rows):
        return [
            [
                float(row["energy"]),
                float(row["shape_complementarity"]),
                float(row["hbond_count"]),
                float(row["buried_sasa"]),
                float(row["clash_count"]),
            ]
            for row in rows
        ]

    train_matrix, (input_matrix,) = _standardize(
        features(train_rows),
        [features(input_rows)],
    )
    weights = _ridge_fit(train_matrix, [float(row["quality"]) for row in train_rows], alpha=1.0)
    predictions = np.clip(_ridge_predict(weights, input_matrix), 0.0, 1.0)

    by_target = {}
    for row, prediction in zip(input_rows, predictions):
        by_target.setdefault(row["target"], []).append((row["id"], float(prediction)))
    shortlist = []
    for items in by_target.values():
        items.sort(key=lambda item: (-item[1], item[0]))
        shortlist.append(items[0][0])
    return {
        "predictions": [
            {"id": row["id"], "value": float(value)}
            for row, value in zip(input_rows, predictions)
        ],
        "shortlist": shortlist,
    }


def _b05_baseline(data):
    input_rows = data["private_input_rows"]
    by_target = {}
    for row in input_rows:
        by_target.setdefault(row["target"], []).append(row)
    shortlist = []
    for target_rows in by_target.values():
        best = min(target_rows, key=lambda row: (float(row["energy"]), row["id"]))
        shortlist.append(best["id"])
    return {
        "predictions": [{"id": row["id"], "value": 0.5} for row in input_rows],
        "shortlist": shortlist,
    }


def _b05_oracle(data):
    truth = {row["id"]: float(row["quality"]) for row in data["truth_rows"]}
    input_rows = data["private_input_rows"]
    by_target = {}
    for row in input_rows:
        by_target.setdefault(row["target"], []).append(row)
    shortlist = []
    for target_rows in by_target.values():
        best = min(target_rows, key=lambda row: (-truth[row["id"]], row["id"]))
        shortlist.append(best["id"])
    return {
        "predictions": [
            {"id": row["id"], "value": truth[row["id"]]}
            for row in input_rows
        ],
        "shortlist": shortlist,
    }


def _b05_diagnostics(data):
    truth = {row["id"]: float(row["quality"]) for row in data["truth_rows"]}
    by_target = {}
    for row in data["private_input_rows"]:
        by_target.setdefault(row["target"], []).append(row)
    gaps = []
    for target_rows in by_target.values():
        baseline = min(target_rows, key=lambda row: (float(row["energy"]), row["id"]))
        oracle = max(truth[row["id"]] for row in target_rows)
        gaps.append(oracle - truth[baseline["id"]])
    return {
        "min_gap": round(min(gaps), 6),
        "mean_gap": round(_mean(gaps), 6),
        "degenerate_targets": sum(1 for gap in gaps if gap <= 1e-9),
        "n_targets": len(gaps),
    }


# ---------------------------------------------------------------------------
# B06: target-group-CV selection between logistic and quadratic logistic
# ---------------------------------------------------------------------------


def _b06_features(rows):
    result = []
    for row in rows:
        charge = float(row["net_charge"])
        interface = float(row["interface_area"])
        result.append([
            interface,
            float(row["hydrophobicity"]),
            charge,
            abs(charge - 1.0),
            interface * float(row["hydrophobicity"]),
        ])
    return result


def _b06_quadratic_features(rows):
    """Add squared and pairwise interaction terms to the linear features."""
    result = []
    for vector in _b06_features(rows):
        squares = [value * value for value in vector]
        interactions = [
            vector[0] * vector[1],
            vector[0] * vector[2],
            vector[0] * vector[3],
            vector[1] * vector[2],
            vector[1] * vector[3],
            vector[2] * vector[3],
        ]
        result.append(vector + squares + interactions)
    return result


def _b06_fit_predict(train_rows, input_rows, feature_fn, alpha, iterations):
    train_matrix, (input_matrix,) = _standardize(
        feature_fn(train_rows),
        [feature_fn(input_rows)],
    )
    weights = _logistic_fit(
        train_matrix,
        [float(row["binder"]) for row in train_rows],
        alpha=alpha,
        iterations=iterations,
        learning_rate=0.5,
    )
    return _logistic_predict(weights, input_matrix)


def _b06_log_loss(labels, probabilities):
    total = 0.0
    for label, probability in zip(labels, probabilities):
        probability = min(max(float(probability), 1e-9), 1.0 - 1e-9)
        total -= label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability)
    return total / float(len(labels)) if labels else 0.0


_B06_CANDIDATES = {
    "logistic": (_b06_features, 1e-3, 2000),
    "quadratic": (_b06_quadratic_features, 1e-2, 3000),
}


def _b06_select_model(train_rows):
    """Select logistic vs quadratic logistic by leave-one-target-out CV.

    Only training targets are used.  The held-out private rows and their truth
    are never consulted, so this is not private-score tuning.
    """
    targets = sorted(set(row["target"] for row in train_rows))
    losses = {name: [] for name in _B06_CANDIDATES}
    for target in targets:
        train_fold = [row for row in train_rows if row["target"] != target]
        test_fold = [row for row in train_rows if row["target"] == target]
        labels = [float(row["binder"]) for row in test_fold]
        for name, (feature_fn, alpha, iterations) in _B06_CANDIDATES.items():
            probabilities = _b06_fit_predict(
                train_fold, test_fold, feature_fn, alpha, iterations
            )
            losses[name].append(_b06_log_loss(labels, probabilities))
    means = {name: _mean(values) for name, values in losses.items()}
    best = min(means, key=lambda name: (means[name], name))
    return best, means


def _b06_reference(data):
    train_rows = data["train_rows"]
    input_rows = data["private_input_rows"]
    selected, _ = _b06_select_model(train_rows)
    feature_fn, alpha, iterations = _B06_CANDIDATES[selected]
    probabilities = _b06_fit_predict(
        train_rows, input_rows, feature_fn, alpha, iterations
    )

    by_target = {}
    for row, probability in zip(input_rows, probabilities):
        by_target.setdefault(row["target"], []).append((row["id"], float(probability)))
    shortlist = []
    for items in by_target.values():
        items.sort(key=lambda item: (-item[1], item[0]))
        shortlist.extend(item[0] for item in items[:10])
    return {
        "predictions": [
            {"id": row["id"], "value": float(value)}
            for row, value in zip(input_rows, probabilities)
        ],
        "shortlist": shortlist,
    }


def _b06_diagnostics(data):
    selected, means = _b06_select_model(data["train_rows"])
    return {
        "selected_model": selected,
        "cv_log_loss": {name: round(value, 6) for name, value in means.items()},
    }


def _b06_baseline(data):
    input_rows = data["private_input_rows"]
    by_target = {}
    for row in input_rows:
        by_target.setdefault(row["target"], []).append(row)
    shortlist = []
    for target_rows in by_target.values():
        ordered = sorted(target_rows, key=lambda row: (float(row["scaffold_rmsd"]), row["id"]))
        shortlist.extend(row["id"] for row in ordered[:10])
    return {
        "predictions": [{"id": row["id"], "value": 0.5} for row in input_rows],
        "shortlist": shortlist,
    }


def _b06_simple(data):
    input_rows = data["private_input_rows"]
    by_target = {}
    for row in input_rows:
        score = float(row["interface_area"]) - 0.1 * abs(float(row["net_charge"]) - 1.0)
        by_target.setdefault(row["target"], []).append((row["id"], score))
    shortlist = []
    for items in by_target.values():
        items.sort(key=lambda item: (-item[1], item[0]))
        shortlist.extend(item[0] for item in items[:10])
    return {
        "predictions": [{"id": row["id"], "value": 0.5} for row in input_rows],
        "shortlist": shortlist,
    }


def _b06_oracle(data):
    truth = {row["id"]: int(float(row["binder"])) for row in data["truth_rows"]}
    input_rows = data["private_input_rows"]
    by_target = {}
    for row in input_rows:
        by_target.setdefault(row["target"], []).append(row)
    shortlist = []
    for target_rows in by_target.values():
        ordered = sorted(target_rows, key=lambda row: (-truth[row["id"]], row["id"]))
        shortlist.extend(row["id"] for row in ordered[:10])
    return {
        "predictions": [
            {"id": row["id"], "value": float(truth[row["id"]])}
            for row in input_rows
        ],
        "shortlist": shortlist,
    }


# ---------------------------------------------------------------------------
# Dispatch and aggregation
# ---------------------------------------------------------------------------


_METHODS = {
    "B01": {
        "baseline": _b01_baseline,
        "simple": _b01_simple,
        "reference": _b01_reference,
        "oracle": _b01_oracle,
    },
    "B02": {
        "baseline": _b02_baseline,
        "simple": _b02_simple,
        "reference": _b02_reference,
        "oracle": _b02_oracle,
    },
    "B03": {
        "baseline": _b03_baseline,
        "simple": _b03_simple,
        "reference": _b03_reference,
        "oracle": _b03_oracle,
    },
    "B04": {
        "baseline": _b04_baseline,
        "simple": _b04_simple,
        "reference": _b04_reference,
        "oracle": _b04_oracle,
    },
    "B05": {
        "baseline": _b05_baseline,
        "simple": _b05_simple,
        "reference": _b05_reference,
        "oracle": _b05_oracle,
    },
    "B06": {
        "baseline": _b06_baseline,
        "simple": _b06_simple,
        "reference": _b06_reference,
        "oracle": _b06_oracle,
    },
}


def _build_output(task_id, method, data):
    return _METHODS[task_id][method](data)


def _calibration_check(methods):
    """Check method validity and the baseline=0 / oracle=100 invariants."""
    errors = []
    for method in ("baseline", "simple", "reference", "oracle"):
        result = methods.get(method)
        if not result:
            errors.append("%s method is missing" % method)
            continue
        if not result.get("valid"):
            errors.append(
                "%s method is invalid: %s"
                % (method, result.get("reason", "unknown reason"))
            )
    baseline = methods.get("baseline", {})
    oracle = methods.get("oracle", {})
    if baseline.get("valid") and abs(float(baseline.get("score", 0.0))) > 1e-9:
        errors.append("baseline score is %r, expected 0" % baseline.get("score"))
    if oracle.get("valid") and abs(float(oracle.get("score", 0.0)) - 100.0) > 1e-9:
        errors.append("oracle score is %r, expected 100" % oracle.get("score"))
    for method, result in methods.items():
        metrics = result.get("metrics", {}) if result else {}
        if metrics.get("calibration_error"):
            errors.append("%s reports calibration_error" % method)
        if metrics.get("panel_calibration_error"):
            errors.append("%s reports panel_calibration_error" % method)
    return {"valid": not errors, "errors": errors}


def evaluate_task(task_id, seed):
    """Evaluate all four reference methods for one task and seed.

    The returned record includes a ``calibration`` block.  A seed is calibrated
    only when every method is valid, the baseline scores exactly 0, the oracle
    scores exactly 100, and no scorer reports a degenerate calibration error.
    """
    data = _generate_task_data(task_id, seed)
    input_rows = data["private_input_rows"]
    truth_rows = data["truth_rows"]
    train_rows = data["train_rows"]

    methods = {}
    for method in ("baseline", "simple", "reference", "oracle"):
        obj = _build_output(task_id, method, data)
        valid, reason, prediction_map, shortlist = _validate_submission(
            task_id, obj, input_rows, truth_rows
        )
        if not valid:
            methods[method] = {
                "valid": False,
                "reason": reason,
                "score": 0.0,
                "metrics": {},
            }
            continue
        score, metrics = _score_task(
            task_id,
            prediction_map,
            shortlist,
            input_rows,
            truth_rows,
            train_rows,
        )
        methods[method] = {
            "valid": True,
            "score": round(float(score), 6),
            "metrics": metrics,
        }

    diagnostics = {}
    if task_id == "B03":
        diagnostics = _b03_diagnostics(data)
    elif task_id == "B05":
        diagnostics = _b05_diagnostics(data)
    elif task_id == "B06":
        diagnostics = _b06_diagnostics(data)

    return {
        "task_id": task_id,
        "seed": seed,
        "methods": methods,
        "diagnostics": diagnostics,
        "calibration": _calibration_check(methods),
    }


def evaluate(seeds, task_ids=None):
    """Evaluate reference methods across seeds and tasks.

    ``seeds`` is an iterable of integers.  ``task_ids`` defaults to all tasks.
    The return value contains raw per-seed records plus per-task/method
    median/min/max summaries and a macro summary per method.
    """
    seeds = [int(seed) for seed in seeds]
    task_ids = list(task_ids or TASK_IDS)
    unknown = set(task_ids) - set(TASK_IDS)
    if unknown:
        raise ValueError("unknown task ids: " + ", ".join(sorted(unknown)))

    records = []
    for seed in seeds:
        for task_id in task_ids:
            record = evaluate_task(task_id, seed)
            if not record["calibration"]["valid"]:
                raise CalibrationError(
                    "calibration failed for task %s seed %s: %s"
                    % (
                        task_id,
                        seed,
                        "; ".join(record["calibration"]["errors"]),
                    )
                )
            records.append(record)

    tasks = {}
    for task_id in task_ids:
        task_records = [record for record in records if record["task_id"] == task_id]
        tasks[task_id] = {}
        for method in ("baseline", "simple", "reference", "oracle"):
            scores = [record["methods"][method]["score"] for record in task_records]
            tasks[task_id][method] = {
                "scores": scores,
                "median": statistics.median(scores),
                "min": min(scores),
                "max": max(scores),
            }

    macro = {}
    for method in ("baseline", "simple", "reference", "oracle"):
        per_seed = []
        for seed in seeds:
            seed_scores = [
                record["methods"][method]["score"]
                for record in records
                if record["seed"] == seed
            ]
            per_seed.append(_mean(seed_scores))
        macro[method] = {
            "per_seed": per_seed,
            "median": statistics.median(per_seed),
            "min": min(per_seed),
            "max": max(per_seed),
        }

    return {
        "seeds": seeds,
        "tasks": tasks,
        "macro": macro,
        "records": records,
    }


def write_report(result, path):
    """Write a reference report as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
