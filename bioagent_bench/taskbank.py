"""Synthetic task bank and grader for BioAgentBench.

The module is intentionally self-contained and uses only the Python standard
library.  It generates six deterministic synthetic computational-biology tasks
and grades agent submissions against sealed held-out labels.

Public API
----------
TASK_IDS
    Tuple of task ids ``("B01", ..., "B06")``.
generate_bank(root, seed=2026)
    Create ``root/tasks/BXX/...`` for every task.
grade_task(task_dir, submission_dir, grade_dir, timeout_seconds=300)
    Run and grade a submission, returning a JSON-serialisable report.
"""

import csv
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

__all__ = [
    "BANK_VERSION",
    "BankMetadataError",
    "TASK_IDS",
    "generate_bank",
    "grade_task",
    "read_bank_metadata",
    "validate_bank",
]

TASK_IDS = ("B01", "B02", "B03", "B04", "B05", "B06")

# Single source of truth for the generated-bank format.  Bump this whenever the
# on-disk task schema or generation semantics change so stale banks are rejected
# instead of silently reused.
BANK_VERSION = "0.2.0"
BANK_METADATA_FILENAME = "bank.json"

_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


class BankMetadataError(RuntimeError):
    """Raised when a generated bank is missing or incompatible."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def _sigmoid(value):
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _rank_normalize(values):
    """Return average ranks scaled to [0, 1], with deterministic tie handling."""
    count = len(values)
    if count <= 1:
        return [0.0] * count
    order = sorted(range(count), key=lambda index: (values[index], index))
    ranks = [0.0] * count
    start = 0
    while start < count:
        end = start
        while end + 1 < count and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = 0.5 * (start + end)
        for position in range(start, end + 1):
            ranks[order[position]] = average / float(count - 1)
        start = end + 1
    return ranks


def _rmse(predicted, truth):
    if not predicted:
        return 0.0
    total = 0.0
    for p, t in zip(predicted, truth):
        diff = p - t
        total += diff * diff
    return math.sqrt(total / float(len(predicted)))


def _write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _tail(data, limit=2000):
    if data is None:
        return ""
    if isinstance(data, bytes):
        text = data.decode("utf-8", errors="replace")
    else:
        text = str(data)
    if len(text) > limit:
        return text[-limit:]
    return text


# ---------------------------------------------------------------------------
# B01: antibody variant binding
# ---------------------------------------------------------------------------


def _gen_b01(rng):
    length = 24
    motifs = (("CDE", 2), ("HIK", 9), ("PQR", 16), ("WYF", 20))
    train_families = ["F%02d" % i for i in range(1, 7)]
    test_families = ["F%02d" % i for i in range(7, 10)]

    family_params = {}
    for family in train_families + test_families:
        family_params[family] = {
            "offset": rng.uniform(-0.12, 0.12),
            "motif1_interaction": rng.uniform(-0.10, 0.10),
        }

    def make_sequence():
        sequence = [rng.choice(_ALPHABET) for _ in range(length)]
        for motif, position in motifs:
            if rng.random() < 0.5:
                for offset, residue in enumerate(motif):
                    sequence[position + offset] = residue
        return "".join(sequence)

    def make_binding(sequence, family):
        present = []
        for motif, position in motifs:
            present.append(1.0 if sequence[position:position + len(motif)] == motif else 0.0)
        params = family_params[family]
        value = 0.45
        value += 0.20 * present[0] + 0.14 * present[1]
        value -= 0.16 * present[2] + 0.10 * present[3]
        value += params["offset"] + params["motif1_interaction"] * present[0]
        value += 0.05 * present[0] * present[1] - 0.04 * present[2] * present[3]
        value += rng.gauss(0.0, 0.05)
        return _clamp(value, 0.0, 1.0)

    train_rows = []
    counter = 0
    for family in train_families:
        for _ in range(100):
            counter += 1
            sequence = make_sequence()
            train_rows.append({
                "id": "ab%05d" % counter,
                "sequence": sequence,
                "family": family,
                "binding": "%.6f" % make_binding(sequence, family),
            })

    public_rows = []
    for family in test_families:
        for _ in range(60):
            counter += 1
            sequence = make_sequence()
            public_rows.append({
                "id": "ab%05d" % counter,
                "sequence": sequence,
                "family": family,
            })

    private_rows = []
    truth_rows = []
    for family in test_families:
        for _ in range(250):
            counter += 1
            sequence = make_sequence()
            row_id = "ab%05d" % counter
            private_rows.append({
                "id": row_id,
                "sequence": sequence,
                "family": family,
            })
            truth_rows.append({
                "id": row_id,
                "binding": "%.6f" % make_binding(sequence, family),
            })

    return {
        "title": "Antibody variant binding",
        "category": "protein_engineering",
        "description": "Predict continuous binding scores for antibody sequence variants from held-out families.",
        "batch_invariant": True,
        "train_fields": ["id", "sequence", "family", "binding"],
        "train_rows": train_rows,
        "input_fields": ["id", "sequence", "family"],
        "public_input_rows": public_rows,
        "private_input_rows": private_rows,
        "truth_fields": ["id", "binding"],
        "truth_rows": truth_rows,
        "brief": _brief_b01(),
    }


def _brief_b01():
    return """# B01 - Antibody variant binding

**Category:** protein_engineering
**Batch-invariance check:** yes

## Goal
Predict a continuous binding score in [0, 1] for antibody sequence variants.
Training data covers six antibody families. The graded input contains variants
from held-out families that do not appear in training.

## Files
You develop against the public copies:

- `data/train.csv` - labeled training rows.
- `data/input.csv` - unlabeled development rows.

At grading time your script runs in a fresh directory containing only a copy of
your `pipeline/`. The grader passes an absolute `INPUT_DIR` containing:

- `INPUT_DIR/train.csv` - identical to `data/train.csv`.
- `INPUT_DIR/input.csv` - the held-out rows to predict, with the same columns as
  `data/input.csv`.

Do not hard-code the public input or assume it is the graded input.

## Columns
`train.csv`:

- `id` - unique row id (string).
- `sequence` - 24-residue amino-acid sequence (uppercase one-letter codes).
- `family` - antibody family label (categorical).
- `binding` - target binding score in [0, 1].

`input.csv`:

- `id`, `sequence`, `family` - same schema, no `binding`.

## Split
Training rows come from families F01-F06. Development and graded rows come from
held-out families F07-F09. The graded input is larger than the public
development input.

## Signal
Binding depends on sequence context, family context, and interactions between
sequence features. Sequence-derived features and the family label are useful.

## Deliverable
Create `pipeline/predict.sh`. It is invoked from the attempt directory as:

    bash pipeline/predict.sh INPUT_DIR OUTPUT_JSON

It must read `INPUT_DIR/train.csv` and `INPUT_DIR/input.csv` and write
`OUTPUT_JSON`.

## Output JSON
```json
{
  "predictions": [
    {"id": "ab00001", "value": 0.42}
  ]
}
```

- `predictions` must contain exactly one entry per row of `INPUT_DIR/input.csv`.
- `id` values must match the input ids exactly, with no duplicates and no
  missing ids.
- `value` must be a finite number in [0, 1].
- `shortlist` is not required for this task.

## Determinism and runtime
- The same `INPUT_DIR` must produce the same output. The grader runs the script
  twice and compares canonical JSON; byte-identical output is preferred.
- The grader also runs the script on a reversed-row copy of the input and checks
  that per-id predictions are unchanged (tolerance 1e-8).
- No network access. Use the Python standard library or packages already
  installed in the grading environment.
- Each run must finish within 300 seconds.

## Scoring
The score is the normalized RMSE improvement over the global training-mean
baseline:

    100 * (baseline_rmse - model_rmse) / baseline_rmse

clamped to [0, 100]. A constant prediction equal to the training mean scores 0.
"""


# ---------------------------------------------------------------------------
# B02: double-mutant epistasis
# ---------------------------------------------------------------------------


def _gen_b02(rng):
    positions = list(range(1, 11))
    wild_type = {position: rng.choice(_ALPHABET) for position in positions}
    alleles = {}
    for position in positions:
        choices = [aa for aa in _ALPHABET if aa != wild_type[position]]
        alleles[position] = rng.sample(choices, 4)

    mutations = []
    for position in positions:
        for allele in alleles[position]:
            mutations.append((position, wild_type[position], allele))

    additive = {mutation: rng.gauss(0.0, 0.40) for mutation in mutations}
    pair_effect = {}
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pair_effect[(positions[i], positions[j])] = rng.gauss(0.0, 0.30)

    def mutation_name(mutation):
        return "%s%d%s" % (mutation[1], mutation[0], mutation[2])

    def mutation_string(mutations_list):
        if not mutations_list:
            return "WT"
        ordered = sorted(mutations_list, key=lambda item: item[0])
        return "+".join(mutation_name(item) for item in ordered)

    def fitness(mutations_list):
        latent = 0.0
        for mutation in mutations_list:
            latent += additive[mutation]
        if len(mutations_list) == 2:
            first, second = sorted((mutations_list[0][0], mutations_list[1][0]))
            latent += pair_effect[(first, second)]
        latent += rng.gauss(0.0, 0.08)
        return _clamp(_sigmoid(latent), 0.0, 1.0)

    # Training doubles: cover every position pair, then fill to 250.
    train_doubles = []
    seen = set()
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            first_position = positions[i]
            second_position = positions[j]
            for _ in range(4):
                first = (first_position, wild_type[first_position], rng.choice(alleles[first_position]))
                second = (second_position, wild_type[second_position], rng.choice(alleles[second_position]))
                key = tuple(sorted((mutation_name(first), mutation_name(second))))
                if key in seen:
                    continue
                seen.add(key)
                train_doubles.append((first, second))
    while len(train_doubles) < 250:
        first_position, second_position = rng.sample(positions, 2)
        first = (first_position, wild_type[first_position], rng.choice(alleles[first_position]))
        second = (second_position, wild_type[second_position], rng.choice(alleles[second_position]))
        key = tuple(sorted((mutation_name(first), mutation_name(second))))
        if key in seen:
            continue
        seen.add(key)
        train_doubles.append((first, second))
    rng.shuffle(train_doubles)

    # Held-out allele combinations at position pairs seen in training.
    remaining = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            first_position = positions[i]
            second_position = positions[j]
            for first_allele in alleles[first_position]:
                for second_allele in alleles[second_position]:
                    first = (first_position, wild_type[first_position], first_allele)
                    second = (second_position, wild_type[second_position], second_allele)
                    key = tuple(sorted((mutation_name(first), mutation_name(second))))
                    if key not in seen:
                        remaining.append((first, second))
    rng.shuffle(remaining)

    train_rows = []
    counter = 0
    counter += 1
    train_rows.append({
        "id": "dm%05d" % counter,
        "mutations": "WT",
        "fitness": "%.6f" % fitness([]),
    })
    for mutation in mutations:
        counter += 1
        train_rows.append({
            "id": "dm%05d" % counter,
            "mutations": mutation_string([mutation]),
            "fitness": "%.6f" % fitness([mutation]),
        })
    for first, second in train_doubles:
        counter += 1
        train_rows.append({
            "id": "dm%05d" % counter,
            "mutations": mutation_string([first, second]),
            "fitness": "%.6f" % fitness([first, second]),
        })

    shuffled_singles = list(mutations)
    rng.shuffle(shuffled_singles)

    def make_heldout_rows(combo_start, combo_count, single_start, single_count):
        rows = []
        nonlocal counter
        for mutation in shuffled_singles[single_start:single_start + single_count]:
            counter += 1
            rows.append({
                "id": "dm%05d" % counter,
                "mutations": mutation_string([mutation]),
            })
        for first, second in remaining[combo_start:combo_start + combo_count]:
            counter += 1
            rows.append({
                "id": "dm%05d" % counter,
                "mutations": mutation_string([first, second]),
            })
        return rows

    public_rows = make_heldout_rows(0, 90, 0, 10)
    private_rows = make_heldout_rows(90, 180, 10, 20)
    truth_rows = []
    for row in private_rows:
        mutations_list = _parse_mutations(row["mutations"])
        parsed = []
        for name in mutations_list:
            for mutation in mutations:
                if mutation_name(mutation) == name:
                    parsed.append(mutation)
                    break
        truth_rows.append({
            "id": row["id"],
            "fitness": "%.6f" % fitness(parsed),
        })

    return {
        "title": "Double-mutant epistasis",
        "category": "protein_fitness",
        "description": "Predict fitness for wild-type, single, and double mutants, including held-out double combinations.",
        "batch_invariant": True,
        "train_fields": ["id", "mutations", "fitness"],
        "train_rows": train_rows,
        "input_fields": ["id", "mutations"],
        "public_input_rows": public_rows,
        "private_input_rows": private_rows,
        "truth_fields": ["id", "fitness"],
        "truth_rows": truth_rows,
        "brief": _brief_b02(),
    }


def _brief_b02():
    return """# B02 - Double-mutant epistasis

**Category:** protein_fitness
**Batch-invariance check:** yes

## Goal
Predict a fitness value in [0, 1] for wild-type, single-mutant, and double-mutant
protein variants. Training data contains the wild type, all single mutants, and
many double mutants. The graded input contains held-out double combinations:
the same position pairs appear in training, but with different amino-acid
allele combinations.

## Files
You develop against the public copies:

- `data/train.csv` - labeled training rows.
- `data/input.csv` - unlabeled development rows.

At grading time your script runs in a fresh directory containing only a copy of
your `pipeline/`. The grader passes an absolute `INPUT_DIR` containing:

- `INPUT_DIR/train.csv` - identical to `data/train.csv`.
- `INPUT_DIR/input.csv` - the held-out rows to predict, with the same columns as
  `data/input.csv`.

## Columns
`train.csv`:

- `id` - unique row id (string).
- `mutations` - `WT`, a single mutation such as `A12V`, or a double mutation
  such as `A12V+K33R`.
- `fitness` - target fitness in [0, 1].

`input.csv`:

- `id`, `mutations` - same schema, no `fitness`.

## Split
Training rows include the wild type, every single mutation, and 250 double
mutants. Development and graded rows are held-out double combinations at
position pairs that occur in training. The graded input is larger than the
public development input.

## Signal
Fitness has additive per-mutation effects plus position-pair epistasis. A model
that only sums single-mutation effects will miss the pairwise component.

## Deliverable
Create `pipeline/predict.sh`. It is invoked from the attempt directory as:

    bash pipeline/predict.sh INPUT_DIR OUTPUT_JSON

It must read `INPUT_DIR/train.csv` and `INPUT_DIR/input.csv` and write
`OUTPUT_JSON`.

## Output JSON
```json
{
  "predictions": [
    {"id": "dm00001", "value": 0.55}
  ]
}
```

- `predictions` must contain exactly one entry per row of `INPUT_DIR/input.csv`.
- `id` values must match the input ids exactly, with no duplicates and no
  missing ids.
- `value` must be a finite number in [0, 1].
- `shortlist` is not required for this task.

## Determinism and runtime
- The same `INPUT_DIR` must produce the same output. The grader runs the script
  twice and compares canonical JSON; byte-identical output is preferred.
- The grader also runs the script on a reversed-row copy of the input and checks
  that per-id predictions are unchanged (tolerance 1e-8).
- No network access. Use the Python standard library or packages already
  installed in the grading environment.
- Each run must finish within 300 seconds.

## Scoring
The baseline is a small ridge-regularized additive least-squares model fitted on
the training rows only, using one indicator per single mutation plus an
intercept. The score is:

    100 * (baseline_rmse - model_rmse) / baseline_rmse

clamped to [0, 100]. A model no better than the additive baseline scores 0.
"""


# ---------------------------------------------------------------------------
# B03: antibody family clustering
# ---------------------------------------------------------------------------


def _gen_b03(rng):
    length = rng.randint(48, 60)
    ancestor = [rng.choice(_ALPHABET) for _ in range(length)]
    parent_names = ["fam%02d" % i for i in range(1, 9)]
    parent_sequences = {}
    daughter_sequences = {}
    mutation_rates = {}

    for parent in parent_names:
        lineage_positions = rng.sample(range(length), 10)
        parent_sequence = list(ancestor)
        for position in lineage_positions:
            choices = [aa for aa in _ALPHABET if aa != parent_sequence[position]]
            parent_sequence[position] = rng.choice(choices)
        parent_sequences[parent] = "".join(parent_sequence)

        # Two daughter founders differ from the parent at disjoint subsets of
        # the lineage positions.  The parent label is visible in training; the
        # daughter labels are sealed.
        split_positions = rng.sample(lineage_positions, 4)
        first_positions = split_positions[:2]
        second_positions = split_positions[2:]
        first = list(parent_sequence)
        for position in first_positions:
            choices = [aa for aa in _ALPHABET if aa != first[position]]
            first[position] = rng.choice(choices)
        second = list(parent_sequence)
        for position in second_positions:
            choices = [aa for aa in _ALPHABET if aa != second[position]]
            second[position] = rng.choice(choices)
        daughter_sequences[parent + "a"] = "".join(first)
        daughter_sequences[parent + "b"] = "".join(second)
        mutation_rates[parent] = rng.uniform(0.10, 0.18)

    def mutate(sequence, rate):
        result = list(sequence)
        for index in range(length):
            if rng.random() < rate:
                choices = [aa for aa in _ALPHABET if aa != result[index]]
                result[index] = rng.choice(choices)
        return "".join(result)

    train_rows = []
    counter = 0
    for parent in parent_names:
        for _ in range(32):
            counter += 1
            train_rows.append({
                "id": "ab%05d" % counter,
                "sequence": mutate(parent_sequences[parent], mutation_rates[parent]),
                "family": parent,
            })

    # Build the stratified public/private split first, then globally shuffle the
    # pooled test rows and assign independent random non-sequential ids.  The
    # old implementation appended private rows daughter-by-daughter and used
    # contiguous ids, so both `row_index // 21` and sorted-id blocks recovered
    # the sealed partition exactly.  Ids now carry no daughter, split, or
    # row-order information.
    test_rows = []
    for parent in parent_names:
        for suffix in ("a", "b"):
            daughter = parent + suffix
            rows = []
            for _ in range(30):
                counter += 1
                rows.append({
                    "id": "ab%05d" % counter,
                    "sequence": mutate(daughter_sequences[daughter], mutation_rates[parent]),
                })
            # One occasional within-daughter duplicate with a fresh id.  There
            # are no exact train/test sequence or id leaks.
            counter += 1
            source = rng.choice(rows)
            rows.append({
                "id": "ab%05d" % counter,
                "sequence": source["sequence"],
            })
            rng.shuffle(rows)
            for index, row in enumerate(rows):
                row["_split"] = "public" if index < 10 else "private"
                row["_family"] = daughter
                test_rows.append(row)

    rng.shuffle(test_rows)
    # 10000-99999 cannot collide with the five-digit train ids (00001-00256).
    id_values = rng.sample(range(10000, 100000), len(test_rows))
    for row, value in zip(test_rows, id_values):
        row["id"] = "ab%05d" % value

    public_rows = [
        {"id": row["id"], "sequence": row["sequence"]}
        for row in test_rows
        if row["_split"] == "public"
    ]
    private_rows = [
        {"id": row["id"], "sequence": row["sequence"]}
        for row in test_rows
        if row["_split"] == "private"
    ]
    truth_rows = [
        {"id": row["id"], "family": row["_family"]}
        for row in test_rows
        if row["_split"] == "private"
    ]
    rng.shuffle(public_rows)
    rng.shuffle(private_rows)
    rng.shuffle(truth_rows)

    return {
        "title": "Antibody family clustering",
        "category": "sequence_clustering",
        "description": "Cluster held-out antibody sequences into unseen daughter subfamilies of known parent lineages without using label names.",
        "batch_invariant": False,
        "train_fields": ["id", "sequence", "family"],
        "train_rows": train_rows,
        "input_fields": ["id", "sequence"],
        "public_input_rows": public_rows,
        "private_input_rows": private_rows,
        "truth_fields": ["id", "family"],
        "truth_rows": truth_rows,
        "brief": _brief_b03(),
    }


def _brief_b03():
    return """# B03 - Antibody family clustering

**Category:** sequence_clustering
**Batch-invariance check:** no

## Goal
Assign each held-out antibody sequence to its subfamily. Training data contains
labeled sequences from eight parent lineages. Each parent lineage has two
unseen daughter subfamilies in the graded input, so the graded partition has
sixteen true groups even though training exposes only eight parent labels.

## Files
You develop against the public copies:

- `data/train.csv` - labeled training rows.
- `data/input.csv` - unlabeled development rows.

At grading time your script runs in a fresh directory containing only a copy of
your `pipeline/`. The grader passes an absolute `INPUT_DIR` containing:

- `INPUT_DIR/train.csv` - identical to `data/train.csv`.
- `INPUT_DIR/input.csv` - the held-out rows to cluster, with the same columns as
  `data/input.csv`.

## Columns
`train.csv`:

- `id` - unique row id (string).
- `sequence` - amino-acid sequence of length 48-60 (uppercase one-letter codes).
- `family` - parent-lineage label (categorical).

`input.csv`:

- `id`, `sequence` - same schema, no `family`.

## Split
Training rows are 32 labeled sequences per parent lineage. Development and
graded rows are held-out sequences from the two unseen daughter subfamilies of
each parent. Both the public development input and the larger graded input
contain rows from all sixteen daughters. A small number of within-daughter
duplicate sequences may appear with fresh ids; there are no exact training-row
leaks.

## Signal
Sequences from the same parent lineage share lineage-specific positions, and
the two daughters of a lineage differ at a subset of those positions. A
reasonable approach is to assign each input sequence broadly to its nearest
parent lineage using the training rows, then cluster the sequences assigned to
each lineage into two groups (for example with two medoids or a centroid
split). Label names are arbitrary.

## Output
Create `pipeline/predict.sh`. It is invoked from the attempt directory as:

    bash pipeline/predict.sh INPUT_DIR OUTPUT_JSON

It must read `INPUT_DIR/train.csv` and `INPUT_DIR/input.csv` and write
`OUTPUT_JSON`.

```json
{
  "predictions": [
    {"id": "ab00001", "value": "cluster_1"}
  ]
}
```

- `predictions` must contain exactly one entry per row of `INPUT_DIR/input.csv`.
- `id` values must match the input ids exactly, with no duplicates and no
  missing ids.
- `value` must be a non-empty string. The grader does not require particular
  label names; only the partition is scored.
- `shortlist` is not required for this task.

## Determinism and runtime
- The same `INPUT_DIR` must produce the same output. The grader runs the script
  twice and compares canonical JSON; byte-identical output is preferred.
- No network access. Use the Python standard library or packages already
  installed in the grading environment.
- Each run must finish within 300 seconds.

## Scoring
The score uses the Adjusted Rand Index (ARI) between your partition and the
sealed daughter labels. The baseline is the best single Hamming-distance cutoff
clustering on the input sequences. The score is:

    100 * (model_ari - baseline_ari) / (1 - baseline_ari)

clamped to [0, 100]. Label names are never compared. If the baseline is
degenerate (ARI at or above 1), the task reports a calibration error instead of
rescaling.
"""


# ---------------------------------------------------------------------------
# B04: gene dependency transfer
# ---------------------------------------------------------------------------


def _gen_b04(rng):
    train_tissues = ["lung", "breast", "colon", "skin"]
    test_tissues = ["colon", "skin", "liver", "brain"]
    train_cell_lines = ["CL%02d" % i for i in range(1, 31)]
    test_cell_lines = ["CL%02d" % i for i in range(31, 43)]
    dev_cell_lines = ["CL%02d" % i for i in range(43, 49)]
    all_cell_lines = train_cell_lines + test_cell_lines + dev_cell_lines

    cell_tissue = {}
    for cell_line in train_cell_lines:
        cell_tissue[cell_line] = rng.choice(train_tissues)
    for cell_line in test_cell_lines:
        cell_tissue[cell_line] = rng.choice(test_tissues)
    for cell_line in dev_cell_lines:
        cell_tissue[cell_line] = rng.choice(test_tissues)

    cell_covariates = {}
    for cell_line in all_cell_lines:
        cell_covariates[cell_line] = {
            "expression": round(rng.uniform(0.0, 1.0), 4),
            "mutation_count": rng.randint(0, 20),
            "copy_number": round(rng.uniform(-2.0, 2.0), 4),
            "lineage_score": round(rng.uniform(0.0, 1.0), 4),
        }

    base_genes = ["G%02d" % i for i in range(1, 41)]
    gene_mean = {gene: rng.uniform(0.15, 0.75) for gene in base_genes}
    all_tissues = sorted(set(train_tissues + test_tissues))
    tissue_effect = {}
    for gene in base_genes:
        for tissue in all_tissues:
            tissue_effect[(gene, tissue)] = rng.gauss(0.0, 0.30)

    # A tempting gene: high observed mean because low values are missing more
    # often, but only an average true dependency. It is present in every cell
    # line so that its missingness is visible and it is easy to over-select.
    tempting_gene = "G_TEMP"
    gene_mean[tempting_gene] = 0.55
    for tissue in all_tissues:
        tissue_effect[(tempting_gene, tissue)] = rng.gauss(0.0, 0.10)

    def dependency(cell_line, gene):
        covariates = cell_covariates[cell_line]
        tissue = cell_tissue[cell_line]
        base = math.log(gene_mean[gene] / (1.0 - gene_mean[gene]))
        latent = base + tissue_effect[(gene, tissue)]
        latent += 0.30 * (covariates["expression"] - 0.5)
        latent += 0.015 * (covariates["mutation_count"] - 10)
        latent += 0.10 * covariates["copy_number"]
        latent += 0.15 * (covariates["lineage_score"] - 0.5)
        latent += rng.gauss(0.0, 0.08)
        return _clamp(_sigmoid(latent), 0.0, 1.0)

    def make_rows(cell_lines, genes_per_line, prefix, labeled):
        rows = []
        truth = []
        counter = 0
        for cell_line in cell_lines:
            chosen = rng.sample(base_genes, genes_per_line - 1) + [tempting_gene]
            rng.shuffle(chosen)
            for gene in chosen:
                counter += 1
                value = dependency(cell_line, gene)
                row = {
                    "id": "%s%05d" % (prefix, counter),
                    "cell_line": cell_line,
                    "tissue": cell_tissue[cell_line],
                    "gene": gene,
                    "expression": "%.4f" % cell_covariates[cell_line]["expression"],
                    "mutation_count": str(cell_covariates[cell_line]["mutation_count"]),
                    "copy_number": "%.4f" % cell_covariates[cell_line]["copy_number"],
                    "lineage_score": "%.4f" % cell_covariates[cell_line]["lineage_score"],
                }
                if labeled:
                    if gene == tempting_gene:
                        missing = rng.random() < (0.90 if value < 0.60 else 0.30)
                    else:
                        missing = rng.random() < 0.10
                    row["dependency"] = "" if missing else "%.6f" % value
                rows.append(row)
                truth.append({"id": row["id"], "dependency": "%.6f" % value})
        return rows, truth

    train_rows, _ = make_rows(train_cell_lines, 20, "gd", True)
    public_rows, _ = make_rows(dev_cell_lines, 20, "gv", False)
    private_rows, truth_rows = make_rows(test_cell_lines, 25, "gh", False)

    return {
        "title": "Gene dependency transfer",
        "category": "functional_genomics",
        "description": "Forecast gene dependency for unseen cell lines and tissues, and select a balanced 20-pair panel.",
        "batch_invariant": True,
        "train_fields": [
            "id", "cell_line", "tissue", "gene", "expression",
            "mutation_count", "copy_number", "lineage_score", "dependency",
        ],
        "train_rows": train_rows,
        "input_fields": [
            "id", "cell_line", "tissue", "gene", "expression",
            "mutation_count", "copy_number", "lineage_score",
        ],
        "public_input_rows": public_rows,
        "private_input_rows": private_rows,
        "truth_fields": ["id", "dependency"],
        "truth_rows": truth_rows,
        "brief": _brief_b04(),
    }


def _brief_b04():
    return """# B04 - Gene dependency transfer

**Category:** functional_genomics
**Batch-invariance check:** yes

## Goal
Forecast a continuous gene-dependency score in [0, 1] for long-format
cell-line/gene pairs, then select a balanced panel of 20 pairs. Training data
covers 30 cell lines and four tissues. The graded input contains unseen cell
lines and some unseen tissues.

## Files
You develop against the public copies:

- `data/train.csv` - labeled training rows.
- `data/input.csv` - unlabeled development rows.

At grading time your script runs in a fresh directory containing only a copy of
your `pipeline/`. The grader passes an absolute `INPUT_DIR` containing:

- `INPUT_DIR/train.csv` - identical to `data/train.csv`.
- `INPUT_DIR/input.csv` - the held-out rows to predict, with the same columns as
  `data/input.csv`.

## Columns
`train.csv`:

- `id` - unique pair id (string).
- `cell_line` - cell-line id (categorical).
- `tissue` - tissue label (categorical).
- `gene` - gene id (categorical).
- `expression` - cell-line expression covariate in [0, 1].
- `mutation_count` - cell-line mutation burden (integer).
- `copy_number` - cell-line copy-number score.
- `lineage_score` - cell-line lineage covariate in [0, 1].
- `dependency` - target dependency in [0, 1], or blank when the label is
  missing.

`input.csv`:

- same columns except `dependency`.

## Split
Training rows come from 30 cell lines and four tissues. Development and graded
rows come from unseen cell lines; the graded input also includes two tissues
that do not appear in training. The graded input is larger than the public
development input.

## Missing labels
Blank `dependency` values in `train.csv` are missing labels. Ignore them when
fitting baselines or models. One gene has a high observed mean but a large
fraction of missing labels; its observed mean is biased upward.

## Deliverable
Create `pipeline/predict.sh`. It is invoked from the attempt directory as:

    bash pipeline/predict.sh INPUT_DIR OUTPUT_JSON

It must read `INPUT_DIR/train.csv` and `INPUT_DIR/input.csv` and write
`OUTPUT_JSON`.

## Output JSON
```json
{
  "predictions": [
    {"id": "gh00001", "value": 0.37}
  ],
  "shortlist": ["gh00001", "gh00002"]
}
```

- `predictions` must contain exactly one entry per row of `INPUT_DIR/input.csv`.
- `id` values must match the input ids exactly, with no duplicates and no
  missing ids.
- `value` must be a finite number in [0, 1].
- `shortlist` is required. It must contain exactly 20 unique ids from
  `INPUT_DIR/input.csv`, with at most 2 ids per gene.

## Determinism and runtime
- The same `INPUT_DIR` must produce the same output. The grader runs the script
  twice and compares canonical JSON; byte-identical output is preferred.
- The grader also runs the script on a reversed-row copy of the input and checks
  that per-id predictions are unchanged (tolerance 1e-8).
- No network access. Use the Python standard library or packages already
  installed in the grading environment.
- Each run must finish within 300 seconds.

## Scoring
The forecast component is the normalized RMSE improvement over a train-only
gene-mean baseline (blank labels ignored, global mean fallback):

    forecast = clamp((baseline_rmse - model_rmse) / baseline_rmse, 0, 1)

The panel component compares your 20 selected pairs with a baseline panel and
the best feasible oracle panel. The baseline panel is exactly 20 ids selected
from the input using train-only gene-mean predictions, with at most 2 ids per
gene and deterministic id tie-breaking. The oracle panel is the best feasible
20-pair panel using sealed truth, also with at most 2 ids per gene:

    panel_skill = clamp(
        (selected_mean - baseline_panel_mean) / (oracle_mean - baseline_panel_mean),
        0, 1)

The final score is the equal-weight additive combination:

    100 * (0.5 * forecast + 0.5 * panel_skill)

clamped to [0, 100]. A model that exactly reproduces the baseline predictions
and baseline panel scores 0; a model that exactly reproduces the truth and
oracle panel scores 100. A perfect forecast with the baseline panel scores 50,
and the baseline forecast with the oracle panel scores 50.
"""


# ---------------------------------------------------------------------------
# B05: docking pose ranking
# ---------------------------------------------------------------------------


def _gen_b05(rng):
    target_types = ["kinase", "protease", "gpcr", "phosphatase", "nuclear"]
    train_targets = ["T%02d" % i for i in range(1, 51)]
    dev_targets = ["T%02d" % i for i in range(51, 61)]
    test_targets = ["T%02d" % i for i in range(61, 91)]

    def assign_types(targets, per_type):
        values = []
        for target_type in target_types:
            values.extend([target_type] * per_type)
        rng.shuffle(values)
        return dict(zip(targets, values))

    target_type = {}
    target_type.update(assign_types(train_targets, 10))
    target_type.update(assign_types(dev_targets, 2))
    target_type.update(assign_types(test_targets, 6))

    # Each target_type has a different relationship between within-target
    # relative pose features and quality.
    type_coefficients = {}
    for target_type_name in target_types:
        coefficients = {
            "shape": 1.2 + rng.gauss(0.0, 0.55),
            "hbond": 0.9 + rng.gauss(0.0, 0.55),
            "buried": 0.8 + rng.gauss(0.0, 0.55),
            "clash": -1.0 + rng.gauss(0.0, 0.55),
            "energy": 0.4 + rng.gauss(0.0, 0.55),
            "shape_buried": rng.gauss(0.0, 0.8),
            "hbond_clash": rng.gauss(0.0, 0.8),
        }
        type_coefficients[target_type_name] = (rng.gauss(0.0, 0.5), coefficients)

    target_parameters = {}
    for target in train_targets + dev_targets + test_targets:
        target_parameters[target] = {
            "energy_offset": rng.gauss(0.0, 2.5),
            "energy_scale": rng.uniform(0.6, 1.6),
            "shape_offset": rng.gauss(0.0, 0.25),
            "shape_scale": rng.uniform(0.5, 1.5),
            "hbond_offset": rng.gauss(0.0, 3.0),
            "hbond_scale": rng.uniform(0.5, 1.5),
            "buried_offset": rng.gauss(0.0, 0.2),
            "buried_scale": rng.uniform(0.5, 1.5),
            "clash_offset": rng.gauss(0.0, 1.5),
            "clash_scale": rng.uniform(0.5, 1.5),
        }

    def make_target_rows(target, prefix, counter_start, labeled):
        rows = []
        truth = []
        parameters = target_parameters[target]
        raw = []
        for pose in range(1, 13):
            energy = rng.gauss(-8.0, 2.0) * parameters["energy_scale"] + parameters["energy_offset"]
            shape = _clamp(
                rng.gauss(0.5, 0.2) * parameters["shape_scale"] + parameters["shape_offset"],
                0.0, 1.0,
            )
            hbond = int(round(_clamp(
                rng.gauss(5.0, 2.5) * parameters["hbond_scale"] + parameters["hbond_offset"],
                0.0, 12.0,
            )))
            buried = _clamp(
                rng.gauss(0.5, 0.2) * parameters["buried_scale"] + parameters["buried_offset"],
                0.0, 1.0,
            )
            clash = int(round(_clamp(
                rng.gauss(1.0, 1.2) * parameters["clash_scale"] + parameters["clash_offset"],
                0.0, 8.0,
            )))
            if rng.random() < 0.25:
                energy -= 2.5 * parameters["energy_scale"]
                clash = int(_clamp(clash + rng.randint(2, 4), 0, 8))
            raw.append({
                "pose_rank": pose,
                "energy": energy,
                "shape": shape,
                "hbond": hbond,
                "buried": buried,
                "clash": clash,
            })

        # The minimum-energy pose is a decoy: give it clashes and poor
        # complementarity so that the baseline is not accidentally the oracle.
        min_energy_index = min(
            range(len(raw)),
            key=lambda index: (raw[index]["energy"], raw[index]["pose_rank"]),
        )
        raw[min_energy_index]["clash"] = max(raw[min_energy_index]["clash"], 6)
        raw[min_energy_index]["shape"] = min(raw[min_energy_index]["shape"], 0.2)
        raw[min_energy_index]["buried"] = min(raw[min_energy_index]["buried"], 0.3)

        shape_rank = _rank_normalize([item["shape"] for item in raw])
        hbond_rank = _rank_normalize([item["hbond"] for item in raw])
        buried_rank = _rank_normalize([item["buried"] for item in raw])
        clash_rank = _rank_normalize([item["clash"] for item in raw])
        energy_rank = _rank_normalize([-item["energy"] for item in raw])
        intercept, coefficients = type_coefficients[target_type[target]]

        for index, item in enumerate(raw):
            features = {
                "shape": shape_rank[index],
                "hbond": hbond_rank[index],
                "buried": buried_rank[index],
                "clash": clash_rank[index],
                "energy": energy_rank[index],
            }
            logit = intercept
            for name, value in features.items():
                logit += coefficients[name] * value
            logit += coefficients["shape_buried"] * features["shape"] * features["buried"]
            logit += coefficients["hbond_clash"] * features["hbond"] * (1.0 - features["clash"])
            logit += rng.gauss(0.0, 0.70)
            quality = _sigmoid(logit)
            row = {
                "id": "%s%05d" % (prefix, counter_start + item["pose_rank"]),
                "target": target,
                "target_type": target_type[target],
                "pose_rank": str(item["pose_rank"]),
                "energy": "%.4f" % item["energy"],
                "shape_complementarity": "%.4f" % item["shape"],
                "hbond_count": str(item["hbond"]),
                "buried_sasa": "%.4f" % item["buried"],
                "clash_count": str(item["clash"]),
            }
            if labeled:
                row["quality"] = "%.6f" % quality
            rows.append(row)
            truth.append({"id": row["id"], "quality": "%.6f" % quality})
        return rows, truth

    train_rows = []
    counter = 0
    for target in train_targets:
        rows, _ = make_target_rows(target, "dp", counter, True)
        train_rows.extend(rows)
        counter += 12

    public_rows = []
    for target in dev_targets:
        rows, _ = make_target_rows(target, "dv", counter, False)
        public_rows.extend(rows)
        counter += 12

    private_rows = []
    truth_rows = []
    for target in test_targets:
        rows, truth = make_target_rows(target, "dh", counter, False)
        private_rows.extend(rows)
        truth_rows.extend(truth)
        counter += 12

    return {
        "title": "Docking pose ranking",
        "category": "structural_biology",
        "description": "Rank docking poses for unseen targets and select one pose per target using target-relative pose comparisons.",
        "batch_invariant": True,
        "train_fields": [
            "id", "target", "target_type", "pose_rank", "energy",
            "shape_complementarity", "hbond_count", "buried_sasa",
            "clash_count", "quality",
        ],
        "train_rows": train_rows,
        "input_fields": [
            "id", "target", "target_type", "pose_rank", "energy",
            "shape_complementarity", "hbond_count", "buried_sasa",
            "clash_count",
        ],
        "public_input_rows": public_rows,
        "private_input_rows": private_rows,
        "truth_fields": ["id", "quality"],
        "truth_rows": truth_rows,
        "brief": _brief_b05(),
    }


def _brief_b05():
    return """# B05 - Docking pose ranking

**Category:** structural_biology
**Batch-invariance check:** yes

## Goal
Predict a pose-quality score in [0, 1] for every docking pose, then select
exactly one pose per unseen target. Training data covers 50 targets with twelve
poses each. The graded input contains 30 unseen targets with twelve poses each.

## Files
You develop against the public copies:

- `data/train.csv` - labeled training rows.
- `data/input.csv` - unlabeled development rows.

At grading time your script runs in a fresh directory containing only a copy of
your `pipeline/`. The grader passes an absolute `INPUT_DIR` containing:

- `INPUT_DIR/train.csv` - identical to `data/train.csv`.
- `INPUT_DIR/input.csv` - the held-out rows to predict, with the same columns as
  `data/input.csv`.

## Columns
`train.csv`:

- `id` - unique pose id (string).
- `target` - target id (categorical).
- `target_type` - target regime label (categorical).
- `pose_rank` - pose index within the target (1-12).
- `energy` - docking energy (lower is not always better).
- `shape_complementarity` - shape score in [0, 1].
- `hbond_count` - number of hydrogen bonds (integer).
- `buried_sasa` - buried surface area fraction in [0, 1].
- `clash_count` - steric clash count (integer).
- `quality` - target pose quality in [0, 1].

`input.csv`:

- same columns except `quality`.

## Split
Training rows come from 50 targets. Development and graded rows come from
disjoint unseen targets. Every split contains all five target_type regimes. The
graded input is larger than the public development input.

## Signal
Absolute feature values shift strongly between targets, so a single global
mapping from raw features to quality transfers poorly. Quality depends on how a
pose compares with the other poses for the same target, and the relationship
differs across target_type regimes. The minimum-energy pose is a useful but
imperfect baseline; some low-energy poses are decoys with clashes and poor
shape or buried-surface complementarity.

## Deliverable
Create `pipeline/predict.sh`. It is invoked from the attempt directory as:

    bash pipeline/predict.sh INPUT_DIR OUTPUT_JSON

It must read `INPUT_DIR/train.csv` and `INPUT_DIR/input.csv` and write
`OUTPUT_JSON`.

## Output JSON
```json
{
  "predictions": [
    {"id": "dh00001", "value": 0.61}
  ],
  "shortlist": ["dh00001", "dh00009"]
}
```

- `predictions` must contain exactly one entry per row of `INPUT_DIR/input.csv`.
- `id` values must match the input ids exactly, with no duplicates and no
  missing ids.
- `value` must be a finite number in [0, 1].
- `shortlist` is required. It must contain exactly one pose id per test target,
  and every test target must be covered exactly once.

## Determinism and runtime
- The same `INPUT_DIR` must produce the same output. The grader runs the script
  twice and compares canonical JSON; byte-identical output is preferred.
- The grader also runs the script on a reversed-row copy of the input and checks
  that per-id predictions are unchanged (tolerance 1e-8).
- No network access. Use the Python standard library or packages already
  installed in the grading environment.
- Each run must finish within 300 seconds.

## Scoring
For each target, the baseline is the minimum-energy pose and the oracle is the
highest-quality pose. The per-target skill is the selected pose's quality
improvement over the baseline divided by the oracle-minus-baseline gap:

    skill = (selected_quality - baseline_quality) / (oracle_quality - baseline_quality)

Each per-target skill is clamped to [-1, 1] before averaging so that targets
with very small baseline-to-oracle gaps cannot dominate. The final score is:

    100 * clamp(mean_targets(skill), 0, 1)

Selecting the minimum-energy pose for every target scores 0; selecting the best
pose for every target scores 100. Targets with a non-positive
baseline-to-oracle gap are reported as degenerate and excluded from the mean.
"""


# ---------------------------------------------------------------------------
# B06: designed binder selection
# ---------------------------------------------------------------------------


def _gen_b06(rng):
    train_targets = ["BT%02d" % i for i in range(1, 9)]
    test_targets = ["BT%02d" % i for i in range(9, 13)]
    dev_targets = ["BT%02d" % i for i in range(13, 15)]
    families = ["kinase", "protease", "gpcr", "phosphatase"]

    target_family = {}
    target_effect = {}
    for target in train_targets + test_targets + dev_targets:
        target_family[target] = rng.choice(families)
        target_effect[target] = rng.gauss(0.0, 0.40)

    def make_target_rows(target, prefix, counter_start, labeled):
        rows = []
        truth = []
        for index in range(30):
            rmsd = rng.uniform(0.5, 5.0)
            interface = rng.uniform(0.0, 1.0)
            charge = rng.gauss(0.0, 2.0)
            hydrophobicity = rng.uniform(0.0, 1.0)
            logit = (
                2.5 * interface
                + 0.8 * hydrophobicity
                - 0.4 * abs(charge - 1.0)
                + target_effect[target]
                - 1.0
            )
            probability = _sigmoid(logit)
            binder = 1 if rng.random() < probability else 0
            row = {
                "id": "%s%05d" % (prefix, counter_start + index),
                "target": target,
                "target_family": target_family[target],
                "scaffold_rmsd": "%.4f" % rmsd,
                "interface_area": "%.4f" % interface,
                "net_charge": "%.4f" % charge,
                "hydrophobicity": "%.4f" % hydrophobicity,
            }
            if labeled:
                row["binder"] = str(binder)
            rows.append(row)
            truth.append({"id": row["id"], "binder": str(binder)})
        return rows, truth

    train_rows = []
    counter = 0
    for target in train_targets:
        rows, _ = make_target_rows(target, "bd", counter, True)
        train_rows.extend(rows)
        counter += 30

    public_rows = []
    for target in dev_targets:
        rows, _ = make_target_rows(target, "bv", counter, False)
        public_rows.extend(rows)
        counter += 30

    private_rows = []
    truth_rows = []
    for target in test_targets:
        rows, truth = make_target_rows(target, "bh", counter, False)
        private_rows.extend(rows)
        truth_rows.extend(truth)
        counter += 30

    return {
        "title": "Designed binder selection",
        "category": "protein_design",
        "description": "Predict binder probabilities for designed proteins and select a balanced 40-design panel across unseen targets.",
        "batch_invariant": True,
        "train_fields": [
            "id", "target", "target_family", "scaffold_rmsd",
            "interface_area", "net_charge", "hydrophobicity", "binder",
        ],
        "train_rows": train_rows,
        "input_fields": [
            "id", "target", "target_family", "scaffold_rmsd",
            "interface_area", "net_charge", "hydrophobicity",
        ],
        "public_input_rows": public_rows,
        "private_input_rows": private_rows,
        "truth_fields": ["id", "binder"],
        "truth_rows": truth_rows,
        "brief": _brief_b06(),
    }


def _brief_b06():
    return """# B06 - Designed binder selection

**Category:** protein_design
**Batch-invariance check:** yes

## Goal
Predict a binder probability in [0, 1] for every designed protein, then select
exactly 40 designs: 10 per unseen test target. Training data covers eight
targets with 30 designs each. The graded input contains four unseen targets with
30 designs each.

## Files
You develop against the public copies:

- `data/train.csv` - labeled training rows.
- `data/input.csv` - unlabeled development rows.

At grading time your script runs in a fresh directory containing only a copy of
your `pipeline/`. The grader passes an absolute `INPUT_DIR` containing:

- `INPUT_DIR/train.csv` - identical to `data/train.csv`.
- `INPUT_DIR/input.csv` - the held-out rows to predict, with the same columns as
  `data/input.csv`.

## Columns
`train.csv`:

- `id` - unique design id (string).
- `target` - target id (categorical).
- `target_family` - target family label (categorical).
- `scaffold_rmsd` - scaffold RMSD (lower is not necessarily better).
- `interface_area` - predicted interface area in [0, 1].
- `net_charge` - design net charge.
- `hydrophobicity` - hydrophobicity score in [0, 1].
- `binder` - binary binder label (0 or 1).

`input.csv`:

- same columns except `binder`.

## Split
Training rows come from eight targets. Development and graded rows come from
unseen targets. The graded input is larger than the public development input.

## Signal
Binder probability depends on interface area, hydrophobicity, net charge, and
target context. Low scaffold RMSD is a tempting but weak selection criterion.

## Deliverable
Create `pipeline/predict.sh`. It is invoked from the attempt directory as:

    bash pipeline/predict.sh INPUT_DIR OUTPUT_JSON

It must read `INPUT_DIR/train.csv` and `INPUT_DIR/input.csv` and write
`OUTPUT_JSON`.

## Output JSON
```json
{
  "predictions": [
    {"id": "bh00001", "value": 0.73}
  ],
  "shortlist": ["bh00001", "bh00002"]
}
```

- `predictions` must contain exactly one entry per row of `INPUT_DIR/input.csv`.
- `id` values must match the input ids exactly, with no duplicates and no
  missing ids.
- `value` must be a finite number in [0, 1] for every row, including rows that
  are not shortlisted.
- `shortlist` is required. It must contain exactly 40 unique ids from
  `INPUT_DIR/input.csv`, with exactly 10 ids per test target.

## Determinism and runtime
- The same `INPUT_DIR` must produce the same output. The grader runs the script
  twice and compares canonical JSON; byte-identical output is preferred.
- The grader also runs the script on a reversed-row copy of the input and checks
  that per-id predictions are unchanged (tolerance 1e-8).
- No network access. Use the Python standard library or packages already
  installed in the grading environment.
- Each run must finish within 300 seconds.

## Scoring
The baseline selects the 10 designs with the lowest `scaffold_rmsd` per target.
The oracle selects the 10 true binders per target (or all binders if fewer than
10). The score is the normalized hit count:

    100 * (model_hits - baseline_hits) / (oracle_hits - baseline_hits)

clamped to [0, 100]. The baseline scores 0; the oracle scores 100.
"""


# ---------------------------------------------------------------------------
# Bank generation
# ---------------------------------------------------------------------------


def _write_task(task_dir, task_id, data):
    public_dir = task_dir / "public"
    private_dir = task_dir / "private"
    private_input_dir = private_dir / "input"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_input_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(public_dir / "train.csv", data["train_fields"], data["train_rows"])
    _write_csv(public_dir / "input.csv", data["input_fields"], data["public_input_rows"])
    _write_csv(private_input_dir / "train.csv", data["train_fields"], data["train_rows"])
    _write_csv(private_input_dir / "input.csv", data["input_fields"], data["private_input_rows"])
    _write_csv(private_dir / "truth.csv", data["truth_fields"], data["truth_rows"])

    (task_dir / "brief.md").write_text(data["brief"], encoding="utf-8")
    meta = {
        "id": task_id,
        "version": BANK_VERSION,
        "title": data["title"],
        "category": data["category"],
        "description": data["description"],
        "batch_invariant": data["batch_invariant"],
    }
    (task_dir / "task.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


_GENERATORS = {
    "B01": _gen_b01,
    "B02": _gen_b02,
    "B03": _gen_b03,
    "B04": _gen_b04,
    "B05": _gen_b05,
    "B06": _gen_b06,
}


def _generate_task_data(task_id, seed):
    """Return the in-memory task data for one task and seed.

    The seed formula is preserved from v0.1: ``seed + 1000 * (index + 1)``
    where ``index`` is the task's position in ``TASK_IDS``.
    """
    if task_id not in _GENERATORS:
        raise ValueError("unknown task id %r" % task_id)
    index = TASK_IDS.index(task_id)
    rng = random.Random(seed + 1000 * (index + 1))
    return _GENERATORS[task_id](rng)


def generate_bank(root, seed=2026):
    """Generate the six-task synthetic bank beneath ``root``.

    The bank is replaced consistently: any existing ``tasks/`` tree is removed
    before the new tasks and ``bank.json`` metadata are written.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tasks_root = root / "tasks"
    if tasks_root.exists():
        shutil.rmtree(tasks_root)
    tasks_root.mkdir(parents=True, exist_ok=True)

    for task_id in TASK_IDS:
        data = _generate_task_data(task_id, seed)
        _write_task(tasks_root / task_id, task_id, data)

    metadata = {
        "benchmark": "bioagent-bench",
        "version": BANK_VERSION,
        "seed": seed,
        "task_ids": list(TASK_IDS),
    }
    (root / BANK_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_bank_metadata(root):
    """Read and return ``root/bank.json`` or raise :class:`BankMetadataError`."""
    path = Path(root) / BANK_METADATA_FILENAME
    if not path.is_file():
        raise BankMetadataError(
            "bank metadata is missing at %s; run "
            "`bioagent-bench generate --force --seed <seed>`" % path
        )
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BankMetadataError(
            "could not read bank metadata at %s: %s; run "
            "`bioagent-bench generate --force --seed <seed>`" % (path, exc)
        )
    if not isinstance(metadata, dict):
        raise BankMetadataError(
            "bank metadata at %s is not a JSON object; run "
            "`bioagent-bench generate --force --seed <seed>`" % path
        )
    return metadata


def validate_bank(root, expected_seed=None):
    """Validate bank metadata and optionally the stored generation seed.

    ``expected_seed=None`` validates only the format version, which is what
    read-only listing needs.  Run paths pass the requested seed so a bank made
    with a different seed is rejected instead of silently reused.
    """
    metadata = read_bank_metadata(root)
    version = metadata.get("version")
    if version != BANK_VERSION:
        raise BankMetadataError(
            "bank at %s has version %r, expected %r; run "
            "`bioagent-bench generate --force --seed <seed>`"
            % (root, version, BANK_VERSION)
        )
    if expected_seed is not None:
        stored_seed = metadata.get("seed")
        if stored_seed != expected_seed:
            raise BankMetadataError(
                "bank at %s was generated with seed %r, but seed %r was "
                "requested; run `bioagent-bench generate --force --seed %r`"
                % (root, stored_seed, expected_seed, expected_seed)
            )
    return metadata


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------


def _parse_mutations(value):
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.upper() in ("WT", "WILDTYPE", "WILD-TYPE"):
        return []
    parts = []
    for chunk in text.replace(",", "+").split("+"):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def _solve_linear_system(matrix, vector):
    size = len(vector)
    augmented = [list(matrix[row]) + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            continue
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    return [augmented[row][size] for row in range(size)]


def _fit_additive_baseline(train_rows):
    vocabulary = {}
    for row in train_rows:
        for mutation in _parse_mutations(row.get("mutations")):
            if mutation not in vocabulary:
                vocabulary[mutation] = len(vocabulary)
    size = len(vocabulary) + 1
    matrix = [[0.0] * size for _ in range(size)]
    vector = [0.0] * size
    for row in train_rows:
        features = [1.0] + [0.0] * len(vocabulary)
        for mutation in _parse_mutations(row.get("mutations")):
            features[1 + vocabulary[mutation]] = 1.0
        target = float(row["fitness"])
        for i in range(size):
            if features[i] == 0.0:
                continue
            vector[i] += features[i] * target
            for j in range(size):
                matrix[i][j] += features[i] * features[j]
    for index in range(1, size):
        matrix[index][index] += 1e-6
    return vocabulary, _solve_linear_system(matrix, vector)


def _additive_predict(vocabulary, coefficients, mutations):
    value = coefficients[0]
    for mutation in _parse_mutations(mutations):
        index = vocabulary.get(mutation)
        if index is not None:
            value += coefficients[1 + index]
    return value


def _adjusted_rand_index(labels_true, labels_pred):
    count = len(labels_true)
    if count < 2:
        return 1.0
    true_counts = {}
    pred_counts = {}
    pair_counts = {}
    for true_label, pred_label in zip(labels_true, labels_pred):
        true_counts[true_label] = true_counts.get(true_label, 0) + 1
        pred_counts[pred_label] = pred_counts.get(pred_label, 0) + 1
        key = (true_label, pred_label)
        pair_counts[key] = pair_counts.get(key, 0) + 1
    sum_pairs = sum(math.comb(value, 2) for value in pair_counts.values())
    sum_true = sum(math.comb(value, 2) for value in true_counts.values())
    sum_pred = sum(math.comb(value, 2) for value in pred_counts.values())
    total_pairs = math.comb(count, 2)
    expected = (sum_true * sum_pred) / float(total_pairs) if total_pairs else 0.0
    maximum = 0.5 * (sum_true + sum_pred)
    if maximum == expected:
        return 1.0
    return (sum_pairs - expected) / (maximum - expected)


def _hamming_distance(first, second):
    return sum(1 for left, right in zip(first, second) if left != right)


def _cluster_from_edges(count, edges):
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for first, second in edges:
        union(first, second)
    return [find(index) for index in range(count)]


def _pairwise_distances(sequences):
    distances = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            distances.append((i, j, _hamming_distance(sequences[i], sequences[j])))
    return distances


def _cluster_from_distances(count, distances, cutoff):
    edges = [(i, j) for i, j, distance in distances if distance <= cutoff]
    return _cluster_from_edges(count, edges)


def _cluster_by_cutoff(sequences, cutoff):
    return _cluster_from_distances(
        len(sequences),
        _pairwise_distances(sequences),
        cutoff,
    )


def _clusterings_by_cutoff(sequences):
    """Return single-linkage labelings for every cutoff from 1 to max length.

    Edges are added incrementally in distance order, so the union-find work is
    proportional to the number of pairs rather than pairs times cutoffs.
    """
    count = len(sequences)
    max_length = max(len(sequence) for sequence in sequences)
    edges = _pairwise_distances(sequences)
    edges.sort(key=lambda item: item[2])
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    result = []
    edge_index = 0
    for cutoff in range(1, max_length + 1):
        while edge_index < len(edges) and edges[edge_index][2] <= cutoff:
            first, second, _ = edges[edge_index]
            union(first, second)
            edge_index += 1
        result.append([find(index) for index in range(count)])
    return result


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


def _validate_submission(task_id, obj, input_rows, truth_rows, check_shortlist=True):
    if not isinstance(obj, dict):
        return False, "output JSON must be an object", None, None
    predictions = obj.get("predictions")
    if not isinstance(predictions, list):
        return False, "output JSON must contain a predictions list", None, None

    input_ids = [row["id"] for row in input_rows]
    input_id_set = set(input_ids)
    prediction_map = {}

    for index, item in enumerate(predictions):
        if not isinstance(item, dict):
            return False, "predictions[%d] is not an object" % index, None, None
        if "id" not in item or "value" not in item:
            return False, "predictions[%d] must have id and value" % index, None, None
        row_id = item["id"]
        if not isinstance(row_id, str):
            return False, "predictions[%d].id must be a string" % index, None, None
        if row_id not in input_id_set:
            return False, "prediction id %r is not in input.csv" % row_id, None, None
        if row_id in prediction_map:
            return False, "duplicate prediction id %r" % row_id, None, None
        value = item["value"]
        if task_id == "B03":
            if not isinstance(value, str) or not value.strip():
                return False, "B03 prediction value for %r must be a non-empty string" % row_id, None, None
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False, "prediction value for %r must be numeric" % row_id, None, None
            try:
                value = float(value)
            except (OverflowError, ValueError):
                return False, "prediction value for %r is not finite" % row_id, None, None
            if not math.isfinite(value):
                return False, "prediction value for %r is not finite" % row_id, None, None
            if value < 0.0 or value > 1.0:
                return False, "prediction value for %r is outside [0, 1]" % row_id, None, None
        prediction_map[row_id] = value

    missing = [row_id for row_id in input_ids if row_id not in prediction_map]
    if missing:
        return False, "missing predictions for %d input ids (e.g. %r)" % (len(missing), missing[0]), None, None

    shortlist = obj.get("shortlist")
    if not check_shortlist:
        return True, "", prediction_map, shortlist

    if task_id in ("B04", "B05", "B06"):
        if not isinstance(shortlist, list):
            return False, "shortlist is required and must be a list", None, None
        if not all(isinstance(item, str) for item in shortlist):
            return False, "shortlist entries must be strings", None, None
        if len(set(shortlist)) != len(shortlist):
            return False, "shortlist contains duplicate ids", None, None
        for row_id in shortlist:
            if row_id not in input_id_set:
                return False, "shortlist id %r is not in input.csv" % row_id, None, None

        if task_id == "B04":
            if len(shortlist) != 20:
                return False, "B04 shortlist must contain exactly 20 ids", None, None
            gene_of = {row["id"]: row["gene"] for row in input_rows}
            counts = {}
            for row_id in shortlist:
                gene = gene_of[row_id]
                counts[gene] = counts.get(gene, 0) + 1
            if any(count > 2 for count in counts.values()):
                return False, "B04 shortlist may contain at most 2 ids per gene", None, None

        elif task_id == "B05":
            target_of = {row["id"]: row["target"] for row in input_rows}
            targets = set(target_of.values())
            if len(shortlist) != len(targets):
                return False, "B05 shortlist must contain exactly one pose per target (%d ids)" % len(targets), None, None
            seen = {}
            for row_id in shortlist:
                target = target_of[row_id]
                if target in seen:
                    return False, "B05 shortlist has more than one pose for target %r" % target, None, None
                seen[target] = row_id
            if set(seen) != targets:
                return False, "B05 shortlist must cover every test target exactly once", None, None

        elif task_id == "B06":
            if len(shortlist) != 40:
                return False, "B06 shortlist must contain exactly 40 ids", None, None
            target_of = {row["id"]: row["target"] for row in input_rows}
            targets = set(target_of.values())
            counts = {}
            for row_id in shortlist:
                target = target_of[row_id]
                counts[target] = counts.get(target, 0) + 1
            if set(counts) != targets:
                return False, "B06 shortlist must cover every test target", None, None
            if any(count != 10 for count in counts.values()):
                return False, "B06 shortlist must contain exactly 10 ids per test target", None, None
    else:
        if shortlist is not None:
            if not isinstance(shortlist, list) or not all(isinstance(item, str) for item in shortlist):
                return False, "shortlist, if present, must be a list of strings", None, None

    return True, "", prediction_map, shortlist


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_b01(prediction_map, shortlist, input_rows, truth_rows, train_rows):
    truth = {row["id"]: float(row["binding"]) for row in truth_rows}
    ids = [row["id"] for row in input_rows]
    model_rmse = _rmse([prediction_map[row_id] for row_id in ids], [truth[row_id] for row_id in ids])
    baseline = _mean([float(row["binding"]) for row in train_rows])
    baseline_rmse = _rmse([baseline] * len(ids), [truth[row_id] for row_id in ids])
    score = 0.0 if baseline_rmse <= 0.0 else 100.0 * (baseline_rmse - model_rmse) / baseline_rmse
    score = _clamp(score, 0.0, 100.0)
    return score, {
        "rmse": round(model_rmse, 6),
        "baseline_rmse": round(baseline_rmse, 6),
        "baseline_mean": round(baseline, 6),
        "n": len(ids),
    }


def _score_b02(prediction_map, shortlist, input_rows, truth_rows, train_rows):
    truth = {row["id"]: float(row["fitness"]) for row in truth_rows}
    vocabulary, coefficients = _fit_additive_baseline(train_rows)
    ids = [row["id"] for row in input_rows]
    baseline_predictions = [
        _clamp(_additive_predict(vocabulary, coefficients, row["mutations"]), 0.0, 1.0)
        for row in input_rows
    ]
    baseline_rmse = _rmse(baseline_predictions, [truth[row_id] for row_id in ids])
    model_rmse = _rmse([prediction_map[row_id] for row_id in ids], [truth[row_id] for row_id in ids])
    score = 0.0 if baseline_rmse <= 0.0 else 100.0 * (baseline_rmse - model_rmse) / baseline_rmse
    score = _clamp(score, 0.0, 100.0)
    return score, {
        "rmse": round(model_rmse, 6),
        "baseline_rmse": round(baseline_rmse, 6),
        "n": len(ids),
    }


def _score_b03(prediction_map, shortlist, input_rows, truth_rows, train_rows):
    truth = {row["id"]: row["family"] for row in truth_rows}
    ids = [row["id"] for row in input_rows]
    true_labels = [truth[row_id] for row_id in ids]
    predicted_labels = [prediction_map[row_id] for row_id in ids]
    model_ari = _adjusted_rand_index(true_labels, predicted_labels)

    sequences = [row["sequence"] for row in input_rows]
    best_ari = -1.0
    best_cutoff = None
    for cutoff, labels in enumerate(_clusterings_by_cutoff(sequences), start=1):
        ari = _adjusted_rand_index(true_labels, labels)
        if ari > best_ari:
            best_ari = ari
            best_cutoff = cutoff

    denominator = 1.0 - best_ari
    calibration_error = denominator <= 1e-9
    if calibration_error:
        score = 0.0
    else:
        score = 100.0 * (model_ari - best_ari) / denominator
    score = _clamp(score, 0.0, 100.0)
    return score, {
        "ari": round(model_ari, 6),
        "baseline_ari": round(best_ari, 6),
        "best_cutoff": best_cutoff,
        "denominator": round(denominator, 6),
        "calibration_error": calibration_error,
        "n": len(ids),
    }


def _b04_gene_means(train_rows):
    gene_values = {}
    all_values = []
    for row in train_rows:
        text = (row.get("dependency") or "").strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        gene_values.setdefault(row["gene"], []).append(value)
        all_values.append(value)
    gene_mean = {gene: _mean(values) for gene, values in gene_values.items()}
    global_mean = _mean(all_values) if all_values else 0.5
    return gene_mean, global_mean


def _b04_baseline_panel(input_rows, train_rows):
    """Exactly 20 ids ranked by train-only gene means, at most 2 per gene."""
    gene_mean, global_mean = _b04_gene_means(train_rows)
    gene_of = {row["id"]: row["gene"] for row in input_rows}
    by_gene = {}
    for row in input_rows:
        by_gene.setdefault(row["gene"], []).append(row["id"])
    candidates = []
    for gene, gene_ids in by_gene.items():
        ordered = sorted(gene_ids, key=lambda row_id: (-gene_mean.get(gene, global_mean), row_id))
        candidates.extend(ordered[:2])
    candidates.sort(key=lambda row_id: (-gene_mean.get(gene_of[row_id], global_mean), row_id))
    return candidates[:20]


def _b04_oracle_panel(input_rows, truth_rows):
    """Best feasible 20-pair panel using sealed truth, at most 2 per gene."""
    truth = {row["id"]: float(row["dependency"]) for row in truth_rows}
    by_gene = {}
    for row in input_rows:
        by_gene.setdefault(row["gene"], []).append(row["id"])
    candidates = []
    for gene_ids in by_gene.values():
        ordered = sorted(gene_ids, key=lambda row_id: (-truth[row_id], row_id))
        candidates.extend(ordered[:2])
    candidates.sort(key=lambda row_id: (-truth[row_id], row_id))
    return candidates[:20]


def _score_b04(prediction_map, shortlist, input_rows, truth_rows, train_rows):
    truth = {row["id"]: float(row["dependency"]) for row in truth_rows}
    gene_mean, global_mean = _b04_gene_means(train_rows)

    ids = [row["id"] for row in input_rows]
    baseline_predictions = [gene_mean.get(row["gene"], global_mean) for row in input_rows]
    baseline_rmse = _rmse(baseline_predictions, [truth[row_id] for row_id in ids])
    model_rmse = _rmse([prediction_map[row_id] for row_id in ids], [truth[row_id] for row_id in ids])
    forecast = 0.0 if baseline_rmse <= 0.0 else _clamp((baseline_rmse - model_rmse) / baseline_rmse, 0.0, 1.0)

    selected_ids = list(shortlist or [])
    selected_mean = _mean([truth[row_id] for row_id in selected_ids])
    baseline_panel = _b04_baseline_panel(input_rows, train_rows)
    baseline_panel_mean = _mean([truth[row_id] for row_id in baseline_panel])
    oracle_panel = _b04_oracle_panel(input_rows, truth_rows)
    oracle_mean = _mean([truth[row_id] for row_id in oracle_panel])

    panel_denominator = oracle_mean - baseline_panel_mean
    panel_calibration_error = panel_denominator <= 1e-9
    if panel_calibration_error:
        panel_skill = 0.0
    else:
        panel_skill = _clamp((selected_mean - baseline_panel_mean) / panel_denominator, 0.0, 1.0)
    panel_ratio = 0.0 if oracle_mean <= 0.0 else _clamp(selected_mean / oracle_mean, 0.0, 1.0)

    score = 100.0 * (0.5 * forecast + 0.5 * panel_skill)
    return score, {
        "rmse": round(model_rmse, 6),
        "baseline_rmse": round(baseline_rmse, 6),
        "forecast": round(forecast, 6),
        "panel_skill": round(panel_skill, 6),
        "selected_mean": round(selected_mean, 6),
        "baseline_panel_mean": round(baseline_panel_mean, 6),
        "oracle_mean": round(oracle_mean, 6),
        "panel_ratio": round(panel_ratio, 6),
        "panel_denominator": round(panel_denominator, 6),
        "panel_calibration_error": panel_calibration_error,
        "forecast_weight": 0.5,
        "panel_weight": 0.5,
        "n": len(ids),
    }


def _score_b05(prediction_map, shortlist, input_rows, truth_rows, train_rows):
    truth = {row["id"]: float(row["quality"]) for row in truth_rows}
    targets = sorted(set(row["target"] for row in input_rows))
    target_of = {row["id"]: row["target"] for row in input_rows}
    selected_by_target = {}
    for row_id in shortlist or []:
        selected_by_target[target_of[row_id]] = row_id

    skills = []
    baseline_values = []
    oracle_values = []
    selected_values = []
    degenerate = 0
    for target in targets:
        rows = [row for row in input_rows if row["target"] == target]
        baseline = min(rows, key=lambda row: (float(row["energy"]), row["id"]))
        baseline_quality = truth[baseline["id"]]
        oracle_quality = max(truth[row["id"]] for row in rows)
        selected_id = selected_by_target.get(target, baseline["id"])
        selected_quality = truth[selected_id]
        gap = oracle_quality - baseline_quality
        if gap <= 1e-9:
            degenerate += 1
        else:
            skill = (selected_quality - baseline_quality) / gap
            skills.append(_clamp(skill, -1.0, 1.0))
        baseline_values.append(baseline_quality)
        oracle_values.append(oracle_quality)
        selected_values.append(selected_quality)

    mean_skill = _mean(skills)
    score = 100.0 * _clamp(mean_skill, 0.0, 1.0)
    return score, {
        "selected_mean": round(_mean(selected_values), 6),
        "baseline_mean": round(_mean(baseline_values), 6),
        "oracle_mean": round(_mean(oracle_values), 6),
        "mean_skill": round(mean_skill, 6),
        "min_skill": round(min(skills), 6) if skills else 0.0,
        "max_skill": round(max(skills), 6) if skills else 0.0,
        "degenerate_targets": degenerate,
        "n_targets": len(targets),
    }


def _score_b06(prediction_map, shortlist, input_rows, truth_rows, train_rows):
    truth = {row["id"]: int(float(row["binder"])) for row in truth_rows}
    rmsd = {row["id"]: float(row["scaffold_rmsd"]) for row in input_rows}
    targets = sorted(set(row["target"] for row in input_rows))

    baseline_hits = 0
    oracle_hits = 0
    for target in targets:
        ids = [row["id"] for row in input_rows if row["target"] == target]
        baseline = sorted(ids, key=lambda row_id: (rmsd[row_id], row_id))[:10]
        baseline_hits += sum(truth[row_id] for row_id in baseline)
        oracle = sorted(ids, key=lambda row_id: (-truth[row_id], row_id))[:10]
        oracle_hits += sum(truth[row_id] for row_id in oracle)

    model_hits = sum(truth[row_id] for row_id in shortlist)
    denominator = oracle_hits - baseline_hits
    score = 0.0 if denominator <= 0 else 100.0 * (model_hits - baseline_hits) / denominator
    score = _clamp(score, 0.0, 100.0)
    return score, {
        "model_hits": model_hits,
        "baseline_hits": baseline_hits,
        "oracle_hits": oracle_hits,
        "n_targets": len(targets),
    }


def _score_task(task_id, prediction_map, shortlist, input_rows, truth_rows, train_rows):
    if task_id == "B01":
        return _score_b01(prediction_map, shortlist, input_rows, truth_rows, train_rows)
    if task_id == "B02":
        return _score_b02(prediction_map, shortlist, input_rows, truth_rows, train_rows)
    if task_id == "B03":
        return _score_b03(prediction_map, shortlist, input_rows, truth_rows, train_rows)
    if task_id == "B04":
        return _score_b04(prediction_map, shortlist, input_rows, truth_rows, train_rows)
    if task_id == "B05":
        return _score_b05(prediction_map, shortlist, input_rows, truth_rows, train_rows)
    if task_id == "B06":
        return _score_b06(prediction_map, shortlist, input_rows, truth_rows, train_rows)
    raise ValueError("unknown task id %r" % task_id)


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


def _clean_env():
    """Start from an allowlist rather than inheriting credentials or startup hooks."""
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    # Empty values discard any inherited proxy credentials. This is not a
    # network firewall; untrusted submissions still need a real offline sandbox.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment[key] = ""
    environment["NO_PROXY"] = environment["no_proxy"] = ""
    return environment


def _prepare_env(grade_dir):
    environment = _clean_env()
    # Prefer the actual venv directory. Do not resolve the symlink: resolving
    # .venv/bin/python selects the base framework Python and loses installed
    # scientific packages.
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
    environment["HOME"] = str(grade_dir / ".home")
    environment["TMPDIR"] = str(grade_dir / ".tmp")
    (grade_dir / ".home").mkdir(parents=True, exist_ok=True)
    (grade_dir / ".tmp").mkdir(parents=True, exist_ok=True)
    return environment


def _copy_pipeline(submission_dir, attempt_dir):
    source = submission_dir / "pipeline"
    if not source.is_dir():
        return False, "submission_dir/pipeline is missing"
    destination = attempt_dir / "pipeline"
    try:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    except (OSError, shutil.Error) as exc:
        return False, "could not copy pipeline: %s" % exc
    return True, ""


def _run_attempt(attempt_dir, input_dir, timeout_seconds, environment):
    output_path = attempt_dir / "output.json"
    environment = dict(environment)
    environment["HOME"] = str(attempt_dir / ".home")
    environment["TMPDIR"] = str(attempt_dir / ".tmp")
    (attempt_dir / ".home").mkdir(parents=True, exist_ok=True)
    (attempt_dir / ".tmp").mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        process = subprocess.Popen(
            ["bash", "pipeline/predict.sh", str(input_dir), str(output_path)],
            cwd=str(attempt_dir),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        return {
            "returncode": None,
            "duration_seconds": round(time.time() - start, 3),
            "stdout_tail": "",
            "stderr_tail": "could not start subprocess: %s" % exc,
            "output_path": output_path,
            "timed_out": False,
        }
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    return {
        "returncode": process.returncode,
        "duration_seconds": round(time.time() - start, 3),
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
        "output_path": output_path,
        "timed_out": timed_out,
    }


def _attempt_record(name, run):
    return {
        "name": name,
        "returncode": run["returncode"],
        "duration_seconds": run["duration_seconds"],
        "timed_out": run["timed_out"],
        "stdout_tail": run["stdout_tail"],
        "stderr_tail": run["stderr_tail"],
    }


def _read_output(path):
    if not path.is_file():
        return None, None, "output file was not created"
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, None, "could not read output: %s" % exc
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return raw, None, "output is not valid UTF-8 JSON: %s" % exc
    return raw, obj, ""


# ---------------------------------------------------------------------------
# Public grading entry point
# ---------------------------------------------------------------------------


def grade_task(task_dir, submission_dir, grade_dir, timeout_seconds=300):
    """Run and grade one submission.

    ``task_dir`` is a generated task directory (``.../tasks/BXX``).
    ``submission_dir`` contains the agent's ``pipeline/`` directory.
    ``grade_dir`` is cleaned and used for fresh attempt directories.
    """
    task_dir = Path(task_dir).resolve()
    submission_dir = Path(submission_dir).resolve()
    grade_dir = Path(grade_dir).resolve()

    result = {
        "task_id": None,
        "score": 0.0,
        "passed_gate": False,
        "reason": "",
        "deterministic": False,
        "batch_invariant": None,
        "metrics": {},
        "attempts": [],
    }

    meta_path = task_dir / "task.json"
    if not meta_path.is_file():
        result["reason"] = "task.json not found in task_dir"
        return result
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["reason"] = "could not read task.json: %s" % exc
        return result
    task_id = meta.get("id")
    if task_id not in TASK_IDS:
        result["reason"] = "unknown task id %r" % task_id
        return result
    if meta.get("version") != BANK_VERSION:
        result["reason"] = (
            "task.json version %r is not supported (expected %r); regenerate "
            "the bank with `bioagent-bench generate --force --seed <seed>`"
            % (meta.get("version"), BANK_VERSION)
        )
        return result
    result["task_id"] = task_id

    try:
        if grade_dir.exists():
            shutil.rmtree(grade_dir)
        grade_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        result["reason"] = "could not prepare grade_dir: %s" % exc
        return result

    private_input_dir = task_dir / "private" / "input"
    try:
        train_rows = _read_csv(private_input_dir / "train.csv")
        input_rows = _read_csv(private_input_dir / "input.csv")
        truth_rows = _read_csv(task_dir / "private" / "truth.csv")
    except OSError as exc:
        result["reason"] = "could not read private task data: %s" % exc
        return result

    if not (submission_dir / "pipeline").is_dir():
        result["reason"] = "submission_dir/pipeline is missing"
        return result

    environment = _prepare_env(grade_dir)
    attempt_results = []
    valid_outputs = []
    failures = []

    for name in ("attempt_1", "attempt_2"):
        attempt_dir = grade_dir / name
        attempt_dir.mkdir(parents=True, exist_ok=True)
        copied, error = _copy_pipeline(submission_dir, attempt_dir)
        if not copied:
            result["reason"] = error
            result["attempts"] = attempt_results
            return result
        run = _run_attempt(attempt_dir, private_input_dir, timeout_seconds, environment)
        attempt_results.append(_attempt_record(name, run))
        raw, obj, error = _read_output(run["output_path"])
        if run["returncode"] != 0:
            failures.append("%s exited with code %r" % (name, run["returncode"]))
            continue
        if error:
            failures.append("%s: %s" % (name, error))
            continue
        valid, reason, prediction_map, shortlist = _validate_submission(
            task_id, obj, input_rows, truth_rows
        )
        if not valid:
            failures.append("%s: %s" % (name, reason))
            continue
        valid_outputs.append({
            "name": name,
            "raw": raw,
            "obj": obj,
            "prediction_map": prediction_map,
            "shortlist": shortlist,
        })

    if len(valid_outputs) < 2:
        result["reason"] = "; ".join(failures) if failures else "both attempts failed validation"
        result["attempts"] = attempt_results
        return result

    first = valid_outputs[0]
    second = valid_outputs[1]
    if first["raw"] == second["raw"]:
        deterministic = True
    else:
        deterministic = _canonical_json(first["obj"]) == _canonical_json(second["obj"])
    if not deterministic:
        failures.append("the two runs produced different outputs")
        result["reason"] = "; ".join(failures)
        result["attempts"] = attempt_results
        return result

    batch_invariant = None
    batch_note = ""
    if task_id != "B03":
        batch_input_dir = grade_dir / "batch_input"
        batch_input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(private_input_dir / "train.csv", batch_input_dir / "train.csv")
        with (private_input_dir / "input.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        with (batch_input_dir / "input.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in reversed(rows):
                writer.writerow(row)

        batch_attempt_dir = grade_dir / "batch_invariance"
        batch_attempt_dir.mkdir(parents=True, exist_ok=True)
        copied, error = _copy_pipeline(submission_dir, batch_attempt_dir)
        if not copied:
            batch_invariant = False
            batch_note = "batch-invariance run could not copy pipeline"
        else:
            run = _run_attempt(batch_attempt_dir, batch_input_dir, timeout_seconds, environment)
            attempt_results.append(_attempt_record("batch_invariance", run))
            raw_batch, obj_batch, error = _read_output(run["output_path"])
            if run["returncode"] != 0:
                batch_invariant = False
                batch_note = "batch-invariance run exited with code %r" % run["returncode"]
            elif error:
                batch_invariant = False
                batch_note = "batch-invariance run: %s" % error
            else:
                valid, reason, batch_predictions, _ = _validate_submission(
                    task_id, obj_batch, input_rows, truth_rows, check_shortlist=False
                )
                if not valid:
                    batch_invariant = False
                    batch_note = "batch-invariance run: %s" % reason
                else:
                    batch_invariant = True
                    for row_id, value in first["prediction_map"].items():
                        if row_id not in batch_predictions:
                            batch_invariant = False
                            batch_note = "batch-invariance run omitted id %r" % row_id
                            break
                        if abs(value - batch_predictions[row_id]) > 1e-8:
                            batch_invariant = False
                            batch_note = "batch-invariance changed prediction for id %r" % row_id
                            break

    if batch_invariant is False:
        result["passed_gate"] = False
        result["score"] = 0.0
        result["deterministic"] = True
        result["batch_invariant"] = False
        result["reason"] = "batch-invariance check failed: " + batch_note
        result["attempts"] = attempt_results
        return result

    score, metrics = _score_task(
        task_id,
        first["prediction_map"],
        first["shortlist"],
        input_rows,
        truth_rows,
        train_rows,
    )

    result["score"] = round(float(score), 6)
    result["passed_gate"] = True
    result["deterministic"] = True
    result["batch_invariant"] = batch_invariant
    result["metrics"] = metrics
    result["attempts"] = attempt_results
    result["reason"] = "ok" if not batch_note else "ok; " + batch_note
    return result
