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

__all__ = ["TASK_IDS", "generate_bank", "grade_task"]

TASK_IDS = ("B01", "B02", "B03", "B04", "B05", "B06")

_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


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
Binding depends on short sequence motifs, family context, and interactions
between motifs. One-hot or k-mer encodings of the sequence are useful; family is
a categorical feature.

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
    length = 30
    ancestor = [rng.choice(_ALPHABET) for _ in range(length)]
    families = ["fam%02d" % i for i in range(1, 9)]
    founders = {}
    for family in families:
        sequence = list(ancestor)
        for _ in range(rng.randint(4, 6)):
            position = rng.randrange(length)
            sequence[position] = rng.choice([aa for aa in _ALPHABET if aa != sequence[position]])
        founders[family] = "".join(sequence)

    def mutate(sequence, rate):
        result = list(sequence)
        for index in range(length):
            if rng.random() < rate:
                result[index] = rng.choice([aa for aa in _ALPHABET if aa != result[index]])
        return "".join(result)

    train_rows = []
    counter = 0
    for family in families:
        for _ in range(40):
            counter += 1
            train_rows.append({
                "id": "ab%05d" % counter,
                "sequence": mutate(founders[family], 0.08),
                "family": family,
            })

    input_rows = []
    input_families = []
    for family in families:
        for _ in range(50):
            counter += 1
            input_rows.append({
                "id": "ab%05d" % counter,
                "sequence": mutate(founders[family], 0.08),
            })
            input_families.append(family)

    # A few exact training rows leak into the input with their original ids.
    leak_count = max(1, int(0.08 * len(input_rows)))
    for index in rng.sample(range(len(input_rows)), leak_count):
        source = rng.choice(train_rows)
        input_rows[index] = {"id": source["id"], "sequence": source["sequence"]}
        input_families[index] = source["family"]

    # A few duplicate sequences within the same family.
    duplicate_count = max(1, int(0.05 * len(input_rows)))
    for index in rng.sample(range(len(input_rows)), duplicate_count):
        same_family = [
            other for other, family in enumerate(input_families)
            if family == input_families[index] and other != index
        ]
        if same_family:
            source_index = rng.choice(same_family)
            input_rows[index]["sequence"] = input_rows[source_index]["sequence"]

    # Keep input ids unique even after leaks.
    seen = set()
    for row in input_rows:
        if row["id"] in seen:
            while True:
                counter += 1
                new_id = "ab%05d" % counter
                if new_id not in seen:
                    break
            row["id"] = new_id
        seen.add(row["id"])

    public_rows = []
    private_rows = []
    private_families = []
    family_counts = {}
    for row, family in zip(input_rows, input_families):
        family_counts[family] = family_counts.get(family, 0) + 1
        if family_counts[family] <= 20:
            public_rows.append(row)
        else:
            private_rows.append(row)
            private_families.append(family)

    truth_rows = [
        {"id": row["id"], "family": family}
        for row, family in zip(private_rows, private_families)
    ]

    return {
        "title": "Antibody family clustering",
        "category": "sequence_clustering",
        "description": "Cluster held-out antibody sequences into their close founder families without using label names.",
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
Assign each held-out antibody sequence to a family. Training data contains
labeled sequences from eight close founder families. The graded input contains
held-out rows from the same families, plus a few duplicate sequences and a few
leaked ids that also appear in training.

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
- `sequence` - 30-residue amino-acid sequence (uppercase one-letter codes).
- `family` - family label (categorical).

`input.csv`:

- `id`, `sequence` - same schema, no `family`.

## Split
Training rows are 40 labeled sequences per family. Development and graded rows
are held-out sequences from the same eight families. The graded input is larger
than the public development input.

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
sealed family labels. The baseline is the best single Hamming-distance cutoff
clustering on the input sequences. If that baseline ARI is below 0.98, the score
is:

    100 * (model_ari - baseline_ari) / (1 - baseline_ari)

clamped to [0, 100]. If the best cutoff baseline is at least 0.98, the score is
`100 * model_ari` with an explicit baseline of zero. Label names are never
compared.
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

The panel component is the mean true dependency of your 20 selected pairs
divided by the mean of the best feasible 20-pair panel (at most 2 per gene),
clamped to [0, 1]. The final score is:

    100 * forecast * panel_ratio

clamped to [0, 100]. Both a good forecast and a good panel are required.
"""


# ---------------------------------------------------------------------------
# B05: docking pose ranking
# ---------------------------------------------------------------------------


def _gen_b05(rng):
    train_targets = ["T%02d" % i for i in range(1, 41)]
    test_targets = ["T%02d" % i for i in range(41, 71)]
    dev_targets = ["T%02d" % i for i in range(71, 81)]

    def make_target_rows(target, prefix, counter_start, labeled):
        rows = []
        truth = []
        offset = rng.gauss(0.0, 0.30)
        for pose in range(1, 9):
            energy = rng.gauss(-8.0, 2.0)
            shape = rng.uniform(0.0, 1.0)
            hbond = rng.randint(0, 10)
            buried = rng.uniform(0.0, 1.0)
            clash = rng.randint(0, 5)
            if rng.random() < 0.25:
                energy -= 2.0
                clash += rng.randint(2, 4)
            quality = _sigmoid(
                2.0 * shape
                + 0.15 * hbond
                - 0.35 * clash
                + 0.8 * buried
                - 0.25 * abs(energy + 8.0)
                + offset
                + rng.gauss(0.0, 0.15)
            )
            row = {
                "id": "%s%05d" % (prefix, counter_start + pose),
                "target": target,
                "pose_rank": str(pose),
                "energy": "%.4f" % energy,
                "shape_complementarity": "%.4f" % shape,
                "hbond_count": str(hbond),
                "buried_sasa": "%.4f" % buried,
                "clash_count": str(clash),
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
        counter += 8

    public_rows = []
    for target in dev_targets:
        rows, _ = make_target_rows(target, "dv", counter, False)
        public_rows.extend(rows)
        counter += 8

    private_rows = []
    truth_rows = []
    for target in test_targets:
        rows, truth = make_target_rows(target, "dh", counter, False)
        private_rows.extend(rows)
        truth_rows.extend(truth)
        counter += 8

    return {
        "title": "Docking pose ranking",
        "category": "structural_biology",
        "description": "Rank docking poses for unseen targets and select one pose per target.",
        "batch_invariant": True,
        "train_fields": [
            "id", "target", "pose_rank", "energy", "shape_complementarity",
            "hbond_count", "buried_sasa", "clash_count", "quality",
        ],
        "train_rows": train_rows,
        "input_fields": [
            "id", "target", "pose_rank", "energy", "shape_complementarity",
            "hbond_count", "buried_sasa", "clash_count",
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
exactly one pose per unseen target. Training data covers 40 targets with eight
poses each. The graded input contains 30 unseen targets with eight poses each.

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
- `pose_rank` - pose index within the target (1-8).
- `energy` - docking energy (lower is not always better).
- `shape_complementarity` - shape score in [0, 1].
- `hbond_count` - number of hydrogen bonds (integer).
- `buried_sasa` - buried surface area fraction in [0, 1].
- `clash_count` - steric clash count (integer).
- `quality` - target pose quality in [0, 1].

`input.csv`:

- same columns except `quality`.

## Split
Training rows come from 40 targets. Development and graded rows come from
unseen targets. The graded input is larger than the public development input.

## Signal
Quality depends on shape complementarity, hydrogen bonding, buried surface,
clashes, and energy. The minimum-energy pose is a useful but imperfect
baseline; some low-energy poses are decoys with clashes.

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
The baseline selects the minimum-energy pose per target. The oracle selects the
highest-quality pose per target. The score is:

    100 * (selected_mean - baseline_mean) / (oracle_mean - baseline_mean)

clamped to [0, 100]. Selecting the minimum-energy pose scores 0; selecting the
best pose for every target scores 100.
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
        "title": data["title"],
        "category": data["category"],
        "description": data["description"],
        "batch_invariant": data["batch_invariant"],
    }
    (task_dir / "task.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def generate_bank(root, seed=2026):
    """Generate the six-task synthetic bank beneath ``root``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tasks_root = root / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)

    generators = {
        "B01": _gen_b01,
        "B02": _gen_b02,
        "B03": _gen_b03,
        "B04": _gen_b04,
        "B05": _gen_b05,
        "B06": _gen_b06,
    }

    for index, task_id in enumerate(TASK_IDS):
        rng = random.Random(seed + 1000 * (index + 1))
        data = generators[task_id](rng)
        _write_task(tasks_root / task_id, task_id, data)


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


def _cluster_by_cutoff(sequences, cutoff):
    edges = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            if _hamming_distance(sequences[i], sequences[j]) <= cutoff:
                edges.append((i, j))
    return _cluster_from_edges(len(sequences), edges)


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
        _additive_predict(vocabulary, coefficients, row["mutations"])
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
    max_length = max(len(sequence) for sequence in sequences)
    distances = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            distances.append((i, j, _hamming_distance(sequences[i], sequences[j])))
    best_ari = -1.0
    best_cutoff = None
    for cutoff in range(1, max_length + 1):
        edges = [(i, j) for i, j, distance in distances if distance <= cutoff]
        labels = _cluster_from_edges(len(sequences), edges)
        ari = _adjusted_rand_index(true_labels, labels)
        if ari > best_ari:
            best_ari = ari
            best_cutoff = cutoff

    if best_ari >= 0.98:
        score = 100.0 * model_ari
        baseline_ari = 0.0
        mode = "direct"
    else:
        score = 100.0 * (model_ari - best_ari) / (1.0 - best_ari)
        baseline_ari = best_ari
        mode = "rescaled"
    score = _clamp(score, 0.0, 100.0)
    return score, {
        "ari": round(model_ari, 6),
        "baseline_ari": round(baseline_ari, 6),
        "best_cutoff": best_cutoff,
        "mode": mode,
        "n": len(ids),
    }


def _score_b04(prediction_map, shortlist, input_rows, truth_rows, train_rows):
    truth = {row["id"]: float(row["dependency"]) for row in truth_rows}

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

    ids = [row["id"] for row in input_rows]
    baseline_predictions = [gene_mean.get(row["gene"], global_mean) for row in input_rows]
    baseline_rmse = _rmse(baseline_predictions, [truth[row_id] for row_id in ids])
    model_rmse = _rmse([prediction_map[row_id] for row_id in ids], [truth[row_id] for row_id in ids])
    forecast = 0.0 if baseline_rmse <= 0.0 else _clamp((baseline_rmse - model_rmse) / baseline_rmse, 0.0, 1.0)

    selected_mean = _mean([truth[row_id] for row_id in shortlist])

    by_gene = {}
    for row in input_rows:
        by_gene.setdefault(row["gene"], []).append(row["id"])
    candidates = []
    for gene_ids in by_gene.values():
        ordered = sorted(gene_ids, key=lambda row_id: (-truth[row_id], row_id))
        candidates.extend(ordered[:2])
    candidates.sort(key=lambda row_id: (-truth[row_id], row_id))
    oracle_ids = candidates[:20]
    oracle_mean = _mean([truth[row_id] for row_id in oracle_ids])
    panel_ratio = 0.0 if oracle_mean <= 0.0 else _clamp(selected_mean / oracle_mean, 0.0, 1.0)

    score = _clamp(100.0 * forecast * panel_ratio, 0.0, 100.0)
    return score, {
        "rmse": round(model_rmse, 6),
        "baseline_rmse": round(baseline_rmse, 6),
        "forecast": round(forecast, 6),
        "selected_mean": round(selected_mean, 6),
        "oracle_mean": round(oracle_mean, 6),
        "panel_ratio": round(panel_ratio, 6),
        "n": len(ids),
    }


def _score_b05(prediction_map, shortlist, input_rows, truth_rows, train_rows):
    truth = {row["id"]: float(row["quality"]) for row in truth_rows}
    targets = sorted(set(row["target"] for row in input_rows))

    baseline_ids = []
    oracle_values = []
    for target in targets:
        rows = [row for row in input_rows if row["target"] == target]
        baseline = min(rows, key=lambda row: (float(row["energy"]), row["id"]))
        baseline_ids.append(baseline["id"])
        oracle_values.append(max(truth[row["id"]] for row in rows))

    baseline_mean = _mean([truth[row_id] for row_id in baseline_ids])
    oracle_mean = _mean(oracle_values)
    selected_mean = _mean([truth[row_id] for row_id in shortlist])
    denominator = oracle_mean - baseline_mean
    score = 0.0 if denominator <= 0.0 else 100.0 * (selected_mean - baseline_mean) / denominator
    score = _clamp(score, 0.0, 100.0)
    return score, {
        "selected_mean": round(selected_mean, 6),
        "baseline_mean": round(baseline_mean, 6),
        "oracle_mean": round(oracle_mean, 6),
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
