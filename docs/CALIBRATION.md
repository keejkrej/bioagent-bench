# v0.2 calibration

This page documents the v0.2 task changes, the in-process reference evaluator,
and the observed calibration envelope. It is a development record, not a
leaderboard result. All numbers below were produced locally with the frozen
synthetic seeds listed here.

> **v0.2 example run.** A single one-replicate Haiku 5.5 run on the v0.2 bank
> (version 0.2.0, seed 2026) scored 59.640212/100 with 6/6 gates passing. See
> [RESULTS_V0_2.md](RESULTS_V0_2.md) for the run configuration, per-task
> metrics, and a candid difficulty assessment. It is one replicate and is not
> numerically comparable with the historical v0.1 run.

## What changed in v0.2

### B03: held-out daughter subfamilies

- Training exposes eight parent lineages with 32 labeled sequences each.
- Each parent has two unseen daughter founders, so the sealed test partition
  has 16 true subfamilies.
- Both the public development input and the larger graded input contain all 16
  daughters. The split is stratified per daughter rather than sequential.
- After the stratified split, the pooled public/private rows are globally
  shuffled and every test row receives an independent random non-sequential id
  (10000-99999). Ids therefore carry no daughter, split, or row-order
  information; positional blocks and sorted-id blocks do not recover the
  partition.
- Sequences are 48-60 residues. Daughters differ at disjoint lineage positions;
  within-daughter mutations are applied across the sequence, including the
  lineage positions, so the two-medoids solution is not exact.
- There are no exact train/test sequence or id leaks. A small number of
  within-daughter duplicate sequences with fresh ids are retained.
- Scoring always uses the baseline-rescaled ARI:
  `100 * (model_ari - baseline_ari) / (1 - baseline_ari)`, clamped to [0, 100].
  The baseline is the best single Hamming-distance cutoff clustering. If the
  denominator is degenerate, the task reports a calibration error instead of
  switching to a direct ARI score. ARI is invariant to label names.

### B05: target-relative pose transfer

- Training covers 50 targets with 12 poses each; development and graded inputs
  use disjoint unseen targets. Every split contains all five `target_type`
  regimes.
- Raw feature offsets and scales are strongly target-dependent. Labels are
  generated from within-target normalized features, with different
  coefficients and interactions per `target_type`.
- The minimum-energy pose is a decoy with clashes and poor shape/buried-surface
  complementarity, so the minimum-energy baseline is imperfect.
- Scoring uses the mean per-target normalized skill:
  `(selected_quality - baseline_quality) / (oracle_quality - baseline_quality)`.
  Per-target skills are clamped to [-1, 1] before averaging to stop targets with
  tiny baseline-to-oracle gaps from dominating. The mean is then clamped to
  [0, 1]. Targets with a non-positive gap are reported as degenerate and
  excluded from the mean.

### B04: baseline-aligned components

- The multiplicative `forecast * panel_ratio` score is replaced by an equal
  additive combination:
  `100 * (0.5 * forecast + 0.5 * panel_skill)`.
- `forecast` is unchanged: normalized RMSE improvement over the train-only
  gene-mean baseline.
- `panel_skill` compares the selected 20-pair panel with a baseline panel and
  the feasible truth oracle:
  `(selected_mean - baseline_panel_mean) / (oracle_mean - baseline_panel_mean)`,
  clamped to [0, 1].
- The baseline panel is exactly 20 ids selected by train-only gene-mean
  predictions, at most 2 per gene, with deterministic id tie-breaking. The
  oracle panel is the existing feasible truth panel.
- The scorer returns `forecast`, `panel_skill`, selected/baseline/oracle means,
  `panel_ratio` as an optional diagnostic, the 0.5/0.5 weights, and the panel
  denominator.

The arithmetic identities hold subject only to the documented clamps:

| Predictions | Panel | Score |
|---|---|---:|
| baseline gene means | baseline panel | 0 |
| truth | oracle panel | 100 |
| truth | baseline panel | 50 |
| baseline gene means | oracle panel | 50 |

### B06: train-only model selection

- The reference method selects between a linear logistic model and a quadratic
  logistic model by leave-one-target-out cross-validation on the training
  targets only. The private rows and sealed truth are never used for selection.
- The quadratic model is selected on one of the five audited seeds and does not
  materially improve the reference envelope. The simple interface-area/charge
  heuristic remains the stronger comparator on every audited seed.
- This is a deliberate negative result: the reference is not tuned to the
  private score, and the simple heuristic is documented as the stronger
  baseline-relative comparator for B06.

## Reference evaluator

The reference evaluator runs in-process. It consumes exactly the same generated
rows as normal grading and uses the production `_validate_submission` and
`_score_task` functions. It does not start an agent, gateway call, or
subprocess.

```bash
.venv/bin/python -m bioagent_bench reference \
  --seeds 2026,2027,2028 \
  --output .cache/reference-v0_2.json
```

`--seeds` accepts any comma-separated integer list. `--tasks` optionally limits
the task ids. The JSON report contains raw per-seed records, per-task/method
median/min/max summaries, and a macro summary per method.

Each record includes a `calibration` block. A seed is calibrated only when all
four methods are valid, the baseline scores exactly 0, the oracle scores exactly
100, and no scorer reports a degenerate calibration error. `evaluate` fails fast
with `CalibrationError` before aggregating an invalid record. The baseline=0 and
oracle=100 identities therefore apply to nondegenerate calibrated seeds, not to
arbitrary seeds; a new seed must be audited before its envelope is quoted.

Methods:

- `baseline`: the documented task baseline.
- `simple`: a deliberately basic heuristic. For B05 this is a raw global
  pointwise ridge model; for B06 it is an interface-area/charge heuristic.
- `reference`: a conventional non-agent ML method. B01 uses regularized
  position/k-mer ridge; B02 uses additive plus position-pair ridge; B03 uses
  lineage nearest-centroid assignment plus within-lineage two-medoids; B04 uses
  a regularized gene/covariate logit model plus a constrained predicted panel;
  B05 uses a target-relative pairwise difference-feature ranker; B06 uses
  train-only target-CV selection between a regularized logistic model and a
  quadratic logistic model, plus a quota.
- `oracle`: sealed truth, used only to verify the 100-point ceiling.

## Calibration and audit seeds

- Calibration seeds: **2026, 2027, 2028**.
- Audit seeds: **3026, 4026**.

The unit tests use the three calibration seeds. The five-seed sweep below is the
recorded calibration/audit envelope. All five seeds pass the calibration
invariants (every method valid, baseline exactly 0, oracle exactly 100, no
degenerate calibration flags).

## Observed envelope

Command:

```bash
.venv/bin/python -m bioagent_bench reference \
  --seeds 2026,2027,2028,3026,4026 \
  --output .cache/reference-v0_2.json
```

Per-task scores (median, min, max over the five seeds):

| Task | baseline | simple | reference | oracle |
|---|---:|---:|---:|---:|
| B01 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 42.812 / 35.823 / 50.078 | 100 / 100 / 100 |
| B02 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 67.864 / 64.542 / 69.333 | 100 / 100 / 100 |
| B03 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 65.524 / 55.702 / 88.645 | 100 / 100 / 100 |
| B04 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 25.605 / 13.758 / 36.874 | 100 / 100 / 100 |
| B05 | 0.000 / 0.000 / 0.000 | 66.432 / 57.287 / 75.194 | 78.288 / 67.916 / 88.827 | 100 / 100 / 100 |
| B06 | 0.000 / 0.000 / 0.000 | 40.909 / 35.294 / 60.000 | 36.364 / 23.529 / 45.000 | 100 / 100 / 100 |

Macro over the six tasks (median, min, max over the five seeds):

| Method | median | min | max |
|---|---:|---:|---:|
| baseline | 0.000 | 0.000 | 0.000 |
| simple | 19.199 | 16.571 | 19.548 |
| reference | 51.293 | 50.761 | 58.214 |
| oracle | 100.000 | 100.000 | 100.000 |

Per-seed reference scores:

| Task | 2026 | 2027 | 2028 | 3026 | 4026 |
|---|---:|---:|---:|---:|---:|
| B01 | 50.078 | 35.823 | 42.812 | 45.424 | 39.815 |
| B02 | 65.482 | 64.542 | 69.333 | 69.137 | 67.864 |
| B03 | 65.524 | 55.702 | 82.879 | 88.645 | 62.801 |
| B04 | 13.758 | 36.874 | 35.435 | 25.605 | 21.729 |
| B05 | 67.916 | 78.288 | 86.825 | 70.624 | 88.827 |
| B06 | 45.000 | 36.364 | 32.000 | 40.909 | 23.529 |

### B03 mechanism diagnostics

| Seed | nearest-parent ARI | best-cutoff ARI | two-medoids ARI |
|---|---:|---:|---:|
| 2026 | 0.625698 | 0.599206 | 0.861821 |
| 2027 | 0.625698 | 0.620736 | 0.831993 |
| 2028 | 0.625698 | 0.610461 | 0.933305 |
| 3026 | 0.625698 | 0.605971 | 0.955257 |
| 4026 | 0.625698 | 0.612856 | 0.855987 |

The nearest-parent diagnostic is weak on every audited seed, the best global
single-cutoff baseline stays below 0.95, and the lineage-plus-two-medoids
reference improves materially without reaching 1.0. Positional blocks
(`row_index // 21`) and sorted-id blocks score ARI near 0 on every calibration
seed after the global shuffle and random-id reassignment; the regression tests
bound them below 0.25.

### B05 mechanism diagnostics

| Seed | simple | reference | min gap | mean gap | degenerate targets |
|---|---:|---:|---:|---:|---:|
| 2026 | 57.287 | 67.916 | 0.000000 | 0.397086 | 1 |
| 2027 | 74.581 | 78.288 | 0.001730 | 0.275378 | 0 |
| 2028 | 75.194 | 86.825 | 0.019356 | 0.442925 | 0 |
| 3026 | 66.432 | 70.624 | 0.031617 | 0.512600 | 0 |
| 4026 | 64.129 | 88.827 | 0.063936 | 0.325545 | 0 |

The raw global pointwise model is below 90 on every audited seed and has a
median below 75. The target-relative reference is materially better on every
audited seed and remains below 90. A small number of targets can have a
zero baseline-to-oracle gap; those targets are reported as degenerate and
excluded from the mean.

### B06 model-selection diagnostics

| Seed | selected model | logistic CV log-loss | quadratic CV log-loss |
|---|---|---:|---:|
| 2026 | logistic | 0.595660 | 0.613338 |
| 2027 | logistic | 0.623757 | 0.636664 |
| 2028 | logistic | 0.638438 | 0.664480 |
| 3026 | logistic | 0.623927 | 0.637829 |
| 4026 | quadratic | 0.618306 | 0.607834 |

Leave-one-target-out CV selects the linear logistic model on four of the five
audited seeds and the quadratic model on one. The quadratic model does not
materially improve the reference envelope, and the simple interface-area/charge
heuristic is stronger on every audited seed (median 40.909 vs 36.364). The
reference is therefore reported honestly as the weaker comparator for B06.

## Limitations

- These are synthetic tasks. The reference envelope is a calibration aid, not a
  claim about frontier biological capability or about any external benchmark.
- The reference methods are deliberately conventional and are not tuned to
  maximize the score. B04 and B06 have weak reference headroom because the
  documented baselines are already strong on those synthetic signals. For B06
  the simple heuristic is the stronger comparator; the CV-selected reference is
  reported as-is rather than tuned on the private score.
- B05 can contain a small number of degenerate targets where the minimum-energy
  pose is also the highest-quality pose. They are excluded from the mean and
  reported in the metrics.
- B03 ARI is label-invariant, but the absolute ARI ceiling depends on the
  daughter separation and within-daughter mutation rate. The frozen seeds above
  are the calibration record; new seeds should be audited before being used for
  comparisons.
- The baseline=0 and oracle=100 identities are guaranteed only for
  nondegenerate calibrated seeds. `evaluate` fails fast with `CalibrationError`
  when a seed violates them, so an arbitrary seed is never silently aggregated
  into a macro score.
- The local grader is not a hardened sandbox. See the README security note.
