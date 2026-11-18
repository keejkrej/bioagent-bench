# v0.2 example: Haiku 5.5 result

> **Current v0.2 example.** This page records a single one-replicate run on the
> v0.2 bank. It is a development example, not a leaderboard result and not a
> model baseline. It is **not numerically comparable** with the historical v0.1
> run in [RESULTS.md](RESULTS.md): the B03/B04/B05 tasks, their scoring, and the
> bank version all changed between v0.1 and v0.2. See
> [CALIBRATION.md](CALIBRATION.md) for the v0.2 mechanisms and the frozen-seed
> reference envelope.

## Configuration

- Date: 18 November 2026
- Run: `runs/haiku55-full-v0_2-01/` (gitignored, local)
- Run id: `haiku55-full-v0_2-01`
- Model: `anthropic/claude-haiku-5.5` through Vercel AI Gateway
- Replicates: **1 per task** (a development run, not a variance estimate)
- Agent budget: at most 60 steps / 1200 seconds per task
- Prediction budget: at most 300 seconds per execution, two determinism executions plus the applicable reversed-batch execution
- Bank: deterministic synthetic **version 0.2.0**, **seed 2026**
- Command:

```bash
.venv/bin/python -m bioagent_bench run \
  --model anthropic/claude-haiku-5.5 \
  --replicates 1 \
  --max-steps 60 \
  --agent-time-limit 1200 \
  --grade-time-limit 300 \
  --seed 2026 \
  --run-id haiku55-full-v0_2-01
```

## Scores

| Task | Score | Gate | Agent | Wall time | Main metrics |
|---|---:|---|---|---:|---|
| B01 Antibody variant binding | 65.415105 | Pass | completed | 190.266 s | RMSE 0.070380 vs 0.203498 baseline |
| B02 Double-mutant epistasis | 65.318940 | Pass | completed | 86.691 s | RMSE 0.022294 vs 0.064283 additive baseline |
| B03 Antibody family clustering | 87.557082 | Pass | completed | 47.553 s | ARI 0.950130 vs 0.599206 best-cutoff baseline |
| B04 Cancer gene dependency | 16.580973 | Pass | completed | 100.417 s | Forecast 0.067590; panel skill 0.264029 (0.5/0.5) |
| B05 Docking pose ranking | 67.969173 | Pass | completed | 86.074 s | Mean skill 0.679692; selected 0.783808 vs 0.464919 baseline and 0.862005 oracle |
| B06 Designed binder selection | 55.000000 | Pass | completed | 50.887 s | 31 hits vs 20 baseline and 40 oracle |
| **Macro score** | **59.640212** | **6/6 pass** | **6/6 completed** | **561.888 s** | 9 min 22 s total |

Every agent status was `completed` and every submission passed output
coverage/type/range checks, two-run canonical-JSON determinism, task-specific
panel constraints, and numeric batch invariance where applicable. There were no
agent errors and no gate failures.

B04's 16.58 is a **valid low score**, not an invalid submission or
infrastructure failure: the agent's forecast component was weak (0.0676) and its
panel skill was modest (0.2640), and the equal-weight 0.5/0.5 combination yields
`100 * (0.5 * 0.067590 + 0.5 * 0.264029) = 16.58`. The agent's own pipeline
emitted `RuntimeWarning` divide-by-zero/overflow messages during grading, but its
output remained finite, in range, deterministic, and batch-invariant, so the
gate passed. Those warnings originate in the submitted pipeline, not in the
benchmark infrastructure.

## Usage

- Input tokens: 817,457
- Output tokens: 61,231
- Total tokens: 878,688
- Gateway-reported upstream inference cost: **$0.080993**
- Gateway-reported market cost: **$0.112361**

This cost covers only the six-task full run, not any preceding connectivity or
smoke calls.

Machine-readable source: `runs/haiku55-full-v0_2-01/summary.json` and
`runs/haiku55-full-v0_2-01/records.json`. Model/tool transcripts, submitted
pipelines, and grading details live below each task's `replicate_1/` directory.
Generated runs are gitignored and remain local.

## Comparative reference context

The v0.2 reference evaluator runs conventional non-agent methods on the same
generated rows. The table below places this run beside the **seed-2026**
reference record and the five-seed median envelope from
[CALIBRATION.md](CALIBRATION.md). These reference numbers come from **frozen
synthetic seeds** and are calibration aids, not model baselines on the same
learning process; the agent and the reference methods are not trained or
evaluated identically, so the comparison is contextual only.

| Task | This run | Reference (seed 2026) | Reference (5-seed median) | Simple (5-seed median) |
|---|---:|---:|---:|---:|
| B01 | 65.415 | 50.078 | 42.812 | 0.000 |
| B02 | 65.319 | 65.482 | 67.864 | 0.000 |
| B03 | 87.557 | 65.524 | 65.524 | 0.000 |
| B04 | 16.581 | 13.758 | 25.605 | 0.000 |
| B05 | 67.969 | 67.916 | 78.288 | 66.432 |
| B06 | 55.000 | 45.000 | 36.364 | 40.909 |
| **Macro** | **59.640** | **51.293** | **51.293** | **19.199** |

The agent is above the seed-2026 reference on B01, B03, B04, and B06, roughly
level on B02 and B05, and below the five-seed reference median on B05. Because
this is one replicate on one seed, none of these gaps should be read as a stable
capability difference.

## Difficulty assessment

- **B03 is now the easiest task for this agent** (87.56). The v0.2 stratified
  daughter split and random ids removed the v0.1 leak that let a trivial
  partition score 100, but a competent clustering pipeline still recovers most
  of the 16-way structure. The reference (65.52) is materially weaker, so the
  task is solvable but no longer free.
- **B01, B02, B05 sit in a moderate band** (65–68). Each is clearly above its
  baseline and below the oracle, with the agent roughly matching the reference
  on B02/B05 and beating it on B01. These look like reasonable discriminators.
- **B04 remains hard** (16.58). The equal-weight forecast/panel score is
  baseline-aligned and the agent's forecast was weak; the reference is also low
  (13.76 on seed 2026, 25.61 median). This task has limited headroom and is
  currently the weakest signal for separating agents.
- **B06 is moderate** (55.00). The agent beat the reference (45.00) but was
  below the simple interface-area/charge heuristic (60.00 on seed 2026, 40.909
  median). B06 is a small, quota-constrained selection task with only four
  targets and 40 oracle hits, so its resolution is coarse and the ordering is
  seed-sensitive.

Overall the v0.2 bank is **not trivially solvable** (no task reaches 100) and
**not uniformly hard** (B03 is easy, B04 is hard). A single replicate cannot
estimate run-to-run variance; use `--replicates 3` for a closer approximation to
the publicly described aggregation before drawing ranking conclusions.

## Interpretation

This is one run on six synthetic tasks. It is not a FrontierBioML result, a
comparison to Valen Labs' models, or evidence of real-world biological
capability. It is **not numerically comparable** with the historical v0.1 run in
[RESULTS.md](RESULTS.md): the bank version, B03/B04/B05 task construction, and
scoring all changed. The v0.1 page is retained as historical evidence only.

Offline validation at the time of this run: **50 unit/integration tests
passing**.
