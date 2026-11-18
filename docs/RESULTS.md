# Historical v0.1 example: Haiku 5.5 result

> **Historical v0.1 result.** This page records the v0.1 bank and grader. It is
> not a v0.2 score and should not be compared with the v0.2 reference tables.
> See [CALIBRATION.md](CALIBRATION.md) for the current v0.2 mechanisms and
> observed reference envelope.

## Configuration

- Date: 18 November 2026
- Run: `runs/haiku55-full-01/`
- Model: `anthropic/claude-haiku-5.5` through Vercel AI Gateway
- Replicates: **1 per task** (a development run, not a variance estimate)
- Agent budget: at most 60 steps / 1200 seconds per task
- Prediction budget: at most 300 seconds per execution, two determinism executions plus the applicable reversed-batch execution
- Bank: deterministic synthetic seed 2026

## Scores

| Task | Score | Gate | Wall time | Main metrics |
|---|---:|---|---:|---|
| B01 Antibody variant binding | 66.304696 | Pass | 155.560 s | RMSE 0.068569 vs 0.203498 baseline |
| B02 Double-mutant epistasis | 53.614088 | Pass | 74.238 s | RMSE 0.029818 vs 0.064283 additive baseline |
| B03 Antibody family clustering | 100.000000 | Pass | 45.417 s | ARI 1.0 vs 0.910422 best-cutoff baseline |
| B04 Cancer gene dependency | 0.704862 | Pass | 71.037 s | Forecast 0.007424; panel ratio 0.949409 |
| B05 Docking pose ranking | 98.455150 | Pass | 80.559 s | Selected quality 0.877079 vs 0.428424 baseline and 0.884119 oracle |
| B06 Designed binder selection | 45.000000 | Pass | 63.595 s | 29 hits vs 20 baseline and 40 oracle |
| **Macro score** | **60.679799** | **6/6 pass** | **490.406 s** | 8 min 10 s total |

Every agent status was `completed`. Every submission passed output coverage/type/range checks, two-run canonical-JSON determinism, task-specific panel constraints, and numeric batch invariance where applicable. B04's score is a valid low score caused by its forecast component, not an invalid submission or infrastructure failure.

## Usage

- Input tokens: 765,440
- Output tokens: 53,270
- Total tokens: 818,710
- Gateway-reported upstream inference cost: **$0.103179**
- Gateway-reported market cost: **$0.103179**

This cost covers only the six-task full run, not the preceding connectivity and B01 smoke calls.

Machine-readable source: `runs/haiku55-full-01/summary.json` and `runs/haiku55-full-01/records.json`. Model/tool transcripts, submitted pipelines, and grading details live below each task's `replicate_1/` directory. Generated runs are gitignored and remain local.

## Interpretation

This is one run on six newly authored synthetic tasks. It is not a FrontierBioML result, a comparison to Valen Labs' models, or evidence of real-world biological capability. A single replicate cannot estimate run-to-run variance. Use `--replicates 3` for closer adherence to the publicly disclosed aggregation pattern. Strong B03/B05 results also show that some synthetic tasks are readily solvable; task difficulty should be calibrated before drawing model-ranking conclusions.

Reproduce with:

```bash
.venv/bin/python -m bioagent_bench run \
  --model anthropic/claude-haiku-5.5 \
  --replicates 1 \
  --max-steps 60 \
  --agent-time-limit 1200 \
  --grade-time-limit 300
```

Offline validation at the time of this run: **25 unit/integration tests passing**.
