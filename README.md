# BioAgentBench

An independent, local benchmark for ML-engineering agents on synthetic computational-biology tasks. It is inspired only by the publicly disclosed design of [FrontierBioML](https://valenlabs.com/blog/frontierbioml); it is not affiliated with Valen Labs and does not reproduce its private tasks, data, or leaderboard.

Current package version: **0.2.0**. The v0.2 changes recalibrate B03, B04, and B05 and add an in-process reference evaluator. See [docs/CALIBRATION.md](docs/CALIBRATION.md) for the v0.2 mechanisms, command usage, and observed reference envelope, and [docs/RESULTS_V0_2.md](docs/RESULTS_V0_2.md) for the current v0.2 example run.

## What is included

| ID | Synthetic task | Main held-out shift / deliverable |
|---|---|---|
| B01 | Antibody variant binding | New antibody families; binding predictions |
| B02 | Double-mutant epistasis | Unseen mutation combinations; fitness predictions |
| B03 | Antibody family clustering | Two unseen daughter subfamilies per known parent lineage; 16-way partition |
| B04 | Cancer gene dependency | New cell lines/tissues; predictions and a baseline-aligned 20-pair panel |
| B05 | Docking pose ranking | New targets with target-relative pose quality; one selected pose per target |
| B06 | Designed binder selection | New targets; probabilities and a quota-constrained 40-design panel |

All data are synthetic and deterministically generated. There is no real patient, pathogen, or wet-lab data.

Each agent receives only a brief and public training/development inputs. It must create `pipeline/predict.sh`. Grading copies only that pipeline into fresh directories, supplies sealed inputs, and executes it twice. Gates enforce successful execution, complete well-typed in-range outputs, exact determinism, required shortlist constraints, and numeric batch invariance where applicable. Scores are task-specific, baseline-relative, and clipped to 0–100; failures count as zero.

See [docs/PUBLIC_SPEC.md](docs/PUBLIC_SPEC.md) for sourced design details, differences, and security limitations.

## Current v0.2 example

A completed one-replicate Haiku 5.5 run on the **v0.2** bank (version 0.2.0, seed 2026) scored **59.640212/100**, passed **6/6 grading gates**, used 878,688 tokens, and cost **$0.112361** (market). See [docs/RESULTS_V0_2.md](docs/RESULTS_V0_2.md) for per-task metrics, timings, reference context, and a candid difficulty assessment. This is a single-replicate development example, not a model baseline, and it is not numerically comparable with the historical v0.1 run below.

## Historical v0.1 example (not current v0.2 scores)

A completed one-replicate Haiku 5.5 run on the **v0.1** bank scored **60.679799/100**, passed **6/6 grading gates**, used 818,710 tokens, and cost **$0.103179**. See [docs/RESULTS.md](docs/RESULTS.md) for per-task metrics, timings, artifact paths, and limitations. That run predates the v0.2 B03/B04/B05 changes and is not comparable with v0.2 reference tables.

## Setup

Requires Python 3.9+ and [`uv`](https://docs.astral.sh/uv/).

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"   # keep downloads/cache inside this project
uv venv .venv --python python3
uv pip install --python .venv/bin/python -e .
```

The scientific dependencies are for submitted pipelines. The gateway client and benchmark runner themselves use only the standard library.

## Generate and inspect

```bash
.venv/bin/python -m bioagent_bench generate --seed 2026
.venv/bin/python -m bioagent_bench tasks
```

This creates `.bench/tasks/` and `.bench/bank.json`. Agent workspaces contain only `BRIEF.md` and `data/`; generated `private/` labels are used later by grading and are never copied to an agent workspace. Generated data and run artifacts are gitignored.

`bank.json` records the bank format version (`0.2.0`) and generation seed. `run` and `tasks` reject a missing or unknown metadata version, and `run` rejects a seed mismatch instead of silently reusing a stale bank; regenerate with `generate --force --seed <seed>`.

## Reference calibration

The in-process reference evaluator runs the documented baseline, a deliberately basic heuristic, a conventional non-agent ML reference, and the sealed oracle. It uses the same generated rows and the production validator/scorer, without agents, gateway calls, or subprocesses:

```bash
.venv/bin/python -m bioagent_bench reference \
  --seeds 2026,2027,2028 \
  --output .cache/reference-v0_2.json
```

`--seeds` accepts any comma-separated integer list. The JSON report contains raw per-seed records plus per-task/method median/min/max summaries. The disclosed baseline and oracle score exactly 0 and 100 for every **nondegenerate calibrated** task/seed; `evaluate` fails fast with `CalibrationError` rather than aggregating a seed that violates those invariants. See [docs/CALIBRATION.md](docs/CALIBRATION.md) for the v0.2 envelope and limitations.

## Run Haiku 5.5 through Vercel AI Gateway

Set the key in the parent shell (it is already detected if your environment provides it):

```bash
export AI_GATEWAY_API_KEY=...
```

One replicate of all six tasks:

```bash
.venv/bin/python -m bioagent_bench run \
  --model anthropic/claude-haiku-5.5 \
  --replicates 1 \
  --max-steps 60 \
  --agent-time-limit 1800
```

For closer adherence to the publicly described aggregation, use `--replicates 3`. A single-task smoke run is available with `--tasks B01`; multiple tasks are comma-separated. The default model is already `anthropic/claude-haiku-5.5`.

The runner prints each score and writes a final suite summary. Task scores are averaged over replicates, then macro-averaged over tasks. A failed/missing/invalid submission scores zero rather than being omitted. Results are written under `runs/<run-id>/`:

- `summary.json`: suite score, per-task means, completion/gate counts, token usage, and gateway-reported cost;
- `records.json`: all replicate outcomes, including explicit failure causes;
- `BXX/replicate_N/agent_result.json`: status, steps, time, and usage;
- `BXX/replicate_N/transcript.json`: model/tool trace (no API key);
- `BXX/replicate_N/grading_result.json`: gate and metric details;
- `BXX/replicate_N/workspace/pipeline/`: the submitted pipeline.

## Output contract

```bash
bash pipeline/predict.sh INPUT_DIR OUTPUT_JSON
```

`INPUT_DIR` contains `train.csv` and unlabeled `input.csv`. Output is JSON:

```json
{
  "predictions": [{"id": "row-id", "value": 0.5}],
  "shortlist": ["row-id"]
}
```

Every input row must have exactly one prediction. Numeric values must be finite and in `[0,1]`; B03 uses arbitrary non-empty string cluster labels. `shortlist` is required only when the task brief says so. A pipeline must package all code/assets it needs because the grader copies only `pipeline/`.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The default suite uses the frozen calibration seeds 2026-2028 and includes reference/anti-ceiling tests. A wider audit sweep can be run with the `reference` command above.

## Security and interpretation

This local version is not a hardened sandbox: path guards and credential scrubbing do not provide kernel-level isolation, and proxy variables are not a network firewall. Run untrusted submissions in an actual offline container/VM. Six synthetic tasks are useful for development and regression testing, not claims of frontier biological capability. Do not compare scores directly across tasks or against FrontierBioML's scores.

v0.2 is a calibration release, not a leaderboard. B03 now has 16 sealed daughter subfamilies, B04 uses an equal-weight forecast/panel score, and B05 uses target-relative pose quality with per-target normalized skill. The reference envelope in [docs/CALIBRATION.md](docs/CALIBRATION.md) is the appropriate baseline for interpreting new runs; the v0.1 Haiku example in [docs/RESULTS.md](docs/RESULTS.md) is historical.
