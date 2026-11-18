# Publicly disclosed design and independent implementation choices

## Sources

Read 18 November 2026:

- Valen Labs, [FrontierBioML announcement](https://valenlabs.com/blog/frontierbioml), 12 November 2026.
- FrontierBioML, [Method](https://www.frontierbioml.com/method.html).

The announcement describes 25 computational-biology ML tasks. The method page discloses an offline agent sandbox, submission as `pipeline/predict.sh`, two fresh offline grading runs on sealed held-out data, determinism and validity gates, rescaled 0–100 scores, and aggregation over three task runs. It also names distribution shifts and batch-invariance/panel checks.

## What this repository uses

- The submission filename, two-run determinism gate, fresh grading directories, held-out distribution shifts, 0–100 baseline-relative scores, zero for invalid submissions, and optional three-replicate aggregation.
- Six new task briefs with deterministic synthetic data. They are motivated by broad categories named publicly, but their data, labels, baselines, and graders are implemented here.
- Tool access limited to a per-task workspace and no browser/search tool during a run. Model API access goes only through the gateway client.
- A provider-neutral OpenAI-compatible gateway adapter, tested with `anthropic/claude-haiku-5.5`.

## Deliberate differences

This is **BioAgentBench**, not FrontierBioML, not a reproduction of its leaderboard, and not affiliated with Valen Labs. It has six synthetic tasks rather than 25 experimental-data tasks, shorter default budgets, one run per task by default, and a lightweight local runner. No private Valen tasks, labels, traces, submissions, or scores are included.

Local subprocesses and workspace path checks are evaluation conveniences, **not a hardened security or network sandbox**. Grading subprocesses have credentials removed, but this local version does not enforce kernel-level filesystem or network isolation. Use containers/VMs for untrusted submissions. Scores within a task are baseline-relative; extra care is needed when comparing different task scores, and this small suite is intended for iteration rather than frontier-model rankings.

## Run accounting

Every attempted run receives a status. Missing scripts, timeouts, exceptions, output-gate failures, and valid below-baseline submissions are recorded separately, while all contribute zero to the suite score. This avoids silently dropping failed runs. Transcripts and usage are saved beneath each replicate directory.
