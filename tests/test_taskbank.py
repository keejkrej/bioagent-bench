import json
import shutil
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from bioagent_bench.taskbank import TASK_IDS, generate_bank, grade_task


DUMMY_SCRIPT = r'''#!/usr/bin/env python3
import csv
import json
import sys

TASK = sys.argv[3]
with open(sys.argv[1] + "/input.csv", newline="") as handle:
    rows = list(csv.DictReader(handle))
id_column = next(column for column in rows[0] if column == "id" or column.endswith("_id"))
predictions = []
for row in rows:
    if TASK == "B03":
        predictions.append({"id": row[id_column], "value": row.get("sequence") or row[id_column]})
    else:
        predictions.append({"id": row[id_column], "value": 0.5})
shortlist = []
if TASK == "B04":
    counts = {}
    for row in rows:
        gene = row["gene"]
        if counts.get(gene, 0) < 2:
            shortlist.append(row[id_column])
            counts[gene] = counts.get(gene, 0) + 1
        if len(shortlist) == 20:
            break
elif TASK == "B05":
    seen = set()
    target_column = "target_id" if "target_id" in rows[0] else "target"
    for row in rows:
        if row[target_column] not in seen:
            shortlist.append(row[id_column])
            seen.add(row[target_column])
elif TASK == "B06":
    counts = {}
    target_column = "target_id" if "target_id" in rows[0] else "target"
    for row in rows:
        target = row[target_column]
        if counts.get(target, 0) < 10:
            shortlist.append(row[id_column])
            counts[target] = counts.get(target, 0) + 1
        if len(shortlist) == 40:
            break
with open(sys.argv[2], "w") as handle:
    json.dump({"predictions": predictions, "shortlist": shortlist}, handle, sort_keys=True)
'''


NONDETERMINISTIC_SCRIPT = r'''#!/usr/bin/env python3
import csv, json, random, sys
with open(sys.argv[1] + "/input.csv", newline="") as handle:
    rows = list(csv.DictReader(handle))
id_column = next(column for column in rows[0] if column == "id" or column.endswith("_id"))
with open(sys.argv[2], "w") as handle:
    json.dump({"predictions": [{"id": row[id_column], "value": random.random()} for row in rows]}, handle)
'''


BATCH_DEPENDENT_SCRIPT = r'''#!/usr/bin/env python3
import csv, json, sys
with open(sys.argv[1] + "/input.csv", newline="") as handle:
    rows = list(csv.DictReader(handle))
id_column = next(column for column in rows[0] if column == "id" or column.endswith("_id"))
with open(sys.argv[2], "w") as handle:
    json.dump({"predictions": [{"id": row[id_column], "value": float(index == 0)} for index, row in enumerate(rows)]}, handle)
'''


HUGE_NUMBER_SCRIPT = r'''#!/usr/bin/env python3
import csv, json, sys
with open(sys.argv[1] + "/input.csv", newline="") as handle:
    rows = list(csv.DictReader(handle))
with open(sys.argv[2], "w") as handle:
    json.dump({"predictions": [{"id": row["id"], "value": 10**400} for row in rows]}, handle)
'''

ORPHAN_SCRIPT = r'''#!/usr/bin/env python3
import subprocess, sys, time
child = "import os,sys,time; time.sleep(2); open(sys.argv[1], 'w').write('survived')"
subprocess.Popen([sys.executable, "-c", child, __MARKER__])
time.sleep(10)
'''

SCIENTIFIC_ENV_SCRIPT = r'''#!/usr/bin/env python3
import csv, json, os, sys
import numpy, pandas, scipy, sklearn
with open(sys.argv[1] + "/input.csv", newline="") as handle:
    rows = list(csv.DictReader(handle))
print("SCIENTIFIC_STACK_OK")
print(os.getenv("DATABASE_URL"), os.getenv("SSH_AUTH_SOCK"), os.getenv("BASH_ENV"))
with open(sys.argv[2], "w") as handle:
    json.dump({"predictions": [{"id": row["id"], "value": 0.5} for row in rows]}, handle)
'''


class TaskBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.bank = cls.root / "bank"
        generate_bank(cls.bank, seed=2026)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def workspace(self, task_id, script=DUMMY_SCRIPT):
        workspace = self.root / ("workspace-" + task_id + "-" + str(len(list(self.root.glob("workspace-*")))))
        (workspace / "pipeline").mkdir(parents=True)
        (workspace / "pipeline" / "predict.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\npython3 \"$(dirname \"$0\")/predict.py\" \"$1\" \"$2\" " + task_id + "\n",
            encoding="utf-8",
        )
        (workspace / "pipeline" / "predict.py").write_text(textwrap.dedent(script), encoding="utf-8")
        return workspace

    def grade(self, task_id, workspace, timeout_seconds=60):
        task_dir = self.bank / "tasks" / task_id
        return grade_task(task_dir, workspace, self.root / ("grading-" + workspace.name), timeout_seconds=timeout_seconds)

    def test_generation_has_six_sealed_deterministic_tasks(self):
        self.assertEqual(len(TASK_IDS), 6)
        for task_id in TASK_IDS:
            with self.subTest(task_id=task_id):
                task = self.bank / "tasks" / task_id
                self.assertTrue((task / "brief.md").is_file())
                self.assertTrue((task / "task.json").is_file())
                self.assertTrue((task / "public" / "train.csv").is_file())
                self.assertTrue((task / "public" / "input.csv").is_file())
                self.assertTrue((task / "private" / "truth.csv").is_file())
                self.assertEqual(
                    (task / "public" / "train.csv").read_bytes(),
                    (task / "private" / "input" / "train.csv").read_bytes(),
                )
                self.assertNotIn("private/", (task / "brief.md").read_text(encoding="utf-8").lower())

    def test_regeneration_is_byte_identical(self):
        second = self.root / "bank-second"
        generate_bank(second, seed=2026)
        for task_id in TASK_IDS:
            for relative in ("public/train.csv", "public/input.csv", "private/input/input.csv", "private/truth.csv"):
                with self.subTest(task_id=task_id, relative=relative):
                    first_path = self.bank / "tasks" / task_id / relative
                    second_path = second / "tasks" / task_id / relative
                    self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
        shutil.rmtree(second)

    def test_valid_dummy_pipeline_passes_every_gate(self):
        for task_id in TASK_IDS:
            with self.subTest(task_id=task_id):
                result = self.grade(task_id, self.workspace(task_id))
                self.assertTrue(result["passed_gate"], result)
                self.assertTrue(result["deterministic"], result)
                if task_id != "B03":
                    self.assertTrue(result["batch_invariant"], result)
                self.assertGreaterEqual(result["score"], 0)
                self.assertLessEqual(result["score"], 100)

    def test_nondeterministic_output_fails_closed(self):
        result = self.grade("B01", self.workspace("B01", NONDETERMINISTIC_SCRIPT))
        self.assertFalse(result["passed_gate"])
        self.assertEqual(result["score"], 0)
        self.assertFalse(result["deterministic"])

    def test_batch_dependent_output_fails_closed(self):
        result = self.grade("B01", self.workspace("B01", BATCH_DEPENDENT_SCRIPT))
        self.assertFalse(result["passed_gate"], result)
        self.assertEqual(result["score"], 0)
        self.assertFalse(result["batch_invariant"])

    def test_huge_json_integer_fails_validation_without_raising(self):
        result = self.grade("B01", self.workspace("B01", HUGE_NUMBER_SCRIPT))
        self.assertFalse(result["passed_gate"])
        self.assertEqual(result["score"], 0)
        self.assertIn("not finite", result["reason"])
        self.assertEqual(len(result["attempts"]), 2)

    def test_timeout_kills_the_entire_process_group(self):
        marker = self.root / "orphan-marker"
        script = ORPHAN_SCRIPT.replace("__MARKER__", repr(str(marker)))
        workspace = self.workspace("B01", script)
        result = self.grade("B01", workspace, timeout_seconds=1)
        self.assertFalse(result["passed_gate"])
        self.assertTrue(all(attempt["timed_out"] for attempt in result["attempts"]), result)
        time.sleep(2.3)
        self.assertFalse(marker.exists(), "a grading child survived its process-group timeout")

    def test_grading_uses_scientific_venv_and_allowlisted_environment(self):
        secrets = {
            "DATABASE_URL": "postgres://user:secret@example/db",
            "SSH_AUTH_SOCK": "/private/secret-agent.sock",
            "BASH_ENV": "/private/secret-startup.sh",
        }
        with patch.dict("os.environ", secrets):
            result = self.grade("B01", self.workspace("B01", SCIENTIFIC_ENV_SCRIPT))
        self.assertTrue(result["passed_gate"], result)
        self.assertIn("SCIENTIFIC_STACK_OK", result["attempts"][0]["stdout_tail"])
        self.assertIn("None None None", result["attempts"][0]["stdout_tail"])
        for secret in secrets.values():
            self.assertNotIn(secret, json.dumps(result))

    def test_missing_pipeline_fails_closed(self):
        empty = self.root / "empty-workspace"
        empty.mkdir()
        result = self.grade("B01", empty)
        self.assertFalse(result["passed_gate"])
        self.assertEqual(result["score"], 0)


if __name__ == "__main__":
    unittest.main()
