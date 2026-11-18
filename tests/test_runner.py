import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bioagent_bench.runner import ensure_bank, run_suite, summarize
from bioagent_bench.taskbank import BANK_VERSION, BankMetadataError, generate_bank


class SummaryTests(unittest.TestCase):
    def test_replicates_are_averaged_within_tasks_then_macro_averaged(self):
        config = {"run_id": "test", "model": "test", "task_ids": ["B01", "B02"], "replicates": 2}
        records = []
        for task, scores in (("B01", [20, 40]), ("B02", [60, 80])):
            for index, score in enumerate(scores, 1):
                records.append({"task_id": task, "replicate": index, "score": score, "passed_gate": True, "agent_status": "completed", "usage": {}})
        result = summarize(config, records)
        self.assertEqual(result["task_scores"], {"B01": 30, "B02": 70})
        self.assertEqual(result["score"], 50)
        self.assertTrue(result["complete"])
        self.assertEqual(result["passed_gate"], 4)

    def test_incomplete_suite_does_not_claim_partial_macro_score(self):
        config = {"run_id": "test", "model": "test", "task_ids": ["B01", "B02"], "replicates": 1}
        records = [{"task_id": "B01", "replicate": 1, "score": 90, "passed_gate": True, "agent_status": "completed", "usage": {}}]
        result = summarize(config, records)
        self.assertEqual(result["score"], 0)
        self.assertFalse(result["complete"])

    def test_duplicate_records_do_not_satisfy_missing_task(self):
        config = {"run_id": "test", "model": "test", "task_ids": ["B01", "B02"], "replicates": 1}
        records = [
            {"task_id": "B01", "replicate": 1, "score": 100, "passed_gate": True, "agent_status": "completed", "usage": {}},
            {"task_id": "B01", "replicate": 1, "score": 100, "passed_gate": True, "agent_status": "completed", "usage": {}},
        ]
        result = summarize(config, records)
        self.assertEqual(result["score"], 0)
        self.assertFalse(result["complete"])

    def test_grader_exception_records_zero_and_continues(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank = root / "bank"
            generate_bank(bank)
            agent_result = {
                "pipeline_present": True,
                "status": "completed",
                "error": None,
                "elapsed_seconds": 0.1,
                "steps": 1,
                "usage": {},
            }
            with patch("bioagent_bench.runner.run_agent", return_value=agent_result), patch(
                "bioagent_bench.runner.grade_task", side_effect=ValueError("injected grader bug")
            ):
                summary = run_suite(bank, root / "runs", task_ids=["B01", "B02"])
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["completed_replicates"], 2)
            self.assertEqual(summary["passed_gate"], 0)
            self.assertEqual(summary["score"], 0)
            records = json.loads((root / "runs" / summary["run_id"] / "records.json").read_text())
            self.assertEqual(len(records), 2)
            self.assertTrue(all("injected grader bug" in record["grading_reason"] for record in records))

    def test_run_config_records_bank_version_and_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank = root / "bank"
            generate_bank(bank, seed=2026)
            agent_result = {
                "pipeline_present": True,
                "status": "completed",
                "error": None,
                "elapsed_seconds": 0.1,
                "steps": 1,
                "usage": {},
            }
            with patch("bioagent_bench.runner.run_agent", return_value=agent_result), patch(
                "bioagent_bench.runner.grade_task", side_effect=ValueError("injected grader bug")
            ):
                summary = run_suite(
                    bank, root / "runs", task_ids=["B01"], seed=2026
                )
            config = json.loads(
                (root / "runs" / summary["run_id"] / "config.json").read_text()
            )
            self.assertEqual(config["bank_version"], BANK_VERSION)
            self.assertEqual(config["bank_seed"], 2026)
            self.assertEqual(config["bank_metadata"]["version"], BANK_VERSION)
            self.assertEqual(config["bank_metadata"]["seed"], 2026)

    def test_run_suite_rejects_seed_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank = root / "bank"
            generate_bank(bank, seed=2026)
            with self.assertRaises(BankMetadataError) as context:
                run_suite(bank, root / "runs", task_ids=["B01"], seed=2027)
            self.assertIn("seed 2027", str(context.exception))

    def test_ensure_bank_rejects_stale_bank_without_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank = root / "bank"
            (bank / "tasks").mkdir(parents=True)
            with self.assertRaises(BankMetadataError) as context:
                ensure_bank(bank, seed=2026)
            self.assertIn("generate --force", str(context.exception))

    def test_ensure_bank_generates_missing_bank_and_validates_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank = root / "bank"
            metadata = ensure_bank(bank, seed=2027)
            self.assertEqual(metadata["version"], BANK_VERSION)
            self.assertEqual(metadata["seed"], 2027)
            with self.assertRaises(BankMetadataError):
                ensure_bank(bank, seed=2026)


if __name__ == "__main__":
    unittest.main()
