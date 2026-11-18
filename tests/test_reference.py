import unittest
from unittest.mock import patch

from bioagent_bench.reference import CalibrationError, evaluate, evaluate_task
from bioagent_bench.taskbank import (
    TASK_IDS,
    _adjusted_rand_index,
    _b04_baseline_panel,
    _b04_gene_means,
    _b04_oracle_panel,
    _generate_task_data,
    _hamming_distance,
    _score_b03,
    _score_b04,
)

CALIBRATION_SEEDS = (2026, 2027, 2028)
AUDIT_SEEDS = (3026, 4026)


def _nearest_centroid(sequence, centroids):
    return min(
        centroids,
        key=lambda label: (_hamming_distance(sequence, centroids[label]), label),
    )


def _centroids(rows, label_key):
    grouped = {}
    for row in rows:
        grouped.setdefault(row[label_key], []).append(row["sequence"])
    result = {}
    for label, sequences in grouped.items():
        length = len(sequences[0])
        centroid = []
        for position in range(length):
            counts = {}
            for sequence in sequences:
                residue = sequence[position]
                counts[residue] = counts.get(residue, 0) + 1
            centroid.append(min(counts, key=lambda residue: (-counts[residue], residue)))
        result[label] = "".join(centroid)
    return result


class ReferenceEvaluationTests(unittest.TestCase):
    def test_baseline_and_oracle_are_exact_for_every_task_and_seed(self):
        for seed in CALIBRATION_SEEDS:
            for task_id in TASK_IDS:
                with self.subTest(seed=seed, task_id=task_id):
                    record = evaluate_task(task_id, seed)
                    baseline = record["methods"]["baseline"]
                    oracle = record["methods"]["oracle"]
                    self.assertTrue(baseline["valid"], baseline)
                    self.assertTrue(oracle["valid"], oracle)
                    self.assertEqual(baseline["score"], 0.0)
                    self.assertEqual(oracle["score"], 100.0)

    def test_reference_and_simple_are_valid_and_bounded(self):
        for seed in CALIBRATION_SEEDS:
            for task_id in TASK_IDS:
                with self.subTest(seed=seed, task_id=task_id):
                    record = evaluate_task(task_id, seed)
                    for method in ("simple", "reference"):
                        result = record["methods"][method]
                        self.assertTrue(result["valid"], result)
                        self.assertGreaterEqual(result["score"], 0.0)
                        self.assertLessEqual(result["score"], 100.0)

    def test_b03_anti_ceiling_and_two_medoids_mechanism(self):
        for seed in CALIBRATION_SEEDS:
            with self.subTest(seed=seed):
                record = evaluate_task("B03", seed)
                diagnostics = record["diagnostics"]
                self.assertEqual(diagnostics["n_daughters"], 16)
                self.assertLess(diagnostics["nearest_parent_ari"], 0.90)
                self.assertLess(diagnostics["best_cutoff_ari"], 0.95)
                self.assertGreater(
                    diagnostics["two_medoids_ari"],
                    diagnostics["nearest_parent_ari"] + 0.05,
                )
                self.assertLess(diagnostics["two_medoids_ari"], 0.99)
                self.assertGreater(record["methods"]["reference"]["score"], 0.0)
                self.assertLess(record["methods"]["reference"]["score"], 100.0)

    def test_b03_split_invariants_and_no_train_lookup(self):
        data = _generate_task_data("B03", 2026)
        train_rows = data["train_rows"]
        public_rows = data["public_input_rows"]
        private_rows = data["private_input_rows"]
        truth_rows = data["truth_rows"]

        self.assertEqual(len(train_rows), 256)
        self.assertEqual(len({row["family"] for row in train_rows}), 8)
        self.assertEqual(len(public_rows), 160)
        self.assertEqual(len(private_rows), 336)
        self.assertEqual(len(truth_rows), 336)

        truth = {row["id"]: row["family"] for row in truth_rows}
        private_labels = {truth[row["id"]] for row in private_rows}
        self.assertEqual(len(private_labels), 16)
        for label in private_labels:
            self.assertGreaterEqual(
                sum(1 for row in private_rows if truth[row["id"]] == label),
                15,
            )

        train_ids = {row["id"] for row in train_rows}
        test_ids = {row["id"] for row in public_rows} | {row["id"] for row in private_rows}
        self.assertFalse(train_ids & test_ids)
        train_sequences = {row["sequence"] for row in train_rows}
        test_sequences = {row["sequence"] for row in public_rows} | {
            row["sequence"] for row in private_rows
        }
        self.assertFalse(train_sequences & test_sequences)

        parent_centroids = _centroids(train_rows, "family")
        public_parents = {
            _nearest_centroid(row["sequence"], parent_centroids) for row in public_rows
        }
        self.assertEqual(len(public_parents), 8)

        daughter_centroids = _centroids(
            [{"sequence": row["sequence"], "family": truth[row["id"]]} for row in private_rows],
            "family",
        )
        public_daughters = {
            _nearest_centroid(row["sequence"], daughter_centroids) for row in public_rows
        }
        self.assertEqual(len(public_daughters), 16)

    def test_b03_positional_and_id_block_partitions_do_not_recover_truth(self):
        for seed in CALIBRATION_SEEDS:
            with self.subTest(seed=seed):
                data = _generate_task_data("B03", seed)
                private_rows = data["private_input_rows"]
                truth = {row["id"]: row["family"] for row in data["truth_rows"]}
                true_labels = [truth[row["id"]] for row in private_rows]
                count = len(private_rows)
                self.assertEqual(count, 336)

                positional = [index // 21 for index in range(count)]
                self.assertLess(
                    _adjusted_rand_index(true_labels, positional), 0.25
                )

                order = sorted(
                    range(count), key=lambda index: private_rows[index]["id"]
                )
                sorted_blocks = [0] * count
                for rank, index in enumerate(order):
                    sorted_blocks[index] = rank // 21
                self.assertLess(
                    _adjusted_rand_index(true_labels, sorted_blocks), 0.25
                )

                numeric_ids = [int(row["id"][2:]) for row in private_rows]
                low = min(numeric_ids)
                id_blocks = [(value - low) // 21 for value in numeric_ids]
                self.assertLess(
                    _adjusted_rand_index(true_labels, id_blocks), 0.25
                )

    def test_b03_inputs_have_no_accidental_leak_fields(self):
        data = _generate_task_data("B03", 2026)
        for row in data["public_input_rows"] + data["private_input_rows"]:
            self.assertEqual(set(row), {"id", "sequence"})
        for row in data["truth_rows"]:
            self.assertEqual(set(row), {"id", "family"})

        public_ids = {int(row["id"][2:]) for row in data["public_input_rows"]}
        private_ids = {int(row["id"][2:]) for row in data["private_input_rows"]}
        self.assertLess(min(public_ids), max(private_ids))
        self.assertLess(min(private_ids), max(public_ids))
        all_ids = sorted(public_ids | private_ids)
        self.assertNotEqual(
            all_ids, list(range(all_ids[0], all_ids[0] + len(all_ids)))
        )

    def test_b05_anti_ceiling_and_target_relative_mechanism(self):
        simple_scores = []
        reference_scores = []
        for seed in CALIBRATION_SEEDS:
            with self.subTest(seed=seed):
                record = evaluate_task("B05", seed)
                simple = record["methods"]["simple"]["score"]
                reference = record["methods"]["reference"]["score"]
                simple_scores.append(simple)
                reference_scores.append(reference)
                self.assertLess(simple, 90.0)
                self.assertGreater(reference, simple)
                self.assertLess(reference, 95.0)
                self.assertEqual(record["methods"]["baseline"]["score"], 0.0)
                self.assertEqual(record["methods"]["oracle"]["score"], 100.0)
                diagnostics = record["diagnostics"]
                self.assertGreaterEqual(diagnostics["min_gap"], 0.0)
                self.assertGreater(diagnostics["mean_gap"], 0.0)
                self.assertLessEqual(diagnostics["degenerate_targets"], 2)
        self.assertLess(sorted(simple_scores)[len(simple_scores) // 2], 75.0)
        self.assertGreater(
            sorted(reference_scores)[len(reference_scores) // 2],
            sorted(simple_scores)[len(simple_scores) // 2] + 3.0,
        )

    def test_b05_split_invariants(self):
        data = _generate_task_data("B05", 2026)
        train_rows = data["train_rows"]
        public_rows = data["public_input_rows"]
        private_rows = data["private_input_rows"]

        train_targets = {row["target"] for row in train_rows}
        public_targets = {row["target"] for row in public_rows}
        private_targets = {row["target"] for row in private_rows}
        self.assertEqual(len(train_targets), 50)
        self.assertEqual(len(public_targets), 10)
        self.assertEqual(len(private_targets), 30)
        self.assertFalse(train_targets & public_targets)
        self.assertFalse(train_targets & private_targets)
        self.assertFalse(public_targets & private_targets)

        for rows in (train_rows, public_rows, private_rows):
            self.assertEqual(len({row["target_type"] for row in rows}), 5)
            counts = {}
            for row in rows:
                counts[row["target"]] = counts.get(row["target"], 0) + 1
            self.assertTrue(all(count == 12 for count in counts.values()))

    def test_b03_degenerate_baseline_reports_calibration_error(self):
        input_rows = [
            {"id": "a", "sequence": "AAAA"},
            {"id": "b", "sequence": "AAAA"},
            {"id": "c", "sequence": "CCCC"},
            {"id": "d", "sequence": "CCCC"},
        ]
        truth_rows = [
            {"id": "a", "family": "x"},
            {"id": "b", "family": "x"},
            {"id": "c", "family": "y"},
            {"id": "d", "family": "y"},
        ]
        prediction_map = {"a": "p", "b": "p", "c": "q", "d": "q"}
        score, metrics = _score_b03(prediction_map, None, input_rows, truth_rows, [])
        self.assertEqual(score, 0.0)
        self.assertTrue(metrics["calibration_error"])
        self.assertAlmostEqual(metrics["baseline_ari"], 1.0, places=9)
        self.assertAlmostEqual(metrics["denominator"], 0.0, places=9)

    def test_b04_additive_component_arithmetic(self):
        data = _generate_task_data("B04", 2026)
        train_rows = data["train_rows"]
        input_rows = data["private_input_rows"]
        truth_rows = data["truth_rows"]
        truth = {row["id"]: float(row["dependency"]) for row in truth_rows}

        gene_mean, global_mean = _b04_gene_means(train_rows)
        baseline_predictions = {
            row["id"]: gene_mean.get(row["gene"], global_mean) for row in input_rows
        }
        oracle_predictions = {row["id"]: truth[row["id"]] for row in input_rows}
        baseline_panel = _b04_baseline_panel(input_rows, train_rows)
        oracle_panel = _b04_oracle_panel(input_rows, truth_rows)

        self.assertEqual(len(baseline_panel), 20)
        self.assertEqual(len(oracle_panel), 20)
        self.assertEqual(baseline_panel, _b04_baseline_panel(input_rows, train_rows))
        self.assertEqual(oracle_panel, _b04_oracle_panel(input_rows, truth_rows))

        gene_of = {row["id"]: row["gene"] for row in input_rows}
        for panel in (baseline_panel, oracle_panel):
            counts = {}
            for row_id in panel:
                gene = gene_of[row_id]
                counts[gene] = counts.get(gene, 0) + 1
            self.assertTrue(all(count <= 2 for count in counts.values()))

        baseline_score, baseline_metrics = _score_b04(
            baseline_predictions, baseline_panel, input_rows, truth_rows, train_rows
        )
        self.assertAlmostEqual(baseline_score, 0.0, places=9)
        self.assertAlmostEqual(baseline_metrics["forecast"], 0.0, places=9)
        self.assertAlmostEqual(baseline_metrics["panel_skill"], 0.0, places=9)
        self.assertGreater(baseline_metrics["panel_denominator"], 0.0)
        self.assertEqual(baseline_metrics["forecast_weight"], 0.5)
        self.assertEqual(baseline_metrics["panel_weight"], 0.5)

        oracle_score, oracle_metrics = _score_b04(
            oracle_predictions, oracle_panel, input_rows, truth_rows, train_rows
        )
        self.assertAlmostEqual(oracle_score, 100.0, places=9)
        self.assertAlmostEqual(oracle_metrics["forecast"], 1.0, places=9)
        self.assertAlmostEqual(oracle_metrics["panel_skill"], 1.0, places=9)

        forecast_only_score, forecast_only_metrics = _score_b04(
            oracle_predictions, baseline_panel, input_rows, truth_rows, train_rows
        )
        self.assertAlmostEqual(forecast_only_score, 50.0, places=9)
        self.assertAlmostEqual(forecast_only_metrics["forecast"], 1.0, places=9)
        self.assertAlmostEqual(forecast_only_metrics["panel_skill"], 0.0, places=9)

        panel_only_score, panel_only_metrics = _score_b04(
            baseline_predictions, oracle_panel, input_rows, truth_rows, train_rows
        )
        self.assertAlmostEqual(panel_only_score, 50.0, places=9)
        self.assertAlmostEqual(panel_only_metrics["forecast"], 0.0, places=9)
        self.assertAlmostEqual(panel_only_metrics["panel_skill"], 1.0, places=9)

    def test_evaluate_task_reports_valid_calibration(self):
        for seed in CALIBRATION_SEEDS:
            for task_id in TASK_IDS:
                with self.subTest(seed=seed, task_id=task_id):
                    record = evaluate_task(task_id, seed)
                    self.assertTrue(
                        record["calibration"]["valid"], record["calibration"]
                    )
                    self.assertEqual(record["calibration"]["errors"], [])

    def test_evaluate_fails_fast_on_invalid_calibration(self):
        invalid = {
            "task_id": "B01",
            "seed": 2026,
            "methods": {},
            "diagnostics": {},
            "calibration": {"valid": False, "errors": ["injected calibration bug"]},
        }
        with patch("bioagent_bench.reference.evaluate_task", return_value=invalid):
            with self.assertRaises(CalibrationError) as context:
                evaluate([2026], ["B01"])
        self.assertIn("injected calibration bug", str(context.exception))

    def test_evaluate_task_reports_degenerate_b03_calibration(self):
        degenerate = {
            "train_rows": [
                {"id": "t1", "sequence": "AAAA", "family": "p1"},
                {"id": "t2", "sequence": "CCCC", "family": "p2"},
            ],
            "public_input_rows": [
                {"id": "a", "sequence": "AAAA"},
                {"id": "b", "sequence": "AAAA"},
                {"id": "c", "sequence": "CCCC"},
                {"id": "d", "sequence": "CCCC"},
            ],
            "private_input_rows": [
                {"id": "a", "sequence": "AAAA"},
                {"id": "b", "sequence": "AAAA"},
                {"id": "c", "sequence": "CCCC"},
                {"id": "d", "sequence": "CCCC"},
            ],
            "truth_rows": [
                {"id": "a", "family": "x"},
                {"id": "b", "family": "x"},
                {"id": "c", "family": "y"},
                {"id": "d", "family": "y"},
            ],
        }
        with patch(
            "bioagent_bench.reference._generate_task_data", return_value=degenerate
        ):
            record = evaluate_task("B03", 2026)
        self.assertFalse(record["calibration"]["valid"])
        self.assertTrue(
            any("oracle" in error for error in record["calibration"]["errors"])
        )
        self.assertTrue(
            any(
                "calibration_error" in error
                for error in record["calibration"]["errors"]
            )
        )

    def test_evaluate_returns_raw_and_summary_components(self):
        result = evaluate([2026, 2027], ["B01", "B03"])
        self.assertEqual(result["seeds"], [2026, 2027])
        self.assertEqual(set(result["tasks"]), {"B01", "B03"})
        for task_id in ("B01", "B03"):
            for method in ("baseline", "simple", "reference", "oracle"):
                summary = result["tasks"][task_id][method]
                self.assertEqual(len(summary["scores"]), 2)
                self.assertLessEqual(summary["min"], summary["median"])
                self.assertLessEqual(summary["median"], summary["max"])
        self.assertEqual(len(result["records"]), 4)
        for method in ("baseline", "simple", "reference", "oracle"):
            self.assertEqual(len(result["macro"][method]["per_seed"]), 2)


if __name__ == "__main__":
    unittest.main()
