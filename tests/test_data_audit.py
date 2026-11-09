import unittest
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import openpyxl

from t2d_aaf.data.audit import BASE, LABS, RAW_COLUMNS, composite_evidence, horizon_state, number, run, target


class TargetAuditTests(unittest.TestCase):
    def test_event_at_horizon_is_case(self):
        self.assertEqual(horizon_state(1, 3, 3), "event_by_h")

    def test_censoring_at_horizon_is_unknown(self):
        self.assertEqual(horizon_state(0, 3, 3), "censored_at_or_before_h")

    def test_event_after_horizon_is_event_free_at_h(self):
        self.assertEqual(horizon_state(1, 3.001, 3), "known_event_free_at_h")

    def test_censor_before_and_after_horizon(self):
        self.assertEqual(horizon_state(0, 2, 3), "censored_at_or_before_h")
        self.assertEqual(horizon_state(0, 4, 3), "known_event_free_at_h")

    def test_invalid_targets_rejected(self):
        for event, duration in ((2, 3), (0.5, 3), (1, 0), (0, -1), ("", 3), (1, "nan")):
            with self.subTest(event=event, duration=duration), self.assertRaises(ValueError):
                target(event, duration)

    def test_invalid_horizons_rejected(self):
        for horizon in (0, -1, float("inf")):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                horizon_state(0, 3, horizon)

    def test_positive_lab_without_flag(self):
        self.assertEqual(composite_evidence(7, None), "positive_component")

    def test_positive_flag_with_missing_or_low_lab(self):
        for lab in (None, 5.1):
            self.assertEqual(composite_evidence(lab, 1), "positive_component")

    def test_missing_components_not_asserted_negative(self):
        self.assertEqual(composite_evidence(None, None), "insufficient_components")
        self.assertEqual(composite_evidence(None, 0), "insufficient_components")

    def test_low_lab_is_only_no_observed_positive_component(self):
        self.assertEqual(composite_evidence(6.99, None), "no_positive_component")

    def test_bad_flags_and_nonfinite_rejected(self):
        for lab, flag in ((6, 2), (float("nan"), None), (7, float("inf"))):
            with self.subTest(lab=lab, flag=flag), self.assertRaises(ValueError):
                composite_evidence(lab, flag)

    def test_blank_values_and_numeric_conversion(self):
        self.assertIsNone(number(" ", missing_ok=True))
        self.assertEqual(number("6.5"), 6.5)
        with self.assertRaises(ValueError):
            number(None)


class IntegrationTests(unittest.TestCase):
    def fixture(self, root, *, mismatch=False):
        (root / "data/raw").mkdir(parents=True)
        (root / "data/processed").mkdir(parents=True)
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "RC"
        sheet.append(list(RAW_COLUMNS.values()))
        sheet.append([1, 1, 2.5, 5, 7.2, None])
        sheet.append([2, 0, 4, 4.8, 5.1, None])
        sheet.append([3, 1, 3, 5.2, None, 1])
        workbook.save(root / "data/raw/RC+Health+Care+Data-20180820.xlsx")
        workbook.close()
        columns = ["participant_id", "split", "event", "duration_years", "site", *BASE, *LABS]
        with (root / "data/processed/primary_cohort.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for pid, split, event, duration, fpg in (
                (1, "train", 0 if mismatch else 1, 2.5, 5),
                (2, "validation", 0, 4, 4.8), (3, "test", 1, 3, 5.2)):
                row = {name: 1 for name in BASE + LABS}
                row.update(participant_id=pid, split=split, event=event,
                           duration_years=duration, site=0, fpg=fpg)
                writer.writerow(row)
        (root / "data/processed/primary_cohort_manifest.json").write_text(
            json.dumps({"raw_row_count": 3, "cohort_row_count": 3}), encoding="utf-8")

    def test_synthetic_end_to_end_has_only_aggregates(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            self.fixture(root)
            result = run(root, root / "audit.json")
            self.assertEqual(result["processed_n"], 3)
            self.assertFalse(any(result["violations"].values()))
            self.assertTrue(result["inputs_unchanged"])
            self.assertFalse(result["participant_records_exported"])
            self.assertEqual(result["horizon_counts"]["all@3"]["event_by_h"], 2)
            self.assertEqual(result["raw_counts"]["events_all_followup"], 2)

    def test_raw_processed_mismatch_is_detected(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            self.fixture(root, mismatch=True)
            result = run(root, root / "audit.json")
            self.assertEqual(result["violations"]["event_mismatch"], 1)

    def test_cannot_write_into_input_tree(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                run(root, root / "data/processed/audit.json")


if __name__ == "__main__":
    unittest.main()
