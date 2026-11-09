from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fpg_comparison", ROOT / "scripts/evaluate_fpg_comparison.py")
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_reproduction_rejects_changed_metric() -> None:
    expected = {"harrell_c_index": 0.8, "ipcw_brier_at_horizon": 0.01}
    module.check_reproduction(expected, expected)
    with pytest.raises(ValueError, match="not reproduced"):
        module.check_reproduction({**expected, "harrell_c_index": 0.81}, expected)


def test_frozen_comparison_and_no_patient_export() -> None:
    document = json.loads((ROOT / "reports/fpg_comparison_metrics.json").read_text(encoding="utf-8"))
    assert document["inputs_unchanged"]
    assert document["models_fitted"] is False
    assert document["policy_retuned"] is False
    assert document["participant_level_output"] is None
    for cohort in document["results"].values():
        assert cohort["archived_adaptive_reproduced"]
        assert cohort["methods"]["base_plus_fpg"]["mean_cost"] == 1
        bootstrap = cohort["bootstrap"]
        assert bootstrap["replicates_requested"] == 200
        assert bootstrap["training_censoring_distribution_refit_per_replicate"] is False
        for metric, contrast in bootstrap["contrasts"]["adaptive_minus_base_plus_fpg"].items():
            points = bootstrap["methods"]
            expected = points["adaptive"][metric]["point_estimate"] - points["base_plus_fpg"][metric]["point_estimate"]
            assert contrast["point_estimate"] == pytest.approx(expected, abs=1e-12)
            assert contrast["valid_replicates"] == 200
            assert contrast["ci95_lower"] <= contrast["ci95_upper"]


def test_fpg_protocol_has_no_selection_step() -> None:
    config = json.loads((ROOT / "configs/fpg_comparison.json").read_text())
    assert config["methods"] == ["base_plus_fpg", "adaptive"]
    assert config["refit_models"] is config["retune_policy"] is config["new_dataset"] is False
    assert config["contrasts"] == [["adaptive", "base_plus_fpg"]]
