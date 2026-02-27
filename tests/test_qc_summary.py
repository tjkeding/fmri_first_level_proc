"""Tests for QC summary JSON output and censor warning thresholds."""

import json
import logging
import os
import pytest
import numpy as np
from unittest.mock import patch

from fmri_first_level_proc.first_level_utils import (
    write_qc_summary,
    create_censor_file,
    check_trial_survival,
    CENSOR_WARN_THRESHOLD,
    CENSOR_HIGH_THRESHOLD,
)

logger = logging.getLogger("test_qc_summary")


# ---------------------------------------------------------------------------
# write_qc_summary
# ---------------------------------------------------------------------------

class TestWriteQCSummary:
    """QC summary JSON should be written with expected keys."""

    def test_writes_json(self, tmp_path):
        qc_data = {
            "analysis_type": "task_act",
            "n_trs_total": 100,
            "n_trs_censored": 10,
            "pct_censored": 10.0,
            "fd_threshold": 0.9,
            "censor_prev_tr": False,
        }
        write_qc_summary(str(tmp_path), "test_subj", qc_data, logger)
        out_path = os.path.join(str(tmp_path), "test_subj_qc_summary.json")
        assert os.path.exists(out_path)

        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["analysis_type"] == "task_act"
        assert loaded["n_trs_total"] == 100
        assert loaded["pct_censored"] == 10.0

    def test_includes_per_condition_data(self, tmp_path):
        qc_data = {
            "analysis_type": "task_conn",
            "per_condition_trial_counts": {"stimA": 20, "stimB": 15},
            "per_condition_surviving_trials": {"stimA": 18, "stimB": 12},
        }
        write_qc_summary(str(tmp_path), "test_subj", qc_data, logger)
        out_path = os.path.join(str(tmp_path), "test_subj_qc_summary.json")
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["per_condition_trial_counts"]["stimA"] == 20
        assert loaded["per_condition_surviving_trials"]["stimB"] == 12

    def test_includes_per_run_censoring(self, tmp_path):
        qc_data = {
            "analysis_type": "rest_conn",
            "per_run_censoring": [
                {"run": 1, "n_trs_total": 300, "n_trs_censored": 15, "pct_censored": 5.0},
                {"run": 2, "n_trs_total": 300, "n_trs_censored": 30, "pct_censored": 10.0},
            ],
        }
        write_qc_summary(str(tmp_path), "test_subj", qc_data, logger)
        out_path = os.path.join(str(tmp_path), "test_subj_qc_summary.json")
        with open(out_path) as f:
            loaded = json.load(f)
        assert len(loaded["per_run_censoring"]) == 2
        assert loaded["per_run_censoring"][0]["pct_censored"] == 5.0


# ---------------------------------------------------------------------------
# Censor threshold warnings
# ---------------------------------------------------------------------------

class TestCensorWarnings:
    """create_censor_file should warn at 30% and 50% thresholds."""

    def _make_censor_file_with_pct(self, tmp_path, pct_censored, n_total=100):
        """Create a censor file with the given censoring percentage."""
        n_censored = int(n_total * pct_censored / 100.0)
        data = np.ones(n_total)
        data[:n_censored] = 0
        motion_path = str(tmp_path / "motion.txt")
        np.savetxt(motion_path, np.random.randn(n_total, 6), fmt="%.8f", delimiter="\t")
        return motion_path

    def test_no_warning_below_threshold(self, tmp_path, caplog):
        """< 30% censored → no warning."""
        censor_path = str(tmp_path / "out" / "test_concat_censor.1D")
        os.makedirs(os.path.dirname(censor_path), exist_ok=True)
        data = np.ones(100)
        data[:10] = 0  # 10% censored
        np.savetxt(censor_path, data, fmt="%d")

        # Skip generation, just test the logging path by calling with existing file
        with caplog.at_level(logging.WARNING):
            result = create_censor_file(
                str(tmp_path / "unused_motion.txt"), 0.9, False,
                str(tmp_path / "out"), "test", "concat", logger)
        assert "HIGH CENSORING" not in caplog.text
        assert "Elevated censoring" not in caplog.text

    def test_warning_at_30_pct(self, tmp_path, caplog):
        """>= 30% censored → elevated censoring warning."""
        censor_path = str(tmp_path / "out" / "test_concat_censor.1D")
        os.makedirs(os.path.dirname(censor_path), exist_ok=True)
        data = np.ones(100)
        data[:35] = 0  # 35% censored
        np.savetxt(censor_path, data, fmt="%d")

        with caplog.at_level(logging.WARNING):
            create_censor_file(
                str(tmp_path / "unused_motion.txt"), 0.9, False,
                str(tmp_path / "out"), "test", "concat", logger)
        assert "Elevated censoring" in caplog.text

    def test_warning_at_50_pct(self, tmp_path, caplog):
        """>= 50% censored → HIGH CENSORING warning."""
        censor_path = str(tmp_path / "out" / "test_concat_censor.1D")
        os.makedirs(os.path.dirname(censor_path), exist_ok=True)
        data = np.ones(100)
        data[:55] = 0  # 55% censored
        np.savetxt(censor_path, data, fmt="%d")

        with caplog.at_level(logging.WARNING):
            create_censor_file(
                str(tmp_path / "unused_motion.txt"), 0.9, False,
                str(tmp_path / "out"), "test", "concat", logger)
        assert "HIGH CENSORING" in caplog.text


# ---------------------------------------------------------------------------
# Trial survival QC
# ---------------------------------------------------------------------------

class TestTrialSurvival:
    """check_trial_survival should report surviving trials per condition."""

    def test_all_trials_survive(self, tmp_path):
        import pandas as pd
        censor_path = str(tmp_path / "censor.1D")
        np.savetxt(censor_path, np.ones(100))  # All TRs uncensored
        stim_df = pd.DataFrame({
            "CONDITION": ["stimA", "stimA", "stimB", "stimB"],
            "ONSET": [0.0, 4.0, 8.0, 12.0],
            "DURATION": [2.0, 2.0, 2.0, 2.0],
        })
        result = check_trial_survival(stim_df, ["stimA", "stimB"], censor_path, 0.8, logger)
        assert result["stimA"] == 2
        assert result["stimB"] == 2

    def test_some_trials_censored_but_enough_survive(self, tmp_path):
        import pandas as pd
        data = np.ones(100)
        data[0] = 0  # Censor TR 0 → stimA onset at 0.0 affected
        censor_path = str(tmp_path / "censor.1D")
        np.savetxt(censor_path, data)
        stim_df = pd.DataFrame({
            "CONDITION": ["stimA", "stimA", "stimA", "stimB", "stimB"],
            "ONSET": [0.0, 4.0, 8.0, 12.0, 16.0],
            "DURATION": [2.0, 2.0, 2.0, 2.0, 2.0],
        })
        result = check_trial_survival(stim_df, ["stimA", "stimB"], censor_path, 0.8, logger)
        assert result["stimA"] == 2  # One trial censored, two survive
        assert result["stimB"] == 2  # Both survive

    def test_fewer_than_2_trials_aborts(self, tmp_path):
        import pandas as pd
        data = np.ones(100)
        data[0] = 0  # Censor TR 0 → stimA onset at 0.0 affected
        censor_path = str(tmp_path / "censor.1D")
        np.savetxt(censor_path, data)
        stim_df = pd.DataFrame({
            "CONDITION": ["stimA", "stimA", "stimB", "stimB"],
            "ONSET": [0.0, 4.0, 8.0, 12.0],
            "DURATION": [2.0, 2.0, 2.0, 2.0],
        })
        with pytest.raises(SystemExit):
            check_trial_survival(stim_df, ["stimA", "stimB"], censor_path, 0.8, logger)
