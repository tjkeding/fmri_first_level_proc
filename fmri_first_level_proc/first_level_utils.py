#!/usr/bin/env python3

# ============================================================================
# SHARED UTILITIES FOR fMRI FIRST-LEVEL PROCESSING
# Common functions used across task_act, task_conn, and rest_conn pipelines.
#
# Author: Taylor J. Keding, Ph.D.
# Version: 2.0
# Last updated: 02/17/26
# ============================================================================

import json
import os
import sys
import shutil
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from io import StringIO

# ----------------------------------------------------------------------------
# Censoring QC thresholds (warn only — never abort based on percentage)
# ----------------------------------------------------------------------------
CENSOR_WARN_THRESHOLD = 30.0    # % censored → warning
CENSOR_HIGH_THRESHOLD = 50.0    # % censored → strong warning

# ----------------------------------------------------------------------------
# General Helpers
# ----------------------------------------------------------------------------

def dir_path_exists(path_string):
    """argparse type: ensure directory exists (create if needed)."""
    if not os.path.isdir(path_string):
        os.makedirs(path_string, exist_ok=True)
    return path_string

def file_path_exists(path_string):
    """argparse type: ensure a single file path exists."""
    if not os.path.isfile(path_string):
        raise argparse.ArgumentTypeError(
            f"[ERROR] The file '{path_string}' does not exist or is not a valid file."
        )
    return path_string

def prepare_motion_file(motion_path, use_columns, out_dir, out_prefix, label, logger):
    """
    Validate and prepare a motion regressor file for AFNI.

    Reads the motion file, validates it has >= use_columns columns, and writes
    a truncated version (first use_columns columns only) to out_dir. Returns
    the path to the prepared file.

    Parameters
    ----------
    motion_path : str
        Path to the original motion file.
    use_columns : int
        Number of columns to keep (6 for base, 12 for +derivs, etc.).
    out_dir : str
        Directory for the prepared file.
    out_prefix : str
        Filename prefix for the prepared file.
    label : str
        Descriptive label for logging (e.g. "run1", "concat").
    logger : logging.Logger

    Returns
    -------
    str
        Path to the prepared motion file.
    """
    try:
        data = np.loadtxt(motion_path)
    except Exception as e:
        logger.error("Cannot read motion file '%s': %s", motion_path, e)
        sys.exit(1)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    ncols = data.shape[1]
    if ncols < use_columns:
        logger.error("Motion file '%s' has %d columns, expected >= %d.",
                     motion_path, ncols, use_columns)
        sys.exit(1)
    # Truncate to requested columns
    truncated = data[:, :use_columns]
    prepared_path = os.path.join(out_dir, f"{out_prefix}_{label}_motion_prepared.txt")
    np.savetxt(prepared_path, truncated, fmt="%.8f", delimiter="\t")
    if ncols > use_columns:
        logger.info("Motion file '%s': using %d of %d columns.", motion_path, use_columns, ncols)
    return prepared_path

def valid_string_list(string_list):
    """argparse type: parse comma-separated string into a list of stripped, non-empty strings."""
    try:
        conds = string_list.split(",")
        conds_s = [cond.strip() for cond in conds]
        for cond in conds_s:
            if cond is None or cond == "":
                raise argparse.ArgumentTypeError(
                    f"[ERROR] '{string_list}' could not be parsed; "
                    "use string-list notation (e.g. 'cond1label,cond2label...')"
                )
        return conds_s
    except argparse.ArgumentTypeError:
        raise
    except (ValueError, IndexError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"[ERROR] '{string_list}' could not be parsed; "
            "use string-list notation (e.g. 'cond1label,cond2label...')"
        )

def valid_stat_type(stat_string):
    """argparse type: validate extract stat is one of z, t, r."""
    valid_types = ["z", "t", "r"]
    if stat_string not in valid_types:
        raise argparse.ArgumentTypeError("[ERROR] --extract_stat must be 'z', 't', or 'r'")
    return stat_string

def valid_ave_type(ave_string):
    """argparse type: validate average type is mean or median."""
    if ave_string not in ["mean", "median"]:
        raise argparse.ArgumentTypeError(
            f"[ERROR] '{ave_string}' must be one of ['mean', 'median']."
        )
    return ave_string

def valid_conn_type(conn_string):
    """argparse type: validate connectivity type."""
    if conn_string not in ["seed_to_voxel", "parcellated"]:
        raise argparse.ArgumentTypeError(
            f"[ERROR] '{conn_string}' must be one of ['seed_to_voxel', 'parcellated']."
        )
    return conn_string

def setup_logging(script_name, log_file=None):
    """
    Configure and return a logger with console output (and optional file output).

    Parameters
    ----------
    script_name : str
        Name used in log format, e.g. "task_act_first_level".
    log_file : str, optional
        Path to a log file. If provided, logs are also written to this file.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

def validate_extract_options(extract_flag_name, average_type, extract_out_file_pre,template_path, scan_path, out_dir, force_diff_atlas,logger=None):

    """Validate extraction-related args for task_conn or rest_conn pipelines.

    Parameters
    ----------
    extract_flag_name : str
        Name of the extract flag for error messages (e.g. '--extract_pbseries').
    average_type : str or None
    extract_out_file_pre : str or None
    template_path : str or None
    scan_path : str
        Path to the scan (used for template validation).
    out_dir : str
    force_diff_atlas : bool
    logger : logging.Logger, optional

    Returns
    -------
    str
        Validated (possibly resampled) template_path.
    """
    _log = logger or logging.getLogger(__name__)

    if average_type is None:
        _log.error("--average_type must be provided if %s is present.", extract_flag_name)
        sys.exit(1)

    if extract_out_file_pre is None:
        _log.error("--extract_out_file_pre must be provided if %s is present.", extract_flag_name)
        sys.exit(1)

    if template_path is None:
        _log.error("%s requires an ROI or ROI set (.nii) be provided to --template_path.", extract_flag_name)
        sys.exit(1)

    return validate_template(scan_path, template_path, out_dir, force_diff_atlas,
                              conn_type="extract", logger=logger)

def validate_connectivity_options(calc_conn, conn_out_file_pre, template_path,scan_path, out_dir, force_diff_atlas,data_label="data", logger=None):

    """Validate connectivity-related args for task_conn or rest_conn pipelines.

    Parameters
    ----------
    calc_conn : str or None
        'seed_to_voxel' or 'parcellated'.
    conn_out_file_pre : str or None
    template_path : str or None
    scan_path : str
        Path to scan (used for template validation).
    out_dir : str
    force_diff_atlas : bool
    data_label : str
        Label for log messages (e.g. 'task beta series', 'residual time series').
    logger : logging.Logger, optional

    Returns
    -------
    str or None
        Validated (possibly resampled) template_path, or None if calc_conn is None.
    """
    _log = logger or logging.getLogger(__name__)

    if calc_conn is None:
        _log.info("Connectivity will not be output after generating %s.", data_label)
        return template_path

    if conn_out_file_pre is None:
        _log.error("--conn_out_file_pre must be provided if --calc_conn is present.")
        sys.exit(1)

    if calc_conn == "parcellated":
        if template_path is None:
            _log.error("--calc_conn '%s' requires an ROI set (.nii) be provided to --template_path.", calc_conn)
            sys.exit(1)
        validated = validate_template(scan_path, template_path, out_dir, force_diff_atlas,
                                       conn_type="parcellated", logger=logger)
        _log.info("Calculating parcellated (ROI-based) connectivity matrices after generating %s.", data_label)
    else:
        if template_path is None:
            _log.error("--calc_conn '%s' requires a seed ROI mask file (binary .nii) be provided to --template_path.", calc_conn)
            sys.exit(1)
        validated = validate_template(scan_path, template_path, out_dir, force_diff_atlas,
                                       conn_type="seed_to_voxel", logger=logger)
        _log.info("Calculating seed-to-voxel whole-brain connectivity (.nii) after generating %s.", data_label)

    return validated

def remove_files_from_dir(dir_string, logger=None):
    """Remove all files, links, and subdirectories from a directory."""
    _log = logger or logging.getLogger(__name__)
    resolved = os.path.realpath(dir_string)
    if resolved == "/" or resolved == os.path.expanduser("~") or len(resolved) < 5:
        _log.error("Refusing to clear dangerous path: '%s' (resolved: '%s').", dir_string, resolved)
        sys.exit(1)
    if not os.path.isdir(dir_string):
        _log.warning("'%s' does not yet exist. --remove_previous will be ignored.", dir_string)
    else:
        for item_name in os.listdir(dir_string):
            item_path = os.path.join(dir_string, item_name)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except OSError as e:
                _log.error("Error deleting '%s': %s", item_path, e)

# ----------------------------------------------------------------------------
# HRF Model Helpers
# ----------------------------------------------------------------------------

# HRF model categories
HRF_DURATION_MODELS = {"GAM", "BLOCK"}       # Use mean_dur in model string
HRF_DM_MODELS = {"dmBLOCK"}                  # Per-trial duration via married timing
HRF_IMPULSE_MODELS = {"SPMG1"}               # No duration parameter
VALID_HRF_MODELS = HRF_DURATION_MODELS | HRF_DM_MODELS | HRF_IMPULSE_MODELS | {"custom"}

# Known AFNI model name prefixes for custom HRF validation (first pass)
AFNI_MODEL_PREFIXES = {
    "GAM", "BLOCK", "dmBLOCK", "dmUBLOCK", "TENT", "TENTzero",
    "CSPLIN", "CSPLINzero", "SPMG1", "SPMG2", "SPMG3",
    "TWOGAMpw", "MION", "MIONN", "WAV", "EXPR", "POLY", "SIN",
}

def validate_custom_hrf(custom_hrf, logger):
    """
    Two-stage validation of a custom HRF string:
    1. Prefix check: verify the string starts with a known AFNI model name
    2. AFNI dry run: use 3dDeconvolve -nodata to validate full syntax

    Returns True if valid, False otherwise (errors logged).
    """
    import tempfile

    # Stage 1: prefix whitelist
    prefix_match = any(custom_hrf == name or custom_hrf.startswith(name + "(")
                       for name in AFNI_MODEL_PREFIXES)
    if not prefix_match:
        known = ", ".join(sorted(AFNI_MODEL_PREFIXES))
        logger.error("custom_hrf '%s' does not start with a known AFNI model name. "
                     "Known models: %s", custom_hrf, known)
        return False

    # Stage 2: AFNI -nodata dry run
    tmp_timing = None
    tmp_x1d = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.1D', delete=False) as f:
            f.write("10\n")
            tmp_timing = f.name
        tmp_x1d = tmp_timing.replace('.1D', '_test.x1D')
        cmd = ["3dDeconvolve", "-nodata", "100", "1",
               "-polort", "-1", "-num_stimts", "1",
               "-stim_times", "1", tmp_timing, custom_hrf,
               "-x1D", tmp_x1d, "-x1D_stop", "-nobucket"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("custom_hrf '%s' failed AFNI validation. "
                         "3dDeconvolve error: %s", custom_hrf,
                         result.stderr.strip())
            return False
        return True
    finally:
        for p in (tmp_timing, tmp_x1d):
            if p and os.path.exists(p):
                os.unlink(p)

def needs_married_timing(hrf_model, custom_hrf=None):
    """Check if the HRF model requires married timing (onset*duration).

    Handles both named models (e.g. 'dmBLOCK') and custom strings
    (e.g. hrf_model='custom', custom_hrf='dmBLOCK(1)').
    """
    if hrf_model in HRF_DM_MODELS:
        return True
    if hrf_model == "custom" and custom_hrf is not None:
        return any(custom_hrf == name or custom_hrf.startswith(name + "(")
                   for name in HRF_DM_MODELS)
    return False

def validate_onset_file_format(onset_file, expects_married, logger=None):
    """Check if onset file format (married vs single-column) matches expectations.

    Married timing files contain '*' characters (e.g. '10.5*3.0').
    Single-column files contain only onset times.

    Returns True if format matches expectations, False if mismatch.
    """
    _log = logger or logging.getLogger(__name__)
    try:
        with open(onset_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    has_star = '*' in line
                    if has_star != expects_married:
                        fmt_found = "married" if has_star else "single-column"
                        fmt_expected = "married" if expects_married else "single-column"
                        _log.warning(
                            "Onset file '%s' is in %s format but %s format is expected. "
                            "This file may be stale from a previous run with a different HRF model. "
                            "Consider using --remove_previous or deleting this file.",
                            onset_file, fmt_found, fmt_expected)
                        return False
                    return True
    except OSError as e:
        _log.warning("Could not read onset file '%s' for format validation: %s", onset_file, e)
    return True  # If file is empty or unreadable, don't block — downstream will catch errors

def build_hrf_string(hrf_model, mean_dur=None, custom_hrf=None):
    """Construct AFNI stim_times model string for the selected HRF."""
    if hrf_model == "custom":
        if not custom_hrf:
            raise ValueError("custom_hrf is required when hrf_model='custom'")
        return custom_hrf
    if hrf_model == "GAM":
        return f"GAM(8.6,.547,{mean_dur})"
    if hrf_model == "BLOCK":
        return f"BLOCK({mean_dur},1)"
    if hrf_model == "dmBLOCK":
        return "dmBLOCK(0)"
    if hrf_model == "SPMG1":
        return "SPMG1"
    raise ValueError(f"Unknown hrf_model: {hrf_model}")

# ----------------------------------------------------------------------------
# Stim Data & Onset File Helpers
# ----------------------------------------------------------------------------

def read_and_validate_stim_data(timing_path, cond_labels, logger=None):
    """Read task timing CSV, validate columns, sort by onset, check conditions exist.

    Parameters
    ----------
    timing_path : str
        Path to the timing CSV with CONDITION, ONSET, DURATION columns.
    cond_labels : list of str
        Expected condition labels (must all exist in the timing file).
    logger : logging.Logger, optional

    Returns
    -------
    pd.DataFrame
        Sorted by ONSET, reset index.
    """
    _log = logger or logging.getLogger(__name__)

    stim_df = pd.read_csv(timing_path, sep=',')
    stim_df.columns = stim_df.columns.str.strip()

    for curr_col in ["CONDITION", "ONSET", "DURATION"]:
        if curr_col not in stim_df.columns.to_list():
            _log.error("Timing file did not contain necessary columns 'CONDITION', 'ONSET', and 'DURATION'")
            sys.exit(1)

    sorted_df = stim_df.sort_values(by='ONSET', ascending=True)

    # Check conditions exist in the timing file
    missing = [c for c in cond_labels if c not in sorted_df['CONDITION'].unique()]
    if missing:
        _log.error("Task condition(s) not found in timing file: %s", missing)
        sys.exit(1)

    # Check for conditions with too few trials
    for cond in cond_labels:
        n_trials = len(sorted_df[sorted_df['CONDITION'] == cond])
        if n_trials == 0:
            _log.error("Condition '%s' has 0 trials in the timing file. "
                       "Cannot create onset file for a condition with no events.", cond)
            sys.exit(1)
        elif n_trials < 2:
            _log.error("Condition '%s' has only %d trial(s). At least 2 trials are required "
                       "for a reliable estimate.", cond, n_trials)
            sys.exit(1)
        elif n_trials < 3:
            _log.warning("Condition '%s' has only %d trial(s). Estimates may be unreliable.", cond, n_trials)

    return sorted_df.reset_index(drop=True)

def write_onset_file(condition_df, filepath, hrf_model, custom_hrf=None, logger=None):
    """Write AFNI-compatible onset file. Validates format on existing files.

    Parameters
    ----------
    condition_df : pd.DataFrame
        Must contain 'ONSET' column, and 'DURATION' if married timing needed.
    filepath : str
        Output onset file path.
    hrf_model : str
        HRF model name.
    custom_hrf : str or None
        Custom HRF string (used when hrf_model='custom').
    logger : logging.Logger, optional

    Returns
    -------
    bool
        True if file was created or already exists with correct format.
    """
    _log = logger or logging.getLogger(__name__)
    married = needs_married_timing(hrf_model, custom_hrf)

    if os.path.exists(filepath):
        validate_onset_file_format(filepath, married, logger=logger)
        _log.info("%s already exists.", filepath)
        return True

    if married:
        with open(filepath, 'w') as f:
            for _, row in condition_df.iterrows():
                f.write(f"{row['ONSET']}*{row['DURATION']}\n")
    else:
        with open(filepath, 'w') as f:
            for onset in condition_df['ONSET']:
                f.write(f"{onset}\n")
    _log.info("Created %s", filepath)
    return True

# ----------------------------------------------------------------------------
# AFNI Helpers
# ----------------------------------------------------------------------------

def create_censor_file(motion_path, fd_threshold, censor_prev_tr, out_dir, out_prefix, label, logger):
    """
    Generate a censor file from motion regressors using AFNI's 1d_tool.py.

    Extracts the first 6 columns (base motion parameters) from the motion file,
    then runs 1d_tool.py to create an FD-based censor file.

    Parameters
    ----------
    motion_path : str
        Path to the original motion file (before prepare_motion_file).
    fd_threshold : float
        Framewise displacement threshold (mm) for censoring.
    censor_prev_tr : bool
        If True, also censor the TR preceding each high-motion TR.
    out_dir : str
        Output directory.
    out_prefix : str
        Filename prefix.
    label : str
        Descriptive label (e.g. "concat", "run1").
    logger : logging.Logger

    Returns
    -------
    str
        Path to the generated censor file.
    """
    # 1d_tool.py auto-appends _censor.1D to the prefix
    censor_prefix = os.path.join(out_dir, f"{out_prefix}_{label}")
    censor_path = f"{censor_prefix}_censor.1D"

    # Skip if censor file already exists (1d_tool.py won't overwrite)
    if os.path.exists(censor_path):
        logger.info("Censor file %s already exists (skipping generation).", censor_path)
    else:
        # Extract first 6 columns (base motion only) for FD calculation
        motion_6col = prepare_motion_file(motion_path, 6, out_dir, out_prefix, f"{label}_censor_src", logger)

        # Build 1d_tool.py command
        cmd = ["1d_tool.py",
               "-infile", motion_6col,
               "-censor_motion", str(fd_threshold), censor_prefix]
        if censor_prev_tr:
            cmd.append("-censor_prev_TR")

        run_afni_command(cmd, description=f"1d_tool.py censor {label}", logger=logger)

        if not os.path.exists(censor_path):
            logger.error("Expected censor file not found: %s", censor_path)
            sys.exit(1)

    # Log count of censored TRs and warn at thresholds
    try:
        censor_data = np.loadtxt(censor_path)
        n_censored = int(np.sum(censor_data == 0))
        n_total = len(censor_data)
        pct_censored = 100.0 * n_censored / n_total if n_total > 0 else 0.0
        logger.info("Censor file %s: %d of %d TRs censored (%.1f%%).",
                     censor_path, n_censored, n_total, pct_censored)
        if pct_censored >= CENSOR_HIGH_THRESHOLD:
            logger.warning("HIGH CENSORING: %.1f%% of TRs censored (>= %.0f%%). "
                           "Results may be unreliable. Review motion QC carefully.",
                           pct_censored, CENSOR_HIGH_THRESHOLD)
        elif pct_censored >= CENSOR_WARN_THRESHOLD:
            logger.warning("Elevated censoring: %.1f%% of TRs censored (>= %.0f%%). "
                           "Consider reviewing motion QC.",
                           pct_censored, CENSOR_WARN_THRESHOLD)
    except Exception:
        pass  # Non-critical; censor file exists and will be used

    return censor_path

def check_trial_survival(stim_df, cond_labels, censor_path, tr, logger):
    """Check how many trials per condition survive motion censoring.

    For each condition, count how many trials have their onset TR uncensored.
    Warns if any condition has few surviving trials. Returns a dict of counts.

    Parameters
    ----------
    stim_df : pd.DataFrame
        Timing data with CONDITION and ONSET columns.
    cond_labels : list of str
        Condition labels to check.
    censor_path : str
        Path to the censor file (binary 1D).
    tr : float
        Repetition time in seconds.
    logger : logging.Logger

    Returns
    -------
    dict
        {condition: n_surviving_trials} for each condition.
    """
    try:
        censor_data = np.loadtxt(censor_path)
    except Exception:
        logger.warning("Could not read censor file for trial survival check.")
        return {}

    survival = {}
    for cond in cond_labels:
        cond_df = stim_df[stim_df['CONDITION'] == cond]
        n_total = len(cond_df)
        n_surviving = 0
        for onset in cond_df['ONSET']:
            tr_idx = int(round(onset / tr))
            if 0 <= tr_idx < len(censor_data) and censor_data[tr_idx] == 1:
                n_surviving += 1
        survival[cond] = n_surviving
        if n_surviving < 2:
            logger.error("Condition '%s': only %d of %d trials survive censoring. "
                         "Cannot reliably estimate this condition (minimum 2 required). Aborting.",
                         cond, n_surviving, n_total)
            sys.exit(1)
        elif n_surviving == 2:
            logger.warning("Condition '%s': only 2 of %d trials survive censoring. "
                           "Estimate will have very low power.", cond, n_total)
        elif n_surviving < 5:
            logger.warning("Condition '%s': only %d of %d trials survive censoring. "
                           "Estimates may be unreliable.", cond, n_surviving, n_total)
        else:
            logger.info("Condition '%s': %d of %d trials survive censoring.",
                        cond, n_surviving, n_total)
    return survival


def notch_filter_motion(motion_path, tr, stopband, out_dir, out_prefix, label, logger):
    """
    Apply a notch (stopband) filter to motion parameters using AFNI's 3dTproject.

    Used to remove respiratory artifacts from motion parameters before FD-based
    censoring (Fair et al., 2020).

    Parameters
    ----------
    motion_path : str
        Path to the original motion file.
    tr : float
        Repetition time in seconds.
    stopband : list of float
        [low, high] Hz band to remove.
    out_dir : str
        Output directory.
    out_prefix : str
        Filename prefix.
    label : str
        Descriptive label (e.g. "run1").
    logger : logging.Logger

    Returns
    -------
    str
        Path to the notch-filtered motion file.
    """
    # Extract first 6 columns (base motion only)
    motion_6col = prepare_motion_file(motion_path, 6, out_dir, out_prefix, f"{label}_notch_src", logger)

    out_path = os.path.join(out_dir, f"{out_prefix}_{label}_notch_motion.1D")

    # 3dTproject on 1D file: use \' (AFNI transpose) so rows=timepoints, cols=params
    cmd = ["3dTproject",
           "-polort", "-1",
           "-dt", str(tr),
           "-input", f"{motion_6col}\\'",
           "-stopband", str(stopband[0]), str(stopband[1]),
           "-prefix", out_path]

    run_afni_command(cmd, description=f"3dTproject notch filter {label}", logger=logger)

    if not os.path.exists(out_path):
        logger.error("Expected notch-filtered motion file not found: %s", out_path)
        sys.exit(1)

    logger.info("Notch-filtered motion parameters saved to %s", out_path)
    return out_path

def compute_tissue_derivative(tissue_path, out_dir, out_prefix, tissue_label, logger):
    """
    Compute the first temporal derivative of a tissue signal file.

    Reads the tissue signal, computes backward difference (zero-padded to
    preserve length), and saves to out_dir.

    Parameters
    ----------
    tissue_path : str
        Path to the tissue signal file (single-column text).
    out_dir : str
        Output directory.
    out_prefix : str
        Filename prefix.
    tissue_label : str
        Label for the tissue type (e.g. "CSF", "WM", "run1_CSF").
    logger : logging.Logger

    Returns
    -------
    str
        Path to the derivative file.
    """
    try:
        data = np.loadtxt(tissue_path)
    except Exception as e:
        logger.error("Cannot read tissue file '%s': %s", tissue_path, e)
        sys.exit(1)
    if data.ndim != 1:
        logger.error("Tissue file '%s' must be single-column, got shape %s.", tissue_path, data.shape)
        sys.exit(1)
    deriv = np.diff(data, prepend=data[0])
    deriv_path = os.path.join(out_dir, f"{out_prefix}_{tissue_label}_deriv.txt")
    np.savetxt(deriv_path, deriv, fmt="%.8f")
    logger.info("Created tissue derivative: %s", deriv_path)
    return deriv_path

def build_decon_base_command(scan_path, censor_path, motion_path,CSF_path=None, WM_path=None, CSF_deriv_path=None, WM_deriv_path=None, tr=None):

    """Build base 3dDeconvolve command with standard nuisance regressors.

    Returns list of str containing: -quiet, -input, -polort A, -censor,
    motion ortvec, optional CSF/WM ortvecs (plus their derivatives),
    and optional -TR_times.
    """
    cmd = ["3dDeconvolve", "-quiet",
           "-input", scan_path,
           "-polort", "A",
           "-censor", censor_path,
           "-ortvec", motion_path, "motion_regressors"]
    if tr is not None:
        cmd.extend(["-TR_times", str(tr)])
    if CSF_path is not None:
        cmd.extend(["-ortvec", CSF_path, "CSF_regressor"])
    if WM_path is not None:
        cmd.extend(["-ortvec", WM_path, "WM_regressor"])
    if CSF_deriv_path is not None:
        cmd.extend(["-ortvec", CSF_deriv_path, "CSF_deriv_regressor"])
    if WM_deriv_path is not None:
        cmd.extend(["-ortvec", WM_deriv_path, "WM_deriv_regressor"])
    return cmd

def clean_deconvolve_err(out_dir=None):
    """Remove 3dDeconvolve.err file if it exists (always generated by AFNI).

    Checks both the current working directory and out_dir (if provided).
    """
    for directory in filter(None, [".", out_dir]):
        err_path = os.path.join(directory, "3dDeconvolve.err")
        if os.path.exists(err_path):
            os.remove(err_path)

def log_hrf_model(hrf_model, custom_hrf, logger):
    """Log the HRF model being used."""
    logger.info("HRF model: %s", hrf_model if hrf_model != "custom" else f"custom ({custom_hrf})")

def log_nuisance_model(CSF_path, WM_path, logger, use_tissue_derivs=False):
    """Log the nuisance regressors being used."""
    nuisance_parts = ["motion"]
    if CSF_path is not None:
        nuisance_parts.append("CSF")
        if use_tissue_derivs:
            nuisance_parts.append("CSF_deriv")
    if WM_path is not None:
        nuisance_parts.append("WM")
        if use_tissue_derivs:
            nuisance_parts.append("WM_deriv")
    logger.info("Nuisance regressors: %s", ", ".join(nuisance_parts))

def run_afni_command(command, capture_output=False, description="", logger=None):
    """
    Run an AFNI command via subprocess with standardized error handling.

    Parameters
    ----------
    command : list of str
        The command and arguments.
    capture_output : bool
        If True, capture and return stdout.
    description : str
        Human-readable description for logging.
    logger : logging.Logger, optional

    Returns
    -------
    subprocess.CompletedProcess or str
        If capture_output is True, returns stdout as a string.
        Otherwise returns the CompletedProcess object.

    Raises
    ------
    subprocess.CalledProcessError
        If the command exits with non-zero status.
    """
    if logger:
        logger.debug("Running: %s", " ".join(command))

    try:
        env = os.environ.copy()
        env['AFNI_COMPRESSOR'] = 'GZIP'
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        if capture_output:
            return result.stdout
        return result
    except subprocess.CalledProcessError as e:
        desc = f" ({description})" if description else ""
        msg = f"AFNI command failed{desc}: {' '.join(command)}"
        if e.stderr:
            msg += f"\n  stderr: {e.stderr.strip()}"
        if logger:
            logger.error(msg)
        raise

def get_3dinfo_properties(nifti_path, logger=None):
    """
    Get key properties from a NIfTI file using targeted 3dinfo flags.

    Returns
    -------
    dict with keys: 'space', 'orient', 'spacing', 'dmax'
    """
    props = {}
    queries = {
        "space": ["-space"],
        "orient": ["-orient"],
        "spacing": ["-ad3"],
        "dmax": ["-dmax"],
    }
    for key, flags in queries.items():
        try:
            out = run_afni_command(
                ["3dinfo"] + flags + [nifti_path],
                capture_output=True,
                description=f"3dinfo {flags[0]} {nifti_path}",
                logger=logger,
            )
            props[key] = out.strip()
        except subprocess.CalledProcessError:
            if logger:
                logger.error("Could not retrieve %s from %s with 3dinfo.", key, nifti_path)
            props[key] = None

    return props

# ----------------------------------------------------------------------------
# Template/ROI Helpers
# ----------------------------------------------------------------------------

def validate_template(scan_path, template_path, out_dir, force_diff_atlas,conn_type=None, logger=None):

    """
    Validate a template NIfTI against a scan, checking atlas space, orientation,
    grid spacing, and ROI content. Resample template if grid mismatch detected.

    Parameters
    ----------
    scan_path : str
        Path to the input scan NIfTI.
    template_path : str
        Path to the template NIfTI.
    out_dir : str
        Output directory (for resampled templates).
    force_diff_atlas : bool
        If True, allow atlas mismatch with a warning.
    conn_type : str or None
        None = skip binary/ROI check (e.g. task_act extraction-only)
        'seed_to_voxel' = require binary mask (max=1)
        'parcellated' = require 2+ ROIs (max>=2)
        'extract' = any non-empty template
    logger : logging.Logger, optional

    Returns
    -------
    str
        Path to the (possibly resampled) template.
    """
    _log = logger or logging.getLogger(__name__)

    # Get template properties
    tpl = get_3dinfo_properties(template_path, logger=logger)
    if any(v is None for v in tpl.values()):
        _log.error("Could not get info from --template_path %s with AFNI's 3dinfo - please check your template.", template_path)
        sys.exit(1)

    # Validate ROI content via dmax
    try:
        dmax_val = float(tpl["dmax"])
        int_upper = int(dmax_val)
    except ValueError:
        _log.error("Template provided to --template_path contained non-numeric data max value ('%s') - please check your template.", tpl["dmax"])
        sys.exit(1)

    if dmax_val != int_upper:
        _log.error("Template provided to --template_path contained floats, and should only contain ints (ROI set or binary mask) - please check your template.")
        sys.exit(1)
    if int_upper == 0:
        _log.error("Template provided to --template_path contained no data - please check your template.")
        sys.exit(1)

    # conn_type-specific checks
    if conn_type == "parcellated" and int_upper < 2:
        _log.error("Template provided to --template_path contained a binary mask. With --calc_conn 'parcellated', template must have at least 2 ROIs.")
        sys.exit(1)
    if conn_type == "seed_to_voxel" and int_upper > 1:
        _log.error("Template provided to --template_path contained an ROI set. With --calc_conn 'seed_to_voxel', template must be a binary mask for a single ROI.")
        sys.exit(1)

    # Get scan properties
    scn = get_3dinfo_properties(scan_path, logger=logger)
    if any(v is None for v in [scn["space"], scn["orient"], scn["spacing"]]):
        _log.error("Could not get info from --scan_path %s with AFNI's 3dinfo - please check your scan.", scan_path)
        sys.exit(1)

    _log.info("Specs for scan %s and template %s", scan_path, template_path)
    _log.info("  orient template: %s | orient scan: %s", tpl["orient"], scn["orient"])
    _log.info("  atlas  template: %s | atlas  scan: %s", tpl["space"], scn["space"])
    _log.info("  spacing template: %s | spacing scan: %s", tpl["spacing"], scn["spacing"])

    # Atlas space check
    if tpl["space"] != scn["space"]:
        if force_diff_atlas:
            _log.warning("%s and %s are in different standard template spaces, but proceeding anyway with --force_diff_atlas.", scan_path, template_path)
        else:
            _log.error("Atlas space mismatch between --scan_path %s and --template_path %s - scan and template must be registered to the same atlas.", scan_path, template_path)
            sys.exit(1)

    # Orientation / grid spacing check — resample if mismatch
    if tpl["orient"] != scn["orient"] or tpl["spacing"] != scn["spacing"]:
        _log.warning("Orientation and/or grid spacing for --scan_path %s and --template_path %s do not match.", scan_path, template_path)
        _log.info("Resampling --template_path %s to match --scan_path %s.", template_path, scan_path)

        template_base = os.path.basename(template_path)
        template_file_pre = template_base.replace(".nii.gz", "").replace(".nii", "")
        resampled_path = os.path.join(out_dir, f"{template_file_pre}_resampled.nii.gz")

        # Stale resampled template check: if it exists, verify grid still matches
        needs_resample = True
        if os.path.exists(resampled_path):
            resamp_props = get_3dinfo_properties(resampled_path, logger=logger)
            if resamp_props["orient"] == scn["orient"] and resamp_props["spacing"] == scn["spacing"]:
                _log.info("Resampled template %s already exists and matches scan grid.", resampled_path)
                needs_resample = False
            else:
                _log.warning("Existing resampled template %s has stale grid. Re-resampling.", resampled_path)
                os.remove(resampled_path)

        if needs_resample:
            try:
                run_afni_command(
                    ["3dresample", "-master", scan_path, "-input", template_path,
                     "-prefix", resampled_path],
                    description="resample template to scan grid",
                    logger=logger,
                )
            except subprocess.CalledProcessError:
                _log.error("Could not resample --template_path %s. Please manually resample and try again.", template_path)
                sys.exit(1)

            if os.path.exists(resampled_path):
                _log.info("Successfully resampled template; saved to %s", resampled_path)
            else:
                _log.error("Could not resample --template_path %s. Please manually resample and try again.", template_path)
                sys.exit(1)

        return resampled_path
    else:
        return template_path

def extract_roi_stats(nifti_path, template_path, average_type, logger=None):
    """
    Extract ROI-level statistics from a NIfTI file using a template via 3dROIstats.

    Parameters
    ----------
    nifti_path : str
        Path to input NIfTI (can include sub-brik selectors).
    template_path : str
        Path to the ROI template NIfTI.
    average_type : str
        "mean" or "median".
    logger : logging.Logger, optional

    Returns
    -------
    pd.DataFrame
        Columns named ROI_1, ROI_2, ... with the extracted statistics.
    """
    _log = logger or logging.getLogger(__name__)

    try:
        extract_cmd = ["3dROIstats", f"-nz{average_type}", "-1Dformat",
                       "-mask", template_path, nifti_path]
        extract_out = run_afni_command(
            extract_cmd,
            capture_output=True,
            description=f"3dROIstats on {nifti_path}",
            logger=logger,
        )
    except subprocess.CalledProcessError:
        _log.error("Could not extract ROI stats from %s with template %s using AFNI's 3dROIstats.", nifti_path, template_path)
        sys.exit(1)

    # Parse the -1Dformat output.
    # Format: line 0 "#File\tSub-brick", line 1 = column headers,
    #         then for each sub-brick: label line (starts with #) + data line.
    #         Data lines are at indices 3, 5, 7, ... (every other line from 3).
    try:
        lines = extract_out.strip().split("\n")
        headers = [h.strip() for h in lines[1].split("\t") if h.strip() and h.strip() != "#"]
        # Collect all data lines (non-comment lines after the header)
        rows = []
        for line in lines[2:]:
            if line.startswith("#"):
                continue
            values = [v.strip() for v in line.split("\t") if v.strip()]
            if not values:
                continue
            if len(headers) != len(values):
                raise ValueError(f"Header count ({len(headers)}) != value count ({len(values)})")
            rows.append([float(v) for v in values])
        if not rows:
            raise ValueError("No data lines found in 3dROIstats output")
        out_df = pd.DataFrame(rows, columns=headers)
    except (IndexError, ValueError) as e:
        _log.error("Could not parse 3dROIstats output for %s: %s", nifti_path, e)
        sys.exit(1)

    # Filter columns based on average type
    ave_type_flag = "NZMed" if average_type == "median" else "NZMean"
    cols_to_keep = [colname for colname in out_df.columns if ave_type_flag in colname]

    if len(cols_to_keep) == 0:
        _log.error("No columns matching '%s' found in 3dROIstats output. Available columns: %s", ave_type_flag, list(out_df.columns))
        sys.exit(1)

    new_df = out_df[cols_to_keep].copy()
    new_df.rename(
        columns={oldname: "ROI_" + str(oldname).split("_")[1] for oldname in new_df.columns},
        inplace=True,
    )
    return new_df
    
# ----------------------------------------------------------------------------
# QC Summary Output
# ----------------------------------------------------------------------------

def write_qc_summary(out_dir, out_file_pre, qc_data, logger):
    """Write a QC summary JSON file.

    Parameters
    ----------
    out_dir : str
        Output directory.
    out_file_pre : str
        Output file prefix.
    qc_data : dict
        QC data to write. Typical keys include:
        - analysis_type, hrf_model, fd_threshold, censor_prev_tr
        - n_trs_total, n_trs_censored, pct_censored
        - per_condition_trial_counts, per_condition_surviving_trials (task only)
        - dof (degrees of freedom)
        - per_run_censoring (rest_conn only)
    logger : logging.Logger
    """
    qc_path = os.path.join(out_dir, f"{out_file_pre}_qc_summary.json")
    try:
        with open(qc_path, 'w') as f:
            json.dump(qc_data, f, indent=2)
        logger.info("QC summary written to %s", qc_path)
    except OSError as e:
        logger.warning("Could not write QC summary to %s: %s", qc_path, e)


def compute_dof(censor_path, n_regressors, logger):
    """Compute degrees of freedom from censor file and regressor count.

    Parameters
    ----------
    censor_path : str
        Path to the censor file (binary 1D).
    n_regressors : int
        Total number of regressors in the model.
    logger : logging.Logger

    Returns
    -------
    int
        Degrees of freedom (n_uncensored - n_regressors).
    """
    try:
        censor_data = np.loadtxt(censor_path)
        n_uncensored = int(np.sum(censor_data == 1))
    except Exception:
        logger.warning("Could not read censor file for DOF check.")
        return None
    dof = n_uncensored - n_regressors
    logger.info("Estimated DOF: %d (uncensored TRs: %d, regressors: %d)",
                dof, n_uncensored, n_regressors)
    if dof < 1:
        logger.error("Insufficient degrees of freedom (DOF=%d). "
                     "Too many TRs censored and/or too many regressors for AFNI "
                     "to produce valid output. Consider relaxing the FD threshold "
                     "or reducing the model complexity.", dof)
        sys.exit(1)
    return dof


# ----------------------------------------------------------------------------
# Connectivity Functions
# ----------------------------------------------------------------------------

def seed_to_voxel_conn(inset_path, out_dir, conn_out_file_pre, template_path,fishZ, pcorr, condition=None, logger=None):

    """
    Run seed-to-voxel functional connectivity using AFNI's 3dNetCorr.

    Parameters
    ----------
    inset_path : str
        Path to the input time series / beta series NIfTI.
    out_dir : str
        Output directory.
    conn_out_file_pre : str
        Output file prefix.
    template_path : str
        Path to the binary seed mask.
    fishZ : bool
        Whether to Fisher Z-transform correlations.
    pcorr : bool
        Whether to use partial correlation.
    condition : str or None
        If provided, included in the output filename (task-based).
        If None, omitted (resting-state).
    logger : logging.Logger, optional
    """
    _log = logger or logging.getLogger(__name__)

    # Build output prefix and final path
    if condition is not None:
        prefix = f"{conn_out_file_pre}_{condition}"
    else:
        prefix = conn_out_file_pre

    out_path = os.path.join(out_dir, prefix)
    if fishZ and pcorr:
        out_path += "_pcorr_fishZ.nii.gz"
    elif fishZ:
        out_path += "_fishZ.nii.gz"
    elif pcorr:
        out_path += "_pcorr.nii.gz"
    else:
        out_path += "_corr.nii.gz"

    if os.path.exists(out_path):
        _log.info("Functional connectivity for %s already exists.", prefix)
        return

    # Build and run 3dNetCorr
    netcorr_prefix = os.path.join(out_dir, prefix)
    netCorr_command = [
        "3dNetCorr", "-prefix", netcorr_prefix,
        "-inset", inset_path,
        "-in_rois", template_path, "-push_thru_many_zeros",
        "-ts_wb_corr", "-nifti",
    ]
    if fishZ:
        netCorr_command.append("-ts_wb_Z")
    if pcorr:
        netCorr_command.append("-part_corr")

    try:
        run_afni_command(netCorr_command, description="3dNetCorr seed-to-voxel", logger=logger)
    except subprocess.CalledProcessError:
        _log.warning("Could not create functional connectivity for seed %s.", template_path)
        return

    # Clean up auxiliary outputs
    for ext in ["_000.niml.dset", "_000.roidat", "_000.netcc"]:
        aux = f"{netcorr_prefix}{ext}"
        if os.path.exists(aux):
            os.remove(aux)

    # Move the connectivity map from the INDIV directory
    indiv_dir = f"{netcorr_prefix}_000_INDIV"
    if os.path.isdir(indiv_dir):
        if fishZ:
            src_file = os.path.join(indiv_dir, "WB_Z_ROI_001.nii.gz")
            alt_file = os.path.join(indiv_dir, "WB_CORR_ROI_001.nii.gz")
            if os.path.exists(src_file):
                os.rename(src_file, out_path)
                if os.path.exists(alt_file):
                    os.remove(alt_file)
            else:
                _log.warning("Could not create functional connectivity for seed %s.", template_path)
        else:
            src_file = os.path.join(indiv_dir, "WB_CORR_ROI_001.nii.gz")
            if os.path.exists(src_file):
                os.rename(src_file, out_path)
            else:
                _log.warning("Could not create functional connectivity for seed %s.", template_path)

        # Remove the (now empty) INDIV directory
        try:
            os.rmdir(indiv_dir)
        except OSError:
            shutil.rmtree(indiv_dir, ignore_errors=True)

        if os.path.exists(out_path):
            _log.info("Seed-to-voxel functional connectivity maps created for %s with seed from template %s.", prefix, template_path)
        else:
            _log.warning("Could not move functional connectivity file for seed %s. Check permissions and rerun.", template_path)
    else:
        _log.warning("Could not create functional connectivity for seed %s.", template_path)

def format_netcorr_mat(netcorr_file, fishZ, pcorr, curr_out_path):
    """
    Parse a 3dNetCorr .netcc output file and save the desired matrix as tab-delimited text.
    """
    if not fishZ and not pcorr:
        entry_to_check = "CC"
        end_to_check = "Z"
    elif not fishZ and pcorr:
        entry_to_check = "PC"
        end_to_check = "PCB"
    elif fishZ and not pcorr:
        entry_to_check = "FZ"
        end_to_check = "Z"
    else:
        entry_to_check = "PCB"
        end_to_check = "Z"

    conn_mat = []
    with open(netcorr_file) as f:
        for line in f:
            if line:
                parts = line.split()
                conn_mat.append([entry.strip() for entry in parts])

    start_row = 0
    end_row = 0
    for i, entry in enumerate(conn_mat):
        if len(entry) == 2 and entry[1] == entry_to_check:
            start_row = i + 1
            for j, next_entry in enumerate(conn_mat[i + 1:]):
                if len(next_entry) == 2 and next_entry[1] == end_to_check:
                    end_row = j + i + 1
                    break
            if end_row == 0:
                end_row = len(conn_mat)
            break

    out_conn_mat = conn_mat[start_row:end_row]
    with open(curr_out_path, "w") as f:
        for row in out_conn_mat:
            f.write("\t".join(str(x) for x in row) + "\n")

def parcellated_conn(inset_path, out_dir, conn_out_file_pre, template_path,fishZ, pcorr, condition=None, logger=None):

    """
    Run parcellated (ROI-to-ROI) functional connectivity using AFNI's 3dNetCorr.

    Parameters
    ----------
    inset_path : str
        Path to the input time series / beta series NIfTI.
    out_dir : str
        Output directory.
    conn_out_file_pre : str
        Output file prefix.
    template_path : str
        Path to the multi-ROI parcellation template.
    fishZ : bool
        Whether to Fisher Z-transform correlations.
    pcorr : bool
        Whether to use partial correlation.
    condition : str or None
        If provided, included in the output filename (task-based).
        If None, omitted (resting-state).
    logger : logging.Logger, optional
    """
    _log = logger or logging.getLogger(__name__)

    # Build output prefix and final path
    if condition is not None:
        prefix = f"{conn_out_file_pre}_{condition}"
    else:
        prefix = conn_out_file_pre

    out_path = os.path.join(out_dir, prefix)
    if fishZ and pcorr:
        out_path += "_pcorr_fishZ_mat.txt"
    elif fishZ:
        out_path += "_fishZ_mat.txt"
    elif pcorr:
        out_path += "_pcorr_mat.txt"
    else:
        out_path += "_corr_mat.txt"

    if os.path.exists(out_path):
        _log.info("Functional connectivity for %s already exists.", prefix)
        return

    # Build and run 3dNetCorr
    netcorr_prefix = os.path.join(out_dir, prefix)
    netCorr_command = [
        "3dNetCorr", "-prefix", netcorr_prefix,
        "-inset", inset_path,
        "-in_rois", template_path, "-push_thru_many_zeros",
    ]
    if fishZ:
        netCorr_command.append("-fish_z")
    if pcorr:
        netCorr_command.append("-part_corr")

    try:
        run_afni_command(netCorr_command, description="3dNetCorr parcellated", logger=logger)
    except subprocess.CalledProcessError:
        _log.warning("Could not create functional connectivity for template %s.", template_path)
        return

    # Clean up auxiliary outputs
    for ext in ["_000.niml.dset", "_000.roidat"]:
        aux = f"{netcorr_prefix}{ext}"
        if os.path.exists(aux):
            os.remove(aux)

    netcc_file = f"{netcorr_prefix}_000.netcc"
    if os.path.exists(netcc_file):
        format_netcorr_mat(netcc_file, fishZ, pcorr, out_path)
        if os.path.exists(out_path):
            os.remove(netcc_file)
            _log.info("Parcellated functional connectivity matrix created for %s with ROIs from template %s.", prefix, template_path)
        else:
            _log.error("Could not format functional connectivity for template %s. Likely an internal bug.", template_path)
            sys.exit(1)
    else:
        _log.error("Could not create functional connectivity for template %s.", template_path)
        sys.exit(1)
