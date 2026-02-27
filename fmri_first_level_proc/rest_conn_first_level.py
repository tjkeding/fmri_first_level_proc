#!/usr/bin/env python3

# ============================================================================
# RESTING-STATE fMRI FIRST-LEVEL AND FUNCTIONAL CONNECTIVITY ANALYSIS
# 1. First-level regression with AFNI's 3dTproject, creating residual dense time series
# 2. Note: Unlike task-based analyses, resting-state runs will NOT be concatenated
# 3. (optional) Extract parcel-level/ROI residual time series with a provided template
# 4. (optional) Runs functional connectivity analysis with AFNI's 3dNetCorr
#
# Author: Taylor J. Keding, Ph.D.
# Version: 2.0
# Last updated: 02/17/26
# ============================================================================
'''
REQUIREMENTS:
- AFNI
- numpy
- pandas

INPUTS:
(required)
-scan_paths: list of global file paths to the dense time series runs (preprocessed UP TO first-level analyses)
    format = list(string(path_to_nii)) (volumetric)
--motion_paths: list of global file paths of the run motion regressors (preprocessed UP TO first-level analyses)
    format = list(string(path_to_motion_txt)) (tab-delimited) with no headers; rows = TR/frame, columns = motion regressors (number of columns controlled by motion_deriv_degree: 6 * degree)
--censor_paths: list of global file paths of the motion/outlier censor files (should have been stripped for dummy scans)
    format = list(string(path_to_censor_txt)) (tab-delimited) with no headers; rows= TR/frame, single column = binary (1=include,0=exclude)
--CSF_paths: list of global file paths to the average cerebrospinal fluid BOLD signal time series
    format = list(string(path_to_CSF_TS_txt)); no headers; rows= TR/frame, single column = mean signal for frame
--WM_paths: list of global file paths to the average white-matter BOLD signal time series
    format = list(string(path_to_WM_TS_txt)); no headers; rows= TR/frame, single column = mean signal for frame
--out_dir: global directory path for derivative intermediates and task-condition beta series; will create if doesn't exist
--out_file_pre: prefix to be prepended to the files stored in out_dir (should usually contain an ID, task name, time point, etc. but NOT the global file path; eg. 'subj001_rest_baseline')
--num_cores: number of CPU cores available for parallel processing (int; defaults to 1)

(optional)
--GS_paths: list of global file paths to the average global BOLD signal time series (one per run); enables global signal regression (GSR)
    format = list(string(path_to_GS_TS_txt)); no headers; rows=TR/frame, single column = mean global signal for frame
    NOTE: GSR is controversial — it can remove neural signal and introduce artifactual anti-correlations. Use with caution and report in methods.
--bandpass: bandpass filter frequencies [low high] in Hz (default: [0.01, 0.1]); set to null in YAML config to disable
--polort: hardcoded to 2 (field consensus with bandpass filtering)
--use_tissue_derivs: (no value) include this option to add first temporal derivatives of tissue signals (CSF/WM, and GS when enabled) as additional nuisance regressors
--remove_previous: (no value) include this option if "out_dir" should have all files removed before processing starts (including connectivity); if not included, will not overwrite pre-existing files
--extract_ptseries: include this option if you want residual time series extracted for parcels/ROIs (requires --template_path be provided)
--average_type: either "mean" or "median" - the type of averaging used when calculating average parcel residual time series
--extract_out_file_pre: if --extract_ptseries is included, the prefix to attach to the extracted ptseries (should NOT include the file path; similar to --out_file_pre e.g. 'subj001_rest_baseline_Shen368_WB')
--calc_conn: either 'seed_to_voxel' or 'parcellated'; included if functional connectivity should be calculated with the given approach
--conn_out_file_pre: if --calc_conn is included, the prefix to attach to connectivity output files (should NOT include the file path, task condition, or connectivity; similar to --out_file_pre e.g. 'subj001_rest_baseline_R_amyg_wholeBrain')
--template_path: template file for --extract_ptseries and/or the option specified in --calc_conn.
    For extract_ptseries, can be a binary mask (only a single ptseries will be output) or a set of N ROIs/masks indicated by unique integers>0 (N ptseries will be output)
    For calc_conn, if 'seed_to_voxel', should be a binary mask with a single ROI; if 'parcellated', should be a set of ROIs (at least 2 required, each labeled with a unique integer>0)
    format = .nii (volumetric and registered to the same atlas as task beta series, unless --force_diff_atlas used; if there is a mismatch in grid spacing, will resample the template to the beta series grid)
--force_diff_atlas: (no value) include this option if you know the template and scan are in different standard spaces, but want to force connectivity analyses anyway (USE WITH CAUTION)
--pcorr: (no value) include this option to use partial correlation instead of standard correlation for functional connectivity
--fishZ: (no value) include this option if correlation 'r' should be Fisher-transformed into z-scores
--keep_run_res_dtseries: (default True) keep per-run residual dtseries files after concatenation; use --no-keep_run_res_dtseries to remove them

OUTPUTS:
(always)
{out_dir}/{out_file_pre}_run{N}_residual_dtseries.nii.gz: first-level output residual dense time series by-run (removed if --no-keep_run_res_dtseries)
{out_dir}/{out_file_pre}_concat_residual_dtseries.nii.gz: first-level output concatenated residual dense time series

(optional)
--extract_ptseries:
    {out_dir}/{extract_out_file_pre}_residual_ptseries.csv: residual BOLD time series averaged by masks in --template_path;
        Rows correspond to the TR/frame (no column/header for TR/frame) and columns are unique masks/ROIs (e.g. "ROI_1" for mask_value == 1)
--calc_conn 'parcellated':
{out_dir}/{conn_out_file_pre}_{fishZ}_{pcorr}_mat.txt: Square, task condition-specific functional connectivity matrix for the ROIs contained in --template_path;
    Row/column numbers correspond to the ROI integers from the template (i.e. cell {1,2} is connectivity between ROI_label=1 and ROI_label=2)
--calc_conn 'seed_to_voxel':
{out_dir}/{conn_out_file_pre}_{fishZ}_{pcorr}.nii.gz: .nii file containing voxelwise functional connectivity from seed contained in --template_path
'''

# Imports
import copy
import os
import sys
import argparse
import time

import numpy as np
import pandas as pd

from .first_level_utils import (
    setup_logging,
    dir_path_exists,
    file_path_exists,
    valid_ave_type,
    valid_conn_type,
    validate_template,
    remove_files_from_dir,
    run_afni_command,
    extract_roi_stats,
    seed_to_voxel_conn,
    parcellated_conn,
    prepare_motion_file,
    compute_tissue_derivative,
    create_censor_file,
    notch_filter_motion,
    write_qc_summary,
    compute_dof,
    validate_extract_options,
    validate_connectivity_options,
)

def gen_residual_ts(args, logger):

    # Check if concatenated residual dtseries exists
    concat_path = os.path.join(args.out_dir, f"{args.out_file_pre}_concat_residual_dtseries.nii.gz")
    per_run_dof = []
    if not os.path.exists(concat_path):

        # Validate and prepare motion files, generate censor files per run
        use_cols = 6 * args.motion_deriv_degree
        prepared_motion = []
        censor_paths = []
        for i, mot_path in enumerate(args.motion_paths):
            # Optionally notch-filter motion for respiratory artifact removal
            censor_motion_src = mot_path
            if args.notch_filter_band is not None:
                censor_motion_src = notch_filter_motion(
                    mot_path, args.tr, args.notch_filter_band,
                    args.out_dir, args.out_file_pre, f"run{i+1}", logger)
            # Generate censor file from (possibly filtered) motion
            censor_path = create_censor_file(
                censor_motion_src, args.fd_threshold, args.censor_prev_tr,
                args.out_dir, args.out_file_pre, f"run{i+1}", logger)
            censor_paths.append(censor_path)
            # Prepare motion for regression (original, unfiltered)
            prepared = prepare_motion_file(
                mot_path, use_cols, args.out_dir, args.out_file_pre, f"run{i+1}", logger)
            prepared_motion.append(prepared)

            # Per-run DOF check
            n_regressors = use_cols + 3  # polort 2 → 3 polynomial regressors
            n_regressors += 2  # CSF + WM (always present)
            if args.GS_paths is not None:
                n_regressors += 1
            if args.use_tissue_derivs:
                n_regressors += 2  # CSF_deriv + WM_deriv
                if args.GS_paths is not None:
                    n_regressors += 1  # GS_deriv
            run_dof = compute_dof(censor_path, n_regressors, logger)
            per_run_dof.append(run_dof)

        completed_runs = []

        # Iterate individual resting-state runs
        for i, run_scan in enumerate(args.scan_paths):

            run_out = os.path.join(args.out_dir, f"{args.out_file_pre}_run{i+1}_residual_dtseries.nii.gz")

            # Check if this run's residual dtseries already exists
            if not os.path.exists(run_out):

                # Build 3dTproject command
                tproj_command = ["3dTproject", "-input", run_scan,
                    "-dt", str(args.tr),
                    "-censor", censor_paths[i],
                    "-cenmode", "ZERO",
                    "-ort", prepared_motion[i],
                    "-ort", args.CSF_paths[i],
                    "-ort", args.WM_paths[i]]
                if args.GS_paths is not None:
                    tproj_command.extend(["-ort", args.GS_paths[i]])
                if args.use_tissue_derivs:
                    csf_deriv = compute_tissue_derivative(args.CSF_paths[i], args.out_dir, args.out_file_pre, f"run{i+1}_CSF", logger)
                    tproj_command.extend(["-ort", csf_deriv])
                    wm_deriv = compute_tissue_derivative(args.WM_paths[i], args.out_dir, args.out_file_pre, f"run{i+1}_WM", logger)
                    tproj_command.extend(["-ort", wm_deriv])
                    if args.GS_paths is not None:
                        gs_deriv = compute_tissue_derivative(args.GS_paths[i], args.out_dir, args.out_file_pre, f"run{i+1}_GS", logger)
                        tproj_command.extend(["-ort", gs_deriv])
                tproj_command.extend(["-polort", "2"])  # Field consensus with bandpass
                if args.bandpass is not None:
                    tproj_command.extend(["-bandpass", str(args.bandpass[0]), str(args.bandpass[1])])
                tproj_command.extend(["-prefix", run_out])

                # Run command and check output
                run_afni_command(tproj_command, description=f"3dTproject run{i+1}", logger=logger)
                if os.path.exists(run_out):
                    logger.info("Successfully created residual dtseries for %s_run%d.", args.out_file_pre, i+1)
                    completed_runs.append(run_out)
                else:
                    logger.error("Failed to create residual dtseries for %s_run%d.", args.out_file_pre, i+1)
                    sys.exit(1)
            else:
                completed_runs.append(run_out)

        # Concatenate residual time series runs
        cat_command = ["3dTcat", "-TR", str(args.tr), "-prefix", concat_path] + completed_runs
        run_afni_command(cat_command, description="3dTcat concat residuals", logger=logger)
        if os.path.exists(concat_path):
            logger.info("Successfully created concatenated residual dtseries for %s.", args.out_file_pre)
        else:
            logger.error("Failed to create concatenated residual dtseries for %s.", args.out_file_pre)
            sys.exit(1)

        # If keep_run_res_dtseries is False (default True), remove per-run residual files
        if not args.keep_run_res_dtseries:
            for comp_run in completed_runs:
                os.remove(comp_run)
            all_removed = all(not os.path.exists(cr) for cr in completed_runs)
            if all_removed:
                logger.info("Successfully removed run-specific residual dtseries.")
            else:
                logger.warning("Unable to remove all run-specific residual dtseries (continuing, but check the output).")
    else:
        logger.info("%s already exists.", concat_path)

    return per_run_dof

def gen_ptseries(args, logger):

    # Check if output already exists
    if not os.path.exists(f"{args.out_dir}/{args.extract_out_file_pre}_residual_ptseries.csv"):

        inset_path = os.path.join(args.out_dir, f"{args.out_file_pre}_concat_residual_dtseries.nii.gz")
        new_df = extract_roi_stats(inset_path, args.template_path, args.average_type, logger=logger)

        # Save ptseries
        new_df.to_csv(f"{args.out_dir}/{args.extract_out_file_pre}_residual_ptseries.csv")
        if os.path.exists(f"{args.out_dir}/{args.extract_out_file_pre}_residual_ptseries.csv"):
            logger.info("Successfully extracted ptseries using ROI template.")
        else:
            logger.error("Could not extract ptseries using ROI template.")
            sys.exit(1)
    else:
        logger.info("%s/%s_residual_ptseries.csv already exists.", args.out_dir, args.extract_out_file_pre)

def gen_conn(args, logger):

    inset_path = os.path.join(args.out_dir, f"{args.out_file_pre}_concat_residual_dtseries.nii.gz")

    # Run connectivity analyses based on the specified flavor (no condition — resting-state)
    if args.calc_conn == "parcellated":
        parcellated_conn(
            inset_path=inset_path,
            out_dir=args.out_dir,
            conn_out_file_pre=args.conn_out_file_pre,
            template_path=args.template_path,
            fishZ=args.fishZ,
            pcorr=args.pcorr,
            condition=None,
            logger=logger,
        )
    else:
        seed_to_voxel_conn(
            inset_path=inset_path,
            out_dir=args.out_dir,
            conn_out_file_pre=args.conn_out_file_pre,
            template_path=args.template_path,
            fishZ=args.fishZ,
            pcorr=args.pcorr,
            condition=None,
            logger=logger,
        )

def run(args, logger):
    """Validate args and execute pipeline. Works from CLI or config runner."""
    args = copy.copy(args)

    # Validate all path lists are the same length (CLI usage may skip config validation)
    path_lengths = {"scan_paths": len(args.scan_paths), "motion_paths": len(args.motion_paths),
                    "CSF_paths": len(args.CSF_paths), "WM_paths": len(args.WM_paths)}
    if args.GS_paths is not None:
        path_lengths["GS_paths"] = len(args.GS_paths)
    if len(set(path_lengths.values())) > 1:
        logger.error("All path lists must be the same length. Got: %s", path_lengths)
        sys.exit(1)

    logger.info("Residual dense time series will be created for resting-state fMRI.")
    if args.use_tissue_derivs:
        tissue_parts = ["CSF", "WM"]
        if args.GS_paths is not None:
            tissue_parts.append("GS")
        logger.info("Tissue signal derivatives will be included as additional regressors: %s", ", ".join(tissue_parts))
    if args.GS_paths is not None:
        logger.warning("Global signal regression (GSR) is enabled. Note: GSR is controversial — "
                       "it can remove neural signal and introduce artifactual anti-correlations. "
                       "Use with caution and report in methods.")

    # If --remove_previous is provided, clear-out the output dir (before template
    # resampling or any other file creation in out_dir)
    if args.remove_previous:
        logger.info("Removing all files from previous analyses in %s (if they exist).", args.out_dir)
        remove_files_from_dir(args.out_dir, logger=logger)

    # Make sure optional args exist if --extract_ptseries is provided
    if args.extract_ptseries:
        args.template_path = validate_extract_options(
            "--extract_ptseries", args.average_type, args.extract_out_file_pre,
            args.template_path, args.scan_paths[0], args.out_dir, args.force_diff_atlas,
            logger=logger)

    # Make sure optional args exist if --calc_conn is provided
    if args.calc_conn is not None:
        args.template_path = validate_connectivity_options(
            args.calc_conn, args.conn_out_file_pre, args.template_path,
            args.scan_paths[0], args.out_dir, args.force_diff_atlas,
            data_label="residual time series", logger=logger)
    else:
        logger.info("Connectivity will not be output after generating residual time series.")

    # ---------------------------------

    # Build QC summary data with per-run censoring stats
    qc_data = {
        "analysis_type": "rest_conn",
        "fd_threshold": args.fd_threshold,
        "censor_prev_tr": args.censor_prev_tr,
        "bandpass": args.bandpass,
        "per_run_censoring": [],
    }

    # Generate residual dense time series (regression for denoising, censoring, and temporal filtering)
    per_run_dof = gen_residual_ts(args, logger)
    if per_run_dof:
        qc_data["per_run_dof"] = per_run_dof

    # Collect per-run censor stats (censor files were created inside gen_residual_ts)
    for i in range(len(args.scan_paths)):
        censor_path = os.path.join(args.out_dir, f"{args.out_file_pre}_run{i+1}_censor.1D")
        run_stats = {"run": i + 1}
        try:
            censor_data = np.loadtxt(censor_path)
            run_stats["n_trs_total"] = len(censor_data)
            run_stats["n_trs_censored"] = int(np.sum(censor_data == 0))
            run_stats["pct_censored"] = round(100.0 * run_stats["n_trs_censored"] / run_stats["n_trs_total"], 2) if run_stats["n_trs_total"] > 0 else 0.0
            logger.info("Run %d censoring: %d of %d TRs censored (%.1f%%).",
                        i + 1, run_stats["n_trs_censored"], run_stats["n_trs_total"], run_stats["pct_censored"])
        except Exception:
            pass
        qc_data["per_run_censoring"].append(run_stats)

    # Aggregate totals
    total_trs = sum(r.get("n_trs_total", 0) for r in qc_data["per_run_censoring"])
    total_censored = sum(r.get("n_trs_censored", 0) for r in qc_data["per_run_censoring"])
    qc_data["n_trs_total"] = total_trs
    qc_data["n_trs_censored"] = total_censored
    qc_data["pct_censored"] = round(100.0 * total_censored / total_trs, 2) if total_trs > 0 else 0.0

    # Extract ptseries if user provided option
    if args.extract_ptseries:
        gen_ptseries(args, logger)

    # Generate connectivity from residual time series if user provided option
    if args.calc_conn is not None:
        gen_conn(args, logger)

    # Write QC summary
    write_qc_summary(args.out_dir, args.out_file_pre, qc_data, logger)

def main():
    """CLI entrypoint: parse --flags, call run()."""

    # Start the timer
    start_time = time.time()

    # ---------------------------------
    # Parse arguments
    # ---------------------------------
    parser = argparse.ArgumentParser()
    # required:
    parser.add_argument("--scan_paths", type=file_path_exists, nargs='+', required=True)
    parser.add_argument("--motion_paths", type=file_path_exists, nargs='+', required=True)
    parser.add_argument("--CSF_paths", type=file_path_exists, nargs='+', required=True)
    parser.add_argument("--WM_paths", type=file_path_exists, nargs='+', required=True)
    parser.add_argument("--GS_paths", type=file_path_exists, nargs='+', default=None, required=False,
                        help="Optional global signal regressors (one per run). Enables global signal regression.")
    parser.add_argument("--out_dir", type=dir_path_exists, required=True)
    parser.add_argument("--out_file_pre", type=str, required=True)
    parser.add_argument("--num_cores", type=int, required=True)
    parser.add_argument("--fd_threshold", type=float, required=True,
                        help="Framewise displacement threshold (mm) for motion censoring.")
    parser.add_argument("--censor_prev_tr", action='store_true', default=False,
                        help="Also censor the TR before each high-motion TR (default: False).")
    parser.add_argument("--notch_filter_band", type=float, nargs=2, default=None, required=False,
                        help="Notch filter Hz band [low high] for respiratory artifact removal (default: disabled).")
    parser.add_argument("--tr", type=float, required=True,
                        help="Repetition time in seconds.")

    # optional:
    parser.add_argument("--bandpass", type=float, nargs=2, default=[0.01, 0.1], required=False,
                        help="Bandpass filter frequencies [low high] in Hz. Omit to use default [0.01, 0.1].")
    parser.add_argument("--remove_previous", action='store_true', required=False)
    parser.add_argument("--extract_ptseries", action='store_true', required=False)
    parser.add_argument("--average_type", type=valid_ave_type, required=False)
    parser.add_argument("--extract_out_file_pre", type=str, required=False)
    parser.add_argument("--calc_conn", type=valid_conn_type, required=False)
    parser.add_argument("--conn_out_file_pre", type=str, required=False)
    parser.add_argument("--template_path", type=file_path_exists, required=False)
    parser.add_argument("--force_diff_atlas", action='store_true', required=False)
    parser.add_argument("--pcorr", action='store_true', required=False)
    parser.add_argument("--fishZ", action='store_true', required=False)
    parser.add_argument("--keep_run_res_dtseries", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep per-run residual dtseries files after concatenation (default: True). Use --no-keep_run_res_dtseries to remove them.")
    parser.add_argument("--use_tissue_derivs", action='store_true', default=False, required=False,
                        help="Include first temporal derivatives of tissue signals (CSF/WM/GS) as additional nuisance regressors.")

    args = parser.parse_args()
    logger = setup_logging("rest_conn_first_level")

    run(args, logger)

    # Display total runtime
    logger.info("Total runtime: %.2f seconds", time.time() - start_time)

if __name__ == "__main__":
    main()
