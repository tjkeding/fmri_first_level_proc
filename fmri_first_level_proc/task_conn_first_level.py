#!/usr/bin/env python3

# ============================================================================
# TASK-BASED fMRI FIRST-LEVEL AND FUNCTIONAL CONNECTIVITY ANALYSIS
# 1. Creates task design matrix (including denoising and censoring) with AFNI's 3dDeconvolve
# 2. Creates full-task beta series with AFNI's 3dLSS
# 3. Creates task condition-specific beta series with AFNI's 3dcalc
# 4. (optional) Extract parcel-level/ROI condition beta series with a provided template
# 5. (optional) Runs functional connectivity analysis for available beta series with AFNI's 3dNetCorr
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
-scan_path: global file path to the run-concatenated dense time series (preprocessed UP TO first-level analyses)
    format = string(path_to_nii) (volumetric)
--task_timing_path: global file path to the run-concatenated stimulus timing file (should have been stripped for dummy scans)
    format = string(path_to_csv); column 1 name = 'CONDITION', column 2 name = 'ONSET' (global times matching --scan_path scan length), column 3 name = 'DURATION'
--motion_path: global file path to the run-concatenated motion regressors (should have been stripped for dummy scans)
    format = string(path_to_txt) (tab-delimited) with no headers; rows = TR/frame, columns = motion regressors (6 base or 12 with derivatives, controlled by include_motion_derivs)
--censor_path: global file path to the run-concatenated frame motion/outlier censor file (should have been stripped for dummy scans)
    format = string(path_to_txt) (tab-delimited) with no headers; rows= TR/frame, single column = binary (1=include,0=exclude)
--cond_beta_labels: list (comma-separated) of task conditions to generate beta series for - all labels should match to rows in the 'CONDITION' column from the timing file
    format = string-list of conditions e.g. 'stimFear,stimSad,stimNeu'
--out_dir: global directory path for derivative intermediates and task-condition beta series; will create if doesn't exist
--out_file_pre: prefix to be prepended to the files stored in out_dir (should usually contain an ID, task name, time point, etc. but NOT the global file path; eg. 'subj001_nback_baseline')
--num_cores: number of CPU cores available for parallel processing (int; defaults to 1)

(optional)
--hrf_model: HRF basis function model to use (default: 'GAM'). Options: 'GAM', 'BLOCK', 'dmBLOCK', 'SPMG1', 'custom'
--custom_hrf: custom AFNI HRF model string (only used when hrf_model='custom', e.g. 'TENT(0,14,8)')
--CSF_path: global file path to the run-concatenated cerebrospinal fluid BOLD signal time series
    format = string(path_to_txt); no headers; rows=TR/frame, single column = mean CSF signal
--WM_path: global file path to the run-concatenated white-matter BOLD signal time series
    format = string(path_to_txt); no headers; rows=TR/frame, single column = mean WM signal
--use_tissue_derivs: (no value) include this option to add first temporal derivatives of tissue signals (CSF/WM) as additional nuisance regressors; only computed for tissue paths that are provided
--remove_previous: (no value) include this option if "out_dir" should have all files removed before processing starts (including connectivity); if not included, will not overwrite pre-existing files
--extract_pbseries: include this option if you want condition-specific beta series extracted for parcels/ROIs (requires --template_path be provided)
--average_type: either "mean" or "median" - the type of averaging used when calculating average parcel beta series
--extract_out_file_pre: if --extract_pbseries is included, the prefix to attach to the extracted pbseries (should NOT include the file path or task condition; similar to --out_file_pre e.g. 'subj001_nback_baseline_Shen368_WB')
--calc_conn: either 'seed_to_voxel' or 'parcellated'; included if functional connectivity should be calculated with the given approach
--conn_out_file_pre: if --calc_conn is included with a valid option, the prefix to attach to connectivity output files (should NOT include the file path or task condition; similar to --out_file_pre e.g. 'subj001_nback_baseline_R_amyg_wholeBrain')
--template_path: template file for --extract_pbseries and/or the option specified in --calc_conn.
    For extract_pbseries, can be a binary mask (only a single pbseries will be output) or a set of N ROIs/masks indicated by unique integers>0 (N pbseries will be output per task condition)
    For calc_conn, if 'seed_to_voxel', should be a binary mask with a single ROI; if 'parcellated', should be a set of ROIs (at least 2 required, each labeled with a unique integer>0)
    format = string(path_to_nii) (volumetric and registered to the same atlas as task beta series, unless --force_diff_atlas used; if there is a mismatch in grid spacing, will resample the template to the beta series grid)
--force_diff_atlas: (no value) include this option if you know the template and scan are in different standard spaces, but want to force connectivity analyses anyway (USE WITH CAUTION)
--pcorr: (no value) include this option to use partial correlation instead of standard correlation for functional connectivity
--fishZ: (no value) include this option if correlation 'r' should be Fisher-transformed into z-scores

OUTPUTS:
(always)
{out_dir}/{out_file_pre}_concat_{condition}_onsets.txt: AFNI-safe version of timing onsets for task conditions NOT included in --cond_beta_labels (1D single column with no headers or row names)
{out_dir}/{out_file_pre}_concat_beta_onsets.txt: AFNI-safe version of timing onsets for task conditions to calculate beta series for (1D single column with no headers or row names)
{out_dir}/{out_file_pre}_concat_bseries_dmat.x1D: AFNI-safe design matrix for the task fMRI beta series first-level analysis
{out_dir}/{out_file_pre}_concat_LSS.nii.gz: first-level output beta series for the entire task (subbrik = stim, in the order they were presented in ..beta_onsets.txt)
{out_dir}/{out_file_pre}_concat_LSS.1d: same as "...concat_LSS.nii.gz", but in text file format (not much space and may be useful later)
{out_dir}/{out_file_pre}_concat_bseries_{condition}.nii.gz: first-level output task condition-specific beta series (in the order they were presented in ..beta_onsets.txt)

(optional)
--extract_pbseries:
    {out_dir}/{extract_out_file_pre}_pbseries_{condition}.csv: condition-specific beta series averaged by masks in --template_path;
        Rows correspond to the TR/frame (no column/header for TR/frame) and columns are unique masks/ROIs (e.g. "ROI_1" for mask_value == 1)
--calc_conn 'parcellated':
{out_dir}/{conn_out_file_pre}_{condition}_{fishZ}_{pcorr}_mat.txt: Square, task condition-specific functional connectivity matrix for the ROIs contained in --template_path;
    Row/column numbers correspond to the ROI integers from the template (i.e. cell {1,2} is connectivity between ROI_label=1 and ROI_label=2)
--calc_conn 'seed_to_voxel':
{out_dir}/{conn_out_file_pre}_{condition}_{fishZ}_{pcorr}.nii.gz: .nii file containing voxelwise, task condition-specific functional connectivity from seed contained in --template_path
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
    valid_string_list,
    valid_ave_type,
    valid_conn_type,
    run_afni_command,
    validate_template,
    remove_files_from_dir,
    extract_roi_stats,
    seed_to_voxel_conn,
    parcellated_conn,
    build_hrf_string,
    needs_married_timing,
    read_and_validate_stim_data,
    write_onset_file,
    build_decon_base_command,
    clean_deconvolve_err,
    log_hrf_model,
    log_nuisance_model,
    prepare_motion_file,
    compute_tissue_derivative,
    create_censor_file,
    check_trial_survival,
    write_qc_summary,
    compute_dof,
    validate_extract_options,
    validate_connectivity_options,
    HRF_DM_MODELS,
    HRF_IMPULSE_MODELS,
    HRF_DURATION_MODELS,
    VALID_HRF_MODELS,
)

def get_stim_data(args, logger):

    # Read, validate, and sort stim timing data
    sorted_df = read_and_validate_stim_data(args.task_timing_path, args.cond_beta_labels, logger=logger)

    # Iterate the different task conditions from the timing file
    betas_onsets = None
    betas_durations = None
    for cond in sorted_df['CONDITION'].unique():
        cond_df = sorted_df[sorted_df['CONDITION'] == cond]

        # If a task condition should only be controlled for (no beta series output), create its own timing file
        if cond not in args.cond_beta_labels:
            onset_file = f"{args.out_dir}/{args.out_file_pre}_concat_{cond}_onsets.txt"
            write_onset_file(cond_df, onset_file, args.hrf_model, args.custom_hrf, logger=logger)
        # If we want a beta series for a task condition, add their onsets to betas_onsets
        else:
            if betas_onsets is None:
                betas_onsets = list(cond_df['ONSET'])
                betas_durations = list(cond_df['DURATION'])
            else:
                betas_onsets.extend(cond_df['ONSET'])
                betas_durations.extend(cond_df['DURATION'])
    betas_df = pd.DataFrame({'ONSET': betas_onsets, 'DURATION': betas_durations})

    # Save all onset times for task conditions requiring beta series
    beta_onset_file = f"{args.out_dir}/{args.out_file_pre}_concat_beta_onsets.txt"
    write_onset_file(betas_df, beta_onset_file, args.hrf_model, args.custom_hrf, logger=logger)

    # Return formatted, sorted stim times
    return sorted_df

def gen_design_matrix(stim_data, args, logger):

    # Check if output already exists
    if not os.path.exists(f"{args.out_dir}/{args.out_file_pre}_concat_bseries_dmat.x1D"):

        # Compute tissue signal derivatives if requested
        CSF_deriv_path = None
        WM_deriv_path = None
        if args.use_tissue_derivs:
            if args.CSF_path is not None:
                CSF_deriv_path = compute_tissue_derivative(args.CSF_path, args.out_dir, args.out_file_pre, "CSF", logger)
            if args.WM_path is not None:
                WM_deriv_path = compute_tissue_derivative(args.WM_path, args.out_dir, args.out_file_pre, "WM", logger)

        # Set-up base command, including temporal detrending, frame censors, and motion regressors
        decon_command = build_decon_base_command(args.scan_path, args.censor_path, args.motion_path,
                                                 CSF_path=args.CSF_path, WM_path=args.WM_path,
                                                 CSF_deriv_path=CSF_deriv_path, WM_deriv_path=WM_deriv_path,
                                                 tr=args.tr)
        decon_command.extend(["-num_stimts", f"{1+len(np.unique(stim_data['CONDITION']))-len(args.cond_beta_labels)}"])

        # Iteratively add stim timing for conditions NOT included in beta series
        stim_count = 1
        for cond in np.unique(stim_data['CONDITION']):
            if cond not in args.cond_beta_labels:
                mean_dur = None
                if args.hrf_model in HRF_DURATION_MODELS:
                    mean_dur = np.mean(stim_data[stim_data['CONDITION'] == cond]['DURATION'])
                hrf_string = build_hrf_string(args.hrf_model, mean_dur=mean_dur, custom_hrf=args.custom_hrf)
                stim_flag = "-stim_times_AM1" if needs_married_timing(args.hrf_model, args.custom_hrf) else "-stim_times"
                decon_command.extend([stim_flag, f"{stim_count}", f"{args.out_dir}/{args.out_file_pre}_concat_{cond}_onsets.txt",
                                     hrf_string, "-stim_label", f"{stim_count}", f"{cond}"])
                stim_count += 1

        # Add stimtimes IM for conditions included in beta series
        # 3dLSS requires -stim_times_IM; config validation rejects married-timing
        # HRFs (e.g. dmBLOCK) that would require -stim_times_AM1.
        mean_dur = None
        if args.hrf_model in HRF_DURATION_MODELS:
            mean_dur = np.mean(stim_data[stim_data['CONDITION'].isin(args.cond_beta_labels)]['DURATION'])
        hrf_string = build_hrf_string(args.hrf_model, mean_dur=mean_dur, custom_hrf=args.custom_hrf)
        beta_stim_flag = "-stim_times_IM"
        decon_command.extend([beta_stim_flag, f"{stim_count}", f"{args.out_dir}/{args.out_file_pre}_concat_beta_onsets.txt",
                              hrf_string, "-stim_label", f"{stim_count}", "beta_stims",
                              "-x1D", f"{args.out_dir}/{args.out_file_pre}_concat_bseries_dmat.x1D",
                              "-x1D_stop", "-nobucket", "-jobs", f"{args.num_cores}"])

        # Run command and be sure we have output
        run_afni_command(decon_command, description="3dDeconvolve design matrix", logger=logger)

        if os.path.exists(f"{args.out_dir}/{args.out_file_pre}_concat_bseries_dmat.x1D"):
            logger.info("Successfully created design matrix %s_concat_bseries_dmat.x1D", args.out_file_pre)
        else:
            logger.error("Failed to create design matrix %s_concat_bseries_dmat.x1D", args.out_file_pre)
            sys.exit(1)
    else:
        logger.info("Design matrix for %s already exists.", args.out_file_pre)

    # Remove err file that always seems to be output
    clean_deconvolve_err(args.out_dir)

def gen_beta_series(stim_data, args, logger):

    # Check if full-task beta series already exists
    if not os.path.exists(f"{args.out_dir}/{args.out_file_pre}_concat_LSS.nii.gz"):

        # Build the 3dLSS command and create full-task beta series
        lss_command = ["3dLSS", "-matrix", f"{args.out_dir}/{args.out_file_pre}_concat_bseries_dmat.x1D",
                       "-input", f"{args.scan_path}",
                       "-save1D", f"{args.out_dir}/{args.out_file_pre}_concat_LSS.1d",
                       "-prefix", f"{args.out_dir}/{args.out_file_pre}_concat_LSS.nii.gz"]
        run_afni_command(lss_command, description="3dLSS beta series", logger=logger)

        if os.path.exists(f"{args.out_dir}/{args.out_file_pre}_concat_LSS.nii.gz"):
            logger.info("Successfully created full-task beta series for %s", args.out_file_pre)
        else:
            logger.error("Failed to create full-task beta series for %s", args.out_file_pre)
            sys.exit(1)
    else:
        logger.info("%s/%s_concat_LSS.nii.gz already exists", args.out_dir, args.out_file_pre)

    # Iterate task conditions in stim timing file
    bseries_out = {"CONDITION": [], "PATH": []}
    total_used = 0
    for curr_cond in np.unique(stim_data['CONDITION']):

        # If we want a condition-specific beta series
        if curr_cond in args.cond_beta_labels:

            # Get onset times for the condition
            cond = stim_data[stim_data['CONDITION'] == curr_cond]['ONSET']

            # Check if condition-specific beta series already exists
            if not os.path.exists(f"{args.out_dir}/{args.out_file_pre}_concat_bseries_{curr_cond}.nii.gz"):

                # Create condition index string
                calc_indices_str = "["
                for n, onset in enumerate(cond):
                    if not n == len(cond) - 1:
                        calc_indices_str = f"{calc_indices_str}{n+total_used},"
                    else:
                        calc_indices_str = f"{calc_indices_str}{n+total_used}]"

                # Build the 3dcalc command and create task condition beta series
                calc_command = ["3dcalc", "-a", f"{args.out_dir}/{args.out_file_pre}_concat_LSS.nii.gz{calc_indices_str}",
                                "-expr", "a",
                                "-prefix", f"{args.out_dir}/{args.out_file_pre}_concat_bseries_{curr_cond}.nii.gz"]
                run_afni_command(calc_command, description=f"3dcalc beta series {curr_cond}", logger=logger)

                if os.path.exists(f"{args.out_dir}/{args.out_file_pre}_concat_bseries_{curr_cond}.nii.gz"):
                    logger.info("Successfully created beta series %s_concat_bseries_%s.nii.gz", args.out_file_pre, curr_cond)
                    bseries_out['CONDITION'].append(curr_cond)
                    bseries_out["PATH"].append(f"{args.out_dir}/{args.out_file_pre}_concat_bseries_{curr_cond}.nii.gz")
                else:
                    logger.warning("Failed to create beta series %s_concat_bseries_%s.nii.gz", args.out_file_pre, curr_cond)
                    logger.warning("Continuing to see if beta series can be created for other task conditions.")
            else:
                logger.info("Beta series %s_concat_bseries_%s.nii.gz already exists.", args.out_file_pre, curr_cond)
                bseries_out['CONDITION'].append(curr_cond)
                bseries_out["PATH"].append(f"{args.out_dir}/{args.out_file_pre}_concat_bseries_{curr_cond}.nii.gz")

            total_used += len(cond)

    # Something went wrong if our output is empty
    if len(bseries_out["CONDITION"]) == 0:
        logger.error("Could not find or create any beta series for %s.", args.out_file_pre)
        sys.exit(1)

    # Return dictionary with those conditions for which we've successfully created beta series
    return bseries_out

def gen_pbseries(bseries_info, args, logger):

    # Iterate available condition-specific beta series
    for i, cond in enumerate(bseries_info['CONDITION']):

        # Check if output already exists
        if not os.path.exists(f"{args.out_dir}/{args.extract_out_file_pre}_pbseries_{cond}.csv"):

            new_df = extract_roi_stats(bseries_info['PATH'][i], args.template_path, args.average_type, logger=logger)

            # Save pbseries
            new_df.to_csv(f"{args.out_dir}/{args.extract_out_file_pre}_pbseries_{cond}.csv")
            if os.path.exists(f"{args.out_dir}/{args.extract_out_file_pre}_pbseries_{cond}.csv"):
                logger.info("Successfully extracted pbseries from %s using ROI template.", bseries_info['PATH'][i])
            else:
                logger.error("Could not extract pbseries from %s using ROI template.", bseries_info['PATH'][i])
                sys.exit(1)
        else:
            logger.info("%s/%s_pbseries_%s.csv already exists (skipping).", args.out_dir, args.extract_out_file_pre, cond)

def gen_conn(bseries_info, args, logger):

    # Iterate task conditions with beta series
    for i, cond in enumerate(bseries_info['CONDITION']):

        # Run connectivity analyses based on the specified flavor
        if args.calc_conn == "parcellated":
            parcellated_conn(
                inset_path=bseries_info['PATH'][i],
                out_dir=args.out_dir,
                conn_out_file_pre=args.conn_out_file_pre,
                template_path=args.template_path,
                fishZ=args.fishZ,
                pcorr=args.pcorr,
                condition=cond,
                logger=logger,
            )
        else:
            seed_to_voxel_conn(
                inset_path=bseries_info['PATH'][i],
                out_dir=args.out_dir,
                conn_out_file_pre=args.conn_out_file_pre,
                template_path=args.template_path,
                fishZ=args.fishZ,
                pcorr=args.pcorr,
                condition=cond,
                logger=logger,
            )

def run(args, logger):
    """Validate args and execute pipeline. Works from CLI or config runner."""
    args = copy.copy(args)

    logger.info("Beta series will be created for the following task conditions: %s", args.cond_beta_labels)
    logger.info("All other task conditions will be controlled for, but won't have betas modeled.")
    log_hrf_model(args.hrf_model, args.custom_hrf, logger)
    log_nuisance_model(args.CSF_path, args.WM_path, logger, use_tissue_derivs=args.use_tissue_derivs)

    # If --remove_previous is provided, clear-out the output dir (before template
    # resampling or any other file creation in out_dir)
    if args.remove_previous:
        logger.info("Removing all files from previous analyses in %s (if they exist).", args.out_dir)
        remove_files_from_dir(args.out_dir, logger=logger)

    # Make sure optional args exist if --extract_pbseries is provided
    if args.extract_pbseries:
        args.template_path = validate_extract_options(
            "--extract_pbseries", args.average_type, args.extract_out_file_pre,
            args.template_path, args.scan_path, args.out_dir, args.force_diff_atlas,
            logger=logger)

    # Make sure optional args exist if --calc_conn is provided
    if args.calc_conn is not None:
        args.template_path = validate_connectivity_options(
            args.calc_conn, args.conn_out_file_pre, args.template_path,
            args.scan_path, args.out_dir, args.force_diff_atlas,
            data_label="task beta series", logger=logger)
    else:
        logger.info("Connectivity will not be output after generating task beta series.")

    # ---------------------------------

    # Generate censor file from motion regressors (before prepare_motion_file overwrites path)
    args.censor_path = create_censor_file(
        args.motion_path, args.fd_threshold, args.censor_prev_tr,
        args.out_dir, args.out_file_pre, "concat", logger)

    # Validate and prepare motion regressors
    use_cols = 12 if args.include_motion_derivs else 6
    args.motion_path = prepare_motion_file(
        args.motion_path, use_cols, args.out_dir, args.out_file_pre, "concat", logger)

    # Pre-flight DOF check (conservative lower bound — actual per-trial regressors are higher)
    n_regressors = use_cols + len(args.cond_beta_labels)
    if args.CSF_path is not None:
        n_regressors += 1
    if args.WM_path is not None:
        n_regressors += 1
    if args.use_tissue_derivs:
        if args.CSF_path is not None:
            n_regressors += 1
        if args.WM_path is not None:
            n_regressors += 1
    dof = compute_dof(args.censor_path, n_regressors, logger)

    # Get stimulus timing data and save single-column .txt file with beta conds stim timing
    stim_data = get_stim_data(args, logger)

    # Per-condition trial survival QC
    trial_survival = check_trial_survival(stim_data, args.cond_beta_labels, args.censor_path, args.tr, logger)

    # Build QC summary data
    qc_data = {
        "analysis_type": "task_conn",
        "hrf_model": args.hrf_model if args.hrf_model != "custom" else f"custom ({args.custom_hrf})",
        "fd_threshold": args.fd_threshold,
        "censor_prev_tr": args.censor_prev_tr,
    }
    try:
        censor_data = np.loadtxt(args.censor_path)
        qc_data["n_trs_total"] = len(censor_data)
        qc_data["n_trs_censored"] = int(np.sum(censor_data == 0))
        qc_data["pct_censored"] = round(100.0 * qc_data["n_trs_censored"] / qc_data["n_trs_total"], 2) if qc_data["n_trs_total"] > 0 else 0.0
    except Exception:
        pass
    qc_data["per_condition_trial_counts"] = {c: int(len(stim_data[stim_data['CONDITION'] == c])) for c in args.cond_beta_labels}
    qc_data["per_condition_surviving_trials"] = trial_survival
    qc_data["dof"] = dof

    # Generate task design matrix, including regressors for denoising, censoring, and stimulus timing
    gen_design_matrix(stim_data, args, logger)

    # Generate full task beta-series and condition-specific beta series
    bseries_out = gen_beta_series(stim_data, args, logger)

    # Extract pbseries if user provided option
    if args.extract_pbseries:
        gen_pbseries(bseries_out, args, logger)

    # Generate connectivity from task beta-series if user provided option
    if args.calc_conn is not None:
        gen_conn(bseries_out, args, logger)

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
    parser.add_argument("--scan_path", type=file_path_exists, required=True)
    parser.add_argument("--task_timing_path", type=file_path_exists, required=True)
    parser.add_argument("--motion_path", type=file_path_exists, required=True)
    parser.add_argument("--cond_beta_labels", type=valid_string_list, required=True)
    parser.add_argument("--out_dir", type=dir_path_exists, required=True)
    parser.add_argument("--out_file_pre", type=str, required=True)
    parser.add_argument("--num_cores", type=int, required=True)
    parser.add_argument("--fd_threshold", type=float, required=True,
                        help="Framewise displacement threshold (mm) for motion censoring.")
    parser.add_argument("--censor_prev_tr", action='store_true', default=True,
                        help="Also censor the TR before each high-motion TR (default: True).")
    parser.add_argument("--tr", type=float, required=True,
                        help="Repetition time in seconds.")

    # optional:
    parser.add_argument("--hrf_model", type=str, choices=sorted(VALID_HRF_MODELS), default="GAM", required=False)
    parser.add_argument("--custom_hrf", type=str, default=None, required=False)
    parser.add_argument("--CSF_path", type=file_path_exists, default=None, required=False)
    parser.add_argument("--WM_path", type=file_path_exists, default=None, required=False)
    parser.add_argument("--remove_previous", action='store_true', required=False)
    parser.add_argument("--extract_pbseries", action='store_true', required=False)
    parser.add_argument("--average_type", type=valid_ave_type, required=False)
    parser.add_argument("--extract_out_file_pre", type=str, required=False)
    parser.add_argument("--calc_conn", type=valid_conn_type, required=False)
    parser.add_argument("--conn_out_file_pre", type=str, required=False)
    parser.add_argument("--template_path", type=file_path_exists, required=False)
    parser.add_argument("--force_diff_atlas", action='store_true', required=False)
    parser.add_argument("--pcorr", action='store_true', required=False)
    parser.add_argument("--fishZ", action='store_true', required=False)
    parser.add_argument("--use_tissue_derivs", action='store_true', default=False, required=False,
                        help="Include first temporal derivatives of tissue signals (CSF/WM) as additional nuisance regressors.")

    args = parser.parse_args()
    logger = setup_logging("task_conn_first_level")

    run(args, logger)

    # Display total runtime
    logger.info("Total runtime: %.2f seconds", time.time() - start_time)

if __name__ == "__main__":
    main()
