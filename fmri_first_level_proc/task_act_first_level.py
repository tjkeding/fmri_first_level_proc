#!/usr/bin/env python3

# ============================================================================
# TASK-BASED fMRI ACTIVATION ANALYSIS
# 1. First-level regression statistical dataset (including denoising and censoring) with AFNI's 3dDeconvolve
# 2. (optional) Creates condition/contrast- and stat-specific datasets with AFNI's 3dcalc
# 3. (optional) Extract parcel-level/ROI activation with a provided template with AFNI's 3dROIstats
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
--cond_labels: string-list (comma-separated) of task conditions to use in first level analyses - labels should match the 'CONDITION' column from the timing file;
    Conditions in cond_labels will be the only ones used in the regression (more could be in task_timing_path file) and the only ones that can be used within contrast_functions (optional)
    format = string-list of conditions e.g. 'stimFear,stimSad,stimNeu'
--out_dir: global directory path for derivative intermediates and statistical datasets; will create if doesn't exist
--out_file_pre: prefix to be prepended to the statistical dataset outputs (should usually contain an ID, task name, time point, etc. but NOT the global file path; eg. 'subj001_nback_baseline')
--num_cores: number of CPU cores available for parallel processing (int)

(optional)
--hrf_model: HRF basis function model to use (default: 'GAM'). Options: 'GAM', 'BLOCK', 'dmBLOCK', 'SPMG1', 'custom'
--custom_hrf: custom AFNI HRF model string (only used when hrf_model='custom', e.g. 'TENT(0,14,8)')
--CSF_path: global file path to the run-concatenated cerebrospinal fluid BOLD signal time series
    format = string(path_to_txt); no headers; rows=TR/frame, single column = mean CSF signal
--WM_path: global file path to the run-concatenated white-matter BOLD signal time series
    format = string(path_to_txt); no headers; rows=TR/frame, single column = mean WM signal
--use_tissue_derivs: (no value) include this option to add first temporal derivatives of tissue signals (CSF/WM) as additional nuisance regressors; only computed for tissue paths that are provided
--remove_previous: (no value) include this option if "out_dir" should have all files removed before processing starts (including connectivity); if not included, will not overwrite pre-existing files
--contrast_functions: list of string formulas for linear contrasts using condition names in cond_labels
    format = list(string-equations) (e.g. if conditions "stimFear", "stimSad", and "stimNeu" exist in cond_labels, "1*stimFear-1*stimNeu,1*stimSad-1*stimNeu" would produce Fear-Neutral and Sad-Neutral constrasts)
--contrast_labels: list of string labels to give to each constrast in contrasts_functions
    format = list(string-labels) (for the example above, "Fear-Neutral,Sad-Neutral")
--extract: (no value) include this option if you want condition/contrast- and stat-specific maps
    If present and no template is provided to template_path, voxelwise whole-brain maps will be output; if present and template_path is valid, parcellated maps will be output
--extract_labels: if --extract is present, list of string labels to extract (each to their own file); MUST have corresponding matches in either --cond_labels or --constrast_labels
    format = list(string-labels) (e.g. "stimFear,stimSad,Fear-Neutral,Sad-Neutral" would save 4 maps specific to each condition/contrast and stat contained in --extract_stat)
--extract_stat: if --extract is present, statistic to extract from the 'bucket' statistical dataset from 3dDeconvolve; options: 'z', 't', 'r'
--extract_out_file_pre: if --extract, the prefix to attach to output files (should NOT include the file path, task condition/contrast, or stat; similar to --out_file_pre e.g. 'subj001_nback_baseline_Shen368')
--template_path: optional template file for --extract; if --extract is present and no template exists, voxelwise maps will be extracted, else parcel-level/ROI averages will be extracted based on the template and extract_stat
    Can be a binary mask (only a single parcel/ROI will be output) or a set of N ROIs/masks indicated by unique integers>0 (full parcel/ROI table will be output)
    format = string(path_to_nii) (volumetric and registered to the same atlas as input scan, unless --force_diff_atlas used; if there is a mismatch in grid spacing, will resample the template to the input scan grid)
--average_type: either "mean" or "median" - the type of averaging used when calculating average parcel activation with template
--extract_resids: (no value) if --extract is present and a valid template is provided, flag to indicate that parcel-level/ROI residuals should also be extracted.
--force_diff_atlas: (no value) include this option if you know the template and scan are in different standard spaces, but want to force connectivity analyses anyway (USE WITH CAUTION)

OUTPUTS:
(always)
{out_dir}/{out_file_pre}_concat_{condition}_onsets.txt: AFNI-safe version of timing onsets for task conditions (1D single column with no headers or row names)
{out_dir}/{out_file_pre}_concat_bucket_stats.nii.gz: first-level output statistical dataset for the entire task (subbrik = stim-stat combination)
{out_dir}/{out_file_pre}_concat_bucket_resids.nii.gz: residuals from the first-level output statistical dataset for the entire task

(optional)
--extract:
    If no template is provided: {out_dir}/{extract_out_file_pre}_{condition/constrast}_{stat}.nii.gz: condition- and stat-specific voxelwise map
    If a template is provided: {out_dir}/{extract_out_file_pre}_{condition/constrast}_{stat}.csv; rows correspond to each parcel/ROI in the template and columns are ["ROI", "ACT_{averageType}_{stat}"]
    If a template is provided and --extract_resids is present: {out_dir}/{extract_out_file_pre}_resids.csv; rows correspond to each parcel/ROI in the template and columns are ["ROI", "ACT_{averageType}_RESID"]
'''

# Imports
import copy
import os
import sys
import subprocess
import argparse
import time
import re

import numpy as np
import pandas as pd

from .first_level_utils import (
    setup_logging,
    dir_path_exists,
    file_path_exists,
    valid_string_list,
    valid_stat_type,
    valid_ave_type,
    run_afni_command,
    validate_template,
    remove_files_from_dir,
    extract_roi_stats,
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
    CENSOR_WARN_THRESHOLD,
    CENSOR_HIGH_THRESHOLD,
    HRF_DM_MODELS,
    HRF_IMPULSE_MODELS,
    HRF_DURATION_MODELS,
    VALID_HRF_MODELS,
)

def valid_contrast_functions(contrast_list, contrast_labs_list, cond_list, logger=None):
    """Parse and validate linear contrast equations.

    Accepted format: ``coef*COND[+-coef*COND...]``
    Examples: ``1*stimA-1*stimB``, ``-1*A+0.5*B+0.5*C``, ``+1*X-1*Y``

    Returns a list of dicts, one per contrast, each with keys 'COEFS' (list of
    str coefficients with signs) and 'CONDS' (list of condition names).
    """

    # Check that contrast_list and contrast_labs_list are same size
    if len(contrast_list) != len(contrast_labs_list):
        logger.error("Every contrast in --contrast_functions must have a label in --contrast_labels")
        sys.exit(1)

    # Regex for a single term: optional sign, a numeric coefficient, *, condition name
    _TERM_RE = re.compile(r'([+-]?\d*\.?\d+)\*(\w+)')
    # Regex that matches the entire valid contrast string (one or more terms)
    _FULL_RE = re.compile(r'^([+-]?\d*\.?\d+\*\w+)([+-]\d*\.?\d+\*\w+)*$')

    out = []
    for contrast in contrast_list:
        raw = contrast
        contrast = contrast.replace(" ", "")

        if not contrast:
            logger.error("Empty contrast equation (original: '%s').", raw)
            sys.exit(1)

        # Validate the entire string matches the expected grammar
        if not _FULL_RE.match(contrast):
            # Provide a targeted diagnostic
            if '*' not in contrast:
                logger.error("Contrast '%s' is missing coefficients. "
                             "Every condition must have an explicit coefficient "
                             "(e.g. '1*stimA-1*stimB', not 'stimA-stimB').", raw)
            elif re.search(r'[*/]{2,}', contrast):
                logger.error("Contrast '%s' has consecutive operators.", raw)
            elif contrast.endswith(('+', '-', '*')):
                logger.error("Contrast '%s' has a trailing operator.", raw)
            elif re.search(r'[^+\-\d.*\w]', contrast):
                bad = re.findall(r'[^+\-\d.*\w]', contrast)
                logger.error("Contrast '%s' contains invalid characters: %s", raw, bad)
            else:
                logger.error("Contrast '%s' could not be parsed. "
                             "Expected format: coef*COND[+-coef*COND...] "
                             "(e.g. '1*stimA-1*stimB').", raw)
            sys.exit(1)

        matches = _TERM_RE.findall(contrast)

        coefs = [m[0] for m in matches]
        conds = [m[1] for m in matches]

        # Check all contrast conditions exist in cond_list
        for item in conds:
            if item not in cond_list:
                logger.error("Contrast '%s' references condition '%s' which is not in --cond_labels (%s).",
                             raw, item, cond_list)
                sys.exit(1)

        # Validate coefficients are valid floats
        for item in coefs:
            try:
                float(item)
            except ValueError:
                logger.error("Contrast '%s' has invalid coefficient '%s'.", raw, item)
                sys.exit(1)

        # Warn about zero coefficients (technically valid but likely a mistake)
        for coef, cond in zip(coefs, conds):
            if float(coef) == 0.0:
                logger.warning("Contrast '%s': coefficient for '%s' is 0 — "
                               "this condition will have no effect.", raw, cond)

        out.append({'COEFS': coefs, 'CONDS': conds})
    return out

def valid_extract_labels(cond_labels, contrast_labels, extract_labels, logger=None):
    combined = list(cond_labels) + (contrast_labels if contrast_labels else [])
    for valid in extract_labels:
        if valid not in combined:
            logger.error("Every entry in --extract_labels must have an exact label in --contrast_labels or --cond_labels.")
            sys.exit(1)

def get_stim_data(args, logger):

    # Read, validate, and sort stim timing data
    sorted_df = read_and_validate_stim_data(args.task_timing_path, args.cond_labels, logger=logger)

    # Create individual (run-concatenated) timing files
    for cond in sorted_df['CONDITION'].unique():
        cond_df = sorted_df[sorted_df['CONDITION'] == cond]
        if cond in args.cond_labels:
            onset_file = f"{args.out_dir}/{args.out_file_pre}_concat_{cond}_onsets.txt"
            write_onset_file(cond_df, onset_file, args.hrf_model, args.custom_hrf, logger=logger)

    return sorted_df

def run_first_level(stim_data, args, logger):

    # Check if outputs exists
    if not os.path.exists(f"{args.out_dir}/{args.out_file_pre}_concat_bucket_stats.nii.gz"):

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
        decon_command.extend(["-num_stimts", f"{len(args.cond_labels)}"])

        # Iteratively add stim timing for conditions
        for i, cond in enumerate(args.cond_labels):
            # Compute mean duration for models that need it
            mean_dur = None
            if args.hrf_model in HRF_DURATION_MODELS:
                mean_dur = np.mean(stim_data[stim_data['CONDITION'] == cond]['DURATION'])
            hrf_string = build_hrf_string(args.hrf_model, mean_dur=mean_dur, custom_hrf=args.custom_hrf)

            # dmBLOCK-family models require -stim_times_AM1 for married timing
            stim_flag = "-stim_times_AM1" if needs_married_timing(args.hrf_model, args.custom_hrf) else "-stim_times"
            decon_command.extend([stim_flag, f"{i+1}", f"{args.out_dir}/{args.out_file_pre}_concat_{cond}_onsets.txt",
                                 hrf_string, "-stim_label", f"{i+1}", f"{cond}"])

        # Iteratively add contrasts
        if args.contrast_functions:
            decon_command.extend(["-num_glt", str(len(args.contrast_functions))])
            for ci, cont in enumerate(args.contrast_functions):
                sym_eq = "SYM: " + " ".join(
                    f"{coef}*{cond}" for coef, cond in zip(cont["COEFS"], cont["CONDS"])
                )
                decon_command.extend(["-gltsym", sym_eq,
                                      "-glt_label", f"{ci+1}", f"{args.contrast_labels[ci]}"])

        # Add statistics output
        decon_command.extend(["-errts", f"{args.out_dir}/{args.out_file_pre}_concat_bucket_resids.nii.gz",
                              "-fout", "-rout", "-tout",
                              "-bucket", f"{args.out_dir}/{args.out_file_pre}_concat_bucket_stats.nii.gz",
                              "-jobs", f"{args.num_cores}"])

        # Run command and be sure we have output
        run_afni_command(decon_command, description="3dDeconvolve activation", logger=logger)

        if os.path.exists(f"{args.out_dir}/{args.out_file_pre}_concat_bucket_stats.nii.gz"):
            logger.info("Successfully created first level activation stats for %s", args.out_file_pre)
        else:
            logger.error("Failed to create first level activation stats for %s", args.out_file_pre)
            sys.exit(1)
    else:
        logger.info("First level activation stats for %s already exists.", args.out_file_pre)

    # Remove err file that always seems to be output
    clean_deconvolve_err(args.out_dir)

def _find_subbrik_index(bucket_split, cond_label, stat_suffix, logger=None):
    """Find the sub-brick index for a condition and stat type in 3dinfo output.

    Parameters
    ----------
    bucket_split : list of str
        Output of ``3dinfo -subbrick_info``, split on whitespace.
    cond_label : str
        The condition or contrast label to match (e.g. ``"ave_negface-posface"``).
    stat_suffix : str
        The stat suffix to match (e.g. ``"Tstat"``, ``"Fstat"``).

    Returns
    -------
    int or None
        The sub-brick index, or None if not found.
    """
    for i, substr in enumerate(bucket_split):
        if cond_label in substr and stat_suffix in substr:
            try:
                return int(bucket_split[i - 1][1:])
            except (ValueError, IndexError):
                if logger:
                    logger.error("Could not parse sub-brick index for %s, %s", cond_label, stat_suffix)
                return None
    return None


def _get_subbrik_stat_aux(bucket_path, subbrik_index, logger=None):
    """Get the statistical auxiliary parameters (e.g. DOF) for a sub-brick.

    Returns
    -------
    list of float
        The auxiliary parameter values (e.g. [DOF] for a T-stat sub-brick).
    """
    try:
        out = run_afni_command(
            ["3dAttribute", "BRICK_STATAUX", bucket_path],
            capture_output=True,
            description="3dAttribute BRICK_STATAUX",
            logger=logger,
        )
    except subprocess.CalledProcessError:
        return []

    # BRICK_STATAUX format: pairs of (subbrik_index stat_type n_params param1 param2 ...)
    # stat_type 3 = T-stat (params: DOF)
    tokens = out.split()
    i = 0
    while i < len(tokens):
        try:
            idx = int(tokens[i])
            stype = int(tokens[i + 1])
            npar = int(tokens[i + 2])
            params = [float(tokens[i + 3 + j]) for j in range(npar)]
            if idx == subbrik_index:
                return params
            i += 3 + npar
        except (ValueError, IndexError):
            break
    return []


def extract_effects(args, logger):

    bucket_path = f"{args.out_dir}/{args.out_file_pre}_concat_bucket_stats.nii.gz"

    # Get subbrik info from bucket stats dataset
    try:
        bucket_info = run_afni_command(
            ['3dinfo', '-subbrick_info', bucket_path],
            capture_output=True,
            description="3dinfo subbrick_info",
            logger=logger,
        )
    except subprocess.CalledProcessError:
        logger.error("Could not open %s with AFNI's 3dinfo - check your first-level outputs.", bucket_path)
        sys.exit(1)

    bucket_split = bucket_info.split()

    # Determine whether T-to-Z conversion is needed.
    # 3dDeconvolve outputs T-stats, not Z-stats. When extract_stat="z",
    # we find the T-stat sub-brick and convert using fitt_t2z().
    need_t2z = args.extract_stat == "z"
    search_stat = "Tstat" if need_t2z else f"{args.extract_stat.upper()}stat"

    # Iterate effects to extract from bucket stats
    for curr_cond in args.extract_labels:

        out_csv = f"{args.out_dir}/{args.extract_out_file_pre}_{curr_cond}_{args.extract_stat}.csv"
        out_nii = f"{args.out_dir}/{args.extract_out_file_pre}_{curr_cond}_{args.extract_stat}.nii.gz"

        # Check if output for this effect already exists
        if not os.path.exists(out_csv) and not os.path.exists(out_nii):

            # Get the subbrik associated with the current effect and stat
            out_sub_brik_indx = _find_subbrik_index(bucket_split, curr_cond, search_stat, logger=logger)
            if out_sub_brik_indx is None:
                logger.error("Could not find appropriate bucket subbrik for %s, %s in %s", curr_cond, search_stat, bucket_path)
                sys.exit(1)

            extract_vol = f"{bucket_path}[{out_sub_brik_indx}]"

            # If Z-stats requested, convert T-stat sub-brick to Z using 3dcalc
            if need_t2z:
                aux_params = _get_subbrik_stat_aux(bucket_path, out_sub_brik_indx, logger=logger)
                if not aux_params:
                    logger.error("Could not retrieve DOF for sub-brick %d in %s; cannot convert T to Z.", out_sub_brik_indx, bucket_path)
                    sys.exit(1)
                dof = int(aux_params[0])
                z_prefix = f"{args.out_dir}/{args.out_file_pre}_{curr_cond}_zconv_tmp.nii.gz"
                calc_cmd = ["3dcalc", "-a", extract_vol,
                            "-expr", f"fitt_t2z(a,{dof})",
                            "-prefix", z_prefix]
                try:
                    run_afni_command(calc_cmd, description=f"T-to-Z conversion for {curr_cond}", logger=logger)
                except subprocess.CalledProcessError:
                    logger.error("Failed to convert T to Z for %s.", curr_cond)
                    sys.exit(1)
                extract_vol = z_prefix
                logger.info("Converted T-stat to Z-stat for %s (DOF=%d).", curr_cond, dof)

            # If no template was given, output specific condition/stat bucket to its own nifti
            if args.template_path == "WB":

                if need_t2z:
                    # Z volume already created as a standalone nifti; rename it
                    os.rename(z_prefix, out_nii)
                else:
                    # Build the 3dcalc command and extract from the identified bucket subbrik
                    calc_cmd = ["3dcalc", "-a", extract_vol,
                                "-expr", "a", "-prefix", out_nii]
                    try:
                        run_afni_command(calc_cmd, description="3dcalc extract subbrik", logger=logger)
                    except subprocess.CalledProcessError:
                        pass
                if not os.path.exists(out_nii):
                    logger.error("Could not create %s from %s", out_nii, bucket_path)
                    sys.exit(1)

            # If a template was given, output average activation condition/stat bucket by ROI to .csv
            else:
                new_act_df = extract_roi_stats(extract_vol, args.template_path, args.average_type, logger=logger)
                new_act_df.rename(index={0: f"{args.extract_stat}"}, inplace=True)

                new_act_df.to_csv(out_csv)
                if os.path.exists(out_csv):
                    logger.info("Successfully extracted %s, %s using ROI template.", curr_cond, args.extract_stat)
                else:
                    logger.error("Could not extract %s, %s using ROI template.", curr_cond, args.extract_stat)
                    sys.exit(1)

                # Clean up temporary Z conversion file
                if need_t2z and os.path.exists(z_prefix):
                    os.remove(z_prefix)
        else:
            logger.info("%s/%s_%s_%s already exists (skipping).", args.out_dir, args.extract_out_file_pre, curr_cond, args.extract_stat)

    # Extract residuals from the template
    if args.extract_resids:
        resid_path = f"{args.out_dir}/{args.out_file_pre}_concat_bucket_resids.nii.gz"
        new_resid_df = extract_roi_stats(resid_path, args.template_path, args.average_type, logger=logger)
        new_resid_df.rename(index={0: f"{args.extract_stat}"}, inplace=True)

        new_resid_df.to_csv(f"{args.out_dir}/{args.extract_out_file_pre}_resids.csv")
        if os.path.exists(f"{args.out_dir}/{args.extract_out_file_pre}_resids.csv"):
            logger.info("Successfully extracted residuals using ROI template.")
        else:
            logger.error("Could not extract residuals using ROI template.")
            sys.exit(1)

def run(args, logger):
    """Validate args and execute pipeline. Works from CLI or config runner."""
    args = copy.copy(args)

    logger.info("First-level regression will use the following task conditions: %s", args.cond_labels)
    logger.info("All other task conditions will not be modeled.")
    log_hrf_model(args.hrf_model, args.custom_hrf, logger)
    log_nuisance_model(args.CSF_path, args.WM_path, logger, use_tissue_derivs=args.use_tissue_derivs)

    # Contrast-related checks
    if args.contrast_labels is not None:
        if args.contrast_functions is None:
            logger.error("--contrast_functions must exist if --contrast_labels exist.")
            sys.exit(1)
    else:
        if args.contrast_functions is not None:
            logger.error("--contrast_labels must exist if --contrast_functions exist.")
            sys.exit(1)
    if args.contrast_functions is not None and args.contrast_labels is not None:
        args.contrast_functions = valid_contrast_functions(args.contrast_functions, args.contrast_labels, args.cond_labels, logger=logger)

    # If --remove_previous is provided, clear-out the output dir (before template
    # resampling or any other file creation in out_dir)
    if args.remove_previous:
        logger.info("Removing all files from previous analyses in %s (if they exist).", args.out_dir)
        remove_files_from_dir(args.out_dir, logger=logger)

    # Check if the user wants to extract data
    if args.extract:

        if args.extract_labels is None or args.extract_stat is None or args.extract_out_file_pre is None:
            logger.error("--extract_labels, --extract_stat, and --extract_out_file_pre must be present and valid with --extract option")
            sys.exit(1)
        else:
            valid_extract_labels(args.cond_labels, args.contrast_labels, args.extract_labels, logger=logger)
        logger.info("Activation will be calculated from %s maps for the following task effects: %s", args.extract_stat, args.extract_labels)

        if args.template_path is None:
            args.template_path = "WB"
            logger.info("No template was provided, so all extractions will be whole-brain, voxelwise maps.")
            if args.extract_resids:
                logger.error("--extract_resids is only valid when --template_path is provided and valid.")
                sys.exit(1)
        else:
            args.template_path = validate_template(args.scan_path, args.template_path, args.out_dir, args.force_diff_atlas, conn_type="extract", logger=logger)
            if args.average_type is None:
                logger.error("--average_type is required when --extract is provided and --template_path is valid.")
                sys.exit(1)
            logger.info("Parcellated %s %s will be extracted based on %s.", args.average_type, args.extract_stat, args.template_path)
    else:
        logger.info("Activation will not be extracted after generating voxelwise maps.")

    # ---------------------------------

    # Generate censor file from motion regressors (before prepare_motion_file overwrites path)
    args.censor_path = create_censor_file(
        args.motion_path, args.fd_threshold, args.censor_prev_tr,
        args.out_dir, args.out_file_pre, "concat", logger)

    # Validate and prepare motion regressors
    use_cols = 12 if args.include_motion_derivs else 6
    args.motion_path = prepare_motion_file(
        args.motion_path, use_cols, args.out_dir, args.out_file_pre, "concat", logger)

    # Pre-flight DOF check
    n_regressors = use_cols + len(args.cond_labels)
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

    # Get stimulus timing data and save AFNI-safe single-column .txt files
    stim_data = get_stim_data(args, logger)

    # Per-condition trial survival QC
    trial_survival = check_trial_survival(stim_data, args.cond_labels, args.censor_path, args.tr, logger)

    # Build QC summary data
    qc_data = {
        "analysis_type": "task_act",
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
    qc_data["per_condition_trial_counts"] = {c: int(len(stim_data[stim_data['CONDITION'] == c])) for c in args.cond_labels}
    qc_data["per_condition_surviving_trials"] = trial_survival
    qc_data["dof"] = dof

    # Run activation first-level analysis
    run_first_level(stim_data, args, logger)

    # Run extraction if indicated
    if args.extract:
        extract_effects(args, logger)

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
    parser.add_argument("--cond_labels", type=valid_string_list, required=True)
    parser.add_argument("--out_dir", type=dir_path_exists, required=True)
    parser.add_argument("--out_file_pre", type=str, required=True)
    parser.add_argument("--num_cores", type=int, required=True)
    parser.add_argument("--fd_threshold", type=float, required=True,
                        help="Framewise displacement threshold (mm) for motion censoring.")
    parser.add_argument("--censor_prev_tr", action='store_true', default=False,
                        help="Also censor the TR before each high-motion TR (default: False).")
    parser.add_argument("--include_motion_derivs", action='store_true', default=False,
                        help="Include temporal derivatives of motion regressors (12 columns instead of 6; default: False).")
    parser.add_argument("--tr", type=float, required=True,
                        help="Repetition time in seconds.")

    # optional:
    parser.add_argument("--hrf_model", type=str, choices=sorted(VALID_HRF_MODELS), default="GAM", required=False)
    parser.add_argument("--custom_hrf", type=str, default=None, required=False)
    parser.add_argument("--CSF_path", type=file_path_exists, default=None, required=False)
    parser.add_argument("--WM_path", type=file_path_exists, default=None, required=False)
    parser.add_argument("--remove_previous", action='store_true', required=False)
    parser.add_argument("--contrast_functions", type=valid_string_list, required=False)
    parser.add_argument("--contrast_labels", type=valid_string_list, required=False)
    parser.add_argument("--extract", action='store_true', required=False)
    parser.add_argument("--extract_labels", type=valid_string_list, required=False)
    parser.add_argument("--extract_stat", type=valid_stat_type, required=False)
    parser.add_argument("--extract_out_file_pre", type=str, required=False)
    parser.add_argument("--template_path", type=file_path_exists, required=False)
    parser.add_argument("--average_type", type=valid_ave_type, required=False)
    parser.add_argument("--extract_resids", action='store_true', required=False)
    parser.add_argument("--force_diff_atlas", action='store_true', required=False)
    parser.add_argument("--use_tissue_derivs", action='store_true', default=False, required=False,
                        help="Include first temporal derivatives of tissue signals (CSF/WM) as additional nuisance regressors.")

    args = parser.parse_args()
    logger = setup_logging("task_act_first_level")

    run(args, logger)

    # Display total runtime
    logger.info("Total runtime: %.2f seconds", time.time() - start_time)

if __name__ == "__main__":
    main()
