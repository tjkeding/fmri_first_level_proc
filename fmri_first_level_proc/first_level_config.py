#!/usr/bin/env python3

# ============================================================================
# YAML CONFIG LOADER AND VALIDATOR FOR fMRI FIRST-LEVEL PROCESSING
# Reads a YAML config file, validates its structure and values, merges global
# defaults into each analysis block, and builds argparse.Namespace objects
# compatible with each script's run() function.
#
# Author: Taylor J. Keding, Ph.D.
# Version: 2.0
# Last updated: 02/17/26
# ============================================================================

import os
import sys
import argparse

import yaml

from .first_level_utils import VALID_HRF_MODELS, validate_custom_hrf, needs_married_timing

VALID_TYPES = {"task_act", "task_conn", "rest_conn"}
VALID_STAT_TYPES = {"z", "t", "r"}
VALID_AVE_TYPES = {"mean", "median"}
VALID_CONN_TYPES = {"seed_to_voxel", "parcellated"}

# Required keys per analysis type
REQUIRED_KEYS = {
    "task_act":  {"paths", "out_dir", "out_file_pre", "cond_labels",
                  "average_type", "hrf_model", "include_motion_derivs", "fd_threshold"},
    "task_conn": {"paths", "out_dir", "out_file_pre", "cond_beta_labels",
                  "average_type", "hrf_model", "include_motion_derivs", "fd_threshold"},
    "rest_conn": {"paths", "out_dir", "out_file_pre",
                  "average_type", "bandpass", "motion_deriv_degree", "fd_threshold"},
}

# Required path sub-keys per analysis type
REQUIRED_PATHS = {
    "task_act": {"scan_path", "task_timing_path", "motion_path"},
    "task_conn": {"scan_path", "task_timing_path", "motion_path"},
    "rest_conn": {"scan_paths", "motion_paths", "CSF_paths", "WM_paths"},
}

# Keys that are actual boolean flags
BOOL_KEYS = ("remove_previous", "include_motion_derivs", "use_tissue_derivs", "censor_prev_tr")

# Keys that are always read from global (block values ignored)
GLOBAL_ONLY_KEYS = ("num_cores", "template_path", "force_diff_atlas", "tr")

def load_config(config_path):
    """Load a YAML config file and return the raw dict."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML parse error in {config_path}: {e}")
    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")
    return raw

def validate_config(raw_config, logger):
    """
    Validate the structure and values of a raw config dict.

    Returns the validated config (with 'global' defaulting to {} if absent).
    """
    # Top-level structure
    if "analyses" not in raw_config:
        logger.error("Config missing required top-level key 'analyses'.")
        sys.exit(1)
    if not isinstance(raw_config["analyses"], list) or len(raw_config["analyses"]) == 0:
        logger.error("'analyses' must be a non-empty list.")
        sys.exit(1)

    global_cfg = raw_config.get("global", {})
    if not isinstance(global_cfg, dict):
        logger.error("'global' must be a dict if provided.")
        sys.exit(1)

    # Validate global.tr (required for all analyses)
    tr_val = global_cfg.get("tr")
    if tr_val is None:
        logger.error("global.tr is required. Set it to the repetition time (seconds) for all scans.")
        sys.exit(1)
    try:
        tr_float = float(tr_val)
    except (TypeError, ValueError):
        logger.error("global.tr must be a positive number.")
        sys.exit(1)
    if tr_float <= 0:
        logger.error("global.tr must be a positive number, got %s.", tr_val)
        sys.exit(1)

    # Validate each analysis block
    for idx, block in enumerate(raw_config["analyses"]):
        label = block.get("name", f"analyses[{idx}]")

        # type
        if "type" not in block:
            logger.error("[%s] Missing required key 'type'.", label)
            sys.exit(1)
        atype = block["type"]
        if atype not in VALID_TYPES:
            logger.error("[%s] Invalid type '%s'. Must be one of %s.", label, atype, VALID_TYPES)
            sys.exit(1)

        # Required keys
        for key in REQUIRED_KEYS[atype]:
            if key not in block:
                logger.error("[%s] Missing required key '%s' for type '%s'.", label, key, atype)
                sys.exit(1)

        # Required path sub-keys
        paths = block.get("paths", {})
        if not isinstance(paths, dict):
            logger.error("[%s] 'paths' must be a dict.", label)
            sys.exit(1)
        for pkey in REQUIRED_PATHS[atype]:
            if pkey not in paths:
                logger.error("[%s] Missing required path key '%s' for type '%s'.", label, pkey, atype)
                sys.exit(1)

        # Validate file paths exist
        if atype in ("task_act", "task_conn"):
            for pkey in REQUIRED_PATHS[atype]:
                fpath = paths[pkey]
                if not os.path.isfile(fpath):
                    logger.error("[%s] File does not exist: paths.%s = '%s'.", label, pkey, fpath)
                    sys.exit(1)
            # Validate optional tissue regressor paths (skip if null)
            for opt_pkey in ("CSF_path", "WM_path"):
                if opt_pkey in paths and paths[opt_pkey] is not None:
                    if not os.path.isfile(paths[opt_pkey]):
                        logger.error("[%s] File does not exist: paths.%s = '%s'.", label, opt_pkey, paths[opt_pkey])
                        sys.exit(1)
            # Warn if use_tissue_derivs is enabled but no tissue paths are provided
            if block.get("use_tissue_derivs", False):
                csf_val = paths.get("CSF_path")
                wm_val = paths.get("WM_path")
                if csf_val is None and wm_val is None:
                    logger.warning("[%s] use_tissue_derivs is true but both CSF_path and WM_path are null — no tissue derivatives will be computed.", label)
        elif atype == "rest_conn":
            for pkey in REQUIRED_PATHS[atype]:
                plist = paths[pkey]
                if not isinstance(plist, list) or len(plist) == 0:
                    logger.error("[%s] paths.%s must be a non-empty list for rest_conn.", label, pkey)
                    sys.exit(1)
                for fpath in plist:
                    if not os.path.isfile(fpath):
                        logger.error("[%s] File does not exist: paths.%s entry '%s'.", label, pkey, fpath)
                        sys.exit(1)

            # Validate optional GS_paths if present and not null
            if "GS_paths" in paths and paths["GS_paths"] is not None:
                gs_list = paths["GS_paths"]
                if not isinstance(gs_list, list) or len(gs_list) == 0:
                    logger.error("[%s] paths.GS_paths must be a non-empty list for rest_conn.", label)
                    sys.exit(1)
                for fpath in gs_list:
                    if not os.path.isfile(fpath):
                        logger.error("[%s] File does not exist: paths.GS_paths entry '%s'.", label, fpath)
                        sys.exit(1)

            # All rest_conn path lists must be the same length (include optional GS_paths if present)
            lengths = {pkey: len(paths[pkey]) for pkey in REQUIRED_PATHS[atype]}
            if "GS_paths" in paths and paths["GS_paths"] is not None:
                lengths["GS_paths"] = len(paths["GS_paths"])
            unique_lens = set(lengths.values())
            if len(unique_lens) > 1:
                logger.error("[%s] All rest_conn path lists must be the same length. Got: %s", label, lengths)
                sys.exit(1)

        # -- Per-block enum validation --
        ave = block.get("average_type")
        if ave is not None and ave not in VALID_AVE_TYPES:
            logger.error("[%s] average_type must be one of %s, got '%s'.", label, VALID_AVE_TYPES, ave)
            sys.exit(1)

        if atype in ("task_act", "task_conn"):
            hrf = block.get("hrf_model")
            if hrf is not None and hrf not in VALID_HRF_MODELS:
                logger.error("[%s] hrf_model must be one of %s, got '%s'.", label, VALID_HRF_MODELS, hrf)
                sys.exit(1)

        # Warn if block contains global-only keys (they will be ignored)
        for key in GLOBAL_ONLY_KEYS:
            if key in block:
                logger.warning("[%s] '%s' is a global-only setting — block value will be ignored.", label, key)

        # rest_conn-specific parameter validation
        if atype == "rest_conn":
            # motion_deriv_degree validation
            mdd = block.get("motion_deriv_degree")
            if mdd is not None:
                try:
                    mdd_int = int(mdd)
                except (TypeError, ValueError):
                    logger.error("[%s] motion_deriv_degree must be a positive integer.", label)
                    sys.exit(1)
                if mdd_int < 1:
                    logger.error("[%s] motion_deriv_degree must be a positive integer, got %s.", label, mdd)
                    sys.exit(1)

            # Bandpass validation — required for rest_conn, must be [low, high] list
            bp = block.get("bandpass")
            if bp is None:
                logger.error("[%s] bandpass is required for rest_conn. Provide [low, high] Hz.", label)
                sys.exit(1)
            if bp is True or bp is False:
                logger.error("[%s] bandpass must be a [low, high] list for rest_conn, not true/false.", label)
                sys.exit(1)
            if not (isinstance(bp, list) and len(bp) == 2):
                logger.error("[%s] bandpass must be a list of 2 floats [low, high].", label)
                sys.exit(1)
            try:
                bp_low, bp_high = float(bp[0]), float(bp[1])
            except (TypeError, ValueError):
                logger.error("[%s] bandpass values must be numeric.", label)
                sys.exit(1)
            if bp_low < 0 or bp_high <= bp_low:
                logger.error("[%s] bandpass must satisfy 0 <= low < high. Got [%s, %s].", label, bp[0], bp[1])
                sys.exit(1)

            # notch_filter_band validation (rest_conn only, optional)
            nfb = block.get("notch_filter_band")
            if nfb is not None:
                if not (isinstance(nfb, list) and len(nfb) == 2):
                    logger.error("[%s] notch_filter_band must be a list of 2 floats [low, high] or null.", label)
                    sys.exit(1)
                try:
                    nfb_low, nfb_high = float(nfb[0]), float(nfb[1])
                except (TypeError, ValueError):
                    logger.error("[%s] notch_filter_band values must be numeric.", label)
                    sys.exit(1)
                if nfb_low <= 0 or nfb_high <= 0 or nfb_low >= nfb_high:
                    logger.error("[%s] notch_filter_band must satisfy 0 < low < high. Got [%s, %s].", label, nfb[0], nfb[1])
                    sys.exit(1)

        # Validate fd_threshold (all types)
        fd = block.get("fd_threshold")
        if fd is not None:
            try:
                fd_float = float(fd)
            except (TypeError, ValueError):
                logger.error("[%s] fd_threshold must be a positive number.", label)
                sys.exit(1)
            if fd_float <= 0:
                logger.error("[%s] fd_threshold must be a positive number, got %s.", label, fd)
                sys.exit(1)

        # Validate out_dir can be created
        out_dir = block["out_dir"]
        if not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except OSError as e:
                logger.error("[%s] Cannot create out_dir '%s': %s", label, out_dir, e)
                sys.exit(1)

        # Contrasts validation (task_act only) — functions: null = disabled
        if "contrasts" in block:
            contrasts = block["contrasts"]
            funcs = contrasts.get("functions")
            labels = contrasts.get("labels")
            if funcs is not None and labels is not None:
                if len(funcs) != len(labels):
                    logger.error("[%s] contrasts.functions and contrasts.labels must be the same length.", label)
                    sys.exit(1)
            elif funcs is not None and labels is None:
                logger.error("[%s] contrasts.functions provided but contrasts.labels is missing.", label)
                sys.exit(1)

        # HRF model validation (task_act and task_conn only)
        if atype in ("task_act", "task_conn"):
            hrf = block.get("hrf_model")
            if hrf is not None:
                if hrf not in VALID_HRF_MODELS:
                    logger.error("[%s] hrf_model must be one of %s, got '%s'.", label, VALID_HRF_MODELS, hrf)
                    sys.exit(1)
                if hrf == "custom":
                    custom = block.get("custom_hrf")
                    if not custom:
                        logger.error("[%s] custom_hrf string is required when hrf_model='custom'.", label)
                        sys.exit(1)
                    else:
                        if not validate_custom_hrf(custom, logger):
                            sys.exit(1)

                # task_conn uses 3dLSS which requires -stim_times_IM;
                # married-timing HRFs (e.g. dmBLOCK) require -stim_times_AM1
                # and are therefore incompatible.
                if atype == "task_conn":
                    custom = block.get("custom_hrf")
                    if needs_married_timing(hrf, custom):
                        logger.error(
                            "[%s] hrf_model '%s' requires married timing (-stim_times_AM1) which is "
                            "incompatible with 3dLSS (-stim_times_IM). Use GAM, SPMG1, or another "
                            "non-married HRF for task_conn analyses.", label, hrf)
                        sys.exit(1)

        # Extraction validation
        if "extraction" in block:
            ext = block["extraction"]
            if atype == "task_act":
                # New extract key is the on/off switch
                do_extract = ext.get("extract", False)
                if do_extract:
                    for ekey in ("extract_labels", "extract_stat", "extract_out_file_pre"):
                        if ekey not in ext:
                            logger.error("[%s] extraction block missing required key '%s' for task_act when extract=true.", label, ekey)
                            sys.exit(1)
                    if ext["extract_stat"] not in VALID_STAT_TYPES:
                        logger.error("[%s] extraction.extract_stat must be one of %s.", label, VALID_STAT_TYPES)
                        sys.exit(1)
            elif atype == "task_conn":
                if ext.get("extract_pbseries", False) and "extract_out_file_pre" not in ext:
                    logger.error("[%s] extraction block missing 'extract_out_file_pre' for task_conn when extract_pbseries=true.", label)
                    sys.exit(1)
            elif atype == "rest_conn":
                if ext.get("extract_ptseries", False) and "extract_out_file_pre" not in ext:
                    logger.error("[%s] extraction block missing 'extract_out_file_pre' for rest_conn when extract_ptseries=true.", label)
                    sys.exit(1)

        # Connectivity validation
        b_conn = block.get("connectivity", {}) or {}
        b_calc = b_conn.get("calc_conn")
        if b_calc is not None:
            if b_calc not in VALID_CONN_TYPES:
                logger.error("[%s] connectivity.calc_conn must be one of %s, got '%s'.", label, VALID_CONN_TYPES, b_calc)
                sys.exit(1)

        # Template path validation — global-only, validate global value
        tpl = global_cfg.get("template_path")
        if tpl is not None and not os.path.isfile(tpl):
            logger.error("[%s] template_path file does not exist: '%s'.", label, tpl)
            sys.exit(1)

    # Check for duplicate (out_dir, out_file_pre) pairs across blocks
    seen_pairs = {}
    for idx, block in enumerate(raw_config["analyses"]):
        label = block.get("name", f"analyses[{idx}]")
        pair = (block["out_dir"], block["out_file_pre"])
        if pair in seen_pairs:
            logger.error("[%s] Duplicate (out_dir, out_file_pre) pair: ('%s', '%s') — "
                         "also used by [%s]. Output files would collide.",
                         label, pair[0], pair[1], seen_pairs[pair])
            sys.exit(1)
        seen_pairs[pair] = label

    raw_config["global"] = global_cfg
    return raw_config

def _merge_global_into_block(block, global_cfg):
    """
    Merge global settings into an analysis block.

    Global-only keys (num_cores, template_path, force_diff_atlas) are injected
    from global. All other settings use direct block values — no inheritance.
    """
    merged = dict(block)

    # Global-only keys: always use global value, ignore block
    for key in GLOBAL_ONLY_KEYS:
        merged[key] = global_cfg.get(key)

    # Boolean flag keys: null → False
    for key in BOOL_KEYS:
        val = merged.get(key)
        if val is None:
            merged[key] = False

    # Connectivity: use block values directly (no global merge)
    b_conn = merged.get("connectivity", {}) or {}
    merged_conn = {}
    for k in ("calc_conn", "conn_out_file_pre", "pcorr", "fishZ"):
        val = b_conn.get(k)
        if k in ("pcorr", "fishZ"):
            merged_conn[k] = bool(val) if val is not None else False
        else:
            merged_conn[k] = val  # None if absent
    merged["connectivity"] = merged_conn

    return merged

def build_namespace(merged_block, logger):
    """
    Flatten a merged YAML block into an argparse.Namespace with the exact
    attribute names each script's run() expects.
    """
    atype = merged_block["type"]
    label = merged_block.get("name", "unnamed")
    paths = merged_block.get("paths", {})
    extraction = merged_block.get("extraction", None)
    connectivity = merged_block.get("connectivity", {}) or {}
    contrasts = merged_block.get("contrasts", None)

    ns = argparse.Namespace()

    # -- Common attributes (null-safe) --
    ns.out_dir = merged_block["out_dir"]
    ns.out_file_pre = merged_block["out_file_pre"]
    val = merged_block.get("num_cores")
    ns.num_cores = val if val is not None else 1
    val = merged_block.get("remove_previous")
    ns.remove_previous = bool(val) if val is not None else False
    ns.template_path = merged_block.get("template_path", None)
    val = merged_block.get("force_diff_atlas")
    ns.force_diff_atlas = bool(val) if val is not None else False
    ns.average_type = merged_block["average_type"]

    # -- Global keys --
    val = merged_block.get("tr")
    ns.tr = float(val) if val is not None else None

    # -- Censor generation keys (all types) --
    ns.fd_threshold = float(merged_block["fd_threshold"])
    cpt = merged_block.get("censor_prev_tr")
    ns.censor_prev_tr = bool(cpt) if cpt is not None else False

    # -- Type-specific attributes --
    if atype == "task_act":
        ns.scan_path = paths["scan_path"]
        ns.task_timing_path = paths["task_timing_path"]
        ns.motion_path = paths["motion_path"]
        ns.cond_labels = merged_block["cond_labels"]
        ns.hrf_model = merged_block["hrf_model"]
        ns.custom_hrf = merged_block.get("custom_hrf", None)
        ns.CSF_path = paths.get("CSF_path", None)
        ns.WM_path = paths.get("WM_path", None)
        ns.include_motion_derivs = bool(merged_block.get("include_motion_derivs", False))
        ns.use_tissue_derivs = bool(merged_block.get("use_tissue_derivs", False))

        # Contrasts — functions: null = disabled
        if contrasts is not None:
            funcs = contrasts.get("functions")
            labels = contrasts.get("labels")
            if funcs is not None and labels is not None:
                ns.contrast_functions = funcs
                ns.contrast_labels = labels
            else:
                ns.contrast_functions = None
                ns.contrast_labels = None
        else:
            ns.contrast_functions = None
            ns.contrast_labels = None

        # Extraction — extract key is the on/off switch
        if extraction is not None:
            ns.extract = bool(extraction.get("extract", False))
            if ns.extract:
                ns.extract_labels = extraction.get("extract_labels")
                ns.extract_stat = extraction.get("extract_stat")
                ns.extract_out_file_pre = extraction.get("extract_out_file_pre")
                ns.extract_resids = bool(extraction.get("extract_resids", False))
            else:
                ns.extract_labels = None
                ns.extract_stat = None
                ns.extract_out_file_pre = None
                ns.extract_resids = False
        else:
            ns.extract = False
            ns.extract_labels = None
            ns.extract_stat = None
            ns.extract_out_file_pre = None
            ns.extract_resids = False

    elif atype == "task_conn":
        ns.scan_path = paths["scan_path"]
        ns.task_timing_path = paths["task_timing_path"]
        ns.motion_path = paths["motion_path"]
        ns.cond_beta_labels = merged_block["cond_beta_labels"]
        ns.hrf_model = merged_block["hrf_model"]
        ns.custom_hrf = merged_block.get("custom_hrf", None)
        ns.CSF_path = paths.get("CSF_path", None)
        ns.WM_path = paths.get("WM_path", None)
        ns.include_motion_derivs = bool(merged_block.get("include_motion_derivs", False))
        ns.use_tissue_derivs = bool(merged_block.get("use_tissue_derivs", False))

        # Extraction
        if extraction is not None:
            ns.extract_pbseries = bool(extraction.get("extract_pbseries", False))
            ns.extract_out_file_pre = extraction.get("extract_out_file_pre")
        else:
            ns.extract_pbseries = False
            ns.extract_out_file_pre = None

        # Connectivity
        ns.calc_conn = connectivity.get("calc_conn", None)
        ns.conn_out_file_pre = connectivity.get("conn_out_file_pre", None)
        val = connectivity.get("pcorr")
        ns.pcorr = bool(val) if val is not None else False
        val = connectivity.get("fishZ")
        ns.fishZ = bool(val) if val is not None else False

    elif atype == "rest_conn":
        ns.scan_paths = paths["scan_paths"]
        ns.motion_paths = paths["motion_paths"]
        ns.CSF_paths = paths["CSF_paths"]
        ns.WM_paths = paths["WM_paths"]

        ns.GS_paths = paths.get("GS_paths", None)
        ns.notch_filter_band = merged_block.get("notch_filter_band", None)
        ns.bandpass = merged_block["bandpass"]   # Required, always a [low, high] list
        ns.motion_deriv_degree = int(merged_block["motion_deriv_degree"])
        ns.use_tissue_derivs = bool(merged_block.get("use_tissue_derivs", False))
        val = merged_block.get("keep_run_res_dtseries")
        ns.keep_run_res_dtseries = val if val is not None else True

        # Extraction
        if extraction is not None:
            ns.extract_ptseries = bool(extraction.get("extract_ptseries", False))
            ns.extract_out_file_pre = extraction.get("extract_out_file_pre")
        else:
            ns.extract_ptseries = False
            ns.extract_out_file_pre = None

        # Connectivity
        ns.calc_conn = connectivity.get("calc_conn", None)
        ns.conn_out_file_pre = connectivity.get("conn_out_file_pre", None)
        val = connectivity.get("pcorr")
        ns.pcorr = bool(val) if val is not None else False
        val = connectivity.get("fishZ")
        ns.fishZ = bool(val) if val is not None else False

    return ns

# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def load_and_validate(config_path, logger):
    """
    Load, validate, merge, and build Namespace objects from a YAML config.

    Returns
    -------
    list of (str, argparse.Namespace, str)
        Each tuple is (analysis_type, namespace, analysis_name).
    """
    raw = load_config(config_path)
    validated = validate_config(raw, logger)

    global_cfg = validated["global"]
    results = []

    for idx, block in enumerate(validated["analyses"]):
        merged = _merge_global_into_block(block, global_cfg)
        ns = build_namespace(merged, logger)
        atype = block["type"]
        name = block.get("name", f"analyses[{idx}]")
        results.append((atype, ns, name))

    return results
