# fmri_first_level_proc

General-purpose first-level fMRI analysis framework built on AFNI.

## Overview

- Three analysis types: task activation, task connectivity (beta series), resting-state connectivity
- YAML config-driven: define all analyses in one file, run with a single command
- pip-installable Python package

## Prerequisites

- Python >= 3.8
- [AFNI](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/main_toc.html) (installed separately; must be on your PATH)
- numpy, pandas, pyyaml

## Installation

pip install:

```
pip install git+https://github.com/tjkeding/fmri_first_level_proc.git
```

Or conda environment:

```
conda env create -f environment.yaml
pip install .
```

## Quick Start

1. Copy `example_config.yaml`
2. Edit paths, settings, and analysis blocks
3. Dry-run: `run-first-level --config my_config.yaml --dry-run`
4. Run: `run-first-level --config my_config.yaml`

## CLI Reference

```
run-first-level --config <path> [options]
```

| Flag | Description |
|---|---|
| `--config` | Path to YAML config file (required) |
| `--dry-run` | Validate config and print plan without executing |
| `--analyses` | Run only specific block indices (0-based), e.g. `--analyses 0 2` |
| `--log-file` | Write logs to file in addition to console |

## Analysis Types

### Task Activation (`task_act`)

First-level GLM via `3dDeconvolve`. Supports multiple HRF models, linear contrasts,
and optional parcel-level stat extraction.

### Task Connectivity (`task_conn`)

Beta series estimation via `3dDeconvolve` + `3dLSS`. Optional parcel beta series
extraction and functional connectivity ('seed-to-voxel' or 'parcellated').

### Resting-State Connectivity (`rest_conn`)

Residual time series via `3dTproject` with bandpass filtering. Per-run processing
with concatenation. Optional parcel time series extraction and functional connectivity.

## Configuration

See `example_config.yaml` for the full reference with all options documented.

### Global Settings

- `num_cores`: CPU cores for AFNI parallel processing
- `tr`: repetition time in seconds (required)
- `template_path`: ROI template for extraction/connectivity
- `force_diff_atlas`: allow atlas mismatch

### Motion Regressors

Task-based analyses (`task_act`, `task_conn`) use `include_motion_derivs`:
- `false`: 6 motion columns (translation + rotation)
- `true`: 12 motion columns (6 base + 6 temporal derivatives)

Resting-state analyses (`rest_conn`) use `motion_deriv_degree`:
- Total columns = 6 * `motion_deriv_degree` (e.g. degree 2 = 12 columns, degree 4 = 24 columns)

Motion regressor files can contain more columns than needed; the pipeline validates
and truncates to the requested number.

### Tissue Signal Derivatives

All three analysis types support `use_tissue_derivs`:
- `false` (default): tissue signals (CSF, WM, and optionally GS for rest_conn) are used as-is
- `true`: first temporal derivatives of tissue signals are computed on-the-fly and included as additional nuisance regressors

Derivatives are only computed for tissue paths that are actually provided (not null).
For rest_conn with GS enabled, GS derivatives are also included.

### HRF Models (`task_act`, `task_conn`)

`GAM`, `BLOCK`, `dmBLOCK`, `SPMG1`, or custom AFNI model strings.

### Connectivity Options (`task_conn`, `rest_conn`)

- `seed_to_voxel` or `parcellated`
- Optional partial correlation (`pcorr`) and Fisher Z-transform (`fishZ`)

## Output Files

Each analysis writes outputs to `out_dir` using the `out_file_pre` prefix. All analysis types produce a QC summary JSON (`{out_file_pre}_qc_summary.json`).

### Task Activation (`task_act`)

- `{out_file_pre}_concat_bucket_stats.nii.gz` — AFNI bucket dataset with condition betas, t-stats, and contrasts
- `{out_file_pre}_concat_censor.1D` — generated censor file
- `{out_file_pre}_concat_motion_prepared.txt` — validated/truncated motion regressors
- `{out_file_pre}_concat_{label}_onsets.txt` — per-condition AFNI onset files
- Extracted parcel stats (if `extraction` enabled): CSV files with parcel-level statistics

### Task Connectivity (`task_conn`)

- `{out_file_pre}_concat_bseries_{condition}.nii.gz` — per-condition trial-level beta series from 3dLSS
- `{out_file_pre}_concat_censor.1D` — generated censor file
- `{out_file_pre}_concat_motion_prepared.txt` — validated/truncated motion regressors
- Extracted parcel beta series (if `extraction` enabled): CSV files
- Connectivity matrices (if `connectivity` enabled): tab-delimited text files (correlation/partial correlation, optionally Fisher Z-transformed)

### Resting-State Connectivity (`rest_conn`)

- `{out_file_pre}_run{N}_residual_dtseries.nii.gz` — per-run residual time series from 3dTproject
- `{out_file_pre}_concat_residual_dtseries.nii.gz` — concatenated residual time series
- `{out_file_pre}_run{N}_censor.1D` — per-run generated censor files
- `{out_file_pre}_run{N}_motion_prepared.txt` — per-run validated/truncated motion regressors
- Extracted parcel time series (if `extraction` enabled): CSV files
- Connectivity matrices (if `connectivity` enabled): tab-delimited text files

## Quality Control

The pipeline performs automatic QC checks at runtime and records results in the QC summary JSON.

- **Censor warnings:** A warning is logged when >= 30% of TRs are censored. A strong warning is logged when >= 50% of TRs are censored.
- **Trial survival** (`task_act`, `task_conn`): Conditions with fewer than 2 surviving (uncensored) trials cause an error and abort. Conditions with exactly 2 surviving trials produce a warning.
- **Degrees of freedom (DOF) pre-flight check:** Before running the main AFNI regression, the pipeline estimates DOF (uncensored TRs minus number of regressors). If DOF < 1, the analysis aborts with an error to prevent AFNI from producing invalid output.

## Programmatic Usage

### Config-driven (recommended)

```python
from fmri_first_level_proc import load_and_validate, setup_logging, DISPATCH

logger = setup_logging("my_analysis")
analyses = load_and_validate("my_config.yaml", logger)
# analyses = [(atype, namespace, name), ...]

for atype, ns, name in analyses:
    logger.info("Running %s (%s)", name, atype)
    run_fn = DISPATCH[atype]
    run_fn(ns, logger)
```

### Direct function calls

```python
import argparse
from fmri_first_level_proc import run_task_act, setup_logging

logger = setup_logging("task_act")
args = argparse.Namespace(
    scan_path="/path/to/scan.nii.gz",
    task_timing_path="/path/to/timing.csv",
    motion_path="/path/to/motion.txt",
    cond_labels=["stimA", "stimB"],
    out_dir="/path/to/output",
    out_file_pre="subj001_task",
    num_cores=4,
    tr=0.8,
    fd_threshold=0.9,
    censor_prev_tr=False,
    hrf_model="GAM",
    custom_hrf=None,
    CSF_path=None,
    WM_path=None,
    include_motion_derivs=False,
    use_tissue_derivs=False,
    remove_previous=False,
    contrast_functions=None,
    contrast_labels=None,
    extract=False,
    extract_labels=None,
    extract_stat=None,
    extract_out_file_pre=None,
    extract_resids=False,
    template_path=None,
    average_type="mean",
    force_diff_atlas=False,
)
run_task_act(args, logger)
```

Each pipeline script also has a CLI entrypoint for standalone use
(e.g. `python -m fmri_first_level_proc.task_act_first_level --help`).
