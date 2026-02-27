# Input Specification

Complete reference for all inputs accepted by `fmri_first_level_proc`. This document covers the YAML config structure, every parameter and its allowed values, and the exact format requirements for each input file.

---

## Table of Contents

- [YAML Config Structure](#yaml-config-structure)
- [Global Settings](#global-settings)
- [Analysis Block: Common Fields](#analysis-block-common-fields)
- [Analysis Block: task\_act](#analysis-block-task_act)
- [Analysis Block: task\_conn](#analysis-block-task_conn)
- [Analysis Block: rest\_conn](#analysis-block-rest_conn)
- [Input File Formats](#input-file-formats)
  - [Scan File](#scan-file)
  - [Task Timing CSV](#task-timing-csv)
  - [Motion Regressor File](#motion-regressor-file)
  - [Censor File (Generated Internally)](#censor-file-generated-internally)
  - [Tissue Regressor Files (CSF, WM, GS)](#tissue-regressor-files-csf-wm-gs)
  - [ROI Template File](#roi-template-file)
- [Output Files](#output-files)
- [Quality Control](#quality-control)
- [CLI Flags](#cli-flags)

---

## YAML Config Structure

The config file has two top-level keys:

```yaml
global:
  # Global settings (applied uniformly to all blocks)

analyses:
  - # Analysis block 1
  - # Analysis block 2
  # ...
```

- `global` (dict, optional): settings that apply to every analysis block. Cannot be overridden per-block.
- `analyses` (list, required): one or more analysis block dicts. Blocks execute in order (top to bottom).

---

## Global Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `num_cores` | int | `1` | CPU cores for AFNI parallel processing. |
| `tr` | float | (required) | Repetition time in seconds. Applies to all scans across all analysis blocks. |
| `template_path` | str or null | `null` | Path to a shared ROI template NIfTI file (`.nii.gz`). Required if any block uses extraction or connectivity. See [ROI Template File](#roi-template-file). |
| `force_diff_atlas` | bool | `false` | Allow atlas space/orientation mismatch between scan and template. Use with caution. |

These four keys are **global-only**. If they appear inside an analysis block, the block value is ignored and a warning is logged.

---

## Analysis Block: Common Fields

Every analysis block must contain these fields:

| Key | Type | Required | Description |
|---|---|---|---|
| `name` | str | No | Human-readable label for the block. Defaults to `analyses[<index>]`. |
| `type` | str | Yes | One of: `task_act`, `task_conn`, `rest_conn`. |
| `paths` | dict | Yes | Input file paths. Required sub-keys vary by type (see below). |
| `out_dir` | str | Yes | Output directory. Created automatically if it does not exist. |
| `out_file_pre` | str | Yes | Output filename prefix for all generated files. |
| `remove_previous` | bool | No | `true` = delete all files in `out_dir` before running. Default: `false`. |
| `average_type` | str | Yes | `"mean"` or `"median"`. Controls parcel-level averaging during extraction. |
| `fd_threshold` | float | Yes | Framewise displacement threshold in mm for motion censoring. TRs with FD above this value are censored. |
| `censor_prev_tr` | bool | No | `true` = also censor the TR immediately before each high-motion TR. Default: `false`. |

---

## Analysis Block: `task_act`

First-level GLM via AFNI `3dDeconvolve`.

### Required Fields

| Key | Type | Description |
|---|---|---|
| `cond_labels` | list of str | Condition labels. Must match entries in the `CONDITION` column of the timing CSV. |
| `hrf_model` | str | HRF model. One of: `GAM`, `BLOCK`, `dmBLOCK`, `SPMG1`, `custom`. |
| `include_motion_derivs` | bool | `false` = 6 motion columns. `true` = 12 motion columns (6 base + 6 temporal derivatives). |

### Required Paths

| Key | Type | Description |
|---|---|---|
| `paths.scan_path` | str | Path to preprocessed, run-concatenated NIfTI scan. See [Scan File](#scan-file). |
| `paths.task_timing_path` | str | Path to task timing CSV. See [Task Timing CSV](#task-timing-csv). |
| `paths.motion_path` | str | Path to motion regressor file. See [Motion Regressor File](#motion-regressor-file). |

### Optional Paths

| Key | Type | Default | Description |
|---|---|---|---|
| `paths.CSF_path` | str or null | `null` | Mean CSF BOLD signal per TR. See [Tissue Regressor Files](#tissue-regressor-files-csf-wm-gs). |
| `paths.WM_path` | str or null | `null` | Mean WM BOLD signal per TR. See [Tissue Regressor Files](#tissue-regressor-files-csf-wm-gs). |

### Optional Fields

| Key | Type | Default | Description |
|---|---|---|---|
| `custom_hrf` | str or null | `null` | Custom AFNI HRF model string (e.g. `"TENT(0,14,8)"`). Required when `hrf_model: custom`. |
| `use_tissue_derivs` | bool | `false` | `true` = include first temporal derivatives of tissue signals (CSF, WM) as additional nuisance regressors. Only computed for tissue paths that are not null. |
| `contrasts` | dict or null | `null` | Linear contrast specification. See below. |
| `extraction` | dict or null | `null` | Parcel-level stat extraction. See below. |

### Contrasts

```yaml
contrasts:
  functions:          # list of str or null; null = contrasts disabled
    - "1*stimFear-1*stimNeu"
    - "1*stimSad-1*stimNeu"
  labels:             # list of str; must be same length as functions
    - "Fear-Neutral"
    - "Sad-Neutral"
```

- `functions`: list of AFNI-style contrast equations using condition labels. Set to `null` to disable.
- `labels`: human-readable names for each contrast. Same length as `functions`.

### Extraction

```yaml
extraction:
  extract: true                    # bool; true = enable, false = skip
  extract_labels:                  # list of str; condition and/or contrast labels to extract
    - "stimFear"
    - "Fear-Neutral"
  extract_stat: "z"                # str; one of: "z", "t", "r"
  extract_out_file_pre: "subj001"  # str; output filename prefix for extracted values
  extract_resids: false            # bool; true = also extract residuals
```

When `extract: true`, all sub-keys (`extract_labels`, `extract_stat`, `extract_out_file_pre`) are required. Requires `template_path` in global settings.

---

## Analysis Block: `task_conn`

Beta series estimation via AFNI `3dDeconvolve` + `3dLSS`.

### Required Fields

| Key | Type | Description |
|---|---|---|
| `cond_beta_labels` | list of str | Conditions for trial-level beta series estimation. Must match entries in the `CONDITION` column of the timing CSV. |
| `hrf_model` | str | HRF model. One of: `GAM`, `BLOCK`, `dmBLOCK`, `SPMG1`, `custom`. |
| `include_motion_derivs` | bool | `false` = 6 motion columns. `true` = 12 motion columns. |

### Required Paths

Same as `task_act`: `scan_path`, `task_timing_path`, `motion_path`.

### Optional Paths

Same as `task_act`: `CSF_path`, `WM_path`.

### Optional Fields

| Key | Type | Default | Description |
|---|---|---|---|
| `custom_hrf` | str or null | `null` | Custom AFNI HRF model string. Required when `hrf_model: custom`. |
| `use_tissue_derivs` | bool | `false` | `true` = include first temporal derivatives of tissue signals (CSF, WM) as additional nuisance regressors. Only computed for tissue paths that are not null. |
| `extraction` | dict or null | `null` | Parcel beta series extraction. See below. |
| `connectivity` | dict or null | `null` | Functional connectivity computation. See below. |

### Extraction

```yaml
extraction:
  extract_pbseries: true           # bool; true = extract parcel beta series
  extract_out_file_pre: "subj001"  # str; required when extract_pbseries is true
```

Requires `template_path` in global settings.

### Connectivity

```yaml
connectivity:
  calc_conn: "parcellated"         # str or null; null = disabled; "seed_to_voxel" or "parcellated"
  conn_out_file_pre: "subj001"     # str; output filename prefix
  pcorr: false                     # bool; true = partial correlation
  fishZ: true                      # bool; true = Fisher Z-transform
```

Requires `template_path` in global settings. For `seed_to_voxel`, the template must be a binary mask (single ROI). For `parcellated`, the template must contain >= 2 integer-labeled ROIs.

---

## Analysis Block: `rest_conn`

Residual time series via AFNI `3dTproject` with bandpass filtering. Processes each run individually, then concatenates.

### Required Fields

| Key | Type | Description |
|---|---|---|
| `bandpass` | list of 2 floats | Bandpass filter bounds in Hz: `[low, high]`. Must satisfy `0 <= low < high`. |
| `motion_deriv_degree` | int (>= 1) | Number of derivative orders. Total motion columns = `6 * motion_deriv_degree`. E.g. `2` = 12 columns (6 base + 6 first derivatives). |

### Required Paths

All path keys are **lists** with one entry per run. All lists must be the **same length**.

| Key | Type | Description |
|---|---|---|
| `paths.scan_paths` | list of str | Preprocessed NIfTI scan per run. See [Scan File](#scan-file). |
| `paths.motion_paths` | list of str | Motion regressor file per run. See [Motion Regressor File](#motion-regressor-file). |
| `paths.CSF_paths` | list of str | Mean CSF BOLD signal per TR, per run. See [Tissue Regressor Files](#tissue-regressor-files-csf-wm-gs). |
| `paths.WM_paths` | list of str | Mean WM BOLD signal per TR, per run. See [Tissue Regressor Files](#tissue-regressor-files-csf-wm-gs). |

### Optional Paths

| Key | Type | Default | Description |
|---|---|---|---|
| `paths.GS_paths` | list of str or null | `null` | Mean global BOLD signal per TR, per run. Enables global signal regression. Must be same length as other path lists. |

### Optional Fields

| Key | Type | Default | Description |
|---|---|---|---|
| `notch_filter_band` | list of 2 floats or null | `null` | Respiration artifact notch filter Hz band: `[low, high]`. Must satisfy `0 < low < high`. Applied to motion regressors before censor file generation. Set to `null` to disable. |
| `keep_run_res_dtseries` | bool | `true` | `true` = retain per-run residual dtseries files. `false` = delete after concatenation. |
| `use_tissue_derivs` | bool | `false` | `true` = include first temporal derivatives of tissue signals (CSF, WM, and GS when enabled) as additional nuisance regressors. |
| `extraction` | dict or null | `null` | Parcel time series extraction. See below. |
| `connectivity` | dict or null | `null` | Functional connectivity computation. See below. |

### Extraction

```yaml
extraction:
  extract_ptseries: true           # bool; true = extract parcel time series
  extract_out_file_pre: "subj001"  # str; required when extract_ptseries is true
```

Requires `template_path` in global settings.

### Connectivity

Same structure as `task_conn`:

```yaml
connectivity:
  calc_conn: "parcellated"         # str or null; null = disabled; "seed_to_voxel" or "parcellated"
  conn_out_file_pre: "subj001"     # str; output filename prefix
  pcorr: false                     # bool; true = partial correlation
  fishZ: true                      # bool; true = Fisher Z-transform
```

---

## Input File Formats

### Scan File

- **Format:** NIfTI (`.nii` or `.nii.gz`)
- **Content:** Preprocessed BOLD fMRI data (4D: x, y, z, time)
- **Space:** Must be in a standard atlas space (e.g. MNI). AFNI `3dinfo` is used to check space, orientation, and grid spacing against the ROI template.
- **Task analyses (`task_act`, `task_conn`):** A single file containing all runs concatenated along the time dimension.
- **Resting-state (`rest_conn`):** One file per run (not concatenated; the pipeline concatenates residuals internally).

### Task Timing CSV

- **Format:** CSV (comma-delimited) with a header row
- **Reader:** `pandas.read_csv(path, sep=',')`
- **Required columns (exact names):**

| Column | Type | Description |
|---|---|---|
| `CONDITION` | str | Trial condition label. Must match entries in `cond_labels` (task_act) or `cond_beta_labels` (task_conn). |
| `ONSET` | float | Trial onset time in seconds, relative to the start of the concatenated scan. |
| `DURATION` | float | Trial duration in seconds. |

- **Optional column:**

| Column | Type | Description |
|---|---|---|
| `AMP` | float | Amplitude modulator. Used by `dmBLOCK` and amplitude-modulated HRF models. |

- **Row ordering:** Rows are sorted by `ONSET` internally; input order does not matter.
- **Condition coverage:** Every label in `cond_labels`/`cond_beta_labels` must appear at least once. A warning is logged for conditions with fewer than 3 trials.
- **Extra conditions:** The timing file may contain conditions not listed in `cond_labels`/`cond_beta_labels`; they are ignored.

Example:

```csv
CONDITION,ONSET,DURATION,AMP
stimFear,25.879,3.985,1
stimSad,5.663,3.968,1
stimNeu,66.462,3.967,1
```

### Motion Regressor File

- **Format:** Plain text, whitespace-delimited (spaces or tabs), no header row, no row names
- **Reader:** `numpy.loadtxt(path)`
- **Rows:** One row per TR (timepoint). Must match the scan length (enforced by AFNI).
- **Columns:** 6 base motion parameters (3 translation + 3 rotation), optionally followed by temporal derivatives.
  - `task_act` / `task_conn` with `include_motion_derivs: false` -> minimum 6 columns used
  - `task_act` / `task_conn` with `include_motion_derivs: true` -> minimum 12 columns used
  - `rest_conn` with `motion_deriv_degree: N` -> minimum `6 * N` columns used
- **Truncation:** If the file contains more columns than needed, extra columns are silently truncated to the required count. A log message is emitted when this occurs.
- **Prepared file:** The pipeline writes a validated, truncated copy to `out_dir` as `{out_file_pre}_concat_motion_prepared.txt` (task) or `{out_file_pre}_run{N}_motion_prepared.txt` (rest_conn). The prepared file is tab-delimited with 8 decimal places.

Example (6-column):

```
0.0123    0.0456    0.0789    -0.0012    0.0034    -0.0056
0.0130    0.0460    0.0792    -0.0010    0.0032    -0.0054
...
```

### Censor File (Generated Internally)

Censor files are **not** user-provided inputs. They are generated automatically by the pipeline from the motion regressor file using `fd_threshold` and `censor_prev_tr`. The pipeline computes framewise displacement (FD) via AFNI's `1d_tool.py` and writes a binary censor file to `out_dir`.

- **Naming:** `{out_file_pre}_concat_censor.1D` (task_act, task_conn) or `{out_file_pre}_run{N}_censor.1D` (rest_conn, one per run).
- **Format:** Plain text, single column of binary values (`1` = include, `0` = censor). One row per TR.
- **Usage:** Passed directly to AFNI (`3dDeconvolve -censor` or `3dTproject -censor`).

### Tissue Regressor Files (CSF, WM, GS)

Applies to: `CSF_path`/`CSF_paths`, `WM_path`/`WM_paths`, `GS_paths`

- **Format:** Plain text, no header row, no row names
- **Rows:** One row per TR (timepoint). Must match the scan length.
- **Columns:** Single column containing the mean BOLD signal for that tissue compartment at each TR.
- **Reader:** Passed directly to AFNI (`3dDeconvolve -ortvec` or `3dTproject -ort`). When `use_tissue_derivs: true`, files are also read by Python (`numpy.loadtxt`) to compute first temporal derivatives.
- **Derivative files:** When `use_tissue_derivs: true`, derivative files are written to `out_dir` as `{out_file_pre}_{tissue}_deriv.txt` (task) or `{out_file_pre}_run{N}_{tissue}_deriv.txt` (rest_conn). Derivatives use backward differencing with zero-padding to preserve length.
- **Required vs. optional:**
  - `task_act`, `task_conn`: CSF and WM are optional (`null` to skip)
  - `rest_conn`: CSF and WM are **required**; GS is optional (`null` to skip global signal regression)

Example:

```
1023.456
1025.789
1022.123
...
```

### ROI Template File

- **Format:** NIfTI (`.nii` or `.nii.gz`)
- **Content:** 3D volume with integer-valued ROI labels
- **Space:** Should match the scan's atlas space and orientation. If there is a grid mismatch, the pipeline will resample the template via `3dresample` to match the scan grid. A space/orientation mismatch causes an error unless `force_diff_atlas: true`.
- **Label constraints:**
  - Must contain only integer values (no floating-point ROI labels)
  - Must contain at least one non-zero voxel
  - For `seed_to_voxel` connectivity: must be a binary mask (max label = 1)
  - For `parcellated` connectivity or extraction: must have >= 2 distinct ROI labels

---

## Output Files

All outputs are written to `out_dir` using the `out_file_pre` prefix. Every analysis type writes a QC summary JSON.

### QC Summary JSON

**File:** `{out_file_pre}_qc_summary.json`

Schema varies by analysis type:

**All types:**
| Key | Type | Description |
|---|---|---|
| `analysis_type` | str | `"task_act"`, `"task_conn"`, or `"rest_conn"` |
| `fd_threshold` | float | FD threshold used for censoring |
| `censor_prev_tr` | bool | Whether previous-TR censoring was enabled |
| `n_trs_total` | int | Total number of TRs |
| `n_trs_censored` | int | Number of censored TRs |
| `pct_censored` | float | Percentage of TRs censored |

**task_act and task_conn additionally include:**
| Key | Type | Description |
|---|---|---|
| `hrf_model` | str | HRF model used (or `"custom (MODEL)"` if custom) |
| `dof` | int or null | Estimated degrees of freedom |
| `per_condition_trial_counts` | dict | Total trial count per condition |
| `per_condition_surviving_trials` | dict | Trials surviving censoring per condition |

**rest_conn additionally includes:**
| Key | Type | Description |
|---|---|---|
| `per_run_dof` | list of int | Degrees of freedom per run |
| `per_run_censoring` | list of dict | Per-run stats: `n_trs_total`, `n_trs_censored`, `pct_censored` |

### Task Activation (`task_act`)

- `{out_file_pre}_concat_bucket_stats.nii.gz` — AFNI bucket dataset with condition betas, t-stats, and contrasts
- `{out_file_pre}_concat_censor.1D` — generated censor file
- `{out_file_pre}_concat_motion_prepared.txt` — validated/truncated motion regressors
- `{out_file_pre}_concat_{label}_onsets.txt` — per-condition AFNI onset files
- `{out_file_pre}_qc_summary.json` — QC summary
- Extracted parcel stats (if `extraction` enabled): CSV files with parcel-level statistics

### Task Connectivity (`task_conn`)

- `{out_file_pre}_concat_bseries_{condition}.nii.gz` — per-condition trial-level beta series from 3dLSS
- `{out_file_pre}_concat_censor.1D` — generated censor file
- `{out_file_pre}_concat_motion_prepared.txt` — validated/truncated motion regressors
- `{out_file_pre}_qc_summary.json` — QC summary
- Extracted parcel beta series (if `extraction` enabled): CSV files
- Connectivity matrices (if `connectivity` enabled): tab-delimited text files

### Resting-State Connectivity (`rest_conn`)

- `{out_file_pre}_run{N}_residual_dtseries.nii.gz` — per-run residual time series from 3dTproject
- `{out_file_pre}_concat_residual_dtseries.nii.gz` — concatenated residual time series
- `{out_file_pre}_run{N}_censor.1D` — per-run generated censor files
- `{out_file_pre}_run{N}_motion_prepared.txt` — per-run validated/truncated motion regressors
- `{out_file_pre}_qc_summary.json` — QC summary
- Extracted parcel time series (if `extraction` enabled): CSV files
- Connectivity matrices (if `connectivity` enabled): tab-delimited text files

---

## Quality Control

The pipeline performs automatic QC checks during execution. All checks are logged and key metrics are recorded in the QC summary JSON.

### Censor Warnings

- **>= 30% TRs censored:** A warning is logged indicating substantial motion artifact.
- **>= 50% TRs censored:** A strong warning is logged indicating the data may be unreliable.

### Trial Survival (`task_act`, `task_conn`)

After censoring, the pipeline checks how many trials per condition still have uncensored onsets:

- **< 2 surviving trials:** The analysis aborts with an error. The condition cannot be reliably estimated.
- **Exactly 2 surviving trials:** A warning is logged. The estimate will have very low power.

### Degrees of Freedom Pre-Flight Check

Before running the main AFNI regression, the pipeline computes estimated degrees of freedom:

`DOF = (uncensored TRs) - (number of regressors)`

- **DOF < 1:** The analysis aborts with an error. This prevents AFNI from producing statistically invalid output. Consider relaxing the FD threshold or reducing model complexity.

---

## CLI Flags

```
run-first-level --config <path> [options]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | str | (required) | Path to the YAML config file. |
| `--dry-run` | flag | off | Validate the config and print the analysis plan without executing any analyses. |
| `--analyses` | list of int | all | Run only specific analysis block indices (0-based). E.g. `--analyses 0 2` runs the first and third blocks. |
| `--log-file` | str | none | Path to a log file. When set, logs are written to both console and file. |
