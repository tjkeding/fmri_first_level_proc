# fmri_first_level_proc

A YAML-driven Python framework for first-level fMRI analysis (Task Activation, Task Connectivity, and Resting-State Connectivity) built on AFNI.

## Key Features

- **Three Analysis Pipelines:**
  - **Task Activation (`task_act`):** GLM via `3dDeconvolve`. Supports multiple HRF models, linear contrasts, and parcel-level stat extraction.
  - **Task Connectivity (`task_conn`):** Beta series estimation via `3dLSS`. Supports parcel beta series extraction, functional connectivity, and connectivity contrasts.
  - **Resting-State Connectivity (`rest_conn`):** Residual time series via `3dTproject` with bandpass filtering. Supports parcel time series extraction and functional connectivity.
- **Config-Driven:** Define complex analysis batches in a single YAML file.
- **Robustness:** Automated QC, motion censoring, trial survival checks, and DOF pre-flight verification.
- **Parallel Processing:** Native AFNI multi-core support.

## Prerequisites

- Python >= 3.8
- [AFNI](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/main_toc.html) (must be on your PATH)
- Dependencies: `numpy`, `pandas`, `pyyaml`

## Installation

```bash
pip install git+https://github.com/tjkeding/fmri_first_level_proc.git
```

## Quick Start

1.  **Configure:** Copy and edit `example_config.yaml` to define your analyses and paths.
2.  **Dry-Run:** Validate your configuration and view the execution plan:
    ```bash
    run-first-level --config my_config.yaml --dry-run
    ```
3.  **Execute:** Run the full pipeline:
    ```bash
    run-first-level --config my_config.yaml
    ```

## CLI Reference

| Flag | Description |
|---|---|
| `--config` | Path to YAML config file (required) |
| `--dry-run` | Validate config and print plan without executing |
| `--analyses` | Run only specific block indices (0-based), e.g. `--analyses 0 2` |
| `--log-file` | Write logs to file in addition to console |

## Detailed Documentation

For exhaustive details on YAML parameters, input file formats, output file naming, and QC rules, see:
- [**INPUT_SPECIFICATION.md**](./INPUT_SPECIFICATION.md)
- [**example_config.yaml**](./example_config.yaml)

## Programmatic Usage

The pipeline is designed to be used both as a CLI tool and a Python library.

### Config-Driven API

```python
from fmri_first_level_proc import load_and_validate, setup_logging, DISPATCH

logger = setup_logging("my_analysis")
# analyses = [(atype, namespace, name), ...]
analyses = load_and_validate("my_config.yaml", logger)

for atype, ns, name in analyses:
    run_fn = DISPATCH[atype]
    run_fn(ns, logger)
```

Each pipeline script also supports standalone CLI execution (e.g., `python -m fmri_first_level_proc.task_act_first_level --help`).