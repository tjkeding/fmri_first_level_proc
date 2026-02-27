#!/usr/bin/env python3

# ============================================================================
# YAML CONFIG RUNNER FOR fMRI FIRST-LEVEL PROCESSING
# Reads a YAML config, validates it, and dispatches each analysis block
# to the appropriate pipeline script's run() function.
#
# Usage:
#   python run_first_level.py --config my_study.yaml
#   python run_first_level.py --config my_study.yaml --dry-run
#   python run_first_level.py --config my_study.yaml --analyses 0 2
#   python run_first_level.py --config my_study.yaml --log-file run.log
#
# Author: Taylor J. Keding, Ph.D.
# Version: 2.0
# Last updated: 02/17/26
# ============================================================================

import sys
import time
import shutil
import argparse

from .first_level_utils import setup_logging
from .first_level_config import load_and_validate

from .task_act_first_level import run as run_task_act
from .task_conn_first_level import run as run_task_conn
from .rest_conn_first_level import run as run_rest_conn

# Dispatch table
DISPATCH = {
    "task_act": run_task_act,
    "task_conn": run_task_conn,
    "rest_conn": run_rest_conn,
}

def main():
    parser = argparse.ArgumentParser(
        description="Run fMRI first-level analyses from a YAML config file."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate the config and print the analysis plan without executing."
    )
    parser.add_argument(
        "--analyses", type=int, nargs="+", default=None,
        help="Run only specific analysis block indices (0-based). Default: run all."
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Optional path to a log file (useful for parallel runs)."
    )

    args = parser.parse_args()
    logger = setup_logging("run_first_level", log_file=args.log_file)

    # Check that AFNI is available on PATH
    if shutil.which("3dinfo") is None:
        logger.error("AFNI not found on PATH. Install AFNI and ensure it is on your PATH before running.")
        sys.exit(1)

    # Load and validate config
    logger.info("Loading config from %s", args.config)
    analysis_list = load_and_validate(args.config, logger)
    logger.info("Config validated: %d analysis block(s) found.", len(analysis_list))

    # Filter to requested indices
    if args.analyses is not None:
        for idx in args.analyses:
            if idx < 0 or idx >= len(analysis_list):
                logger.error("--analyses index %d is out of range (0-%d).", idx, len(analysis_list) - 1)
                sys.exit(1)
        analysis_list = [(atype, ns, name) for i, (atype, ns, name) in enumerate(analysis_list) if i in args.analyses]
        logger.info("Running %d selected analysis block(s).", len(analysis_list))

    # Print analysis plan
    logger.info("=" * 60)
    logger.info("ANALYSIS PLAN")
    logger.info("=" * 60)
    for i, (atype, ns, name) in enumerate(analysis_list):
        logger.info("  [%d] %s (type: %s)", i, name, atype)
        logger.info("       out_dir: %s", ns.out_dir)
        logger.info("       out_file_pre: %s", ns.out_file_pre)
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("Dry run complete. No analyses were executed.")
        return

    # Execute each analysis block
    start_time = time.time()
    for i, (atype, ns, name) in enumerate(analysis_list):
        logger.info("-" * 60)
        logger.info("Starting analysis [%d]: %s (type: %s)", i, name, atype)
        logger.info("-" * 60)

        run_fn = DISPATCH[atype]
        block_start = time.time()
        try:
            run_fn(ns, logger)
        except SystemExit as e:
            if e.code != 0:
                logger.error("Analysis [%d] '%s' failed. Stopping.", i, name)
                sys.exit(1)
        except Exception as e:
            logger.error("Analysis [%d] '%s' raised an unexpected error: %s", i, name, e)
            sys.exit(1)

        logger.info("Analysis [%d] '%s' completed in %.2f seconds.", i, name, time.time() - block_start)

    logger.info("=" * 60)
    logger.info("All analyses completed successfully in %.2f seconds.", time.time() - start_time)

if __name__ == "__main__":
    main()
