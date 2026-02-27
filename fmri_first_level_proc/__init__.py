"""
fmri_first_level_proc — First-level fMRI analysis framework.

A general-purpose Python package for first-level (within-participant)
fMRI analyses including task activation, task connectivity (beta series),
and resting-state connectivity. Built on AFNI.
"""

__version__ = "2.0.0"

# Public API
from .first_level_utils import (
    setup_logging,
    build_hrf_string,
    validate_custom_hrf,
    needs_married_timing,
    VALID_HRF_MODELS,
    HRF_DURATION_MODELS,
    HRF_DM_MODELS,
    HRF_IMPULSE_MODELS,
)
from .first_level_config import load_and_validate
from .run_first_level import DISPATCH

from .task_act_first_level import run as run_task_act
from .task_conn_first_level import run as run_task_conn
from .rest_conn_first_level import run as run_rest_conn
