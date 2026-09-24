"""
rsoxr_harness
-------------
Declarative, project-directory based front end to the Refltools RSoXR
pipeline, designed to be driven from plain-text requests (see
.claude/skills/rsoxr-fit/SKILL.md).

    from rsoxr_harness import FitProject, LayerSpec, InstrumentSpec, pm_offset
"""

import os as _os
import sys as _sys

_REFLTOOLS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REFLTOOLS not in _sys.path:
    _sys.path.insert(0, _REFLTOOLS)

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_ENABLE_X64", "1")

from .recipe import (LayerSpec, InstrumentSpec, ModelRecipe,  # noqa: E402
                     derive, diff, layer_table, validate, pm, pm_offset)
from .materials import MaterialSource  # noqa: E402
from .project import FitProject, ProjectError, free_parameter_table  # noqa: E402

__all__ = ["FitProject", "ProjectError", "LayerSpec", "InstrumentSpec",
           "ModelRecipe", "MaterialSource", "derive", "diff", "layer_table",
           "validate", "pm", "pm_offset", "free_parameter_table"]
