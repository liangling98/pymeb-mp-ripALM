"""Compatibility exports for the core pymeb building blocks.

The implementation is split by role:

``problem.py`` stores MEB data and primal-dual state.
``cone.py`` contains second-order cone projection utilities.
``subproblem.py`` evaluates ripALM subproblems and stopping checks.
``newton.py`` contains the semismooth Newton direction and line search.
"""

from ._types import ScalarLike, TensorLike
from .cone import SOCProjection, project_soc, soc_jvp
from .newton import (
    LineSearchResult,
    NewtonDirection,
    line_search_newton_step,
    semismooth_newton_direction,
)
from .mixed_precision import (
    MixedPrecisionOptions,
    MixedPrecisionPhaseTimings,
    MixedPrecisionRefinementInfo,
    MixedPrecisionResult,
    mixed_precision_ripalm,
    screen_by_boundary_ratio,
)
from .problem import MEBProblem, MEBState
from .subproblem import (
    RelativeInexactCheck,
    RipALMSubproblemCandidate,
    check_relative_inexact_candidate,
    check_relative_inexact_condition,
    evaluate_subproblem_candidate,
    subproblem_hessian_matrix,
)

__all__ = [
    "ScalarLike",
    "TensorLike",
    "SOCProjection",
    "project_soc",
    "soc_jvp",
    "LineSearchResult",
    "NewtonDirection",
    "line_search_newton_step",
    "semismooth_newton_direction",
    "MixedPrecisionOptions",
    "MixedPrecisionPhaseTimings",
    "MixedPrecisionRefinementInfo",
    "MixedPrecisionResult",
    "mixed_precision_ripalm",
    "screen_by_boundary_ratio",
    "MEBProblem",
    "MEBState",
    "RelativeInexactCheck",
    "RipALMSubproblemCandidate",
    "check_relative_inexact_candidate",
    "check_relative_inexact_condition",
    "evaluate_subproblem_candidate",
    "subproblem_hessian_matrix",
]
