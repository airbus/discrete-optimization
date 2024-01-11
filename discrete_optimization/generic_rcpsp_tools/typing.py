#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Union

from discrete_optimization.generic_tools.do_problem import Problem, Solution
from discrete_optimization.rcpsp import (
    RCPSPModel,
    RCPSPModelPreemptive,
    RCPSPModelSpecialConstraintsPreemptive,
)
from discrete_optimization.rcpsp.rcpsp_model_preemptive import RCPSPSolutionPreemptive
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from discrete_optimization.rcpsp.solver import CP_MRCPSP_MZN, CP_RCPSP_MZN
from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_MRCPSP_MZN_PREEMPTIVE,
    CP_RCPSP_MZN_PREEMPTIVE,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPModel,
    MS_RCPSPModel_Variant,
    MS_RCPSPSolution,
    MS_RCPSPSolution_Preemptive,
    MS_RCPSPSolution_Preemptive_Variant,
    MS_RCPSPSolution_Variant,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import (
    CP_MS_MRCPSP_MZN,
    CP_MS_MRCPSP_MZN_PARTIAL_PREEMPTIVE,
    CP_MS_MRCPSP_MZN_PREEMPTIVE,
)

ANY_CLASSICAL_RCPSP = Union[
    RCPSPModel,
    RCPSPModelPreemptive,
    RCPSPModelSpecialConstraintsPreemptive,
]
ANY_RCPSP = Union[
    RCPSPModel,
    RCPSPModelPreemptive,
    RCPSPModelSpecialConstraintsPreemptive,
    MS_RCPSPModel,
    MS_RCPSPModel_Variant,
]
ANY_MSRCPSP = Union[MS_RCPSPModel_Variant, MS_RCPSPModel]
ANY_SOLUTION = Union[
    RCPSPSolutionPreemptive,
    RCPSPSolution,
    MS_RCPSPSolution,
    MS_RCPSPSolution_Variant,
    MS_RCPSPSolution_Preemptive,
    MS_RCPSPSolution_Preemptive_Variant,
]
ANY_SOLUTION_UNPREEMPTIVE = Union[
    RCPSPSolution, MS_RCPSPSolution, MS_RCPSPSolution_Variant
]
ANY_SOLUTION_PREEMPTIVE = Union[RCPSPSolutionPreemptive, MS_RCPSPSolution_Preemptive]
ANY_SOLUTION_CLASSICAL_RCPSP = Union[RCPSPSolution, RCPSPSolutionPreemptive]
ANY_SOLUTION_MSRCPSP = Union[
    MS_RCPSPSolution,
    MS_RCPSPSolution_Variant,
    MS_RCPSPSolution_Preemptive,
    MS_RCPSPSolution_Preemptive_Variant,
]
ANY_CP_SOLVER = Union[
    CP_RCPSP_MZN_PREEMPTIVE,
    CP_RCPSP_MZN,
    CP_MRCPSP_MZN,
    CP_MRCPSP_MZN_PREEMPTIVE,
    CP_MS_MRCPSP_MZN,
    CP_MS_MRCPSP_MZN_PREEMPTIVE,
    CP_MS_MRCPSP_MZN_PARTIAL_PREEMPTIVE,
]


def is_instance_any_rcpsp_problem(problem: Problem) -> bool:
    return isinstance(
        problem,
        (
            RCPSPModel,
            RCPSPModelPreemptive,
            RCPSPModelSpecialConstraintsPreemptive,
            MS_RCPSPModel,
            MS_RCPSPModel_Variant,
        ),
    )


def is_instance_any_rcpsp_solution(solution: Solution) -> bool:
    return isinstance(
        solution,
        (
            RCPSPSolutionPreemptive,
            RCPSPSolution,
            MS_RCPSPSolution,
            MS_RCPSPSolution_Variant,
            MS_RCPSPSolution_Preemptive,
            MS_RCPSPSolution_Preemptive_Variant,
        ),
    )
