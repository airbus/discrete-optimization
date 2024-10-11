#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Union

from discrete_optimization.generic_tools.do_problem import Problem, Solution
from discrete_optimization.rcpsp import (
    PreemptiveRcpspProblem,
    RcpspProblem,
    SpecialConstraintsPreemptiveRcpspProblem,
)
from discrete_optimization.rcpsp.problem_preemptive import PreemptiveRcpspSolution
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp_multiskill.problem import (
    MultiskillRcpspProblem,
    MultiskillRcpspSolution,
    PreemptiveMultiskillRcpspSolution,
    VariantMultiskillRcpspProblem,
    VariantMultiskillRcpspSolution,
    VariantPreemptiveMultiskillRcpspSolution,
)

ANY_CLASSICAL_RCPSP = Union[
    RcpspProblem,
    PreemptiveRcpspProblem,
    SpecialConstraintsPreemptiveRcpspProblem,
]
ANY_RCPSP = Union[
    RcpspProblem,
    PreemptiveRcpspProblem,
    SpecialConstraintsPreemptiveRcpspProblem,
    MultiskillRcpspProblem,
    VariantMultiskillRcpspProblem,
]
ANY_MSRCPSP = Union[VariantMultiskillRcpspProblem, MultiskillRcpspProblem]
ANY_SOLUTION = Union[
    PreemptiveRcpspSolution,
    RcpspSolution,
    MultiskillRcpspSolution,
    VariantMultiskillRcpspSolution,
    PreemptiveMultiskillRcpspSolution,
    VariantPreemptiveMultiskillRcpspSolution,
]
ANY_SOLUTION_UNPREEMPTIVE = Union[
    RcpspSolution, MultiskillRcpspSolution, VariantMultiskillRcpspSolution
]
ANY_SOLUTION_PREEMPTIVE = Union[
    PreemptiveRcpspSolution, PreemptiveMultiskillRcpspSolution
]
ANY_SOLUTION_CLASSICAL_RCPSP = Union[RcpspSolution, PreemptiveRcpspSolution]
ANY_SOLUTION_MSRCPSP = Union[
    MultiskillRcpspSolution,
    VariantMultiskillRcpspSolution,
    PreemptiveMultiskillRcpspSolution,
    VariantPreemptiveMultiskillRcpspSolution,
]


if TYPE_CHECKING:
    from discrete_optimization.rcpsp.solvers.cp_mzn import (
        CpMultimodePreemptiveRcpspSolver,
        CpMultimodeRcpspSolver,
        CpPreemptiveRcpspSolver,
        CpRcpspSolver,
    )
    from discrete_optimization.rcpsp_multiskill.solvers.cp_mzn import (
        CpMultiskillRcpspSolver,
        CpPartialPreemptiveMultiskillRcpspSolver,
        CpPreemptiveMultiskillRcpspSolver,
    )

    ANY_CP_SOLVER = Union[
        CpPreemptiveRcpspSolver,
        CpRcpspSolver,
        CpMultimodeRcpspSolver,
        CpMultimodePreemptiveRcpspSolver,
        CpMultiskillRcpspSolver,
        CpPreemptiveMultiskillRcpspSolver,
        CpPartialPreemptiveMultiskillRcpspSolver,
    ]
else:
    ANY_CP_SOLVER = Union[
        "CpPreemptiveRcpspSolver",
        "CpRcpspSolver",
        "CpMultimodeRcpspSolver",
        "CpMultimodePreemptiveRcpspSolver",
        "CpMultiskillRcpspSolver",
        "CpPreemptiveMultiskillRcpspSolver",
        "CpPartialPreemptiveMultiskillRcpspSolver",
    ]


def is_instance_any_rcpsp_problem(problem: Problem) -> bool:
    return isinstance(
        problem,
        (
            RcpspProblem,
            PreemptiveRcpspProblem,
            SpecialConstraintsPreemptiveRcpspProblem,
            MultiskillRcpspProblem,
            VariantMultiskillRcpspProblem,
        ),
    )


def is_instance_any_rcpsp_solution(solution: Solution) -> bool:
    return isinstance(
        solution,
        (
            PreemptiveRcpspSolution,
            RcpspSolution,
            MultiskillRcpspSolution,
            VariantMultiskillRcpspSolution,
            PreemptiveMultiskillRcpspSolution,
            VariantPreemptiveMultiskillRcpspSolution,
        ),
    )
