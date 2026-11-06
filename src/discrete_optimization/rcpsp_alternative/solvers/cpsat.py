#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tasks_tools.allocation import (
    NoUnaryResource,
)
from discrete_optimization.generic_tasks_tools.exclusions import NoExclusionResource
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.skill import (
    NonSkillCumulativeResource,
    NoSkill,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
)
from discrete_optimization.rcpsp.problem import Task
from discrete_optimization.rcpsp.transformations.generic_scheduling_impl import (
    transform_solution_from_raw_generic_to_rcpsp,
)
from discrete_optimization.rcpsp_alternative.problem import (
    RcpspWithAlternativePath,
)


class CpsatRcpspWithAlternativePathSolver(
    GenericSchedulingAutoCpSatSolver[
        Task,
        NoUnaryResource,
        NoSkill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        NoExclusionResource,
    ]
):
    problem: RcpspWithAlternativePath
    additional_variables: dict

    def convert_task_variables_to_solution(
        self, raw_sol: RawSolution[Task, NoUnaryResource, NoSkill]
    ) -> GenericSchedulingSolution[
        Task,
        NoUnaryResource,
        NoSkill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        NoExclusionResource,
    ]:
        return transform_solution_from_raw_generic_to_rcpsp(
            raw_sol=raw_sol, problem=self.problem
        )
