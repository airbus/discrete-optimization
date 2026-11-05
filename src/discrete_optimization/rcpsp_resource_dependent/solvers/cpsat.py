#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tasks_tools.allocation import (
    NoUnaryResource,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.exclusions import ExclusionResource
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.skill import (
    NoSkill,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
)
from discrete_optimization.rcpsp_resource_dependent.problem import (
    NonRenewableResource,
    NonSkillCumulativeResource,
    RcpspResourceDependentProblem,
    RcpspResourceDependentSolution,
    Task,
)


class CpSatRcpspResourceDependentSolver(
    GenericSchedulingAutoCpSatSolver[
        Task,
        NoUnaryResource,
        NoSkill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ]
):
    problem: RcpspResourceDependentProblem

    def convert_task_variables_to_solution(
        self, raw_sol: RawSolution[Task, NoUnaryResource, NoSkill]
    ) -> GenericSchedulingSolution[
        Task,
        NoUnaryResource,
        NoSkill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ]:
        return RcpspResourceDependentSolution(
            problem=self.problem,
            schedule={
                t: (
                    raw_sol.task_variables[t].get_start_or_end(
                        start_or_end=StartOrEnd.START
                    ),
                    raw_sol.task_variables[t].get_start_or_end(
                        start_or_end=StartOrEnd.END
                    ),
                )
                for t in self.problem.tasks_list
            },
            modes={t: raw_sol.task_variables[t].mode for t in self.problem.tasks_list},
        )
