#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tasks_tools.allocation import (
    NoUnaryResource,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
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
    Skill,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
)
from discrete_optimization.rcpsp_resource_dependent.problem import (
    RcpspResourceDependentProblem,
    RcpspResourceDependentSolution,
)


class CpSatRcpspResourceDependentSolver(
    GenericSchedulingAutoCpSatSolver[
        Task, NoUnaryResource, NoSkill, NonSkillCumulativeResource, NonRenewableResource
    ]
):
    problem: RcpspResourceDependentProblem

    def needs_duration_variables(self) -> bool:
        return True

    def convert_task_variables_to_solution(
        self, raw_sol: RawSolution[Task, UnaryResource, Skill]
    ) -> GenericSchedulingSolution[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
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
