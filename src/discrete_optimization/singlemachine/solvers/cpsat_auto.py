#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.skill import Skill
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
    Task,
)
from discrete_optimization.singlemachine.problem import (
    ExclusionResource,
    NoNonRenewableResource,
    NonSkillCumulativeResource,
    NoSkill,
    NoUnaryResource,
    Task,
)
from discrete_optimization.singlemachine.transformations.to_generic import (
    transform_solution_from_raw_generic_to_wt,
)


class CpSatAutoWTSolver(
    GenericSchedulingAutoCpSatSolver[
        Task,
        NoUnaryResource,
        NoSkill,
        NonSkillCumulativeResource,
        NoNonRenewableResource,
        ExclusionResource,
    ],
):
    def convert_task_variables_to_solution(
        self, raw_sol: RawSolution[Task, UnaryResource, Skill]
    ) -> GenericSchedulingSolution[
        Task,
        UnaryResource,
        Skill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ]:
        return transform_solution_from_raw_generic_to_wt(
            raw_sol=raw_sol, problem=self.problem
        )
