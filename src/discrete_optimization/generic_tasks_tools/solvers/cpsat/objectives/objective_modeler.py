#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from typing import Generic

from ortools.sat.python.cp_model import LinearExpr

from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
    NonRenewableResource,
    NonSkillCumulativeResource,
    Skill,
    Task,
    UnaryResource,
)


class ObjectiveModelerCpSat(
    Generic[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ],
    ABC,
):
    objective_computer: ObjectiveComputer
    solver: GenericSchedulingAutoCpSatSolver[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ]

    def __init__(
        self,
        solver: GenericSchedulingAutoCpSatSolver[
            Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
        ],
        objective_computer: ObjectiveComputer,
    ) -> None:
        self.objective_computer = objective_computer
        self.solver = solver

    @abstractmethod
    def get_objective_expr(self) -> LinearExpr: ...
