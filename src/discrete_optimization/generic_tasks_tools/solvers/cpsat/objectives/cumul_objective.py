#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from enum import Enum

from ortools.sat.python.cp_model import Domain, LinearExpr

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationProblem,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
)
from discrete_optimization.generic_tasks_tools.multimode import MultimodeProblem
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.objectives.cumul_cost import (
    CumulCostComputer,
)
from discrete_optimization.generic_tasks_tools.skill import (
    NonSkillCumulativeResource,
    Skill,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.allocation import (
    AllocationCpSatSolver,
    Task,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.multimode import (
    MultimodeCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.objective_modeler import (
    ObjectiveModelerCpSat,
)


class MultimodeAllocationProblem(
    MultimodeProblem[Task], AllocationProblem[Task, UnaryResource]
):
    pass


class MultimodeAllocationSolver(
    AllocationCpSatSolver[Task, UnaryResource], MultimodeCpSatSolver[Task]
):
    pass


class ModelisationDispersion(Enum):
    EXACT = 0
    MAX_DIFF = 1
    PROXY_MAX_MIN = 2
    PROXY_MIN_MAX = 3
    PROXY_SUM = 4
    PROXY_SLACK = 5


class CumulativeObjective:
    problem: GenericSchedulingProblem

    def __init__(
        self, problem: MultimodeAllocationProblem, solver: MultimodeAllocationSolver
    ):
        self.problem = problem
        self.solver = solver
        self.cumul_value_per_ur = {}
        self.cumul_value_per_ur_nz = {}
        self.cumul_value_per_ur_implication = {}
        self.max_value = {}
        self.min_value_nz = {}

    def create_variables(
        self, val_per_task_per_mode: dict[Task, dict[int, int]], name_value: str
    ):
        cumul_value_per_ur, cumul_value_per_ur_nz = self.create_cumul_value_duplicated(
            val_per_task_per_mode, name_value
        )
        self.cumul_value_per_ur[name_value] = cumul_value_per_ur
        self.cumul_value_per_ur_nz[name_value] = cumul_value_per_ur_nz
        cumul_value_per_ur_impl = self.create_cumul_value_implication(
            val_per_task_per_mode, name_value
        )
        self.cumul_value_per_ur_implication[name_value] = cumul_value_per_ur_impl

    def create_cumul_value_implication(
        self, val_per_task_per_mode: dict[Task, dict[int, int]], name_value: str
    ):
        number_ur = len(self.problem.unary_resources_list)
        upper_bound_value = int(
            sum([max(val_per_task_per_mode[t].values()) for t in val_per_task_per_mode])
        )
        cumul_value_per_ur = [
            self.solver.cp_model.NewIntVar(
                lb=0,
                ub=upper_bound_value,
                name=f"cumulated_value_{name_value}_{i}_impl",
            )
            for i in range(number_ur)
        ]
        for index_ur in range(number_ur):
            contribution_per_task = []
            for t in self.problem.tasks_list:
                modes = list(self.problem.get_task_modes(t))
                if len(modes) == 1:
                    contribution_per_task.append(
                        self.solver.get_task_unary_resource_is_present_variable(
                            task=t,
                            unary_resource=self.problem.unary_resources_list[index_ur],
                        )
                        * val_per_task_per_mode[t][modes[0]]
                    )
                else:
                    contrib_t = self.solver.cp_model.NewIntVarFromDomain(
                        domain=Domain.FromValues(
                            [0]
                            + [
                                val_per_task_per_mode[t][m]
                                for m in val_per_task_per_mode[t]
                            ]
                        ),
                        name=f"contrib_{t}_{index_ur}",
                    )
                    for m in modes:
                        self.solver.cp_model.add(
                            contrib_t == val_per_task_per_mode[t][m]
                        ).only_enforce_if(
                            self.solver.get_task_unary_resource_is_present_variable(
                                t, self.problem.unary_resources_list[index_ur]
                            ),
                            self.solver.get_task_mode_is_present_variable(t, m),
                        )
                        self.solver.cp_model.add(contrib_t == 0).only_enforce_if(
                            ~self.solver.get_task_unary_resource_is_present_variable(
                                t, self.problem.unary_resources_list[index_ur]
                            )
                        )
                    contribution_per_task.append(contrib_t)
            load = sum(contribution_per_task)
            self.solver.cp_model.add(
                cumul_value_per_ur[index_ur] == load
            ).OnlyEnforceIf(
                self.solver.used_variables[self.problem.unary_resources_list[index_ur]]
            )
        return cumul_value_per_ur

    def create_cumul_value_duplicated(
        self, val_per_task_per_mode: dict[Task, dict[int, int]], name_value: str
    ):
        number_ur = len(self.problem.unary_resources_list)
        upper_bound_value = int(
            sum([max(val_per_task_per_mode[t].values()) for t in val_per_task_per_mode])
        )
        cumul_value_per_ur = [
            self.solver.cp_model.NewIntVar(
                lb=0, ub=upper_bound_value, name=f"cumulated_value_{name_value}_{i}"
            )
            for i in range(number_ur)
        ]
        cumul_value_per_ur_nz = [
            self.solver.cp_model.NewIntVar(
                lb=0, ub=upper_bound_value, name=f"cumulated_value_nz_{name_value}_{i}"
            )
            for i in range(number_ur)
        ]
        for index_ur in range(number_ur):
            contribution_per_task = []
            for t in self.problem.tasks_list:
                modes = list(self.problem.get_task_modes(t))
                if len(modes) == 1:
                    contribution_per_task.append(
                        self.solver.get_task_unary_resource_is_present_variable(
                            task=t,
                            unary_resource=self.problem.unary_resources_list[index_ur],
                        )
                        * val_per_task_per_mode[t][modes[0]]
                    )
                else:
                    contrib_t = self.solver.cp_model.NewIntVarFromDomain(
                        domain=Domain.FromValues(
                            [0]
                            + [
                                val_per_task_per_mode[t][m]
                                for m in val_per_task_per_mode[t]
                            ]
                        ),
                        name=f"contrib_{t}_{index_ur}",
                    )
                    for m in modes:
                        self.solver.cp_model.add(
                            contrib_t == val_per_task_per_mode[t][m]
                        ).only_enforce_if(
                            self.solver.get_task_unary_resource_is_present_variable(
                                t, self.problem.unary_resources_list[index_ur]
                            ),
                            self.solver.get_task_mode_is_present_variable(t, m),
                        )
                        self.solver.cp_model.add(contrib_t == 0).only_enforce_if(
                            ~self.solver.get_task_unary_resource_is_present_variable(
                                t, self.problem.unary_resources_list[index_ur]
                            )
                        )
                    contribution_per_task.append(contrib_t)
            load = sum(contribution_per_task)
            self.solver.cp_model.add(cumul_value_per_ur[index_ur] == load)
            self.solver.cp_model.add(
                cumul_value_per_ur_nz[index_ur] == load
            ).only_enforce_if(
                self.solver.used_variables[self.problem.unary_resources_list[index_ur]]
            )
            self.solver.cp_model.add(
                cumul_value_per_ur_nz[index_ur] == upper_bound_value
            ).only_enforce_if(
                self.solver.used_variables[
                    self.problem.unary_resources_list[index_ur]
                ].Not()
            )
        return cumul_value_per_ur, cumul_value_per_ur_nz

    def create_dispersion_objective(
        self,
        val_per_task_per_mode: dict[Task, dict[int, int]],
        name_value: str,
        modelisation_dispersion: ModelisationDispersion = ModelisationDispersion.EXACT,
    ):
        self.solver.create_used_variables()
        self.create_variables(val_per_task_per_mode, name_value)
        upper_bound_value = int(
            sum([max(val_per_task_per_mode[t].values()) for t in val_per_task_per_mode])
        )
        if modelisation_dispersion == ModelisationDispersion.EXACT:
            max_value = self.solver.cp_model.NewIntVar(
                lb=0,
                ub=upper_bound_value,
                name=f"max_value_{name_value}",
            )
            min_value = self.solver.cp_model.NewIntVar(
                lb=0,  # upper_bound//len(used_team),
                ub=upper_bound_value,
                name=f"min_value_{name_value}",
            )
            self.solver.cp_model.add_min_equality(
                min_value, self.cumul_value_per_ur_nz[name_value]
            )
            self.solver.cp_model.add_max_equality(
                max_value, self.cumul_value_per_ur[name_value]
            )
            self.min_value_nz[name_value] = min_value
            self.max_value[name_value] = max_value
            return max_value - min_value
        elif modelisation_dispersion == ModelisationDispersion.MAX_DIFF:
            max_diff = self.solver.cp_model.NewIntVar(
                lb=0, ub=upper_bound_value, name=f"max_diff_{name_value}"
            )
            self.solver.cp_model.AddMaxEquality(
                max_diff,
                [
                    x - y
                    for x in self.cumul_value_per_ur_implication[name_value]
                    for y in self.cumul_value_per_ur_implication[name_value]
                ],
            )
            return max_diff
        elif modelisation_dispersion == ModelisationDispersion.PROXY_MAX_MIN:
            max_value = self.solver.cp_model.NewIntVar(
                lb=0,  # upper_bound // len(used_team),
                ub=upper_bound_value,
                name=f"max_value_{name_value}",
            )
            self.solver.cp_model.AddMaxEquality(
                max_value, self.cumul_value_per_ur[name_value]
            )
            return max_value
        elif modelisation_dispersion == ModelisationDispersion.PROXY_MIN_MAX:
            min_value = self.solver.cp_model.NewIntVar(
                lb=0,  # upper_bound // len(used_team),
                ub=upper_bound_value,
                name=f"min_value_{name_value}",
            )
            self.solver.cp_model.AddMinEquality(
                min_value, self.cumul_value_per_ur_nz[name_value]
            )
            return -min_value
        elif modelisation_dispersion == ModelisationDispersion.PROXY_SUM:
            abs_deltas = [
                {
                    j: self.solver.cp_model.NewIntVar(
                        lb=0, ub=upper_bound_value, name=f"delta_{i}_{j}_{name_value}"
                    )
                    for j in range(
                        i + 1, len(self.cumul_value_per_ur_implication[name_value])
                    )
                }
                for i in range(len(self.cumul_value_per_ur_implication[name_value]))
            ]
            for i in range(len(abs_deltas)):
                for j in abs_deltas[i]:
                    self.solver.cp_model.AddAbsEquality(
                        abs_deltas[i][j],
                        self.cumul_value_per_ur_implication[name_value][i]
                        - self.cumul_value_per_ur_implication[name_value][j],
                    )

            return sum(
                [
                    abs_deltas[i][j]
                    for i in range(len(abs_deltas))
                    for j in abs_deltas[i]
                ]
            )
        elif modelisation_dispersion == ModelisationDispersion.PROXY_SLACK:
            some_expected_value = self.solver.cp_model.NewIntVar(
                lb=0, ub=upper_bound_value, name=f"expected_value_{name_value}"
            )
            slack = self.solver.cp_model.NewIntVar(
                lb=0, ub=upper_bound_value, name=f"slack_{name_value}"
            )
            for i in range(len(self.cumul_value_per_ur[name_value])):
                (
                    self.solver.cp_model.Add(
                        self.cumul_value_per_ur[name_value][i]
                        <= some_expected_value + slack
                    ).OnlyEnforceIf(
                        self.solver.used_variables[self.problem.unary_resources_list[i]]
                    )
                )
                (
                    self.solver.cp_model.Add(
                        self.cumul_value_per_ur[name_value][i]
                        >= some_expected_value - slack
                    ).OnlyEnforceIf(
                        self.solver.used_variables[self.problem.unary_resources_list[i]]
                    )
                )
            return slack
        else:
            raise NotImplementedError(f"Method {modelisation_dispersion} unknown")


class CumulCostModelerCpSat(ObjectiveModelerCpSat):
    objective_computer: CumulCostComputer

    def __init__(
        self,
        solver: GenericSchedulingAutoCpSatSolver[
            Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
        ],
        objective_computer: CumulCostComputer,
        modelisation_dispersion: ModelisationDispersion,
    ) -> None:
        super().__init__(solver, objective_computer)
        self.modelisation_dispersion = modelisation_dispersion
        self.cumul_objective = []
        self.objs = []

    def get_objective_expr(self) -> LinearExpr:
        for dim in self.objective_computer.cumul_dimensions:
            cumul_obj = CumulativeObjective(
                problem=self.solver.problem, solver=self.solver
            )
            obj = cumul_obj.create_dispersion_objective(
                val_per_task_per_mode={
                    t: {
                        m: self.objective_computer.get_value_dimension_task_mode(
                            dimension=dim, task=t, mode=m
                        )
                        for m in self.solver.problem.get_task_modes(t)
                    }
                    for t in self.solver.problem.tasks_list
                    if t in self.objective_computer.value_tasks[dim]
                    or any(
                        x[0] == t
                        for x in self.objective_computer.value_tasks_per_mode[dim]
                    )
                },
                name_value=dim,
                modelisation_dispersion=self.modelisation_dispersion,
            )
            self.cumul_objective.append(cumul_obj)
            self.objs.append(obj)
        return sum(self.objs)
