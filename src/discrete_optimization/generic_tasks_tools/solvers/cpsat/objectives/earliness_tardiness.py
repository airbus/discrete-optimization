#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic

from ortools.sat.python.cp_model import LinearExpr

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.objectives.earliness_tardiness import (
    EarlinessTardinessComputer,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    Task,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.objective_modeler import (
    ObjectiveModelerCpSat,
)


class EarlinessTardinessCpSatModeler(ObjectiveModelerCpSat, Generic[Task]):
    objective_computer: EarlinessTardinessComputer[Task]
    earliness_tardiness_vars_initialized: bool = False
    earliness_start_vars: dict
    earliness_end_vars: dict
    tardiness_start_vars: dict
    tardiness_end_vars: dict

    def __init__(
        self,
        problem: SchedulingProblem[Task],
        weight_objective: float = 1.0,
        max_start_and_weight_for_tardiness: dict[
            Task, tuple[int | None, int | None]
        ] = None,
        max_end_and_weight_for_tardiness: dict[
            Task, tuple[int | None, int | None]
        ] = None,
        min_start_and_weight_for_earliness: dict[
            Task, tuple[int | None, int | None]
        ] = None,
        min_end_and_weight_for_earliness: dict[
            Task, tuple[int | None, int | None]
        ] = None,
    ):
        super().__init__(problem, weight_objective)
        if max_start_and_weight_for_tardiness is None:
            self.max_start_and_weight_for_tardiness = {}
        else:
            self.max_start_and_weight_for_tardiness = max_start_and_weight_for_tardiness
        if max_end_and_weight_for_tardiness is None:
            self.max_end_and_weight_for_tardiness = {}
        else:
            self.max_end_and_weight_for_tardiness = max_end_and_weight_for_tardiness
        if min_start_and_weight_for_earliness is None:
            self.min_start_and_weight_for_earliness = {}
        else:
            self.min_start_and_weight_for_earliness = min_start_and_weight_for_earliness
        if min_end_and_weight_for_earliness is None:
            self.min_end_and_weight_for_earliness = {}
        else:
            self.min_end_and_weight_for_earliness = min_end_and_weight_for_earliness

    def _create_earliness_tardiness_vars(self):
        cp_model = self.solver.cp_model
        self.earliness_start_vars = {}
        self.earliness_end_vars = {}
        self.tardiness_start_vars = {}
        self.tardiness_end_vars = {}
        for task in self.objective_computer.get_tasks_having_min_start_for_earliness():
            start_for_earliness = self.objective_computer.get_min_start_for_earliness(
                task
            )
            lb_s = self.solver.get_task_start_or_end_lower_bound(
                task=task, start_or_end=StartOrEnd.START
            )
            max_earliness = start_for_earliness - lb_s
            self.earliness_start_vars[task] = cp_model.new_int_var(
                lb=0, ub=max_earliness, name=f"earliness_start_{task}"
            )
            cp_model.add_max_equality(
                self.earliness_start_vars[task],
                [
                    0,
                    start_for_earliness
                    - self.solver.get_task_start_or_end_variable(
                        task, StartOrEnd.START
                    ),
                ],
            )
        for task in self.objective_computer.get_tasks_having_min_end_for_earliness():
            end_for_earliness = self.objective_computer.get_min_end_for_earliness(task)
            lb_e = self.solver.get_task_start_or_end_lower_bound(
                task=task, start_or_end=StartOrEnd.END
            )
            max_earliness = end_for_earliness - lb_e
            self.earliness_end_vars[task] = cp_model.new_int_var(
                lb=0, ub=max_earliness, name=f"earliness_end_{task}"
            )
            cp_model.add_max_equality(
                self.earliness_start_vars[task],
                [
                    0,
                    end_for_earliness
                    - self.solver.get_task_start_or_end_variable(task, StartOrEnd.END),
                ],
            )

        for task in self.objective_computer.get_tasks_having_max_start_for_tardiness():
            start_for_tardiness = self.objective_computer.get_max_start_for_tardiness(
                task
            )
            ub_s = self.solver.get_task_start_or_end_upper_bound(
                task=task, start_or_end=StartOrEnd.START
            )
            max_tardiness = ub_s - start_for_tardiness
            self.tardiness_start_vars[task] = cp_model.new_int_var(
                lb=0, ub=max_tardiness, name=f"tardiness_start_{task}"
            )
            cp_model.add_max_equality(
                self.earliness_start_vars[task],
                [
                    0,
                    self.solver.get_task_start_or_end_variable(task, StartOrEnd.START)
                    - start_for_tardiness,
                ],
            )
        for task in self.objective_computer.get_tasks_having_max_end_for_tardiness():
            end_for_tardiness = self.objective_computer.get_max_end_for_tardiness(task)
            ub_e = self.solver.get_task_start_or_end_upper_bound(
                task=task, start_or_end=StartOrEnd.END
            )
            max_tardiness = ub_e - end_for_tardiness
            self.tardiness_end_vars[task] = cp_model.new_int_var(
                lb=0, ub=max_tardiness, name=f"tardiness_end_{task}"
            )
            cp_model.add_max_equality(
                self.earliness_start_vars[task],
                [
                    0,
                    self.solver.get_task_start_or_end_variable(task, StartOrEnd.END)
                    - end_for_tardiness,
                ],
            )
        self.earliness_tardiness_vars_initialized = True

    def get_objective_expr(self) -> LinearExpr:
        if not self.earliness_tardiness_vars_initialized:
            self._create_earliness_tardiness_vars()
        sum_start_earliness = sum(
            [
                self.earliness_start_vars[task]
                * self.objective_computer.get_weight_start_for_earliness(task)
                for task in self.earliness_start_vars
            ]
        )
        sum_end_earliness = sum(
            [
                self.earliness_start_vars[task]
                * self.objective_computer.get_weight_start_for_earliness(task)
                for task in self.earliness_start_vars
            ]
        )
        sum_start_earliness = sum(
            [
                self.earliness_start_vars[task]
                * self.objective_computer.get_weight_start_for_earliness(task)
                for task in self.earliness_start_vars
            ]
        )
        sum_start_earliness = sum(
            [
                self.earliness_start_vars[task]
                * self.objective_computer.get_weight_start_for_earliness(task)
                for task in self.earliness_start_vars
            ]
        )
