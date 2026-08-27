#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from ortools.sat.python.cp_model import LinearExpr

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.objectives.schedule_changes import (
    ScheduleChangesComputer,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.objective_modeler import (
    ObjectiveModelerCpSat,
)


class ScheduleChangesModelerCpSat(ObjectiveModelerCpSat):
    objective_computer: ScheduleChangesComputer
    variables: dict
    schedule_changes_initialized: bool = False

    def _create_vars_schedule_changes(self):
        self.variables = {}
        problem = self.solver.problem
        base_solution = self.objective_computer.base_scheduling_solution
        base_problem: SchedulingProblem = base_solution.problem
        cp_model = self.solver.cp_model
        common_tasks = list(
            set(base_problem.tasks_list).intersection(problem.tasks_list)
        )
        tasks_of_interest = [
            t
            for t in common_tasks
            if self.objective_computer.cost_any_shift(t) != 0
            or self.objective_computer.cost_unit_deviation(t) != 0
        ]
        len_tasks_of_interest = len(tasks_of_interest)
        delta_starts = {t: None for t in tasks_of_interest}
        delta_starts_abs = {
            t: None
            for t in tasks_of_interest
            if self.objective_computer.cost_unit_deviation(t) != 0
        }
        is_shifted = {
            t: cp_model.new_bool_var(name=f"shifted_{t}")
            for t in tasks_of_interest
            if self.objective_computer.cost_any_shift(t) != 0
        }
        self.variables["is_shifted"] = is_shifted
        for i in range(len_tasks_of_interest):
            tt = tasks_of_interest[i]
            delta_starts[tt] = self.solver.get_task_start_or_end_variable(
                task=tt, start_or_end=StartOrEnd.START
            ) - base_solution.get_start_time(tt)
            if tt in is_shifted:
                cp_model.add(delta_starts[tt] != 0).only_enforce_if(is_shifted[tt])
                cp_model.add(delta_starts[tt] == 0).only_enforce_if(
                    is_shifted[tt].Not()
                )
            if tt in delta_starts_abs:
                delta_starts_abs[tt] = cp_model.new_int_var(
                    lb=0,
                    ub=problem.get_makespan_upper_bound(),
                    name=f"delta_abs_starts_{tt}",
                )
                cp_model.add_abs_equality(delta_starts_abs[tt], delta_starts[tt])
        self.variables["delta_starts_abs"] = delta_starts_abs
        self.variables["delta_starts"] = delta_starts
        # TODO specify data in the objectivecomputer to specify max_delta cost (on which tasks it's computed?)
        # max_delta_start = cp_model.new_int_var(
        #    lb=0,
        #    ub=problem.get_makespan_upper_bound(),
        #    name=f"max_delta_starts"
        # )
        # self.variables["max_delta_start"] = max_delta_start
        # cp_model.add_max_equality(max_delta_start, ])
        self.schedule_changes_initialized = True

    def get_objective_expr(self) -> LinearExpr:
        if not self.schedule_changes_initialized:
            self._create_vars_schedule_changes()
        return sum(
            [
                self.variables["is_shifted"][tt]
                * self.objective_computer.cost_any_shift(tt)
                for tt in self.variables["is_shifted"].keys()
            ]
        ) + sum(
            [
                self.variables["delta_starts_abs"][tt]
                * self.objective_computer.cost_unit_deviation(tt)
                for tt in self.variables["delta_starts_abs"].keys()
            ]
        )
