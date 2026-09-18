#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from ortools.sat.python.cp_model import LinearExpr

from discrete_optimization.generic_tasks_tools.enums import MinOrMax, StartOrEnd
from discrete_optimization.generic_tasks_tools.objectives.soft_time_penalty import (
    SoftTimePenaltyComputer,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.objective_modeler import (
    ObjectiveModelerCpSat,
)


class SoftTimePenaltyModelerCpSat(ObjectiveModelerCpSat):
    objective_computer: SoftTimePenaltyComputer
    variables: dict
    initialized_variables: bool = False

    def _create_vars(self):
        # LB/UB vars
        self.variables = {}
        start_lb_violation = {}
        end_lb_violation = {}
        start_ub_violation = {}
        end_ub_violation = {}
        problem = self.solver.problem
        tasks_list = problem.tasks_list
        cp_model = self.solver.cp_model
        for task in tasks_list:
            start_lb = problem.get_task_start_or_end_lower_bound(
                task=task, start_or_end=StartOrEnd.START
            )
            end_lb = problem.get_task_start_or_end_lower_bound(
                task=task, start_or_end=StartOrEnd.END
            )
            start_ub = problem.get_task_start_or_end_upper_bound(
                task=task, start_or_end=StartOrEnd.START
            )
            end_ub = problem.get_task_start_or_end_upper_bound(
                task=task, start_or_end=StartOrEnd.END
            )
            start_lb_violation[task] = cp_model.new_int_var(
                lb=0,
                ub=problem.get_makespan_upper_bound(),
                name=f"start_lb_violation_{task}",
            )

            end_lb_violation[task] = cp_model.new_int_var(
                lb=0,
                ub=problem.get_makespan_upper_bound(),
                name=f"end_lb_violation_{task}",
            )

            start_ub_violation[task] = cp_model.new_int_var(
                lb=0,
                ub=problem.get_makespan_upper_bound(),
                name=f"start_ub_violation_{task}",
            )

            end_ub_violation[task] = cp_model.new_int_var(
                lb=0,
                ub=problem.get_makespan_upper_bound(),
                name=f"end_ub_violation_{task}",
            )
            cp_model.add_max_equality(
                start_lb_violation[task],
                [
                    0,
                    start_lb
                    - self.solver.get_task_start_or_end_variable(
                        task, StartOrEnd.START
                    ),
                ],
            )
            cp_model.add_max_equality(
                end_lb_violation[task],
                [
                    0,
                    end_lb
                    - self.solver.get_task_start_or_end_variable(task, StartOrEnd.END),
                ],
            )
            cp_model.add_max_equality(
                start_ub_violation[task],
                [
                    0,
                    self.solver.get_task_start_or_end_variable(task, StartOrEnd.START)
                    - start_ub,
                ],
            )
            cp_model.add_max_equality(
                end_ub_violation[task],
                [
                    0,
                    self.solver.get_task_start_or_end_variable(task, StartOrEnd.END)
                    - end_ub,
                ],
            )
        self.variables["start_lb_violation"] = start_lb_violation
        self.variables["end_lb_violation"] = end_lb_violation
        self.variables["start_ub_violation"] = start_ub_violation
        self.variables["end_ub_violation"] = end_ub_violation
        # TIMELAGS
        time_lags = {}
        cnt_time_lags = 0
        for task1_start_or_end in StartOrEnd:
            for task2_start_or_end in StartOrEnd:
                for min_or_max in MinOrMax:
                    for task1, task2, offset in problem.get_original_time_lags(
                        task1_start_or_end=task1_start_or_end,
                        task2_start_or_end=task2_start_or_end,
                        min_or_max=min_or_max,
                    ):
                        var1 = self.solver.get_task_start_or_end_variable(
                            task=task1, start_or_end=task1_start_or_end
                        )
                        var2 = self.solver.get_task_start_or_end_variable(
                            task=task2, start_or_end=task2_start_or_end
                        )
                        time_lags[
                            (
                                task1,
                                task2,
                                offset,
                                min_or_max,
                                task1_start_or_end,
                                task2_start_or_end,
                            )
                        ] = cp_model.new_int_var(
                            lb=0,
                            ub=problem.get_makespan_upper_bound(),
                            name=f"time_lags_violation_{cnt_time_lags}",
                        )
                        cnt_time_lags += 1
                        if min_or_max == MinOrMax.MIN:
                            cp_model.add_max_equality(
                                time_lags[
                                    (
                                        task1,
                                        task2,
                                        offset,
                                        min_or_max,
                                        task1_start_or_end,
                                        task2_start_or_end,
                                    )
                                ],
                                [0, var1 + offset - var2],
                            )
                        else:
                            cp_model.add_max_equality(
                                time_lags[
                                    (
                                        task1,
                                        task2,
                                        offset,
                                        min_or_max,
                                        task1_start_or_end,
                                        task2_start_or_end,
                                    )
                                ],
                                [0, var2 - (offset - var1)],
                            )
        self.variables["time_lags"] = time_lags
        self.initialized_variables = True

    def get_objective_expr(self) -> LinearExpr:
        if not self.initialized_variables:
            self._create_vars()
        return sum([sum(self.variables[k].values()) for k in self.variables])
