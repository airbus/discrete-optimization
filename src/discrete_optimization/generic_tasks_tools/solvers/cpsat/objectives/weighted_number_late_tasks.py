#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic

from ortools.sat.python.cp_model import IntVar, LinearExpr

from discrete_optimization.generic_tasks_tools.entities import VariableBatchEntity
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.objectives.weighted_number_late_tasks import (
    WeightedNumberLateComputer,
    WeightedNumberLateTasksInBatchComputer,
)
from discrete_optimization.generic_tasks_tools.scheduling import Task
from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.objective_modeler import (
    ObjectiveModelerCpSat,
)


class WeightedNumberLateCpSatModeler(ObjectiveModelerCpSat, Generic[Task]):
    """
    CP-SAT modeler for weighted late tasks.
    """

    objective_computer: WeightedNumberLateComputer[Task]
    late_boolean_variables: dict[Task, IntVar]
    initialized_late_boolean_variables: bool = False

    def init_late_boolean_variables(self):
        self.late_boolean_variables = {}
        for task in self.objective_computer.max_end_and_weight:
            self.late_boolean_variables[task] = self.solver.cp_model.new_bool_var(
                f"late_{task}"
            )
            end_var = self.solver.get_task_start_or_end_variable(
                task, start_or_end=StartOrEnd.END
            )
            active = self.solver.get_task_is_present_variable(task)
            if isinstance(active, int) and active == 1:
                (
                    self.solver.cp_model.add(
                        end_var > self.objective_computer.max_end_and_weight[task][0]
                    ).only_enforce_if(self.late_boolean_variables[task])
                )
                (
                    self.solver.cp_model.add(
                        end_var <= self.objective_computer.max_end_and_weight[task][0]
                    ).only_enforce_if(~self.late_boolean_variables[task])
                )
            else:
                (
                    self.solver.cp_model.add(
                        end_var > self.objective_computer.max_end_and_weight[task][0]
                    ).only_enforce_if(self.late_boolean_variables[task], active)
                )
                (
                    self.solver.cp_model.add(
                        end_var <= self.objective_computer.max_end_and_weight[task][0]
                    ).only_enforce_if(~self.late_boolean_variables[task], active)
                )
                # When not active, it's not late.
                self.solver.cp_model.add_implication(
                    ~active, ~self.late_boolean_variables[task]
                )

    def get_objective_expr(self) -> LinearExpr:
        if not self.initialized_late_boolean_variables:
            self.init_late_boolean_variables()
        return sum(
            [
                self.objective_computer.max_end_and_weight[task][1]
                * self.late_boolean_variables[task]
                for task in self.late_boolean_variables
            ]
        )


class WeightedNumberLateInBatchCpSatModeler(ObjectiveModelerCpSat, Generic[Task]):
    """
    CP-SAT modeler for weighted late tasks.
    """

    objective_computer: WeightedNumberLateTasksInBatchComputer[Task]
    batch_late_boolean_variables: dict[Task, IntVar]
    corrected_end_variables: dict[Task, IntVar]
    initialized_batch_late_boolean_variables: bool = False

    def init_corrected_end_variables(self):
        batching_task_categ = (
            self.objective_computer.problem.get_batching_task_categories()
        )
        # "mandatory": mandatory_batch,
        # "optional": optional_batch,
        # "forbidden": forbidden_batch,
        self.corrected_end_variables = {}
        for task in self.objective_computer.max_end_and_weight:
            if (
                task in batching_task_categ["mandatory"]
                or task in batching_task_categ["optional"]
            ):
                if task in self.solver.task_to_batch_vars:
                    lb = self.solver.get_task_start_or_end_lower_bound(
                        task, start_or_end=StartOrEnd.END
                    )
                    ub = self.solver.get_task_start_or_end_upper_bound(
                        task, start_or_end=StartOrEnd.END
                    )
                    self.corrected_end_variables[task] = (
                        self.solver.cp_model.new_int_var(
                            lb=lb, ub=ub, name=f"corrected_end_{task}"
                        )
                    )
                    for batch_id in self.solver.task_to_batch_vars[task]:
                        # TODO : util function or change the storing of task_to_batch...
                        entity = [
                            ent
                            for ent in self.solver._starts_entity
                            if isinstance(ent, VariableBatchEntity)
                            and ent.batch_id == batch_id
                        ][0]
                        (
                            self.solver.cp_model.add(
                                self.corrected_end_variables[task]
                                == self.solver.get_entity_start_or_end_variable(
                                    entity, start_or_end=StartOrEnd.END
                                )
                            ).only_enforce_if(
                                self.solver.task_to_batch_vars[task][batch_id]
                            )
                        )
            else:
                self.corrected_end_variables[task] = (
                    self.solver.get_task_start_or_end_variable(task, StartOrEnd.END)
                )

    def init_batch_late_boolean_variables(self):
        self.init_corrected_end_variables()
        self.batch_late_boolean_variables = {}
        for task in self.objective_computer.max_end_and_weight:
            self.batch_late_boolean_variables[task] = self.solver.cp_model.new_bool_var(
                f"late_{task}"
            )
            end_var = self.corrected_end_variables[task]
            active = self.solver.get_task_is_present_variable(task)
            if isinstance(active, int) and active == 1:
                (
                    self.solver.cp_model.add(
                        end_var > self.objective_computer.max_end_and_weight[task][0]
                    ).only_enforce_if(self.batch_late_boolean_variables[task])
                )
                (
                    self.solver.cp_model.add(
                        end_var <= self.objective_computer.max_end_and_weight[task][0]
                    ).only_enforce_if(~self.batch_late_boolean_variables[task])
                )
            else:
                (
                    self.solver.cp_model.add(
                        end_var > self.objective_computer.max_end_and_weight[task][0]
                    ).only_enforce_if(self.batch_late_boolean_variables[task], active)
                )
                (
                    self.solver.cp_model.add(
                        end_var <= self.objective_computer.max_end_and_weight[task][0]
                    ).only_enforce_if(~self.batch_late_boolean_variables[task], active)
                )
                # When not active, it's not late.
                self.solver.cp_model.add_implication(
                    ~active, ~self.batch_late_boolean_variables[task]
                )

    def get_objective_expr(self) -> LinearExpr:
        if not self.initialized_batch_late_boolean_variables:
            self.init_batch_late_boolean_variables()
        return sum(
            [
                self.objective_computer.max_end_and_weight[task][1]
                * self.batch_late_boolean_variables[task]
                for task in self.batch_late_boolean_variables
            ]
        )
