#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Generic

from ortools.sat.python.cp_model import LinearExpr

from discrete_optimization.generic_tasks_tools.exclusions import (
    ExclusionProblem,
    ExclusionResource,
    Task,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.multimode_scheduling import (
    MultimodeSchedulingCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.utils import (
    ModeToValueModeling,
    create_variable_function_of_mode_on_solver,
)

logger = logging.getLogger(__name__)


class ExclusionCpSatSolver(
    MultimodeSchedulingCpSatSolver[Task], Generic[Task, ExclusionResource]
):
    problem: ExclusionProblem[Task, ExclusionResource]
    exclusion_resource_demand_vars: dict[tuple[Task, ExclusionResource], LinearExpr]
    exclusion_vars: dict[tuple[Task, ExclusionResource], LinearExpr]
    exclusion_resource_vars_initialized: bool = False

    def initialize_exclusion_resource_demand_vars(self):
        """
        Build either expression or variable array for resource demand.
        For task for which resource demand only depends on its own mode, this is a simple expression,
        While for dependent consumption based of other task mode, additional variable is added.
        """
        self.exclusion_resource_demand_vars = {}
        self.exclusion_vars = {}
        for resource in self.problem.exclusion_resources_list:
            for task in self.problem.get_tasks_possibly_excluded(resource):
                mode2value = {
                    m: self.problem.get_task_consumption_exclusion_resource(
                        resource=resource, task=task, mode=m
                    )
                    for m in self.problem.get_task_modes(task)
                }
                self.exclusion_resource_demand_vars[task, resource] = (
                    create_variable_function_of_mode_on_solver(
                        solver=self,
                        name=f"conso_{task}_{resource}",
                        mode2value=mode2value,
                        task=task,
                        modeling=ModeToValueModeling.ENFORCE_IF,
                    )
                )
            for task in self.problem.get_tasks_possibly_exclude_others(resource):
                mode2value = {
                    m: self.problem.is_task_mode_excluding_others(
                        resource=resource, task=task, mode=m
                    )
                    for m in self.problem.get_task_modes(task)
                }
                self.exclusion_vars[task, resource] = (
                    create_variable_function_of_mode_on_solver(
                        solver=self,
                        name=f"blocking_others_{task}_{resource}",
                        mode2value=mode2value,
                        task=task,
                        modeling=ModeToValueModeling.ENFORCE_IF,
                    )
                )

        self.exclusion_resource_vars_initialized = True

    def get_exclusion_resource_demand_variable(
        self, task: Task, resource: ExclusionResource
    ) -> LinearExpr:
        if not self.exclusion_resource_vars_initialized:
            self.initialize_exclusion_resource_demand_vars()
        return self.exclusion_resource_demand_vars[task, resource]

    def create_exclusion_constraints(self):
        if not self.exclusion_resource_vars_initialized:
            self.initialize_exclusion_resource_demand_vars()
        for r in self.problem.exclusion_resources_list:
            tasks_possibly_blocked = list(self.problem.get_tasks_possibly_excluded(r))
            intervals_consuming = [
                self.get_task_interval(task) for task in tasks_possibly_blocked
            ]
            demands_consuming = [
                self.get_exclusion_resource_demand_variable(task=task, resource=r)
                for task in tasks_possibly_blocked
            ]
            # Classical cumulative constraint.
            # TODO : create a custom object to specify somehow infinite resource
            if (capacity := self.problem.get_capacity_exclusion_resource(r)) < float(
                "inf"
            ):
                self.cp_model.add_cumulative(
                    intervals_consuming, demands_consuming, capacity
                )
            # Actual exclusion constraint.
            tasks_possibly_blocking = list(
                self.problem.get_tasks_possibly_exclude_others(r)
            )
            if capacity == 1:
                # This is a common case, where there is a nice compact cumulative formulation
                nb_tasks_blocking = len(tasks_possibly_blocking)
                intervals_ = [
                    self.get_task_interval(task) for task in tasks_possibly_blocking
                ]
                demands_ = [
                    self.exclusion_vars[task, r] for task in tasks_possibly_blocking
                ]
                intervals_.extend(intervals_consuming)
                demands_.extend([nb_tasks_blocking * d for d in demands_consuming])
                # Blocked tasks consume NB_TASKS_BLOCKING
                # Blocking tasks consume 1
                # Therefore, no blocked task can overlap between each other (coherent with capacity=1)
                # And wont overlap with the blocking tasks neither.
                # However blocking tasks can still overlap.
                self.cp_model.add_cumulative(intervals_, demands_, nb_tasks_blocking)
            else:
                # We need to put some manual no overlap...
                intervals_blocking = [
                    self.get_task_interval(task) for task in tasks_possibly_blocking
                ]
                demands_blocking = [
                    self.exclusion_vars[task, r] for task in tasks_possibly_blocking
                ]
                for i_blocked in range(len(intervals_consuming)):
                    for i_blocking in range(len(intervals_blocking)):
                        blocking = demands_blocking[i_blocking]
                        blocked = demands_consuming[i_blocked]
                        if (
                            isinstance(blocking, int)
                            and blocking == 1
                            and isinstance(blocked, int)
                            and blocked == 1
                        ):
                            logging.info("Blocking and blocker are always there")
                            self.cp_model.add_no_overlap(
                                [
                                    intervals_blocking[i_blocking],
                                    intervals_consuming[i_blocked],
                                ]
                            )
                        else:
                            self.cp_model.add_cumulative(
                                [
                                    intervals_blocking[i_blocking],
                                    intervals_consuming[i_blocked],
                                ],
                                [blocking, blocked],
                                1,
                            )
