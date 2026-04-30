#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Any, Iterable, Optional

from ortools.sat.python.cp_model import IntVar, LinearExprT

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.scheduling import SchedulingCpSolver
from discrete_optimization.generic_tools.cp_tools import SignEnum
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver


class SchedulingCpSatSolver(OrtoolsCpSatSolver, SchedulingCpSolver[Task]):
    """Base class for most ortools/cpsat solvers handling scheduling problems.

    Allows to have common code.

    """

    _makespan: Optional[IntVar] = None
    """Internal variable use to define the global makespan."""

    _subtasks_makespan: Optional[IntVar] = None
    """Internal variable use to define the partial makespan."""

    constraints_on_makespan: Optional[list[Any]] = None
    """Constraints on partial makespan so that it can be considered as the objective."""

    def init_model(self, **kwargs: Any) -> None:
        """Init cp model and reset stored variables if any."""
        super().init_model(**kwargs)
        self._makespan = None
        self._subtasks_makespan = None
        self.constraints_on_makespan = None

    @abstractmethod
    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExprT:
        """Retrieve the variable storing the start or end time of given task.

        Args:
            task:
            start_or_end:

        Returns:

        """
        ...

    def add_constraint_on_task(
        self, task: Task, start_or_end: StartOrEnd, sign: SignEnum, time: int
    ) -> list[Any]:
        var = self.get_task_start_or_end_variable(task=task, start_or_end=start_or_end)
        return self.add_bound_constraint(var=var, sign=sign, value=time)

    def add_constraint_chaining_tasks(self, task1: Task, task2: Task) -> list[Any]:
        var1 = self.get_task_start_or_end_variable(
            task=task1, start_or_end=StartOrEnd.END
        )
        var2 = self.get_task_start_or_end_variable(
            task=task2, start_or_end=StartOrEnd.START
        )
        return [self.cp_model.add(var1 == var2)]

    def _get_makespan_var(self) -> IntVar:
        """Get the makespan variable used to track global makespan."""
        if self._makespan is None:
            self._makespan = self.cp_model.NewIntVar(
                lb=self.get_makespan_lower_bound(),
                ub=self.get_makespan_upper_bound(),
                name="makespan",
            )
        return self._makespan

    def _get_subtasks_makespan_var(self) -> IntVar:
        """Get the makespan variable used to track subtasks makespan."""
        if self._subtasks_makespan is None:
            self._subtasks_makespan = self.cp_model.NewIntVar(
                lb=0,  # lower bound for any tasks subset
                ub=self.get_makespan_upper_bound(),
                name="subtasks_makespan",
            )
        return self._subtasks_makespan

    def remove_constraints_on_objective(self) -> None:
        if self.constraints_on_makespan is not None:
            self.remove_constraints(self.constraints_on_makespan)

    def get_global_makespan_variable(self) -> Any:
        # remove previous constraints on makespan variable from cp model
        self.remove_constraints_on_objective()
        # get makespan variable
        makespan = self._get_makespan_var()
        # update those constraints
        self.constraints_on_makespan = [
            self.cp_model.AddMaxEquality(
                makespan,
                [
                    self.get_task_start_or_end_variable(task, StartOrEnd.END)
                    for task in self.problem.get_last_tasks()
                ],
            )
        ]
        return makespan

    def get_subtasks_makespan_variable(self, subtasks: Iterable[Task]) -> Any:
        # remove previous constraints on makespan variable from cp model
        self.remove_constraints_on_objective()
        # get makespan variable
        makespan = self._get_subtasks_makespan_var()
        # update those constraints
        self.constraints_on_makespan = [
            self.cp_model.AddMaxEquality(
                makespan,
                [
                    self.get_task_start_or_end_variable(task, StartOrEnd.END)
                    for task in subtasks
                ],
            )
        ]
        return makespan

    def get_subtasks_sum_end_time_variable(self, subtasks: Iterable[Task]) -> Any:
        self.remove_constraints_on_objective()
        return sum(
            self.get_task_start_or_end_variable(task, StartOrEnd.END)
            for task in subtasks
        )

    def get_subtasks_sum_start_time_variable(self, subtasks: Iterable[Task]) -> Any:
        self.remove_constraints_on_objective()
        return sum(
            self.get_task_start_or_end_variable(task, StartOrEnd.START)
            for task in subtasks
        )
