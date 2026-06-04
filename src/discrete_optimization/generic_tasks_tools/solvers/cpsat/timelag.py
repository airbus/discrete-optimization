#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat.scheduling import (
    SchedulingCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.timelag import TimelagProblem


class TimelagCpSatSolver(SchedulingCpSatSolver[Task]):
    """Mixin for cpsat solvers dealing with scheduling problems with time lag constraints."""

    problem: TimelagProblem[Task]

    def _create_timelag_constraints(
        self,
        tasks_n_offsets_min: list[tuple[Task, Task, int]],
        tasks_n_offsets_max: list[tuple[Task, Task, int]],
        task1_start_or_end: StartOrEnd,
        task2_start_or_end: StartOrEnd,
    ) -> None:
        tasks_n_offsets_common = set(tasks_n_offsets_min).intersection(
            set(tasks_n_offsets_max)
        )
        tasks_n_offsets_min_only = set(tasks_n_offsets_min).difference(
            set(tasks_n_offsets_max)
        )
        tasks_n_offsets_max_only = set(tasks_n_offsets_max).difference(
            set(tasks_n_offsets_min)
        )
        for task1, task2, offset in tasks_n_offsets_common:
            self.cp_model.add(
                self.get_task_start_or_end_variable(
                    task=task1, start_or_end=task1_start_or_end
                )
                + offset
                == self.get_task_start_or_end_variable(
                    task=task2, start_or_end=task2_start_or_end
                )
            )
        for task1, task2, offset in tasks_n_offsets_min_only:
            self.cp_model.add(
                self.get_task_start_or_end_variable(
                    task=task1, start_or_end=task1_start_or_end
                )
                + offset
                <= self.get_task_start_or_end_variable(
                    task=task2, start_or_end=task2_start_or_end
                )
            )
        for task1, task2, offset in tasks_n_offsets_max_only:
            self.cp_model.add(
                self.get_task_start_or_end_variable(
                    task=task1, start_or_end=task1_start_or_end
                )
                + offset
                >= self.get_task_start_or_end_variable(
                    task=task2, start_or_end=task2_start_or_end
                )
            )

    def create_timelag_constraints(self) -> None:
        """Add precedence constraints to cp model."""
        self._create_timelag_constraints(
            tasks_n_offsets_min=self.problem.get_start_to_start_min_time_lags(),
            tasks_n_offsets_max=self.problem.get_start_to_start_max_time_lags(),
            task1_start_or_end=StartOrEnd.START,
            task2_start_or_end=StartOrEnd.START,
        )
        self._create_timelag_constraints(
            tasks_n_offsets_min=self.problem.get_end_to_start_min_time_lags(),
            tasks_n_offsets_max=self.problem.get_end_to_start_max_time_lags(),
            task1_start_or_end=StartOrEnd.END,
            task2_start_or_end=StartOrEnd.START,
        )
        self._create_timelag_constraints(
            tasks_n_offsets_min=self.problem.get_end_to_end_min_time_lags(),
            tasks_n_offsets_max=self.problem.get_end_to_end_max_time_lags(),
            task1_start_or_end=StartOrEnd.END,
            task2_start_or_end=StartOrEnd.END,
        )
        self._create_timelag_constraints(
            tasks_n_offsets_min=self.problem.get_start_to_end_min_time_lags(),
            tasks_n_offsets_max=self.problem.get_start_to_end_max_time_lags(),
            task1_start_or_end=StartOrEnd.START,
            task2_start_or_end=StartOrEnd.END,
        )
