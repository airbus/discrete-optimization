#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import MinOrMax, StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat.scheduling import (
    SchedulingCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.timelag import TimelagProblem


class TimelagCpSatSolver(SchedulingCpSatSolver[Task]):
    """Mixin for cpsat solvers dealing with scheduling problems with time lag constraints."""

    problem: TimelagProblem[Task]

    def _create_timelag_constraints(
        self,
        task1_start_or_end: StartOrEnd,
        task2_start_or_end: StartOrEnd,
    ) -> None:
        # after normalization only offsets >=0 (min time lags) or >0 (max time lags)
        min_timelags = self.problem.get_consolidated_time_lags(
            task1_start_or_end=task1_start_or_end,
            task2_start_or_end=task2_start_or_end,
            min_or_max=MinOrMax.MIN,
        )
        max_timelags = self.problem.get_consolidated_time_lags(
            task1_start_or_end=task1_start_or_end,
            task2_start_or_end=task2_start_or_end,
            min_or_max=MinOrMax.MAX,
        )

        # equality constraints: two cases with consolidated time lags
        # - offset > 0: (t1, t2, offset) in min and max time lags
        # - offset==0: (t1, t2, 0) in min time lags and (t2, t1, 0) in reversed tasks min time lags
        # set computation
        equality_timelags_positive_offset = set(min_timelags).intersection(max_timelags)
        min_timelags_0_offset = [
            (t1, t2, 0) for (t1, t2, offset) in min_timelags if offset == 0
        ]
        max_timelags_0_offset = [
            (t1, t2, 0)
            for (t2, t1, offset) in self.problem.get_consolidated_time_lags(
                task1_start_or_end=task2_start_or_end,
                task2_start_or_end=task1_start_or_end,
                min_or_max=MinOrMax.MIN,
            )
            if offset == 0
        ]
        equality_timelags_0_offset = set()
        for t1, t2, _ in min_timelags_0_offset:
            if (t1, t2, 0) in max_timelags_0_offset and (
                # avoid adding twice an equality (start(t1) == start(t2) and start(t2) == start(t1))
                task1_start_or_end != task2_start_or_end
                or (t2, t1, 0) not in equality_timelags_0_offset
            ):
                equality_timelags_0_offset.add((t1, t2, 0))
        equality_timelags = equality_timelags_positive_offset.union(
            equality_timelags_0_offset
        )
        # add corresponding constraints to cp_model
        for task1, task2, offset in equality_timelags:
            self.cp_model.add(
                self.get_task_start_or_end_variable(
                    task=task1, start_or_end=task1_start_or_end
                )
                + offset
                == self.get_task_start_or_end_variable(
                    task=task2, start_or_end=task2_start_or_end
                )
            )

        # min only constraints
        min_only_timelags = set(min_timelags).difference(
            set(max_timelags).union(max_timelags_0_offset)
        )
        for task1, task2, offset in min_only_timelags:
            self.cp_model.add(
                self.get_task_start_or_end_variable(
                    task=task1, start_or_end=task1_start_or_end
                )
                + offset
                <= self.get_task_start_or_end_variable(
                    task=task2, start_or_end=task2_start_or_end
                )
            )
        # max only constraints
        max_only_timelags = set(max_timelags).difference(min_timelags)
        for task1, task2, offset in max_only_timelags:
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
        for task1_start_or_end in StartOrEnd:
            for task2_start_or_end in StartOrEnd:
                if (task1_start_or_end, task2_start_or_end) == (
                    StartOrEnd.START,
                    StartOrEnd.END,
                ):
                    # already covered by (StartOrEnd.END, StartOrEnd.START)
                    continue
                self._create_timelag_constraints(
                    task1_start_or_end=task1_start_or_end,
                    task2_start_or_end=task2_start_or_end,
                )
