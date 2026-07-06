#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.no_overlap import (
    NoOverlapProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.scheduling import (
    SchedulingCpSatSolver,
)


class NoOverlapCpSatSolver(SchedulingCpSatSolver[Task]):
    """Mixin for cpsat solvers dealing with scheduling problems
    with no overlap constraint between set of tasks"""

    problem: NoOverlapProblem[Task]

    def create_no_overlap_constraints(self):
        """Add no overlap constraints to cp model."""
        for tasks in self.problem.get_no_overlap():
            intervals = [self.get_task_interval(task) for task in tasks]
            self.cp_model.add_no_overlap(intervals)

    def create_forbidden_intervals_constraints(self):
        for task in self.problem.tasks_list:
            intervals = self.problem.get_forbidden_intervals(task)
            if len(intervals) > 0:
                self.cp_model.add_no_overlap(
                    [self.get_task_interval(task)]
                    + [
                        self.cp_model.new_interval_var(
                            start=start, end=end, size=end - start, name=""
                        )
                        for start, end in intervals
                    ]
                )
