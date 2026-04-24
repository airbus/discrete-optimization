#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.precedence_scheduling import (
    PrecedenceSchedulingProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.scheduling import (
    SchedulingCpSatSolver,
)


class PrecedenceSchedulingCpSatSolver(SchedulingCpSatSolver[Task]):
    """Mixin for cpsat solvers dealing with scheduling problems with precedence constraints."""

    problem: PrecedenceSchedulingProblem[Task]

    def create_precedence_constraints(self):
        """Add precedence constraints to cp model."""
        for task1, successors in self.problem.get_precedence_constraints().items():
            for task2 in successors:
                self.cp_model.add(
                    self.get_task_start_or_end_variable(
                        task=task1, start_or_end=StartOrEnd.END
                    )
                    <= self.get_task_start_or_end_variable(
                        task=task2, start_or_end=StartOrEnd.START
                    )
                )
