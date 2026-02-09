#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Optional

from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    SchedulingOptalSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)

try:
    import optalcp as cp
except ImportError:
    cp = None
from discrete_optimization.jsp.problem import JobShopProblem, JobShopSolution, Task


class OptalJspSolver(SchedulingOptalSolver[Task]):
    """Solver for JSP using the OptalCP TypeScript API (fallback if Python API is not available)"""

    problem: JobShopProblem

    def __init__(
        self,
        problem: JobShopProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self._all_intervals = []

    def init_model(self, **kwargs):
        """Builds the OptalCP model for the JSP problem."""
        self.cp_model = cp.Model()
        nb_jobs = self.problem.n_jobs
        nb_machines = self.problem.n_machines
        # Placeholders for machine assignments and intervals
        machines = [[] for _ in range(nb_machines)]
        self._all_intervals = [[] for _ in range(nb_jobs)]
        ends = []
        for i, job in enumerate(self.problem.list_jobs):
            prev = None
            for j, subjob in enumerate(job):
                # Create an interval variable for each operation
                operation = self.cp_model.interval_var(
                    length=subjob.processing_time,
                    name=f"J{i + 1}O{j + 1}M{subjob.machine_id + 1}",
                )
                machines[subjob.machine_id].append(operation)
                self._all_intervals[i].append(operation)
                # Add precedence constraint with the previous operation in the same job
                if prev is not None:
                    self.cp_model.end_before_start(prev, operation)
                prev = operation
            if prev is not None:
                ends.append(prev.end())
        # Add no-overlap constraints for each machine
        for m in range(nb_machines):
            self.cp_model.no_overlap(machines[m])
        # Objective: minimize makespan (max of end times of last operations)
        self.cp_model.minimize(self.cp_model.max(ends))

    def get_task_interval_variable(self, task: Task) -> cp.IntervalVar:
        return self._all_intervals[task[0]][task[1]]

    def retrieve_solution(self, result: cp.SolveResult) -> Solution:
        schedule = []
        for i in range(self.problem.n_jobs):
            sched_i = []
            for k in range(len(self.problem.list_jobs[i])):
                sched_i.append(
                    result.solution.get_value(self.get_task_interval_variable((i, k)))
                )
            schedule.append(sched_i)
        return JobShopSolution(problem=self.problem, schedule=schedule)
