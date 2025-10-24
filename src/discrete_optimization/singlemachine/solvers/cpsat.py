#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any

from ortools.sat.python.cp_model import (
    CpSolverSolutionCallback,
    LinearExpr,
    LinearExprT,
)

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat import (
    SchedulingCpSatSolver,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.singlemachine.problem import (
    Task,
    WeightedTardinessProblem,
    WTSolution,
)

logger = logging.getLogger(__name__)


class CpsatWTSolver(SchedulingCpSatSolver[Task], WarmstartMixin):
    problem: WeightedTardinessProblem
    variables: dict

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        schedule = []
        for i in range(self.problem.num_jobs):
            st = cpsolvercb.value(self.variables["starts"][i])
            end = st + self.problem.processing_times[i]
            schedule.append((st, end))
        logger.info(f"Obj = {cpsolvercb.objective_value}")
        return WTSolution(problem=self.problem, schedule=schedule)

    def init_model(self, **args: Any) -> None:
        self.variables = {"starts": [], "intervals": [], "lateness": []}
        super().init_model(**args)
        max_time = sum(self.problem.processing_times)
        for i in range(self.problem.num_jobs):
            start = self.cp_model.NewIntVar(lb=0, ub=max_time, name=f"start_{i}")
            interval = self.cp_model.NewFixedSizeIntervalVar(
                start=start, size=self.problem.processing_times[i], name=f"interval_{i}"
            )
            self.variables["starts"].append(start)
            self.variables["intervals"].append(interval)
            lateness = self.cp_model.NewIntVar(
                lb=0,
                ub=max(max_time - self.problem.due_dates[i], 0),
                name=f"lateness_{i}",
            )
            self.cp_model.AddMaxEquality(
                lateness,
                [
                    start
                    + self.problem.processing_times[i]
                    - self.problem.due_dates[i],
                    0,
                ],
            )
            self.variables["lateness"].append(lateness)
        self.cp_model.AddNoOverlap(self.variables["intervals"])
        self.cp_model.Minimize(
            LinearExpr.weighted_sum(self.variables["lateness"], self.problem.weights)
        )

    def set_warm_start(self, solution: WTSolution) -> None:
        self.cp_model.clear_hints()
        for i in range(self.problem.num_jobs):
            self.cp_model.add_hint(self.variables["starts"][i], solution.schedule[i][0])
            self.cp_model.add_hint(
                self.variables["lateness"][i],
                max(0, solution.schedule[i][1] - self.problem.due_dates[i]),
            )

    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExprT:
        if start_or_end == StartOrEnd.START:
            return self.variables["starts"][task]
        if start_or_end == StartOrEnd.END:
            return self.variables["starts"][task] + self.problem.processing_times[task]
