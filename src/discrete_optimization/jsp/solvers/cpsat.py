#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Adaptation of
#  https://github.com/erachelson/seq_dec_mak/blob/main/scheduling_newcourse/correction/nb2_jobshopsolver.py
import logging
from typing import Any

from ortools.sat.python.cp_model import CpModel, CpSolverSolutionCallback, LinearExprT

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat import (
    SchedulingCpSatSolver,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.jsp.problem import JobShopProblem, JobShopSolution, Task

logger = logging.getLogger(__name__)


class CpSatJspSolver(
    SchedulingCpSatSolver[Task],
    WarmstartMixin,
):
    problem: JobShopProblem

    def __init__(self, problem: JobShopProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables = {}

    def get_makespan_upper_bound(self) -> int:
        return self._max_time

    def init_model(self, **args: Any) -> None:
        super().init_model(**args)
        # dummy value, todo : compute a better bound
        max_time = args.get(
            "max_time",
            self.problem.get_makespan_upper_bound(),
        )
        self._max_time = max_time  # will be used by the makespan variable
        # Write variables, constraints
        starts = [
            [
                self.cp_model.NewIntVar(0, max_time, f"starts_{j, k}")
                for k in range(len(self.problem.list_jobs[j]))
            ]
            for j in range(self.problem.n_jobs)
        ]
        # Same idea for ends
        ends = [
            [
                self.cp_model.NewIntVar(0, max_time, f"ends_{j, k}")
                for k in range(len(self.problem.list_jobs[j]))
            ]
            for j in range(self.problem.n_jobs)
        ]
        # Create the interval variables
        intervals = [
            [
                self.cp_model.NewIntervalVar(
                    start=starts[j][k],
                    size=self.problem.list_jobs[j][k].processing_time,
                    end=ends[j][k],
                    name=f"task_{j, k}",
                )
                for k in range(len(self.problem.list_jobs[j]))
            ]
            for j in range(self.problem.n_jobs)
        ]
        # Precedence constraint between sub-parts of each job.
        for j in range(self.problem.n_jobs):
            for k in range(1, len(self.problem.list_jobs[j])):
                self.cp_model.Add(starts[j][k] >= ends[j][k - 1])
        # No overlap task on the same machine.
        for machine in self.problem.job_per_machines:
            self.cp_model.AddNoOverlap(
                [intervals[x[0]][x[1]] for x in self.problem.job_per_machines[machine]]
            )
        # Store the variables in some dictionaries.
        self.variables["starts"] = starts
        self.variables["ends"] = ends
        self.variables["intervals"] = intervals

        objective = self.get_global_makespan_variable()
        self.minimize_variable(objective)

    def set_warm_start(self, solution: JobShopSolution) -> None:
        self.cp_model.clear_hints()
        for job_index in range(len(solution.schedule)):
            for subjob_index in range(len(solution.schedule[job_index])):
                self.cp_model.AddHint(
                    self.variables["starts"][job_index][subjob_index],
                    solution.schedule[job_index][subjob_index][0],
                )
                self.cp_model.AddHint(
                    self.variables["ends"][job_index][subjob_index],
                    solution.schedule[job_index][subjob_index][1],
                )

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> JobShopSolution:
        logger.info(
            f"Objective ={cpsolvercb.ObjectiveValue()}, bound = {cpsolvercb.BestObjectiveBound()}"
        )
        schedule = []
        for job_index in range(len(self.variables["starts"])):
            sched_job = []
            for subjob_index in range(len(self.variables["starts"][job_index])):
                sched_job.append(
                    (
                        cpsolvercb.Value(
                            self.variables["starts"][job_index][subjob_index]
                        ),
                        cpsolvercb.Value(
                            self.variables["ends"][job_index][subjob_index]
                        ),
                    )
                )
            schedule.append(sched_job)
        return JobShopSolution(problem=self.problem, schedule=schedule)

    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExprT:
        if start_or_end == StartOrEnd.START:
            var_label = "starts"
        else:
            var_label = "ends"
        j, k = task
        return self.variables[var_label][j][k]
