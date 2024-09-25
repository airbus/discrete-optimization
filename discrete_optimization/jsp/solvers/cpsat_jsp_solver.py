#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Adaptation of
#  https://github.com/erachelson/seq_dec_mak/blob/main/scheduling_newcourse/correction/nb2_jobshopsolver.py
import logging
from typing import Any

from ortools.sat.python.cp_model import CpModel, CpSolverSolutionCallback, Domain

from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCPSatSolver
from discrete_optimization.jsp.job_shop_problem import JobShopProblem, SolutionJobshop

logger = logging.getLogger(__name__)


class CPSatJspSolver(OrtoolsCPSatSolver, WarmstartMixin):
    problem: JobShopProblem

    def __init__(self, problem: JobShopProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables = {}

    def init_model(self, **args: Any) -> None:
        self.cp_model = CpModel()
        # dummy value, todo : compute a better bound
        max_time = args.get(
            "max_time",
            sum(
                sum(subjob.processing_time for subjob in job)
                for job in self.problem.list_jobs
            ),
        )
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
        # Objective value variable
        makespan = self.cp_model.NewIntVar(0, max_time, name="makespan")
        self.cp_model.AddMaxEquality(makespan, [ends[i][-1] for i in range(len(ends))])
        self.cp_model.Minimize(makespan)
        # Store the variables in some dictionaries.
        self.variables["starts"] = starts
        self.variables["ends"] = ends
        self.variables["intervals"] = intervals

    def set_warm_start(self, solution: SolutionJobshop) -> None:
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
    ) -> SolutionJobshop:
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
        return SolutionJobshop(problem=self.problem, schedule=schedule)
