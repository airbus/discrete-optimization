#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Any

from ortools.sat.python.cp_model import CpModel, CpSolverSolutionCallback, Domain

from discrete_optimization.fjsp.problem import FJobShopProblem, FJobShopSolution
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver

logger = logging.getLogger(__name__)


class CpSatFjspSolver(OrtoolsCpSatSolver, WarmstartMixin):
    hyperparameters = [
        CategoricalHyperparameter(
            name="duplicate_temporal_var", choices=[True, False], default=False
        ),
        CategoricalHyperparameter(
            name="add_cumulative_constraint", choices=[True, False], default=False
        ),
    ]
    problem: FJobShopProblem

    def __init__(self, problem: FJobShopProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables = {}

    def init_model(self, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        self.cp_model = CpModel()
        self.create_vars(**args)
        self.create_precedence_constraints()
        self.create_is_present_constraints()
        self.create_disjunctive_constraints(**args)
        max_time = args.get("max_time", self.problem.horizon)
        makespan = self.cp_model.NewIntVar(0, max_time, name="makespan")
        self.cp_model.AddMaxEquality(
            makespan,
            [
                self.variables["ends"][(i, len(self.problem.list_jobs[i].sub_jobs) - 1)]
                for i in range(self.problem.n_jobs)
            ],
        )
        self.variables["makespan"] = makespan
        self.cp_model.Minimize(makespan)

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        logger.info(
            f"Objective ={cpsolvercb.ObjectiveValue()}, bound = {cpsolvercb.BestObjectiveBound()}"
        )
        schedule = []
        for job_index in range(self.problem.n_jobs):
            sched_job = []
            for subjob_index in range(len(self.problem.list_jobs[job_index].sub_jobs)):
                start = cpsolvercb.Value(
                    self.variables["starts"][(job_index, subjob_index)]
                )
                end = cpsolvercb.Value(
                    self.variables["ends"][(job_index, subjob_index)]
                )
                machine = None
                if len(self.problem.list_jobs[job_index].sub_jobs[subjob_index]) == 1:
                    machine = (
                        self.problem.list_jobs[job_index]
                        .sub_jobs[subjob_index][0]
                        .machine_id
                    )
                else:
                    for k in self.variables["keys_per_subjob"][
                        (job_index, subjob_index)
                    ]:
                        if cpsolvercb.Value(self.variables["presents"][k]):
                            machine = self.variables["key_to_machine"][k]
                            break
                sched_job.append((start, end, machine))
            schedule.append(sched_job)
        return FJobShopSolution(problem=self.problem, schedule=schedule)

    def create_vars(self, **args):
        duplicate_temporal_var = args["duplicate_temporal_var"]
        max_time = args.get("max_time", self.problem.horizon)
        starts_var = {}
        ends_var = {}
        durations_var = {}
        intervals_var = {}

        is_present_var = {}
        opt_starts_var = {}
        opt_intervals_var = {}
        keys_per_machines = {m: set() for m in range(self.problem.n_machines)}
        keys_per_subjob = {}
        key_to_machine = {}
        for i in range(self.problem.n_jobs):
            lb = 0  # bit more advanced lb computation than just putting 0
            for sub_i in range(len(self.problem.list_jobs[i].sub_jobs)):
                options = self.problem.list_jobs[i].sub_jobs[sub_i]
                possible_durations = [opt.processing_time for opt in options]
                durations_var[(i, sub_i)] = self.cp_model.NewIntVarFromDomain(
                    domain=Domain.FromValues(possible_durations), name=f"dur_{i, sub_i}"
                )
                starts_var[(i, sub_i)] = self.cp_model.NewIntVar(
                    lb=lb, ub=max_time, name=f"start_{i, sub_i}"
                )
                ends_var[(i, sub_i)] = self.cp_model.NewIntVar(
                    lb=lb + min(possible_durations), ub=max_time, name=f"end_{i, sub_i}"
                )
                intervals_var[(i, sub_i)] = self.cp_model.NewIntervalVar(
                    start=starts_var[(i, sub_i)],
                    size=durations_var[(i, sub_i)],
                    end=ends_var[(i, sub_i)],
                    name=f"interval_{i, sub_i}",
                )
                if len(options) == 1:
                    # 1 alternative
                    opt_intervals_var[(i, sub_i, 0)] = intervals_var[(i, sub_i)]
                    keys_per_machines[options[0].machine_id].add((i, sub_i, 0))
                    key_to_machine[(i, sub_i, 0)] = options[0].machine_id
                else:
                    for opt_i in range(len(options)):
                        is_present_var[(i, sub_i, opt_i)] = self.cp_model.NewBoolVar(
                            name=f"is_present_{i, sub_i, opt_i}"
                        )
                        st = starts_var[(i, sub_i)]
                        end = ends_var[(i, sub_i)]
                        if duplicate_temporal_var:
                            opt_starts_var[(i, sub_i, opt_i)] = self.cp_model.NewIntVar(
                                lb=lb, ub=max_time, name=f"start_{i, sub_i, opt_i}"
                            )
                            opt_intervals_var[
                                (i, sub_i, opt_i)
                            ] = self.cp_model.NewOptionalFixedSizeIntervalVar(
                                start=opt_starts_var[(i, sub_i, opt_i)],
                                size=options[opt_i].processing_time,
                                is_present=is_present_var[(i, sub_i, opt_i)],
                                name=f"opt_interval_{i,sub_i,opt_i}",
                            )
                        else:
                            opt_intervals_var[
                                (i, sub_i, opt_i)
                            ] = self.cp_model.NewOptionalIntervalVar(
                                start=st,
                                size=options[opt_i].processing_time,
                                end=end,
                                is_present=is_present_var[(i, sub_i, opt_i)],
                                name=f"opt_interval_{i, sub_i, opt_i}",
                            )
                        keys_per_machines[options[opt_i].machine_id].add(
                            (i, sub_i, opt_i)
                        )
                        key_to_machine[(i, sub_i, opt_i)] = options[opt_i].machine_id
                keys_per_subjob[(i, sub_i)] = {
                    (i, sub_i, opt_i) for opt_i in range(len(options))
                }
                lb += min(possible_durations)
        self.variables = {
            "starts": starts_var,
            "ends": ends_var,
            "durations": durations_var,
            "intervals": intervals_var,
            "presents": is_present_var,
            "opt_starts": opt_starts_var,
            "opt_intervals": opt_intervals_var,
            "keys_per_machine": keys_per_machines,
            "keys_per_subjob": keys_per_subjob,
            "key_to_machine": key_to_machine,
        }

    def create_precedence_constraints(self):
        for i in range(self.problem.n_jobs):
            for j in range(1, len(self.problem.list_jobs[i].sub_jobs)):
                self.cp_model.Add(
                    self.variables["starts"][(i, j)]
                    >= self.variables["ends"][(i, j - 1)]
                )

    def create_is_present_constraints(self):
        for subjob in self.variables["keys_per_subjob"]:
            if len(self.variables["keys_per_subjob"][subjob]) <= 1:
                continue
            # One way of doing the subjob should be selected !
            self.cp_model.AddExactlyOne(
                [
                    self.variables["presents"][key]
                    for key in self.variables["keys_per_subjob"][subjob]
                ]
            )
            for key in self.variables["keys_per_subjob"][subjob]:
                self.cp_model.Add(
                    self.variables["durations"][subjob]
                    == self.problem.list_jobs[subjob[0]]
                    .sub_jobs[subjob[1]][key[2]]
                    .processing_time
                ).OnlyEnforceIf(self.variables["presents"][key])
                if key in self.variables["opt_starts"]:
                    self.cp_model.Add(
                        self.variables["opt_starts"][key]
                        == self.variables["starts"][subjob]
                    ).OnlyEnforceIf(self.variables["presents"][key])

    def create_disjunctive_constraints(self, **args):
        add_cumulative_constraint = args["add_cumulative_constraint"]
        for m in self.variables["keys_per_machine"]:
            self.cp_model.AddNoOverlap(
                [
                    self.variables["opt_intervals"][key]
                    for key in self.variables["keys_per_machine"][m]
                ]
            )
            if add_cumulative_constraint:
                self.cp_model.AddCumulative(
                    [
                        self.variables["opt_intervals"][key]
                        for key in self.variables["keys_per_machine"][m]
                    ],
                    [1 for _ in self.variables["keys_per_machine"][m]],
                    1,
                )

    def set_warm_start(self, solution: FJobShopSolution) -> None:
        self.cp_model.ClearHints()
        for job_index in range(len(solution.schedule)):
            for subjob_index in range(len(solution.schedule[job_index])):
                options = self.problem.list_jobs[job_index].sub_jobs[subjob_index]
                self.cp_model.AddHint(
                    self.variables["starts"][(job_index, subjob_index)],
                    solution.schedule[job_index][subjob_index][0],
                )
                self.cp_model.AddHint(
                    self.variables["ends"][(job_index, subjob_index)],
                    solution.schedule[job_index][subjob_index][1],
                )
                self.cp_model.AddHint(
                    self.variables["durations"][(job_index, subjob_index)],
                    solution.schedule[job_index][subjob_index][1]
                    - solution.schedule[job_index][subjob_index][0],
                )
                if len(options) > 1:
                    for opt_i in range(len(options)):
                        if (
                            options[opt_i].machine_id
                            == solution.schedule[job_index][subjob_index][2]
                        ):
                            self.cp_model.AddHint(
                                self.variables["presents"][
                                    (job_index, subjob_index, opt_i)
                                ],
                                1,
                            )
                        else:
                            self.cp_model.AddHint(
                                self.variables["presents"][
                                    (job_index, subjob_index, opt_i)
                                ],
                                0,
                            )
