#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from enum import Enum
from typing import Any

from ortools.sat.python.cp_model import CpSolverSolutionCallback, LinearExprT

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat import (
    AllocationCpSatSolver,
    SchedulingCpSatSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.salbp.problem import (
    Resource,
    SalbpProblem,
    SalbpSolution,
    Task,
)


class ModelingCpsatSalbp(Enum):
    SCHEDULING = 0
    BINARY = 1


class CpSatSalbpSolver(
    AllocationCpSatSolver[Task, Resource], SchedulingCpSatSolver[Task]
):
    problem: SalbpProblem
    hyperparameters = [
        EnumHyperparameter(
            name="modeling",
            enum=ModelingCpsatSalbp,
            default=ModelingCpsatSalbp.SCHEDULING,
        )
    ]

    def __init__(
        self,
        problem: SalbpProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.modeling: ModelingCpsatSalbp = None

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        super().init_model(**kwargs)
        self.modeling = kwargs["modeling"]
        if self.modeling == ModelingCpsatSalbp.SCHEDULING:
            self.init_model_scheduling(**kwargs)
        if self.modeling == ModelingCpsatSalbp.BINARY:
            self.init_model_binary(**kwargs)

    def init_model_scheduling(self, **kwargs: Any) -> None:
        upper_bound = kwargs.get("upper_bound", self.problem.get_makespan_upper_bound())
        starts = {}
        intervals = {}
        for t in self.problem.tasks:
            starts[t] = self.cp_model.new_int_var(
                lb=0, ub=upper_bound, name=f"start_{t}"
            )
            intervals[t] = self.cp_model.new_fixed_size_interval_var(
                start=starts[t], size=1, name=f"intervals_{t}"
            )
        for t in self.problem.adj:
            for succ in self.problem.adj[t]:
                self.cp_model.add(starts[succ] >= starts[t])
        self.cp_model.add_cumulative(
            [intervals[t] for t in self.problem.tasks],
            [self.problem.task_times[t] for t in self.problem.tasks],
            self.problem.cycle_time,
        )
        self.variables["starts"] = starts
        self.variables["intervals"] = intervals
        makespan = self.get_global_makespan_variable()
        self.cp_model.minimize(makespan)

    def init_model_binary(self, **kwargs: Any) -> None:
        upper_bound = kwargs.get("upper_bound", self.problem.get_makespan_upper_bound())
        binary_alloc = {}
        stations = {}
        for station in range(0, upper_bound):
            for t in self.problem.tasks:
                binary_alloc[(station, t)] = self.cp_model.new_bool_var(
                    name=f"alloc_{t}_{station}"
                )
            self.cp_model.add(
                sum(
                    binary_alloc[(station, t)] * self.problem.task_times[t]
                    for t in self.problem.tasks
                )
                <= self.problem.cycle_time
            )

        for t in self.problem.tasks:
            stations[t] = sum(
                [
                    binary_alloc[(station, t)] * station
                    for station in range(0, upper_bound)
                ]
            )
            self.cp_model.add_exactly_one(
                [binary_alloc[(station, t)] for station in range(0, upper_bound)]
            )
        for t in self.problem.adj:
            for succ in self.problem.adj[t]:
                self.cp_model.add(stations[succ] >= stations[t])
        self.variables["stations"] = stations
        self.variables["binary_alloc"] = binary_alloc
        max_station = self.cp_model.new_int_var(
            lb=0, ub=upper_bound, name="max_station"
        )
        self.cp_model.add_max_equality(max_station, [stations[t] for t in stations])
        self.cp_model.minimize(max_station)

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExprT:
        if self.modeling == ModelingCpsatSalbp.SCHEDULING:
            raise NotImplementedError
        if self.modeling == ModelingCpsatSalbp.BINARY:
            return self.variables["binary_alloc"][(unary_resource, task)]
        return None

    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExprT:
        if self.modeling == ModelingCpsatSalbp.SCHEDULING:
            if start_or_end == StartOrEnd.START:
                return self.variables["starts"][task]
            else:
                return self.variables["starts"][task] + 1
        if self.modeling == ModelingCpsatSalbp.BINARY:
            if start_or_end == StartOrEnd.START:
                return self.variables["stations"][task]
            else:
                return self.variables["stations"][task] + 1
        return None

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        allocation_to_station = [
            cpsolvercb.value(
                self.get_task_start_or_end_variable(
                    task=t, start_or_end=StartOrEnd.START
                )
            )
            for t in self.problem.tasks
        ]
        return SalbpSolution(
            problem=self.problem, allocation_to_station=allocation_to_station
        )
