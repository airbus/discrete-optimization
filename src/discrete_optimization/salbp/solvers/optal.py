#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Iterable

from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    SchedulingOptalSolver,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

try:
    import optalcp as cp
except ImportError:
    cp = None
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.salbp.problem import (
    SalbpProblem,
    SalbpProblem_1_2,
    SalbpSolution,
    Task,
    calculate_salbp_lower_bounds,
)


class OptalSalbpSolver(SchedulingOptalSolver[Task], WarmstartMixin):
    hyperparameters = [
        CategoricalHyperparameter("use_lb", choices=[True, False], default=True)
    ]
    problem: SalbpProblem

    def __init__(
        self,
        problem: SalbpProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        self.init_model_scheduling(**kwargs)

    def init_model_scheduling(self, **kwargs: Any) -> None:
        self.cp_model = cp.Model()
        upper_bound = kwargs.get("upper_bound", self.problem.get_makespan_upper_bound())
        lb = None
        use_lb = kwargs["use_lb"]
        if use_lb:
            lb = calculate_salbp_lower_bounds(self.problem)
        intervals = {}
        for t in self.problem.tasks:
            intervals[t] = self.cp_model.interval_var(
                start=(0, upper_bound - 1),
                end=(0, upper_bound),
                length=1,
                name=f"intervals_{t}",
            )
        for t in self.problem.adj:
            for succ in self.problem.adj[t]:
                self.cp_model.start_before_start(intervals[t], intervals[succ])
        self.cp_model.enforce(
            self.cp_model.sum(
                [
                    self.cp_model.pulse(intervals[t], self.problem.task_times[t])
                    for t in self.problem.tasks
                ]
            )
            <= self.problem.cycle_time
        )
        self.variables["intervals"] = intervals
        if use_lb:
            makespan = self.cp_model.max(
                [self.cp_model.end(intervals[t]) for t in self.problem.get_last_tasks()]
            )
            self.cp_model.enforce(makespan <= upper_bound)
            self.cp_model.enforce(makespan >= lb)
        else:
            makespan = self.get_global_makespan_variable()
        self.cp_model.minimize(makespan)

    def retrieve_solution(self, result: "cp.SolveResult") -> Solution:
        allocation = [
            int(result.solution.get_start(self.get_task_interval_variable(task)))
            for task in self.problem.tasks
        ]
        return SalbpSolution(problem=self.problem, allocation_to_station=allocation)

    def get_task_interval_variable(self, task: Task) -> "cp.IntervalVar":
        return self.variables["intervals"][task]

    def set_warm_start(self, solution: SalbpSolution) -> None:
        self.use_warm_start = True
        self.warm_start_solution = cp.Solution()
        for t in self.problem.tasks:
            self.warm_start_solution.set_value(
                self.get_task_interval_variable(t),
                solution.get_start_time(t),
                solution.get_end_time(t),
            )
        nb_station = self.problem.evaluate(solution)["nb_stations"]
        self.warm_start_solution.set_objective(nb_station)


class OptalSalbp12Solver(SchedulingOptalSolver[Task], WarmstartMixin):
    problem: SalbpProblem_1_2

    def __init__(
        self,
        problem: SalbpProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.current_obj_str = None

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        self.init_model_scheduling(**kwargs)

    def init_model_scheduling(self, **kwargs: Any) -> None:
        self.cp_model = cp.Model()
        upper_bound = kwargs.get("upper_bound", self.problem.get_makespan_upper_bound())
        intervals = {}
        for t in self.problem.tasks:
            intervals[t] = self.cp_model.interval_var(
                start=(0, upper_bound - 1),
                end=(0, upper_bound),
                length=1,
                name=f"intervals_{t}",
            )
        for t in self.problem.adj:
            for succ in self.problem.adj[t]:
                self.cp_model.start_before_start(intervals[t], intervals[succ])
        task_times = [self.problem.task_times[t] for t in self.problem.tasks]
        max_cycle_time = sum(task_times)
        cycle_time = self.cp_model.int_var(
            min=min(task_times), max=max_cycle_time, name="variable_cycle_time"
        )
        self.cp_model.enforce(
            self.cp_model.sum(
                [
                    self.cp_model.pulse(intervals[t], self.problem.task_times[t])
                    for t in self.problem.tasks
                ]
                + [
                    self.cp_model.pulse(
                        self.cp_model.interval_var(
                            start=cp.IntervalMin, end=cp.IntervalMax, optional=False
                        ),
                        max_cycle_time - cycle_time,
                    )
                ]
            )
            <= max_cycle_time
        )
        self.variables["intervals"] = intervals
        makespan = self.get_global_makespan_variable()
        self.variables["objs"] = {}
        objs = []
        weights = []
        for obj, weight in zip(
            self.params_objective_function.objectives,
            self.params_objective_function.weights,
        ):
            if obj == "cycle_time":
                objs.append(cycle_time)
                weights.append(weight)
                self.variables["objs"][obj] = cycle_time
            if obj == "nb_stations":
                objs.append(makespan)
                weights.append(weight)
                self.variables["objs"][obj] = makespan
        self.cp_model.minimize(
            self.cp_model.sum([w * o for w, o in zip(objs, weights)])
        )

    def retrieve_solution(self, result: "cp.SolveResult") -> Solution:
        allocation = [
            int(result.solution.get_start(self.get_task_interval_variable(task)))
            for task in self.problem.tasks
        ]
        return SalbpSolution(problem=self.problem, allocation_to_station=allocation)

    def implements_lexico_api(self) -> bool:
        return True

    def get_lexico_objectives_available(self) -> list[str]:
        return ["cycle_time", "nb_stations"]

    def set_lexico_objective(self, obj: str) -> None:
        self.cp_model.minimize(self.variables["objs"][obj])
        self.current_obj_str = obj

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        sol = res[-1][0]
        kpis = self.problem.evaluate(sol)
        return kpis[obj]

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        self.cp_model.enforce(self.variables["objs"][obj] <= value)

    def get_task_interval_variable(self, task: Task) -> "cp.IntervalVar":
        return self.variables["intervals"][task]

    def set_warm_start(self, solution: SalbpSolution) -> None:
        self.use_warm_start = True
        self.warm_start_solution = cp.Solution()
        for t in self.problem.tasks:
            self.warm_start_solution.set_value(
                self.get_task_interval_variable(t),
                solution.get_start_time(t),
                solution.get_end_time(t),
            )
        if self.current_obj_str is not None:
            self.warm_start_solution.set_objective(
                self.problem.evaluate(solution)[self.current_obj_str]
            )
        else:
            self.warm_start_solution.set_objective(self.aggreg_from_sol(solution))
