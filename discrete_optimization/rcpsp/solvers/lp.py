#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable
from itertools import product
from typing import Any, Optional, Union

from ortools.math_opt.python import mathopt

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.lp_tools import (
    CplexMilpSolver,
    GurobiMilpSolver,
    MilpSolver,
    OrtoolsMathOptMilpSolver,
    VariableType,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solution import PartialSolution, RcpspSolution
from discrete_optimization.rcpsp.solvers import RcpspSolver
from discrete_optimization.rcpsp.solvers.pile import (
    GreedyChoice,
    PileCalendarRcpspSolver,
    PileRcpspSolver,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    import gurobipy as gurobi

try:
    import docplex
except ImportError:
    cplex_available = False
else:
    cplex_available = True
    import docplex.mp.dvar as cplex_var
    import docplex.mp.model as cplex


logger = logging.getLogger(__name__)


class _BaseLpRcpspSolver(MilpSolver, RcpspSolver):
    problem: RcpspProblem
    hyperparameters = [
        CategoricalHyperparameter(
            name="greedy_start", choices=[True, False], default=True
        ),
    ]

    def __init__(
        self,
        problem: RcpspProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        if problem.is_rcpsp_multimode():
            raise ValueError("this solver is meant for single mode problems")

        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            **kwargs,
        )
        self.variable_decision = {}
        self.constraints_dict = {"lns": []}
        self.start_solution: Optional[RcpspSolution] = None

    def init_model(self, **kwargs):
        greedy_start = kwargs.get("greedy_start", True)
        start_solution = kwargs.get("start_solution", None)
        if start_solution is None:
            if greedy_start:
                logger.info("Computing greedy solution")
                greedy_solver = PileRcpspSolver(self.problem)
                store_solution = greedy_solver.solve(
                    greedy_choice=GreedyChoice.MOST_SUCCESSORS
                )
                self.start_solution = store_solution.get_best_solution_fit()[0]
                makespan = self.problem.evaluate(self.start_solution)["makespan"]
            else:
                logger.info("Get dummy solution")
                solution = self.problem.get_dummy_solution()
                self.start_solution = solution
                makespan = self.problem.evaluate(solution)["makespan"]
        else:
            self.start_solution = start_solution
            makespan = self.problem.evaluate(start_solution)["makespan"]
        sorted_tasks = self.problem.tasks_list
        resources = self.problem.resources_list
        p = [int(self.problem.mode_details[key][1]["duration"]) for key in sorted_tasks]
        u = [
            [self.problem.mode_details[t][1].get(r, 0) for r in resources]
            for t in sorted_tasks
        ]
        c = [self.problem.resources[r] for r in resources]
        S = []
        logger.debug(f"successors: {self.problem.successors}")
        for task in sorted_tasks:
            for suc in self.problem.successors[task]:
                S.append([task, suc])
        # we have a better self.T to limit the number of variables :
        self.index_time = range(int(makespan + 1))
        self.model = self.create_empty_model()
        self.x: list[list[VariableType]] = [
            [self.add_binary_variable(name=f"x({task},{t})") for t in self.index_time]
            for task in sorted_tasks
        ]
        self.index_in_var = {
            t: self.problem.return_index_task(task=t, offset=0) for t in sorted_tasks
        }
        # set objective
        self.set_model_objective(
            self.construct_linear_sum(
                self.x[self.index_in_var[self.problem.sink_task]][t] * t
                for t in self.index_time
            ),
            minimize=True,
        )
        self.index_task = range(self.problem.n_jobs)
        self.index_resource = range(len(resources))
        for task in self.index_task:
            self.add_linear_constraint(
                self.construct_linear_sum(self.x[task][t] for t in self.index_time) == 1
            )

        for (r, t) in product(self.index_resource, self.index_time):
            self.add_linear_constraint(
                self.construct_linear_sum(
                    u[j][r] * self.x[j][t2]
                    for j in self.index_task
                    for t2 in range(max(0, t - p[j] + 1), t + 1)
                )
                <= c[r]
            )

        for (j, s) in S:
            self.add_linear_constraint(
                self.construct_linear_sum(
                    t * self.x[self.index_in_var[s]][t]
                    - t * self.x[self.index_in_var[j]][t]
                    for t in self.index_time
                )
                >= p[self.index_in_var[j]]
            )
        p_s: Optional[PartialSolution] = kwargs.get("partial_solution", None)
        self.partial_solution = p_s
        self.constraints_partial_solutions = []
        if p_s is not None:
            constraints = []
            if p_s.start_times is not None:
                for task in p_s.start_times:
                    constraints += [
                        self.add_linear_constraint(
                            self.construct_linear_sum(
                                [
                                    j * self.x[self.index_in_var[task]][j]
                                    for j in self.index_time
                                ]
                            )
                            == p_s.start_times[task]
                        )
                    ]
                    constraints += [
                        self.add_linear_constraint(
                            self.x[self.index_in_var[task]][p_s.start_times[task]] == 1
                        )
                    ]

            if p_s.partial_permutation is not None:
                for t1, t2 in zip(
                    p_s.partial_permutation[:-1], p_s.partial_permutation[1:]
                ):
                    constraints += [
                        self.add_linear_constraint(
                            self.construct_linear_sum(
                                [
                                    t * self.x[self.index_in_var[t1]][t]
                                    - t * self.x[self.index_in_var[t2]][t]
                                    for t in self.index_time
                                ]
                            )
                            <= 0
                        )
                    ]
            if p_s.list_partial_order is not None:
                for l in p_s.list_partial_order:
                    for t1, t2 in zip(l[:-1], l[1:]):
                        constraints += [
                            self.add_linear_constraint(
                                self.construct_linear_sum(
                                    [
                                        t * self.x[self.index_in_var[t1]][t]
                                        - t * self.x[self.index_in_var[t2]][t]
                                        for t in self.index_time
                                    ]
                                )
                                <= 0
                            )
                        ]
            self.starts = {}
            for j in self.index_task:
                self.starts[j] = self.add_continuous_variable(
                    name="start_" + str(j), lb=0, ub=makespan
                )
                self.add_linear_constraint(
                    self.construct_linear_sum(t * self.x[j][t] for t in self.index_time)
                    == self.starts[j]
                )
            if p_s.start_at_end is not None:
                for i, j in p_s.start_at_end:
                    constraints += [
                        self.add_linear_constraint(
                            self.starts[self.index_in_var[j]]
                            == self.starts[self.index_in_var[i]]
                            + p[self.index_in_var[i]]
                        )
                    ]
            if p_s.start_together is not None:
                for i, j in p_s.start_together:
                    constraints += [
                        self.add_linear_constraint(
                            self.starts[self.index_in_var[j]]
                            == self.starts[self.index_in_var[i]]
                        )
                    ]
            if p_s.start_after_nunit is not None:
                for t1, t2, delta in p_s.start_after_nunit:
                    constraints += [
                        self.add_linear_constraint(
                            self.starts[self.index_in_var[t2]]
                            >= self.starts[self.index_in_var[t1]] + delta
                        )
                    ]
            if p_s.start_at_end_plus_offset is not None:
                for t1, t2, delta in p_s.start_at_end_plus_offset:
                    constraints += [
                        self.add_linear_constraint(
                            self.starts[self.index_in_var[t2]]
                            >= self.starts[self.index_in_var[t1]]
                            + delta
                            + p[self.index_in_var[t1]]
                        )
                    ]
            self.constraints_partial_solutions = constraints
        # take into account "warmstart" w/o calling set_warmstart (would cause a recursion issue here)
        self.set_warm_start(self.start_solution)

    def convert_to_variable_values(
        self, solution: RcpspSolution
    ) -> dict[VariableType, float]:
        hinted_values: dict[VariableType, float] = {}
        for j in self.index_task:
            for t in self.index_time:
                if (
                    self.start_solution.rcpsp_schedule[self.problem.tasks_list[j]][
                        "start_time"
                    ]
                    == t
                ):
                    hinted_values[self.x[j][t]] = 1
                else:
                    hinted_values[self.x[j][t]] = 0
        return hinted_values

    def set_warm_start(self, solution: Solution) -> None:
        """Make the solver warm start from the given solution.

        Implemented by OrtoolsMathOptMilpSolver or GurobiMilpSolver.

        """
        raise NotImplementedError()

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> RcpspSolution:
        rcpsp_schedule = {}
        for (task_index, time) in product(self.index_task, self.index_time):
            value = get_var_value_for_current_solution(self.x[task_index][time])
            if value >= 0.5:
                task = self.problem.tasks_list[task_index]
                rcpsp_schedule[task] = {
                    "start_time": time,
                    "end_time": time + self.problem.mode_details[task][1]["duration"],
                }
        logger.debug(f"Size schedule : {len(rcpsp_schedule.keys())}")
        return RcpspSolution(
            problem=self.problem,
            rcpsp_schedule=rcpsp_schedule,
            rcpsp_schedule_feasible=True,
        )


class MathOptRcpspSolver(OrtoolsMathOptMilpSolver, _BaseLpRcpspSolver):
    hyperparameters = _BaseLpRcpspSolver.hyperparameters
    problem: RcpspProblem

    def convert_to_variable_values(
        self, solution: RcpspSolution
    ) -> dict[mathopt.Variable, float]:
        return _BaseLpRcpspSolver.convert_to_variable_values(self, solution=solution)


class _BaseLpMultimodeRcpspSolver(MilpSolver, RcpspSolver):
    problem: RcpspProblem
    hyperparameters = [
        CategoricalHyperparameter(
            name="greedy_start", choices=[True, False], default=True
        )
    ]

    x: dict[tuple[Hashable, int, int], VariableType]
    max_horizon: Optional[int] = None
    partial_solution: Optional[PartialSolution] = None

    def __init__(
        self,
        problem: RcpspProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.variable_decision = {}
        self.constraints_dict = {"lns": []}

    def init_model(self, **args):
        args = self.complete_with_default_hyperparameters(args)
        greedy_start = args["greedy_start"]
        start_solution = args.get("start_solution", None)
        max_horizon = args.get("max_horizon", None)
        self.max_horizon = max_horizon
        if start_solution is None:
            if greedy_start:
                logger.info("Computing greedy solution")
                if self.problem.is_varying_resource():
                    greedy_solver = PileCalendarRcpspSolver(self.problem)
                else:
                    greedy_solver = PileRcpspSolver(self.problem)
                store_solution = greedy_solver.solve(
                    greedy_choice=GreedyChoice.MOST_SUCCESSORS
                )
                self.start_solution = store_solution.get_best_solution_fit()[0]
                makespan = self.problem.evaluate(self.start_solution)["makespan"]
            else:
                logger.info("Get dummy solution")
                solution = self.problem.get_dummy_solution()
                self.start_solution = solution
                makespan = self.problem.evaluate(solution)["makespan"]
        else:
            self.start_solution = start_solution
            makespan = self.problem.evaluate(start_solution)["makespan"]
        sorted_tasks = self.problem.tasks_list
        resources = self.problem.resources_list
        p = [
            int(
                max(
                    [
                        self.problem.mode_details[key][mode]["duration"]
                        for mode in self.problem.mode_details[key]
                    ]
                )
            )
            for key in sorted_tasks
        ]
        renewable = {
            r: self.problem.resources[r]
            for r in self.problem.resources
            if r not in self.problem.non_renewable_resources
        }
        non_renewable = {
            r: self.problem.resources[r] for r in self.problem.non_renewable_resources
        }
        S = []
        logger.debug(f"successors: {self.problem.successors}")
        for task in sorted_tasks:
            for suc in self.problem.successors[task]:
                S.append([task, suc])
        self.index_time = list(range(sum(p)))
        # we have a better self.T to limit the number of variables :
        if self.start_solution.rcpsp_schedule_feasible:
            self.index_time = list(range(int(makespan + 1)))
        if max_horizon is not None:
            self.index_time = list(range(max_horizon + 1))
        self.model = self.create_empty_model(name="MRCPSP")
        self.x: dict[tuple[Hashable, int, int], VariableType] = {}
        last_task = self.problem.sink_task
        variable_per_task = {}
        keys_for_t = {}
        for task in sorted_tasks:
            if task not in variable_per_task:
                variable_per_task[task] = []
            for mode in self.problem.mode_details[task]:
                for t in self.index_time:
                    self.x[(task, mode, t)] = self.add_binary_variable(
                        name=f"x({task},{mode}, {t})",
                    )
                    for tt in range(
                        t, t + self.problem.mode_details[task][mode]["duration"]
                    ):
                        if tt not in keys_for_t:
                            keys_for_t[tt] = set()
                        keys_for_t[tt].add((task, mode, t))
                    variable_per_task[task] += [(task, mode, t)]
        self.set_model_objective(
            self.construct_linear_sum(
                self.x[key] * key[2] for key in variable_per_task[last_task]
            ),
            minimize=True,
        )
        for j in variable_per_task:
            self.add_linear_constraint(
                self.construct_linear_sum(self.x[key] for key in variable_per_task[j])
                == 1
            )

        if self.problem.is_varying_resource():
            renewable_quantity = {r: renewable[r] for r in renewable}
        else:
            renewable_quantity = {
                r: [renewable[r]] * len(self.index_time) for r in renewable
            }

        if self.problem.is_varying_resource():
            non_renewable_quantity = {r: non_renewable[r] for r in non_renewable}
        else:
            non_renewable_quantity = {
                r: [non_renewable[r]] * len(self.index_time) for r in non_renewable
            }

        for (r, t) in product(renewable, self.index_time):
            self.add_linear_constraint(
                self.construct_linear_sum(
                    int(self.problem.mode_details[key[0]][key[1]][r]) * self.x[key]
                    for key in keys_for_t[t]
                )
                <= renewable_quantity[r][t]
            )

        for r in non_renewable:
            self.add_linear_constraint(
                self.construct_linear_sum(
                    int(self.problem.mode_details[key[0]][key[1]][r]) * self.x[key]
                    for key in self.x
                )
                <= non_renewable_quantity[r][0]
            )
        durations = {
            j: self.add_integer_variable(name="duration_" + str(j))
            for j in variable_per_task
        }
        self.durations = durations
        self.variable_per_task = variable_per_task
        for j in variable_per_task:
            self.add_linear_constraint(
                self.construct_linear_sum(
                    self.problem.mode_details[key[0]][key[1]]["duration"] * self.x[key]
                    for key in variable_per_task[j]
                )
                == durations[j]
            )
        for (j, s) in S:
            self.add_linear_constraint(
                self.construct_linear_sum(
                    key[2] * self.x[key] for key in variable_per_task[s]
                )
                - self.construct_linear_sum(
                    key[2] * self.x[key] for key in variable_per_task[j]
                )
                >= durations[j]
            )

        self.starts = {}
        for task in sorted_tasks:
            self.starts[task] = self.add_integer_variable(
                name=f"start({task})",
                lb=0,
                ub=self.index_time[-1],
            )
            self.add_linear_constraint(
                self.construct_linear_sum(
                    [self.x[key] * key[2] for key in variable_per_task[task]]
                )
                == self.starts[task]
            )

        p_s: Optional[PartialSolution] = args.get("partial_solution", None)
        self.partial_solution = p_s
        self.constraints_partial_solutions = []
        if p_s is not None:
            constraints = []
            if p_s.start_times is not None:
                constraints = [
                    self.add_linear_constraint(
                        self.construct_linear_sum(
                            self.x[k]
                            for k in self.variable_per_task[task]
                            if k[2] == p_s.start_times[task]
                        )
                        == 1
                    )
                    for task in p_s.start_times
                ]

            if p_s.partial_permutation is not None:
                for t1, t2 in zip(
                    p_s.partial_permutation[:-1], p_s.partial_permutation[1:]
                ):
                    constraints += [
                        self.add_linear_constraint(
                            self.construct_linear_sum(
                                [key[2] * self.x[key] for key in variable_per_task[t1]]
                                + [
                                    -key[2] * self.x[key]
                                    for key in variable_per_task[t2]
                                ]
                            )
                            <= 0
                        )
                    ]
            if p_s.list_partial_order is not None:
                for l in p_s.list_partial_order:
                    for t1, t2 in zip(l[:-1], l[1:]):
                        constraints += [
                            self.add_linear_constraint(
                                self.construct_linear_sum(
                                    [
                                        key[2] * self.x[key]
                                        for key in variable_per_task[t1]
                                    ]
                                    + [
                                        -key[2] * self.x[key]
                                        for key in variable_per_task[t2]
                                    ]
                                )
                                <= 0
                            )
                        ]
            if p_s.start_at_end is not None:
                for i, j in p_s.start_at_end:
                    constraints += [
                        self.add_linear_constraint(
                            self.starts[j] == self.starts[i] + durations[i]
                        )
                    ]
            if p_s.start_together is not None:
                for i, j in p_s.start_together:
                    constraints += [
                        self.add_linear_constraint(self.starts[j] == self.starts[i])
                    ]
            if p_s.start_after_nunit is not None:
                for t1, t2, delta in p_s.start_after_nunit:
                    constraints += [
                        self.add_linear_constraint(
                            self.starts[t2] >= self.starts[t1] + delta
                        )
                    ]
            if p_s.start_at_end_plus_offset is not None:
                for t1, t2, delta in p_s.start_at_end_plus_offset:
                    constraints += [
                        self.add_linear_constraint(
                            self.starts[t2] >= self.starts[t1] + delta + durations[t1]
                        )
                    ]
            self.constraints_partial_solutions = constraints
            logger.debug(
                f"Partial solution constraints : {self.constraints_partial_solutions}"
            )

        self.update_model()

        # take into account "warmstart" w/o calling set_warmstart (would cause a recursion issue here)
        self.set_warm_start(self.start_solution)

    def set_warm_start(self, solution: Solution) -> None:
        """Make the solver warm start from the given solution.

        Implemented by OrtoolsMathOptMilpSolver or GurobiMilpSolver.

        """
        raise NotImplementedError()

    def update_model(self) -> None:
        """Update model (especially for gurobi).

        It ensures the well defintion of variables.

        """
        ...

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> RcpspSolution:
        rcpsp_schedule = {}
        modes: dict[Hashable, Union[str, int]] = {}
        for (task, mode, t), x in self.x.items():
            value = get_var_value_for_current_solution(x)
            if value >= 0.5:
                rcpsp_schedule[task] = {
                    "start_time": t,
                    "end_time": t + self.problem.mode_details[task][mode]["duration"],
                }
                modes[task] = mode
        logger.debug(f"Size schedule : {len(rcpsp_schedule.keys())}")
        modes_vec = [modes[k] for k in self.problem.tasks_list_non_dummy]
        return RcpspSolution(
            problem=self.problem,
            rcpsp_schedule=rcpsp_schedule,
            rcpsp_modes=modes_vec,
            rcpsp_schedule_feasible=True,
        )

    def convert_to_variable_values(
        self, solution: RcpspSolution
    ) -> dict[VariableType, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().

        """
        hinted_values: dict[VariableType, float] = {}
        for task in self.problem.tasks_list:
            if task in solution.rcpsp_schedule:
                hinted_values[self.starts[task]] = solution.rcpsp_schedule[task][
                    "start_time"
                ]
        modes_dict = self.problem.build_mode_dict(solution.rcpsp_modes)
        for j in solution.rcpsp_schedule:
            start_time_j = solution.rcpsp_schedule[j]["start_time"]
            hinted_values[self.durations[j]] = self.problem.mode_details[j][
                modes_dict[j]
            ]["duration"]
            for k in self.variable_per_task[j]:
                task, mode, time = k
                if start_time_j == time and mode == modes_dict[j]:
                    hinted_values[self.x[k]] = 1
                else:
                    hinted_values[self.x[k]] = 0

        return hinted_values


class GurobiMultimodeRcpspSolver(GurobiMilpSolver, _BaseLpMultimodeRcpspSolver):
    hyperparameters = _BaseLpMultimodeRcpspSolver.hyperparameters

    def update_model(self) -> None:
        """Update model (especially for gurobi).

        It ensures the well defintion of variables.

        """
        self.model.update()

    def convert_to_variable_values(
        self, solution: RcpspSolution
    ) -> dict[gurobipy.Var, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().

        """
        return _BaseLpMultimodeRcpspSolver.convert_to_variable_values(self, solution)


class MathOptMultimodeRcpspSolver(
    OrtoolsMathOptMilpSolver, _BaseLpMultimodeRcpspSolver
):
    hyperparameters = _BaseLpMultimodeRcpspSolver.hyperparameters

    max_horizon: Optional[int] = None
    partial_solution: Optional[PartialSolution] = None

    def convert_to_variable_values(
        self, solution: RcpspSolution
    ) -> dict[mathopt.Variable, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().

        """
        return _BaseLpMultimodeRcpspSolver.convert_to_variable_values(self, solution)


class CplexMultimodeRcpspSolver(CplexMilpSolver, _BaseLpMultimodeRcpspSolver):
    hyperparameters = _BaseLpMultimodeRcpspSolver.hyperparameters

    def init_model(self, **args):
        greedy_start = args.get("greedy_start", True)
        start_solution = args.get("start_solution", None)
        max_horizon = args.get("max_horizon", None)
        if start_solution is None:
            if greedy_start:
                logger.info("Computing greedy solution")
                if self.problem.is_varying_resource():
                    greedy_solver = PileCalendarRcpspSolver(self.problem)
                else:
                    greedy_solver = PileRcpspSolver(self.problem)
                store_solution = greedy_solver.solve(
                    greedy_choice=GreedyChoice.MOST_SUCCESSORS
                )
                self.start_solution = store_solution.get_best_solution_fit()[0]
                makespan = self.problem.evaluate(self.start_solution)["makespan"]
            else:
                logger.info("Get dummy solution")
                solution = self.problem.get_dummy_solution()
                self.start_solution = solution
                makespan = self.problem.evaluate(solution)["makespan"]
        else:
            self.start_solution = start_solution
            makespan = self.problem.evaluate(start_solution)["makespan"]
        sorted_tasks = self.problem.tasks_list
        renewable = {
            r: self.problem.resources[r]
            for r in self.problem.resources_list
            if r not in self.problem.non_renewable_resources
        }
        non_renewable = {
            r: self.problem.resources[r] for r in self.problem.non_renewable_resources
        }
        S = []
        logger.debug(f"successors: {self.problem.successors}")
        for task in sorted_tasks:
            for suc in self.problem.successors[task]:
                S.append([task, suc])
        if self.start_solution.rcpsp_schedule_feasible:
            self.index_time = list(range(int(makespan + 1)))
        if max_horizon is not None:
            self.index_time = list(range(max_horizon + 1))
        self.model: "cplex.Model" = cplex.Model("MRCPSP")
        self.x: dict[tuple[Hashable, int, int], "cplex_var.Var"] = {}
        last_task = self.problem.sink_task
        variable_per_task = {}
        keys_for_t = {}
        for task in sorted_tasks:
            if task not in variable_per_task:
                variable_per_task[task] = []
            for mode in self.problem.mode_details[task]:
                for t in self.index_time:
                    self.x[(task, mode, t)] = self.model.binary_var(
                        name=f"x({task},{mode}, {t})",
                    )
                    for tt in range(
                        t, t + self.problem.mode_details[task][mode]["duration"]
                    ):
                        if tt not in keys_for_t:
                            keys_for_t[tt] = set()
                        keys_for_t[tt].add((task, mode, t))
                    variable_per_task[task] += [(task, mode, t)]
        self.model.minimize(
            self.model.sum(self.x[key] * key[2] for key in variable_per_task[last_task])
        )

        # Only one (mode, t) switched on per task, which means the task starts at time=t with a given mode.
        self.model.add_constraints(
            (
                self.model.sum(self.x[key] for key in variable_per_task[j]) == 1
                for j in variable_per_task
            )
        )

        if self.problem.is_varying_resource():
            renewable_quantity = {r: renewable[r] for r in renewable}
        else:
            renewable_quantity = {
                r: [renewable[r]] * len(self.index_time) for r in renewable
            }

        if self.problem.is_varying_resource():
            non_renewable_quantity = {r: non_renewable[r] for r in non_renewable}
        else:
            non_renewable_quantity = {
                r: [non_renewable[r]] * len(self.index_time) for r in non_renewable
            }

        # Cumulative constraint for renewable resource.
        self.model.add_constraints(
            (
                self.model.sum(
                    int(self.problem.mode_details[key[0]][key[1]].get(r, 0))
                    * self.x[key]
                    for key in keys_for_t[t]
                )
                <= renewable_quantity[r][t]
                for (r, t) in product(renewable, self.index_time)
            )
        )

        self.model.add_constraints(
            (
                self.model.sum(
                    int(self.problem.mode_details[key[0]][key[1]][r]) * self.x[key]
                    for key in self.x
                )
                <= non_renewable_quantity[r][0]
                for r in non_renewable
            )
        )
        if self.problem.is_rcpsp_multimode():
            durations = {
                j: self.model.integer_var(
                    lb=min(
                        [
                            self.problem.mode_details[j][m]["duration"]
                            for m in self.problem.mode_details[j]
                        ]
                    ),
                    ub=max(
                        [
                            self.problem.mode_details[j][m]["duration"]
                            for m in self.problem.mode_details[j]
                        ]
                    ),
                    name="duration_" + str(j),
                )
                for j in variable_per_task
            }
            self.model.add_constraints(
                (
                    self.model.sum(
                        self.problem.mode_details[key[0]][key[1]]["duration"]
                        * self.x[key]
                        for key in variable_per_task[j]
                    )
                    == durations[j]
                    for j in variable_per_task
                )
            )
        else:
            durations = {
                j: self.problem.mode_details[j][1]["duration"]
                for j in variable_per_task
            }
        self.durations = durations
        self.variable_per_task = variable_per_task
        # Precedence constraints.
        self.model.add_constraints(
            (
                self.model.sum(
                    [key[2] * self.x[key] for key in variable_per_task[s]]
                    + [-key[2] * self.x[key] for key in variable_per_task[j]]
                )
                >= durations[j]
                for (j, s) in S
            )
        )

        self.starts = {}
        for task in sorted_tasks:
            self.starts[task] = self.model.integer_var(
                lb=0, ub=self.index_time[-1], name=f"start({task})"
            )
            self.model.add_constraint(
                self.model.sum(
                    [self.x[key] * key[2] for key in variable_per_task[task]]
                )
                == self.starts[task]
            )
        modes_dict = self.problem.build_mode_dict(self.start_solution.rcpsp_modes)
        if greedy_start:
            warmstart = self.model.new_solution()
            for j in self.start_solution.rcpsp_schedule:
                start_time_j = self.start_solution.rcpsp_schedule[j]["start_time"]
                for k in self.variable_per_task[j]:
                    task, mode, time = k
                    if start_time_j == time and mode == modes_dict[j]:
                        warmstart.add_var_value(self.x[k], 1)
                        warmstart.add_var_value(self.starts[task], time)
                    else:
                        warmstart.add_var_value(self.x[k], 0)
            self.model.add_mip_start(warmstart)
