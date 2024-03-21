#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from itertools import product
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple, Union

from mip import BINARY, INTEGER, MINIMIZE, Model, Var, xsum

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.lp_tools import (
    CplexMilpSolver,
    GurobiMilpSolver,
    MilpSolver,
    MilpSolverName,
    ParametersMilp,
    PymipMilpSolver,
    map_solver,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_solution import PartialSolution, RCPSPSolution
from discrete_optimization.rcpsp.solver.rcpsp_pile import (
    GreedyChoice,
    PileSolverRCPSP,
    PileSolverRCPSP_Calendar,
)
from discrete_optimization.rcpsp.solver.rcpsp_solver import SolverRCPSP

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


class LP_RCPSP(PymipMilpSolver, SolverRCPSP):
    problem: RCPSPModel

    def __init__(
        self,
        problem: RCPSPModel,
        lp_solver: MilpSolverName = MilpSolverName.CBC,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        if problem.is_rcpsp_multimode():
            raise ValueError("this solver is meant for single mode problems")
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.variable_decision = {}
        self.constraints_dict = {"lns": []}
        self.lp_solver = lp_solver
        self.solver_name = map_solver[lp_solver]
        self.start_solution: Optional[RCPSPSolution] = None

    def init_model(self, **args):
        greedy_start = args.get("greedy_start", True)
        start_solution = args.get("start_solution", None)
        if start_solution is None:
            if greedy_start:
                logger.info("Computing greedy solution")
                greedy_solver = PileSolverRCPSP(self.problem)
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
        self.model = Model(sense=MINIMIZE, solver_name=self.solver_name)
        self.x: List[List[Var]] = [
            [
                self.model.add_var(name=f"x({task},{t})", var_type=BINARY)
                for t in self.index_time
            ]
            for task in sorted_tasks
        ]
        self.index_in_var = {
            t: self.problem.return_index_task(task=t, offset=0) for t in sorted_tasks
        }
        self.model.objective = xsum(
            self.x[self.index_in_var[self.problem.sink_task]][t] * t
            for t in self.index_time
        )
        self.index_task = range(self.problem.n_jobs)
        self.index_resource = range(len(resources))
        for task in self.index_task:
            self.model += xsum(self.x[task][t] for t in self.index_time) == 1

        for (r, t) in product(self.index_resource, self.index_time):
            self.model += (
                xsum(
                    u[j][r] * self.x[j][t2]
                    for j in self.index_task
                    for t2 in range(max(0, t - p[j] + 1), t + 1)
                )
                <= c[r]
            )

        for (j, s) in S:
            self.model += (
                xsum(
                    t * self.x[self.index_in_var[s]][t]
                    - t * self.x[self.index_in_var[j]][t]
                    for t in self.index_time
                )
                >= p[self.index_in_var[j]]
            )
        start = []
        for j in self.index_task:
            for t in self.index_time:
                if (
                    self.start_solution.rcpsp_schedule[self.problem.tasks_list[j]][
                        "start_time"
                    ]
                    == t
                ):
                    start += [(self.x[j][t], 1)]
                else:
                    start += [(self.x[j][t], 0)]
        self.model.start = start
        p_s: Optional[PartialSolution] = args.get("partial_solution", None)
        self.constraints_partial_solutions = []
        if p_s is not None:
            constraints = []
            if p_s.start_times is not None:
                for task in p_s.start_times:
                    constraints += [
                        self.model.add_constr(
                            xsum(
                                [
                                    j * self.x[self.index_in_var[task]][j]
                                    for j in self.index_time
                                ]
                            )
                            == p_s.start_times[task]
                        )
                    ]
                    constraints += [
                        self.model.add_constr(
                            self.x[self.index_in_var[task]][p_s.start_times[task]] == 1
                        )
                    ]

            if p_s.partial_permutation is not None:
                for t1, t2 in zip(
                    p_s.partial_permutation[:-1], p_s.partial_permutation[1:]
                ):
                    constraints += [
                        self.model.add_constr(
                            xsum(
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
                            self.model.add_constr(
                                xsum(
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
                self.starts[j] = self.model.add_var(
                    name="start_" + str(j), lb=0, ub=makespan
                )
                self.model.add_constr(
                    xsum(t * self.x[j][t] for t in self.index_time) == self.starts[j]
                )
            if p_s.start_at_end is not None:
                for i, j in p_s.start_at_end:
                    constraints += [
                        self.model.add_constr(
                            self.starts[self.index_in_var[j]]
                            == self.starts[self.index_in_var[i]]
                            + p[self.index_in_var[i]]
                        )
                    ]
            if p_s.start_together is not None:
                for i, j in p_s.start_together:
                    constraints += [
                        self.model.add_constr(
                            self.starts[self.index_in_var[j]]
                            == self.starts[self.index_in_var[i]]
                        )
                    ]
            if p_s.start_after_nunit is not None:
                for t1, t2, delta in p_s.start_after_nunit:
                    constraints += [
                        self.model.add_constr(
                            self.starts[self.index_in_var[t2]]
                            >= self.starts[self.index_in_var[t1]] + delta
                        )
                    ]
            if p_s.start_at_end_plus_offset is not None:
                for t1, t2, delta in p_s.start_at_end_plus_offset:
                    constraints += [
                        self.model.add_constr(
                            self.starts[self.index_in_var[t2]]
                            >= self.starts[self.index_in_var[t1]]
                            + delta
                            + p[self.index_in_var[t1]]
                        )
                    ]
            self.constraints_partial_solutions = constraints

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> RCPSPSolution:
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
        return RCPSPSolution(
            problem=self.problem,
            rcpsp_schedule=rcpsp_schedule,
            rcpsp_schedule_feasible=True,
        )


class _BaseLP_MRCPSP(MilpSolver, SolverRCPSP):
    problem: RCPSPModel
    x: Dict[Tuple[Hashable, int, int], Any]

    def __init__(
        self,
        problem: RCPSPModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.variable_decision = {}
        self.constraints_dict = {"lns": []}

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> RCPSPSolution:
        rcpsp_schedule = {}
        modes: Dict[Hashable, Union[str, int]] = {}
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
        return RCPSPSolution(
            problem=self.problem,
            rcpsp_schedule=rcpsp_schedule,
            rcpsp_modes=modes_vec,
            rcpsp_schedule_feasible=True,
        )


class LP_MRCPSP(PymipMilpSolver, _BaseLP_MRCPSP):
    def __init__(
        self,
        problem: RCPSPModel,
        lp_solver: MilpSolverName = MilpSolverName.CBC,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            **kwargs,
        )
        self.lp_solver = lp_solver
        self.solver_name = map_solver[lp_solver]

    def init_model(self, **args):
        greedy_start = args.get("greedy_start", True)
        start_solution = args.get("start_solution", None)
        if start_solution is None:
            if greedy_start:
                logger.info("Computing greedy solution")
                greedy_solver = PileSolverRCPSP(self.problem)
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
        resources = self.problem.resources_list
        renewable = {
            r: self.problem.resources[r]
            for r in self.problem.resources
            if r not in self.problem.non_renewable_resources
        }
        non_renewable = {
            r: self.problem.resources[r] for r in self.problem.non_renewable_resources
        }
        S = []
        for task in sorted_tasks:
            for suc in self.problem.successors[task]:
                S.append([task, suc])
        self.index_time = range(sum(p))
        self.index_task = range(self.problem.n_jobs)
        self.index_resource = range(len(resources))
        # we have a better self.T to limit the number of variables :
        if self.start_solution.rcpsp_schedule_feasible:
            self.index_time = range(int(makespan + 1))
        self.model = Model(sense=MINIMIZE, solver_name=self.solver_name)
        self.x: Dict[Tuple[Hashable, int, int], Var] = {}
        last_task = self.problem.sink_task
        variable_per_task = {}
        for task in sorted_tasks:
            if task not in variable_per_task:
                variable_per_task[task] = []
            for mode in self.problem.mode_details[task]:
                for t in self.index_time:
                    self.x[(task, mode, t)] = self.model.add_var(
                        name=f"x({task},{mode}, {t})", var_type=BINARY
                    )
                    variable_per_task[task] += [(task, mode, t)]
        self.model.objective = xsum(
            self.x[key] * key[2] for key in variable_per_task[last_task]
        )
        for j in variable_per_task:
            self.model += xsum(self.x[key] for key in variable_per_task[j]) == 1
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
            self.model.add_constr(
                xsum(
                    int(self.problem.mode_details[key[0]][key[1]][r]) * self.x[key]
                    for key in self.x
                    if key[2]
                    <= t
                    < key[2]
                    + int(self.problem.mode_details[key[0]][key[1]]["duration"])
                    and self.problem.mode_details[key[0]][key[1]].get(r, 0) > 0
                )
                <= renewable_quantity[r][t]
            )
        for r in non_renewable:
            self.model.add_constr(
                xsum(
                    int(self.problem.mode_details[key[0]][key[1]][r]) * self.x[key]
                    for key in self.x
                    if self.problem.mode_details[key[0]][key[1]].get(r, 0) > 0
                )
                <= non_renewable_quantity[r][0]
            )
        durations = {
            j: self.model.add_var(name="duration_" + str(j), var_type=INTEGER)
            for j in variable_per_task
        }
        self.durations = durations
        self.variable_per_task = variable_per_task
        for j in variable_per_task:
            self.model.add_constr(
                xsum(
                    self.problem.mode_details[key[0]][key[1]]["duration"] * self.x[key]
                    for key in variable_per_task[j]
                )
                == durations[j]
            )
        for (j, s) in S:
            self.model.add_constr(
                xsum(
                    [key[2] * self.x[key] for key in variable_per_task[s]]
                    + [-key[2] * self.x[key] for key in variable_per_task[j]]
                )
                >= durations[j]
            )

        start = []
        modes_dict = self.problem.build_mode_dict(self.start_solution.rcpsp_modes)
        for j in self.start_solution.rcpsp_schedule:
            start_time_j = self.start_solution.rcpsp_schedule[j]["start_time"]
            mode_j = modes_dict[j]
            start += [
                (
                    self.durations[j],
                    self.problem.mode_details[j][mode_j]["duration"],
                )
            ]
            for k in self.variable_per_task[j]:
                task, mode, time = k
                if start_time_j == time and mode == mode_j:
                    start += [(self.x[k], 1)]
                else:
                    start += [(self.x[k], 0)]
        self.model.start = start
        p_s: Optional[PartialSolution] = args.get("partial_solution", None)
        self.constraints_partial_solutions = []
        if p_s is not None:
            constraints = []
            if p_s.start_times is not None:
                for task in p_s.start_times:
                    constraints += [
                        self.model.add_constr(
                            xsum(
                                [
                                    self.x[k]
                                    for k in self.variable_per_task[task]
                                    if k[2] == p_s.start_times[task]
                                ]
                            )
                            == 1
                        )
                    ]
            if p_s.partial_permutation is not None:
                for t1, t2 in zip(
                    p_s.partial_permutation[:-1], p_s.partial_permutation[1:]
                ):
                    constraints += [
                        self.model.add_constr(
                            xsum(
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
                            self.model.add_constr(
                                xsum(
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
            self.constraints_partial_solutions = constraints
            logger.debug(
                f"Partial solution constraints : {self.constraints_partial_solutions}"
            )

    def solve(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs
    ) -> ResultStorage:
        if self.model is None:
            self.init_model(greedy_start=False, **kwargs)
        return super().solve(parameters_milp=parameters_milp, **kwargs)


class LP_MRCPSP_GUROBI(GurobiMilpSolver, _BaseLP_MRCPSP):
    def init_model(self, **args):
        greedy_start = args.get("greedy_start", True)
        start_solution = args.get("start_solution", None)
        max_horizon = args.get("max_horizon", None)
        if start_solution is None:
            if greedy_start:
                logger.info("Computing greedy solution")
                if self.problem.is_varying_resource():
                    greedy_solver = PileSolverRCPSP_Calendar(self.problem)
                else:
                    greedy_solver = PileSolverRCPSP(self.problem)
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
        self.model = gurobi.Model("MRCPSP")
        self.x: Dict[Tuple[Hashable, int, int], gurobi.Var] = {}
        last_task = self.problem.sink_task
        variable_per_task = {}
        keys_for_t = {}
        for task in sorted_tasks:
            if task not in variable_per_task:
                variable_per_task[task] = []
            for mode in self.problem.mode_details[task]:
                for t in self.index_time:
                    self.x[(task, mode, t)] = self.model.addVar(
                        name=f"x({task},{mode}, {t})",
                        vtype=gurobi.GRB.BINARY,
                    )
                    for tt in range(
                        t, t + self.problem.mode_details[task][mode]["duration"]
                    ):
                        if tt not in keys_for_t:
                            keys_for_t[tt] = set()
                        keys_for_t[tt].add((task, mode, t))
                    variable_per_task[task] += [(task, mode, t)]
        self.model.update()
        self.model.setObjective(
            gurobi.quicksum(
                self.x[key] * key[2] for key in variable_per_task[last_task]
            )
        )
        self.model.addConstrs(
            gurobi.quicksum(self.x[key] for key in variable_per_task[j]) == 1
            for j in variable_per_task
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

        self.model.addConstrs(
            gurobi.quicksum(
                int(self.problem.mode_details[key[0]][key[1]][r]) * self.x[key]
                for key in keys_for_t[t]
            )
            <= renewable_quantity[r][t]
            for (r, t) in product(renewable, self.index_time)
        )

        self.model.addConstrs(
            gurobi.quicksum(
                int(self.problem.mode_details[key[0]][key[1]][r]) * self.x[key]
                for key in self.x
            )
            <= non_renewable_quantity[r][0]
            for r in non_renewable
        )
        self.model.update()
        durations = {
            j: self.model.addVar(name="duration_" + str(j), vtype=gurobi.GRB.INTEGER)
            for j in variable_per_task
        }
        self.durations = durations
        self.variable_per_task = variable_per_task
        self.model.addConstrs(
            gurobi.quicksum(
                self.problem.mode_details[key[0]][key[1]]["duration"] * self.x[key]
                for key in variable_per_task[j]
            )
            == durations[j]
            for j in variable_per_task
        )
        self.model.addConstrs(
            gurobi.quicksum(
                [key[2] * self.x[key] for key in variable_per_task[s]]
                + [-key[2] * self.x[key] for key in variable_per_task[j]]
            )
            >= durations[j]
            for (j, s) in S
        )

        start = []
        self.starts = {}
        for task in sorted_tasks:
            self.starts[task] = self.model.addVar(
                name=f"start({task})",
                vtype=gurobi.GRB.INTEGER,
                lb=0,
                ub=self.index_time[-1],
            )
            self.starts[task].start = self.start_solution.rcpsp_schedule[task][
                "start_time"
            ]
            self.model.addLConstr(
                gurobi.quicksum(
                    [self.x[key] * key[2] for key in variable_per_task[task]]
                )
                == self.starts[task]
            )
        modes_dict = self.problem.build_mode_dict(self.start_solution.rcpsp_modes)
        for j in self.start_solution.rcpsp_schedule:
            start_time_j = self.start_solution.rcpsp_schedule[j]["start_time"]
            start += [
                (
                    self.durations[j],
                    self.problem.mode_details[j][modes_dict[j]]["duration"],
                )
            ]
            for k in self.variable_per_task[j]:
                task, mode, time = k
                if start_time_j == time and mode == modes_dict[j]:
                    start += [(self.x[k], 1)]
                    self.x[k].start = 1
                else:
                    start += [(self.x[k], 0)]
                    self.x[k].start = 0

        p_s: Optional[PartialSolution] = args.get("partial_solution", None)
        self.constraints_partial_solutions = []
        self.model.update()
        if p_s is not None:
            constraints = []
            if p_s.start_times is not None:
                constraints = self.model.addConstrs(
                    gurobi.quicksum(
                        [
                            self.x[k]
                            for k in self.variable_per_task[task]
                            if k[2] == p_s.start_times[task]
                        ]
                    )
                    == 1
                    for task in p_s.start_times
                )
            if p_s.partial_permutation is not None:
                for t1, t2 in zip(
                    p_s.partial_permutation[:-1], p_s.partial_permutation[1:]
                ):
                    constraints += [
                        self.model.addLConstr(
                            gurobi.quicksum(
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
                            self.model.addLConstr(
                                gurobi.quicksum(
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
                        self.model.addLConstr(
                            self.starts[j] == self.starts[i] + durations[i]
                        )
                    ]
            if p_s.start_together is not None:
                for i, j in p_s.start_together:
                    constraints += [
                        self.model.addLConstr(self.starts[j] == self.starts[i])
                    ]
            if p_s.start_after_nunit is not None:
                for t1, t2, delta in p_s.start_after_nunit:
                    constraints += [
                        self.model.addLConstr(
                            self.starts[t2] >= self.starts[t1] + delta
                        )
                    ]
            if p_s.start_at_end_plus_offset is not None:
                for t1, t2, delta in p_s.start_at_end_plus_offset:
                    constraints += [
                        self.model.addLConstr(
                            self.starts[t2] >= self.starts[t1] + delta + durations[t1]
                        )
                    ]
            self.constraints_partial_solutions = constraints
            logger.debug(
                f"Partial solution constraints : {self.constraints_partial_solutions}"
            )
            self.model.update()

    def solve(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs
    ) -> ResultStorage:
        if self.model is None:
            self.init_model(greedy_start=False, **kwargs)
        return super().solve(parameters_milp=parameters_milp, **kwargs)


class LP_RCPSP_CPLEX(CplexMilpSolver, _BaseLP_MRCPSP):
    def init_model(self, **args):
        greedy_start = args.get("greedy_start", True)
        start_solution = args.get("start_solution", None)
        max_horizon = args.get("max_horizon", None)
        if start_solution is None:
            if greedy_start:
                logger.info("Computing greedy solution")
                if self.problem.is_varying_resource():
                    greedy_solver = PileSolverRCPSP_Calendar(self.problem)
                else:
                    greedy_solver = PileSolverRCPSP(self.problem)
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
        self.x: Dict[Tuple[Hashable, int, int], "cplex_var.Var"] = {}
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

    def solve(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs
    ) -> ResultStorage:
        if self.model is None:
            self.init_model(greedy_start=False, **kwargs)
        return super().solve(parameters_milp=parameters_milp, **kwargs)
