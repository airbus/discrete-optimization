#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
from mip import BINARY, MINIMIZE, Model, xsum

from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    MilpSolver,
    MilpSolverName,
    PymipMilpSolver,
    map_solver,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from discrete_optimization.rcpsp.solver.rcpsp_solver import SolverRCPSP

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    import gurobipy as gurobi


logger = logging.getLogger(__name__)


def intersect(i1, i2):
    if i2[0] >= i1[1] or i1[0] >= i2[1]:
        return None
    else:
        s = max(i1[0], i2[0])
        e = min(i1[1], i2[1])
        return [s, e]


class ConstraintTaskIndividual:
    list_tuple: List[Tuple[str, int, int, bool]]
    # task, ressource, ressource_individual, has or has not to do a task
    # indicates constraint for a given resource individual that has to do a tas
    def __init__(self, list_tuple):
        self.list_tuple = list_tuple


class ConstraintWorkDuration:
    ressource: str
    individual: int
    time_bounds: Tuple[int, int]
    working_time_upper_bound: int

    def __init__(self, ressource, individual, time_bounds, working_time_upper_bound):
        self.ressource = ressource
        self.individual = individual
        self.time_bounds = time_bounds
        self.working_time_upper_bound = working_time_upper_bound


class _Base_LP_MRCPSP_GANTT(MilpSolver, SolverRCPSP):
    problem: RCPSPModel

    def __init__(
        self,
        problem: RCPSPModel,
        rcpsp_solution: RCPSPSolution,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        if problem.calendar_details is None:
            raise ValueError(
                "rcpsp_model.calendar_details cannot be None for this solver"
            )
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.rcpsp_solution = rcpsp_solution
        self.jobs = sorted(list(problem.mode_details.keys()))
        self.modes_dict = {
            i + 2: self.rcpsp_solution.rcpsp_modes[i]
            for i in range(len(self.rcpsp_solution.rcpsp_modes))
        }
        self.modes_dict[1] = 1
        self.modes_dict[self.jobs[-1]] = 1
        self.rcpsp_schedule = self.rcpsp_solution.rcpsp_schedule
        self.start_times_dict = {}
        for task in self.rcpsp_schedule:
            t = self.rcpsp_schedule[task]["start_time"]
            if t not in self.start_times_dict:
                self.start_times_dict[t] = set()
            self.start_times_dict[t].add((task, t))
        self.graph_intersection_time = nx.Graph()
        for t in self.jobs:
            self.graph_intersection_time.add_node(t)
        for t in self.jobs:
            intersected_jobs = [
                task
                for task in self.rcpsp_schedule
                if intersect(
                    [
                        self.rcpsp_schedule[task]["start_time"],
                        self.rcpsp_schedule[task]["end_time"],
                    ],
                    [
                        self.rcpsp_schedule[t]["start_time"],
                        self.rcpsp_schedule[t]["end_time"],
                    ],
                )
                is not None
                and t != task
            ]
            for tt in intersected_jobs:
                self.graph_intersection_time.add_edge(t, tt)
        cliques = [c for c in nx.find_cliques(self.graph_intersection_time)]
        self.cliques = cliques
        self.sense_optim = ModeOptim.MINIMIZATION
        self.params_objective_function.sense_function = self.sense_optim
        self.constraint_additionnal = {}

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> Tuple[Dict[Any, Dict[Any, Dict[Any, Any]]], float]:
        objective = get_obj_value_for_current_solution()
        resource_id_usage = {
            k: {
                individual: {
                    task: get_var_value_for_current_solution(resource_usage)
                    for task, resource_usage in self.ressource_id_usage[k][
                        individual
                    ].items()
                }
                for individual in self.ressource_id_usage[k]
            }
            for k in self.ressource_id_usage
        }
        return (resource_id_usage, objective)


class LP_MRCPSP_GANTT(PymipMilpSolver, _Base_LP_MRCPSP_GANTT):
    def __init__(
        self,
        problem: RCPSPModel,
        rcpsp_solution: RCPSPSolution,
        lp_solver=MilpSolverName.CBC,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem,
            rcpsp_solution=rcpsp_solution,
            params_objective_function=params_objective_function,
            **kwargs,
        )
        self.lp_solver = lp_solver

    def init_model(self, **args):
        self.model = Model(sense=MINIMIZE, solver_name=map_solver[self.lp_solver])
        self.ressource_id_usage = {
            k: {i: {} for i in range(len(self.problem.calendar_details[k]))}
            for k in self.problem.calendar_details.keys()
        }
        variables_per_task = {}
        variables_per_individual = {}
        constraints_ressource_need = {}

        for task in self.jobs:
            start = self.rcpsp_schedule[task]["start_time"]
            end = self.rcpsp_schedule[task]["end_time"]
            for k in self.ressource_id_usage:  # typically worker
                needed_ressource = (
                    self.problem.mode_details[task][self.modes_dict[task]][k] > 0
                )
                if needed_ressource:
                    for individual in self.ressource_id_usage[k]:
                        available = all(
                            self.problem.calendar_details[k][individual][time]
                            for time in range(start, end)
                        )
                        if available:
                            key_variable = (k, individual, task)
                            self.ressource_id_usage[k][individual][
                                task
                            ] = self.model.add_var(
                                name=str(key_variable),
                                var_type=BINARY,
                                obj=random.random(),
                            )
                            if task not in variables_per_task:
                                variables_per_task[task] = set()
                            if k not in variables_per_individual:
                                variables_per_individual[k] = {}
                            if individual not in variables_per_individual[k]:
                                variables_per_individual[k][individual] = set()
                            variables_per_task[task].add(key_variable)
                            variables_per_individual[k][individual].add(key_variable)
                    ressource_needed = self.problem.mode_details[task][
                        self.modes_dict[task]
                    ][k]
                    if k not in constraints_ressource_need:
                        constraints_ressource_need[k] = {}
                    constraints_ressource_need[k][task] = self.model.add_constr(
                        xsum(
                            [
                                self.ressource_id_usage[k][key[1]][key[2]]
                                for key in variables_per_task[task]
                                if key[0] == k
                            ]
                        )
                        == ressource_needed,
                        name="ressource_" + str(k) + "_" + str(task),
                    )
        overlaps_constraints = {}

        for i in range(len(self.cliques)):
            tasks = set(self.cliques[i])
            for k in variables_per_individual:
                for individual in variables_per_individual[k]:
                    keys_variable = [
                        variable
                        for variable in variables_per_individual[k][individual]
                        if variable[2] in tasks
                    ]
                    if len(keys_variable) > 0:
                        overlaps_constraints[
                            (i, k, individual)
                        ] = self.model.add_constr(
                            xsum(
                                [
                                    self.ressource_id_usage[key[0]][key[1]][key[2]]
                                    for key in keys_variable
                                ]
                            )
                            <= 1
                        )


# gurobi solver which is ussefull to get a pool of solution (indeed, using the other one we dont have usually a lot of
# ssolution since we converge rapidly to the "optimum" (we don't have an objective value..)
class LP_MRCPSP_GANTT_GUROBI(GurobiMilpSolver, _Base_LP_MRCPSP_GANTT):
    def init_model(self, **args):
        self.model = gurobi.Model("Gantt")
        self.ressource_id_usage = {
            k: {i: {} for i in range(len(self.problem.calendar_details[k]))}
            for k in self.problem.calendar_details.keys()
        }
        variables_per_task = {}
        variables_per_individual = {}
        constraints_ressource_need = {}

        for task in self.jobs:
            start = self.rcpsp_schedule[task]["start_time"]
            end = self.rcpsp_schedule[task]["end_time"]
            for k in self.ressource_id_usage:  # typically worker
                needed_ressource = (
                    self.problem.mode_details[task][self.modes_dict[task]][k] > 0
                )
                if needed_ressource:
                    for individual in self.ressource_id_usage[k]:
                        available = all(
                            self.problem.calendar_details[k][individual][time]
                            for time in range(start, end)
                        )
                        if available:
                            key_variable = (k, individual, task)
                            self.ressource_id_usage[k][individual][
                                task
                            ] = self.model.addVar(
                                name=str(key_variable), vtype=gurobi.GRB.BINARY
                            )
                            if task not in variables_per_task:
                                variables_per_task[task] = set()
                            if k not in variables_per_individual:
                                variables_per_individual[k] = {}
                            if individual not in variables_per_individual[k]:
                                variables_per_individual[k][individual] = set()
                            variables_per_task[task].add(key_variable)
                            variables_per_individual[k][individual].add(key_variable)
                    ressource_needed = self.problem.mode_details[task][
                        self.modes_dict[task]
                    ][k]
                    if k not in constraints_ressource_need:
                        constraints_ressource_need[k] = {}
                    constraints_ressource_need[k][task] = self.model.addLConstr(
                        gurobi.quicksum(
                            [
                                self.ressource_id_usage[k][key[1]][key[2]]
                                for key in variables_per_task[task]
                                if key[0] == k
                            ]
                        )
                        == ressource_needed,
                        name="ressource_" + str(k) + "_" + str(task),
                    )
        overlaps_constraints = {}

        for i in range(len(self.cliques)):
            tasks = set(self.cliques[i])
            for k in variables_per_individual:
                for individual in variables_per_individual[k]:
                    keys_variable = [
                        variable
                        for variable in variables_per_individual[k][individual]
                        if variable[2] in tasks
                    ]
                    if len(keys_variable) > 0:
                        overlaps_constraints[
                            (i, k, individual)
                        ] = self.model.addLConstr(
                            gurobi.quicksum(
                                [
                                    self.ressource_id_usage[key[0]][key[1]][key[2]]
                                    for key in keys_variable
                                ]
                            )
                            <= 1
                        )
        self.model.modelSense = gurobi.GRB.MINIMIZE

    def adding_constraint(
        self,
        constraint_description: Union[ConstraintTaskIndividual, ConstraintWorkDuration],
        constraint_name: str = "",
    ):
        if isinstance(constraint_description, ConstraintTaskIndividual):
            if constraint_name == "":
                constraint_name = str(ConstraintTaskIndividual.__name__)
            for tupl in constraint_description.list_tuple:
                ressource, ressource_individual, task, has_to_do = tupl
                if ressource in self.ressource_id_usage:
                    if ressource_individual in self.ressource_id_usage[ressource]:
                        if (
                            task
                            in self.ressource_id_usage[ressource][ressource_individual]
                        ):
                            if constraint_name not in self.constraint_additionnal:
                                self.constraint_additionnal[constraint_name] = []
                            self.constraint_additionnal[constraint_name] += [
                                self.model.addLConstr(
                                    self.ressource_id_usage[ressource][
                                        ressource_individual
                                    ][task]
                                    == has_to_do
                                )
                            ]
        if isinstance(constraint_description, ConstraintWorkDuration):
            if constraint_name == "":
                constraint_name = str(ConstraintWorkDuration.__name__)
            if constraint_name not in self.constraint_additionnal:
                self.constraint_additionnal[constraint_name] = []
            tasks_of_interest = [
                t
                for t in self.rcpsp_schedule
                if t
                in self.ressource_id_usage.get(
                    constraint_description.ressource, {}
                ).get(constraint_description.individual, {})
                and (
                    constraint_description.time_bounds[0]
                    <= self.rcpsp_schedule[t]["start_time"]
                    <= constraint_description.time_bounds[1]
                    or constraint_description.time_bounds[0]
                    <= self.rcpsp_schedule[t]["end_time"]
                    <= constraint_description.time_bounds[1]
                )
            ]
            logger.debug(tasks_of_interest)
            self.constraint_additionnal[constraint_name] += [
                self.model.addLConstr(
                    gurobi.quicksum(
                        [
                            self.ressource_id_usage[constraint_description.ressource][
                                constraint_description.individual
                            ][t]
                            * (
                                min(
                                    constraint_description.time_bounds[1],
                                    self.rcpsp_schedule[t]["end_time"],
                                )
                                - max(
                                    constraint_description.time_bounds[0],
                                    self.rcpsp_schedule[t]["start_time"],
                                )
                            )
                            for t in tasks_of_interest
                        ]
                    )
                    <= constraint_description.working_time_upper_bound
                )
            ]
            self.model.update()

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> Tuple[Dict[Any, Dict[Any, Dict[Any, Any]]], float]:
        objective = get_obj_value_for_current_solution()
        resource_id_usage = {
            k: {
                individual: {
                    task: get_var_value_for_current_solution(resource_usage)
                    for task, resource_usage in self.ressource_id_usage[k][
                        individual
                    ].items()
                }
                for individual in self.ressource_id_usage[k]
            }
            for k in self.ressource_id_usage
        }
        return (resource_id_usage, objective)

    def build_objective_function_from_a_solution(
        self,
        ressource_usage: Dict[str, Dict[int, Dict[int, bool]]],
        ignore_tuple: Set[Tuple[str, int, int]] = None,
    ):
        objective = gurobi.LinExpr(0.0)
        if ignore_tuple is None:
            ignore_tuple = set()
        for k in ressource_usage:
            for individual in ressource_usage[k]:
                for task in ressource_usage[k][individual]:
                    if ressource_usage[k][individual][task] >= 0.5:
                        objective.add(1 - self.ressource_id_usage[k][individual][task])
        logger.debug("Setting new objectives = Change task objective")
        self.model.setObjective(objective)
