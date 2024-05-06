#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

import networkx as nx
import numpy as np

from discrete_optimization.generic_rcpsp_tools.graph_tools_rcpsp import (
    GraphRCPSP,
    GraphRCPSPSpecialConstraints,
)
from discrete_optimization.generic_rcpsp_tools.ls_solver import (
    LS_SOLVER,
    LS_RCPSP_Solver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.lns_mip import PostProcessSolution
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model_preemptive import (
    RCPSPModelPreemptive,
    RCPSPSolutionPreemptive,
)
from discrete_optimization.rcpsp.specialized_rcpsp.rcpsp_specialized_constraints import (
    RCPSPModelSpecialConstraintsPreemptive,
    RCPSPSolutionSpecialPreemptive,
)

logger = logging.getLogger(__name__)


def last_opti_solution(last_result_store: ResultStorage):
    current_solution, fit = next(
        (
            last_result_store.list_solution_fits[j]
            for j in range(len(last_result_store.list_solution_fits))
            if "opti_from_cp"
            in last_result_store.list_solution_fits[j][0].__dict__.keys()
        ),
        (None, None),
    )
    if current_solution is None or fit != last_result_store.get_best_solution_fit()[1]:
        current_solution, fit = last_result_store.get_last_best_solution()
    return current_solution


class PostProLeftShift(PostProcessSolution):
    def __init__(
        self,
        problem: RCPSPModelPreemptive,
        params_objective_function: ParamsObjectiveFunction = None,
        do_ls: bool = False,
        **kwargs
    ):
        self.problem = problem
        self.params_objective_function = params_objective_function
        (
            self.aggreg_from_sol,
            self.aggreg_from_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem,
            params_objective_function=self.params_objective_function,
        )
        self.graph = self.problem.compute_graph()
        if isinstance(problem, RCPSPModelSpecialConstraintsPreemptive):
            self.graph_rcpsp = GraphRCPSPSpecialConstraints(problem=self.problem)
            self.special_constraints = True
        else:
            self.graph_rcpsp = GraphRCPSP(problem=self.problem)
            self.special_constraints = False
        self.successors = {
            n: nx.algorithms.descendants(self.graph.graph_nx, n)
            for n in self.graph.graph_nx.nodes()
        }
        self.predecessors = {
            n: nx.algorithms.descendants(self.graph.graph_nx, n)
            for n in self.graph.graph_nx.nodes()
        }
        self.immediate_predecessors = {
            n: self.graph.get_predecessors(n) for n in self.graph.nodes_name
        }
        self.do_ls = do_ls
        self.dict_params = kwargs

    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        new_solution = sgs_variant(
            solution=last_opti_solution(result_storage),
            problem=self.problem,
            predecessors_dict=self.immediate_predecessors,
        )
        fit = self.aggreg_from_sol(new_solution)
        result_storage.add_solution(new_solution, fit)
        if self.do_ls:
            solver = LS_RCPSP_Solver(problem=self.problem, ls_solver=LS_SOLVER.SA)
            s = result_storage.get_best_solution().copy()
            if self.problem != s.problem:
                s.change_problem(self.problem)
            result_store = solver.solve(
                nb_iteration_max=self.dict_params.get("nb_iteration_max", 200),
                init_solution=s,
            )
            solution, f = result_store.get_last_best_solution()
            result_storage.list_solution_fits += [
                (solution, self.aggreg_from_sol(solution))
            ]
        return result_storage


def sgs_variant(
    solution: RCPSPSolutionPreemptive, problem: RCPSPModelPreemptive, predecessors_dict
):
    new_proposed_schedule = {}
    new_horizon = min(solution.get_end_time(problem.sink_task) * 3, problem.horizon)
    resource_avail_in_time = {}
    modes_dict = problem.build_mode_dict(solution.rcpsp_modes)
    for r in problem.resources_list:
        if problem.is_varying_resource():
            resource_avail_in_time[r] = np.copy(problem.resources[r][:new_horizon])
        else:
            resource_avail_in_time[r] = problem.resources[r] * np.ones(new_horizon)
    sorted_tasks = sorted(
        solution.rcpsp_schedule.keys(), key=lambda x: (solution.get_start_time(x), x)
    )
    for task in list(sorted_tasks):
        if len(solution.rcpsp_schedule[task]["starts"]) > 1:
            new_proposed_schedule[task] = {
                "starts": solution.rcpsp_schedule[task]["starts"],
                "ends": solution.rcpsp_schedule[task]["ends"],
            }
            for s, e in zip(
                new_proposed_schedule[task]["starts"],
                new_proposed_schedule[task]["ends"],
            ):
                for res in problem.resources_list:
                    resource_avail_in_time[res][s:e] -= problem.mode_details[task][
                        modes_dict[task]
                    ].get(res, 0)
            sorted_tasks.remove(task)

    def is_startable(tt):
        if isinstance(problem, RCPSPModelSpecialConstraintsPreemptive):
            return (
                all(
                    x in new_proposed_schedule
                    for x in problem.special_constraints.dict_start_after_nunit_reverse.get(
                        tt, {}
                    )
                )
                and all(
                    x in new_proposed_schedule
                    for x in problem.special_constraints.dict_start_at_end_reverse.get(
                        tt, {}
                    )
                )
                and all(
                    x in new_proposed_schedule
                    for x in problem.special_constraints.dict_start_at_end_offset_reverse.get(
                        tt, {}
                    )
                )
                and all(t in new_proposed_schedule for t in predecessors_dict[tt])
            )
        return all(t in new_proposed_schedule for t in predecessors_dict[tt])

    while True:
        task = next((t for t in sorted_tasks if is_startable(t)))
        sorted_tasks.remove(task)
        if len(solution.rcpsp_schedule[task]["starts"]) > 1:
            continue
        times_predecessors = [
            new_proposed_schedule[t]["ends"][-1]
            if t in new_proposed_schedule
            else solution.rcpsp_schedule[t]["ends"][-1]
            for t in predecessors_dict[task]
        ]
        if isinstance(problem, RCPSPModelSpecialConstraintsPreemptive):
            times_predecessors += [
                new_proposed_schedule[t]["starts"][0]
                if t in new_proposed_schedule
                else solution.rcpsp_schedule[t]["starts"][0]
                for t in problem.special_constraints.dict_start_together.get(
                    task, set()
                )
            ]
            times_predecessors += [
                new_proposed_schedule[t]["starts"][0]
                + problem.special_constraints.dict_start_after_nunit_reverse.get(task)[
                    t
                ]
                if t in new_proposed_schedule
                else solution.rcpsp_schedule[t]["starts"][0]
                + problem.special_constraints.dict_start_after_nunit_reverse.get(task)[
                    t
                ]
                for t in problem.special_constraints.dict_start_after_nunit_reverse.get(
                    task, {}
                )
            ]
            times_predecessors += [
                new_proposed_schedule[t]["ends"][-1]
                if t in new_proposed_schedule
                else solution.rcpsp_schedule[t]["ends"][-1]
                for t in problem.special_constraints.dict_start_at_end_reverse.get(
                    task, {}
                )
            ]
            times_predecessors += [
                new_proposed_schedule[t]["ends"][-1]
                + problem.special_constraints.dict_start_at_end_offset_reverse.get(
                    task
                )[t]
                if t in new_proposed_schedule
                else solution.rcpsp_schedule[t]["ends"][-1]
                + problem.special_constraints.dict_start_at_end_offset_reverse.get(
                    task
                )[t]
                for t in problem.special_constraints.dict_start_at_end_offset_reverse.get(
                    task, {}
                )
            ]
        if len(times_predecessors) > 0:
            min_time = max(times_predecessors)
        else:
            min_time = 0
        if solution.get_start_time(task) == solution.get_end_time(task):
            new_proposed_schedule[task] = {"starts": [min_time], "ends": [min_time]}
        if len(solution.rcpsp_schedule[task]["starts"]) > 1:
            new_proposed_schedule[task] = {
                "starts": solution.rcpsp_schedule[task]["starts"],
                "ends": solution.rcpsp_schedule[task]["ends"],
            }
        else:
            for t in range(min_time, problem.horizon):
                if all(
                    resource_avail_in_time[res][time]
                    >= problem.mode_details[task][modes_dict[task]].get(res, 0)
                    for res in problem.resources_list
                    for time in range(
                        t, t + problem.mode_details[task][modes_dict[task]]["duration"]
                    )
                ):
                    new_starting_time = t
                    break
            new_proposed_schedule[task] = {
                "starts": [new_starting_time],
                "ends": [
                    new_starting_time
                    + problem.mode_details[task][modes_dict[task]]["duration"]
                ],
            }
            for s, e in zip(
                new_proposed_schedule[task]["starts"],
                new_proposed_schedule[task]["ends"],
            ):
                for res in problem.resources_list:
                    resource_avail_in_time[res][s:e] -= problem.mode_details[task][
                        modes_dict[task]
                    ].get(res, 0)
        if len(sorted_tasks) == 0:
            break
    new_solution = RCPSPSolutionSpecialPreemptive(
        problem=problem,
        rcpsp_schedule=new_proposed_schedule,
        rcpsp_schedule_feasible=True,
        rcpsp_modes=solution.rcpsp_modes,
    )
    logger.debug(
        ("New : ", problem.evaluate(new_solution), problem.satisfy(new_solution))
    )
    logger.debug(("Old : ", problem.evaluate(solution), problem.satisfy(solution)))
    return new_solution
