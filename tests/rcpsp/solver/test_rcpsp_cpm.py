#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from copy import deepcopy

import networkx as nx
import numpy as np

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_problem import ModeOptim
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_utils import plot_ressource_view, plot_task_gantt
from discrete_optimization.rcpsp.solver.cp_solvers import CP_RCPSP_MZN
from discrete_optimization.rcpsp.solver.cpm import CPM, run_partial_classic_cpm


def test_cpm_sm():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    dummy = rcpsp_problem.get_dummy_solution()
    fit_dummy = -rcpsp_problem.evaluate(dummy)["makespan"]

    solver = CP_RCPSP_MZN(rcpsp_problem)
    solver.init_model()
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 5
    result_storage = solver.solve(parameters_cp=parameters_cp)
    solution_cp, fit_cp = result_storage.get_best_solution_fit()
    assert fit_dummy <= fit_cp

    cpm = CPM(problem=rcpsp_problem)
    cpath = cpm.run_classic_cpm()
    order = cpm.return_order_cpm()
    permutation_sgs = [o - 2 for o in order]
    permutation_sgs.remove(min(permutation_sgs))
    permutation_sgs.remove(max(permutation_sgs))
    solution_sgs = RCPSPSolution(
        problem=rcpsp_problem,
        rcpsp_permutation=permutation_sgs,
        rcpsp_modes=[1 for i in range(rcpsp_problem.n_jobs)],
    )
    rcpsp_problem.evaluate(solution_sgs)

    order = sorted(
        cpm.map_node,
        key=lambda x: (
            cpm.map_node[x]._LSD,
            cpm.map_node[x]._LSD - cpm.map_node[x]._ESD,
        ),
    )
    schedule, link_to_add, effects_on_delay, causes_of_delay = cpm.run_sgs_on_order(
        map_nodes=cpm.map_node, critical_path=cpath, total_order=order
    )

    def compute_graph_conflict(
        effects_on_delay, causes_of_delay, cpm, predecessors_map
    ):
        graph_of_possible_conflicts = nx.DiGraph()
        for task in effects_on_delay:
            if task not in graph_of_possible_conflicts:
                graph_of_possible_conflicts.add_node(task)
            for t in effects_on_delay[task]["task_causes"]:
                if t not in graph_of_possible_conflicts:
                    graph_of_possible_conflicts.add_node(t)
                if t not in predecessors_map[task]:
                    graph_of_possible_conflicts.add_edge(t, task)
        for j in causes_of_delay:
            delayed = schedule[j]["start_time"] > cpm.map_node[j]._ESD
            if delayed:
                for res, time, set_task in causes_of_delay[j]["res_t_other_task"]:

                    if time >= cpm.map_node[j]._LSD - 5:
                        if j not in graph_of_possible_conflicts:
                            graph_of_possible_conflicts.add_node(j)
                        for task in set_task:
                            if task not in predecessors_map[j]:
                                if task not in graph_of_possible_conflicts:
                                    graph_of_possible_conflicts.add_node(task)
                                graph_of_possible_conflicts.add_edge(task, j)
        graph_of_possible_conflicts_reverse = nx.reverse(
            graph_of_possible_conflicts, copy=True
        )
        return graph_of_possible_conflicts_reverse

    def compute_link(schedule, cpm, cpath, causes_of_delay, effects_on_delay):
        start_times_critical_path = np.array([schedule[j]["start_time"] for j in cpath])
        expected = np.array([cpm.map_node[j]._ESD for j in cpath])
        delta_expected = np.diff(expected)
        delta_actual = np.diff(start_times_critical_path)
        link_to_add = []
        for k in range(len(delta_expected)):
            if delta_actual[k] > delta_expected[k]:
                task_delayed = cpath[k + 1]
                if task_delayed in effects_on_delay:
                    task_that_caused = effects_on_delay[task_delayed]["task_causes"]
                    for tcaused in task_that_caused:
                        delayed = (
                            schedule[tcaused]["start_time"] > cpm.map_node[tcaused]._ESD
                        )
                        if delayed:
                            if tcaused in causes_of_delay:
                                for res, time, set_task in causes_of_delay[tcaused][
                                    "res_t_other_task"
                                ]:
                                    if time >= cpm.map_node[tcaused]._LSD - 5:
                                        for task in set_task:
                                            link_to_add += [(tcaused, task)]
                            elif tcaused in effects_on_delay:
                                tcaused_2step = effects_on_delay[tcaused]["task_causes"]
                                link_to_add += [(tt, tcaused) for tt in tcaused_2step]
                else:
                    delayed = (
                        schedule[task_delayed]["start_time"]
                        > cpm.map_node[task_delayed]._ESD
                    )
                    if delayed:
                        if task_delayed in causes_of_delay:
                            for res, time, set_task in causes_of_delay[task_delayed][
                                "res_t_other_task"
                            ]:
                                if time >= cpm.map_node[task_delayed]._LSD - 1:
                                    for task in set_task:
                                        link_to_add += [(task, task_delayed)]
        return link_to_add

    def compute_link_second_version(
        schedule, cpm, cpath, causes_of_delay, effects_on_delay, graph_reverse
    ):
        start_times_critical_path = np.array([schedule[j]["start_time"] for j in cpath])
        expected = np.array([cpm.map_node[j]._ESD for j in cpath])
        delta_expected = np.diff(expected)
        delta_actual = np.diff(start_times_critical_path)
        link_to_add = []
        for k in range(len(delta_expected)):
            if delta_actual[k] > delta_expected[k]:
                task_delayed = cpath[k + 1]
                if task_delayed in graph_reverse:
                    ddd = nx.single_source_shortest_path_length(
                        graph_reverse, task_delayed, cutoff=15
                    )
                    subgraph = nx.subgraph(graph_reverse, list(ddd.keys()))
                    for edge in subgraph.edges():
                        if edge[0] != edge[1]:
                            link_to_add += [(edge[0], edge[1])]
        return link_to_add

    graph_conflict = compute_graph_conflict(
        effects_on_delay,
        causes_of_delay,
        cpm,
        {n: cpm.predecessors_map[n]["succs"] for n in cpm.predecessors_map},
    )
    link_to_add = compute_link_second_version(
        schedule, cpm, cpath, causes_of_delay, effects_on_delay, graph_conflict
    )

    original_successors = deepcopy(rcpsp_problem.successors)
    results = []
    for k in range(3):
        modified_model = rcpsp_problem.copy()
        link_to_add = [
            (task, j)
            for j in cpm.unlock_task_transition
            for task in cpm.unlock_task_transition[j]
        ]
        for i1, i2 in random.sample(link_to_add, int(0.8 * len(link_to_add))):
            if i2 not in modified_model.successors[i1]:
                modified_model.successors[i1].append(i2)
        gg = modified_model.compute_graph()
        has_loop = gg.check_loop()
        while has_loop is not None:
            if has_loop is not None:
                for e0, e1, sense in has_loop:
                    if e1 not in original_successors[e0]:
                        modified_model.successors[e0].remove(e1)
            gg = modified_model.compute_graph()
            has_loop = gg.check_loop()
        cpm = CPM(problem=modified_model)
        cpath = cpm.run_classic_cpm()
        order = sorted(
            cpm.map_node,
            key=lambda x: (
                cpm.map_node[x]._LSD,
                -len(cpm.successors_map[x]["succs"]),
                cpm.map_node[x]._LSD - cpm.map_node[x]._ESD,
            ),
        )
        schedule, _, effects_on_delay, causes_of_delay = cpm.run_sgs_on_order(
            map_nodes=cpm.map_node,
            critical_path=cpath,
            total_order=order,
            cut_sgs_by_critical=False,
        )
        solution = RCPSPSolution(
            problem=rcpsp_problem,
            rcpsp_schedule=schedule,
            rcpsp_modes=[1 for i in range(rcpsp_problem.n_jobs)],
        )
        fit = rcpsp_problem.evaluate(solution)
        results += [(solution, -fit["makespan"])]
        theoric = cpm.map_node[cpm.sink]
    result_storage = ResultStorage(
        list_solution_fits=results, mode_optim=ModeOptim.MAXIMIZATION
    )
    solution, fit = result_storage.get_best_solution_fit()
    assert rcpsp_problem.satisfy(solution)
    assert rcpsp_problem.satisfy(solution_cp)
    plot_ressource_view(rcpsp_problem, solution, title_figure="cpath")
    plot_task_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution_cp, title_figure="cp")
    plot_task_gantt(rcpsp_problem, solution_cp)


def test_cpm_partial():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    cpm = CPM(problem=rcpsp_problem)
    sol = cpm.solve()
    cpm.map_node[rcpsp_problem.sink_task]
    best_sol: RCPSPSolution = sol.get_best_solution_fit()[0]
    old_map_cpm = cpm.map_node
    for k in range(best_sol.get_start_time(rcpsp_problem.sink_task)):
        partial_schedule = {
            t: (
                best_sol.rcpsp_schedule[t]["start_time"],
                best_sol.rcpsp_schedule[t]["end_time"],
            )
            for t in best_sol.rcpsp_schedule
            if best_sol.rcpsp_schedule[t]["start_time"] <= k
        }

        additional_cost_method_fast = max(
            [
                partial_schedule[t][1] - old_map_cpm[t]._LFD
                for t in partial_schedule
                if partial_schedule[t][1]
            ],
            default=0,
        )

        cp, map_node = run_partial_classic_cpm(
            partial_schedule=partial_schedule, cpm_solver=cpm
        )
        assert (
            map_node[rcpsp_problem.sink_task]._EFD
            - old_map_cpm[rcpsp_problem.sink_task]._EFD
        ) == additional_cost_method_fast


if __name__ == "__main__":
    test_cpm_sm()
