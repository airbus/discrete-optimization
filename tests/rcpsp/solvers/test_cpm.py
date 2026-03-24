#  Copyright (c) 2022-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from copy import deepcopy

from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.cpm import (
    CpmRcpspSolver,
    run_partial_classic_cpm,
)


def test_cpm_sm():
    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file)

    cpm = CpmRcpspSolver(problem=rcpsp_problem)
    cpath = cpm.run_classic_cpm()
    order = cpm.return_order_cpm()
    permutation_sgs = [o - 2 for o in order]
    permutation_sgs.remove(min(permutation_sgs))
    permutation_sgs.remove(max(permutation_sgs))
    solution_sgs = RcpspSolution(
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
    original_successors = deepcopy(rcpsp_problem.successors)
    results = []
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
    cpm = CpmRcpspSolver(problem=modified_model)
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
    solution = RcpspSolution(
        problem=rcpsp_problem,
        rcpsp_schedule=schedule,
        rcpsp_modes=[1 for i in range(rcpsp_problem.n_jobs)],
    )
    fit = rcpsp_problem.evaluate(solution)
    results += [(solution, -fit["makespan"])]


def test_cpm_partial():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    cpm = CpmRcpspSolver(problem=rcpsp_problem)
    sol = cpm.solve()
    cpm.map_node[rcpsp_problem.sink_task]
    best_sol: RcpspSolution = sol.get_best_solution_fit()[0]
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
