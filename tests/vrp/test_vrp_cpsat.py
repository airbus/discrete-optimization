#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from itertools import product

import pytest

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.lns_cp import LNS_OrtoolsCPSat
from discrete_optimization.generic_tools.lns_tools import ConstraintHandlerMix
from discrete_optimization.tsp.tsp_model import SolutionTSP
from discrete_optimization.vrp.solver.vrp_cpsat_lns import (
    ConstraintHandlerSubpathVRP,
    ConstraintHandlerVRP,
)
from discrete_optimization.vrp.solver.vrp_cpsat_solver import CpSatVrpSolver
from discrete_optimization.vrp.vrp_model import Customer2D, VrpProblem2D, VrpSolution
from discrete_optimization.vrp.vrp_parser import get_data_available, parse_file


def compute_nb_nodes_in_path(solution: VrpSolution):
    expected_nodes = solution.problem.customer_count - len(
        set(solution.problem.start_indexes + solution.problem.end_indexes)
    )
    actual_nb_nodes = sum(len(x) for x in solution.list_paths)
    assert expected_nodes == actual_nb_nodes


@pytest.mark.parametrize(
    "optional_node,cut_transition", list(product([True, False], repeat=2))
)
def test_cpsat_vrp(optional_node, cut_transition):
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file)
    solver = CpSatVrpSolver(problem=problem)
    solver.init_model(optional_node=optional_node, cut_transition=cut_transition)
    p = ParametersCP.default_cpsat()
    p.nb_process = 10
    res = solver.solve(parameters_cp=p, time_limit=10)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(problem.evaluate(sol))
    assert problem.satisfy(sol)
    if not optional_node:
        compute_nb_nodes_in_path(sol)

    # test warm start
    # start_solution = GreedyVRPSolver(problem=vrp_model).solve(time_limit=20).get_best_solution_fit()[0]
    start_solution = res[1][0]

    # first solution is not start_solution
    assert res[0][0].list_paths != start_solution.list_paths

    # warm start at first solution
    solver = CpSatVrpSolver(problem=problem)
    solver.init_model(optional_node=optional_node, cut_transition=cut_transition)
    solver.set_warm_start(start_solution)
    # force first solution to be the hinted one
    res = solver.solve(
        parameters_cp=p,
        time_limit=10,
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
    )
    assert res[0][0].list_paths == start_solution.list_paths


def test_cpsat_lns_vrp():
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file)
    problem.vehicle_capacities = [
        problem.vehicle_capacities[i] for i in range(problem.vehicle_count)
    ]
    print(problem)
    solver = CpSatVrpSolver(problem=problem)
    solver.init_model(optional_node=False, cut_transition=False)
    solver_lns = LNS_OrtoolsCPSat(
        problem=problem,
        subsolver=solver,
        initial_solution_provider=None,
        constraint_handler=ConstraintHandlerMix(
            problem=problem,
            list_constraints_handler=[
                ConstraintHandlerVRP(problem, 0.5),
                ConstraintHandlerSubpathVRP(problem, 0.5),
            ],
            list_proba=[0.5, 0.5],
        ),
    )
    p = ParametersCP.default_cpsat()
    res = solver_lns.solve(
        skip_initial_solution_provider=True,
        nb_iteration_lns=30,
        parameters_cp=p,
        time_limit_subsolver=10,
    )
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(problem.evaluate(sol))
    assert problem.satisfy(sol)
    compute_nb_nodes_in_path(sol)


@pytest.mark.parametrize(
    "optional_node,diff_start_end", list(product([True, False], repeat=2))
)
def test_cpsat_vrp_on_tsp(optional_node, diff_start_end):
    from discrete_optimization.tsp.tsp_parser import (
        TSPModel2D,
        get_data_available,
        parse_file,
    )

    file = [f for f in get_data_available() if "tsp_51_1" in f][0]
    if diff_start_end:
        problem_tsp: TSPModel2D = parse_file(
            file_path=file, start_index=0, end_index=10
        )
    else:
        problem_tsp: TSPModel2D = parse_file(file_path=file, start_index=0, end_index=0)
    problem = VrpProblem2D(
        vehicle_count=1,
        vehicle_capacities=[100000],
        customer_count=problem_tsp.node_count,
        customers=[
            Customer2D(
                name=str(i),
                demand=0,
                x=problem_tsp.list_points[i].x,
                y=problem_tsp.list_points[i].y,
            )
            for i in range(len(problem_tsp.list_points))
        ],
        start_indexes=[problem_tsp.start_index],
        end_indexes=[problem_tsp.end_index],
    )
    solver = CpSatVrpSolver(problem=problem)
    solver.init_model(optional_node=optional_node, cut_transition=False)
    p = ParametersCP.default_cpsat()
    p.nb_process = 10
    res = solver.solve(parameters_cp=p, time_limit=20)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    assert problem.satisfy(sol)
    if not optional_node:
        compute_nb_nodes_in_path(sol)
        sol_tsp = SolutionTSP(
            problem=problem_tsp,
            start_index=problem_tsp.start_index,
            end_index=problem_tsp.end_index,
            permutation=sol.list_paths[0],
        )
        assert problem_tsp.satisfy(sol_tsp)
