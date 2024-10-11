#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import random

import numpy as np
import pytest
from ortools.math_opt.python import mathopt

import discrete_optimization.tsp.parser as tsp_parser
import discrete_optimization.vrp.parser as vrp_parser
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.gpdp.builders.instance_builders import create_selective_tsp
from discrete_optimization.gpdp.plot import plot_gpdp_solution
from discrete_optimization.gpdp.problem import (
    GpdpSolution,
    ProxyClass,
    build_pruned_problem,
)
from discrete_optimization.gpdp.solvers.lp_iterative import (
    GurobiLazyConstraintLinearFlowGpdpSolver,
    GurobiLinearFlowGpdpSolver,
    MathOptLinearFlowGpdpSolver,
)

try:
    import gurobipy as grb
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True

epsilon = 0.000001


@pytest.fixture
def random_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture(
    params=[
        GurobiLinearFlowGpdpSolver,
        MathOptLinearFlowGpdpSolver,
        GurobiLazyConstraintLinearFlowGpdpSolver,
    ]
)
def solver_class(request):
    solver_class = request.param
    if issubclass(solver_class, GurobiMilpSolver) and not gurobi_available:
        pytest.skip("You need Gurobi to test this solver.")
    return solver_class


@pytest.mark.parametrize(
    "subtour_do_order, subtour_use_indicator, include_subtour",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (False, False, True),
    ],
)
@pytest.mark.parametrize(
    "subtour_consider_only_first_component",
    [True, False],
)
def test_tsp_new_api(
    solver_class,
    subtour_do_order,
    subtour_use_indicator,
    subtour_consider_only_first_component,
    include_subtour,
):
    files_available = tsp_parser.get_data_available()
    file_path = [f for f in files_available if "tsp_5_1" in f][0]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_to_gpdp(tsp_model=tsp_model, compute_graph=True)
    linear_flow_solver = solver_class(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True,
        include_capacity=False,
        include_time_evolution=False,
        subtour_do_order=subtour_do_order,
        subtour_use_indicator=subtour_use_indicator,
        subtour_consider_only_first_component=subtour_consider_only_first_component,
        include_subtour=include_subtour,
    )
    res = linear_flow_solver.solve(
        time_limit_subsolver=100,
        do_lns=False,
        nb_iteration_max=20,
        include_subtour=False,
    )
    assert isinstance(res, ResultStorage)
    sol = res.get_best_solution()
    assert isinstance(sol, GpdpSolution)
    assert len(sol.times) == 0
    # check origin and target for each trajectory
    for v, trajectory in sol.trajectories.items():
        assert trajectory[0] == gpdp.origin_vehicle[v]
        assert trajectory[-1] == gpdp.target_vehicle[v]
    # check size of trajectories
    nb_nodes_visited = sum([len(traj) for traj in sol.trajectories.values()])
    assert nb_nodes_visited == len(gpdp.all_nodes)


def test_tsp_cb(solver_class, random_seed):
    files_available = tsp_parser.get_data_available()
    file_path = [f for f in files_available if "tsp_5_1" in f][0]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_to_gpdp(tsp_model=tsp_model, compute_graph=True)
    linear_flow_solver = solver_class(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    linear_flow_solver.set_random_seed(random_seed)
    iteration_stopper = NbIterationStopper(nb_iteration_max=2)
    res = linear_flow_solver.solve(
        time_limit_subsolver=100,
        do_lns=False,
        nb_iteration_max=20,
        include_subtour=False,
        callbacks=[iteration_stopper],
    )
    assert (
        iteration_stopper.nb_iteration > 0
        and iteration_stopper.nb_iteration <= iteration_stopper.nb_iteration_max
    )


def test_tsp_new_api_with_time(solver_class):
    files_available = tsp_parser.get_data_available()
    file_path = [f for f in files_available if "tsp_5_1" in f][0]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_to_gpdp(tsp_model=tsp_model, compute_graph=True)
    linear_flow_solver = solver_class(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=True
    )
    res = linear_flow_solver.solve(
        time_limit_subsolver=100,
        do_lns=False,
        nb_iteration_max=20,
        include_subtour=False,
        mathopt_enable_output=True,
        mathopt_solver_type=mathopt.SolverType.GSCIP,
    )
    assert isinstance(res, ResultStorage)
    sol = res.get_best_solution()
    assert isinstance(sol, GpdpSolution)
    # check origin + target + times increasing for each trajectory
    for v, trajectory in sol.trajectories.items():
        assert trajectory[0] == gpdp.origin_vehicle[v]
        assert trajectory[-1] == gpdp.target_vehicle[v]
        for i in range(len(trajectory) - 1):
            assert (
                sol.times[trajectory[i]]
                + gpdp.time_delta[trajectory[i]][trajectory[i + 1]]
                <= sol.times[trajectory[i + 1]] + epsilon
            )
    # check size of trajectories
    nb_nodes_visited = sum([len(traj) for traj in sol.trajectories.values()])
    assert nb_nodes_visited == len(gpdp.all_nodes)


def test_tsp(solver_class):
    files_available = tsp_parser.get_data_available()
    file_path = [f for f in files_available if "tsp_5_1" in f][0]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_to_gpdp(tsp_model=tsp_model, compute_graph=True)
    linear_flow_solver = solver_class(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    res = linear_flow_solver.solve_iterative(
        time_limit_subsolver=100,
        do_lns=False,
        nb_iteration_max=20,
        include_subtour=False,
    )
    sol: GpdpSolution = res.get_best_solution()
    plot_gpdp_solution(sol, gpdp)


def test_tsp_simplified(solver_class):
    files_available = tsp_parser.get_data_available()
    file_path = [f for f in files_available if "tsp_5_1" in f][0]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_to_gpdp(tsp_model=tsp_model, compute_graph=True)
    gpdp = build_pruned_problem(gpdp, compute_graph=True)
    linear_flow_solver = solver_class(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    res = linear_flow_solver.solve_iterative(
        time_limit_subsolver=100,
        do_lns=False,
        nb_iteration_max=20,
        include_subtour=False,
    )
    sol: GpdpSolution = res.get_best_solution()
    plot_gpdp_solution(sol, gpdp)


def test_vrp(solver_class, random_seed):
    files_available = vrp_parser.get_data_available()
    file_path = [f for f in files_available if "vrp_16_3_1" in f][0]
    vrp_problem = vrp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_vrp_to_gpdp(vrp_problem=vrp_problem, compute_graph=True)
    linear_flow_solver = solver_class(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    linear_flow_solver.set_random_seed(random_seed)
    res = linear_flow_solver.solve_iterative(
        time_limit_subsolver=100,
        do_lns=False,
        nb_iteration_max=20,
        include_subtour=False,
        mathopt_solver_type=mathopt.SolverType.GSCIP,
        mathopt_enable_output=True,
    )
    sol: GpdpSolution = res.get_best_solution()
    plot_gpdp_solution(sol, gpdp)


@pytest.mark.skip(reason="build_pruned_problem() is buggy for now.")
def test_vrp_simplified(solver_class):
    files_available = vrp_parser.get_data_available()
    file_path = [f for f in files_available if "vrp_16_3_1" in f][0]
    vrp_problem = vrp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_vrp_to_gpdp(vrp_problem=vrp_problem, compute_graph=True)
    gpdp = build_pruned_problem(gpdp, compute_graph=True)
    linear_flow_solver = solver_class(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    res = linear_flow_solver.solve_iterative(
        time_limit_subsolver=100,
        do_lns=False,
        nb_iteration_max=20,
        include_subtour=False,
    )
    sol: GpdpSolution = res.get_best_solution()
    plot_gpdp_solution(sol, gpdp)


@pytest.mark.parametrize(
    "subtour_do_order, subtour_use_indicator",
    [(False, False), (True, False), (True, True)],
)
@pytest.mark.parametrize(
    "subtour_consider_only_first_component",
    [True, False],
)
def test_selective_tsp(
    random_seed,
    solver_class,
    subtour_do_order,
    subtour_use_indicator,
    subtour_consider_only_first_component,
):
    gpdp = create_selective_tsp(nb_nodes=20, nb_vehicles=1, nb_clusters=4)
    linear_flow_solver = solver_class(problem=gpdp)
    kwargs_init_model = dict(
        one_visit_per_node=False,
        one_visit_per_cluster=True,
        include_capacity=False,
        include_time_evolution=False,
        subtour_do_order=subtour_do_order,
        subtour_use_indicator=subtour_use_indicator,
        subtour_consider_only_first_component=subtour_consider_only_first_component,
    )
    linear_flow_solver.init_model(**kwargs_init_model)
    linear_flow_solver.set_random_seed(random_seed)
    kwargs_solve = dict(
        time_limit_subsolver=100,
        do_lns=False,
        nb_iteration_max=20,
        include_subtour=False,
    )
    result_storage = linear_flow_solver.solve_iterative(**kwargs_solve)
    sol: GpdpSolution = result_storage.get_best_solution()
    plot_gpdp_solution(sol, gpdp)

    # test warm start
    start_solution: GpdpSolution = result_storage[1][0]

    # first solution is not start_solution
    assert result_storage[0][0].trajectories != start_solution.trajectories

    # warm start at first solution
    linear_flow_solver = solver_class(problem=gpdp)
    linear_flow_solver.init_model(**kwargs_init_model)
    linear_flow_solver.set_random_seed(random_seed)
    linear_flow_solver.set_warm_start(start_solution)

    # force first solution to be the hinted one
    result_storage2 = linear_flow_solver.solve(**kwargs_solve)

    print([sol.trajectories for sol, fit in result_storage])
    print([sol.trajectories for sol, fit in result_storage2])

    assert result_storage2[0][0].trajectories == start_solution.trajectories


def test_selective_vrp(solver_class):
    gpdp = create_selective_tsp(nb_nodes=20, nb_vehicles=3, nb_clusters=4)
    linear_flow_solver = solver_class(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=False,
        one_visit_per_cluster=True,
        include_capacity=False,
        include_time_evolution=False,
    )
    res = linear_flow_solver.solve_iterative(
        time_limit_subsolver=100,
        do_lns=False,
        nb_iteration_max=20,
        include_subtour=False,
    )
    sol: GpdpSolution = res.get_best_solution()
    plot_gpdp_solution(sol, gpdp)


def test_selective_vrp_new_api_with_time(solver_class):
    nb_nodes = 10
    nb_vehicles = 2
    nb_clusters = 4
    gpdp = create_selective_tsp(
        nb_nodes=nb_nodes, nb_vehicles=nb_vehicles, nb_clusters=nb_clusters
    )
    linear_flow_solver = solver_class(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=False,
        one_visit_per_cluster=True,
        include_capacity=False,
        include_time_evolution=True,
    )
    res = linear_flow_solver.solve(
        time_limit_subsolver=100,
        do_lns=False,
        nb_iteration_max=20,
    )
    assert isinstance(res, ResultStorage)
    sol = res.get_best_solution()
    assert isinstance(sol, GpdpSolution)

    # check origin + target + times increasing for each trajectory
    for v, trajectory in sol.trajectories.items():
        assert trajectory[0] == gpdp.origin_vehicle[v]
        assert trajectory[-1] == gpdp.target_vehicle[v]
        for i in range(len(trajectory) - 1):
            assert (
                sol.times[trajectory[i]]
                + gpdp.time_delta[trajectory[i]][trajectory[i + 1]]
                <= sol.times[trajectory[i + 1]] + epsilon
            )
    #  check clusters visited at least once
    node_visited = set()
    for trajectory in sol.trajectories.values():
        node_visited.update(trajectory)
    nb_visit_per_cluster = {
        cluster: len([node for node in nodes if node in node_visited])
        for cluster, nodes in gpdp.clusters_to_node.items()
    }
    for cluster, nb_visit in nb_visit_per_cluster.items():
        assert nb_visit > 0
