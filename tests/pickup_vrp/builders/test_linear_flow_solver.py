#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest
from matplotlib import pyplot as plt

import discrete_optimization.tsp.tsp_parser as tsp_parser
import discrete_optimization.vrp.vrp_parser as vrp_parser
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.pickup_vrp.builders.instance_builders import (
    create_selective_tsp,
)
from discrete_optimization.pickup_vrp.gpdp import (
    GPDPSolution,
    ProxyClass,
    build_pruned_problem,
)
from discrete_optimization.pickup_vrp.plots.gpdp_plot_utils import plot_gpdp_solution
from discrete_optimization.pickup_vrp.solver.lp_solver import (
    LinearFlowSolver,
    ParametersMilp,
)

try:
    import gurobipy as grb
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True

epsilon = 0.000001


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_tsp_new_api():
    files_available = tsp_parser.get_data_available()
    file_path = [f for f in files_available if "tsp_5_1" in f][0]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_model_gpdp(tsp_model=tsp_model, compute_graph=True)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    p = ParametersMilp.default()

    p.time_limit = 100
    res = linear_flow_solver.solve(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    assert isinstance(res, ResultStorage)
    sol = res.get_best_solution()
    assert isinstance(sol, GPDPSolution)
    assert len(sol.times) == 0
    # check origin and target for each trajectory
    for v, trajectory in sol.trajectories.items():
        assert trajectory[0] == gpdp.origin_vehicle[v]
        assert trajectory[-1] == gpdp.target_vehicle[v]
    # check size of trajectories
    nb_nodes_visited = sum([len(traj) for traj in sol.trajectories.values()])
    assert nb_nodes_visited == len(gpdp.all_nodes)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_tsp_cb():
    files_available = tsp_parser.get_data_available()
    file_path = [f for f in files_available if "tsp_5_1" in f][0]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_model_gpdp(tsp_model=tsp_model, compute_graph=True)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    p = ParametersMilp.default()

    p.time_limit = 100
    iteration_stopper = NbIterationStopper(nb_iteration_max=2)
    res = linear_flow_solver.solve(
        parameters_milp=p,
        do_lns=False,
        nb_iteration_max=20,
        include_subtour=False,
        callbacks=[iteration_stopper],
    )
    assert (
        iteration_stopper.nb_iteration > 0
        and iteration_stopper.nb_iteration <= iteration_stopper.nb_iteration_max
    )


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_tsp_new_api_with_time():
    files_available = tsp_parser.get_data_available()
    file_path = [f for f in files_available if "tsp_5_1" in f][0]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_model_gpdp(tsp_model=tsp_model, compute_graph=True)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=True
    )
    p = ParametersMilp.default()

    p.time_limit = 100
    res = linear_flow_solver.solve(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    assert isinstance(res, ResultStorage)
    sol = res.get_best_solution()
    assert isinstance(sol, GPDPSolution)
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


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_tsp():
    files_available = tsp_parser.get_data_available()
    file_path = [f for f in files_available if "tsp_5_1" in f][0]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_model_gpdp(tsp_model=tsp_model, compute_graph=True)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    p = ParametersMilp.default()

    p.time_limit = 100
    res = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    sol: GPDPSolution = res.get_best_solution()
    plot_gpdp_solution(sol, gpdp)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_tsp_simplified():
    files_available = tsp_parser.get_data_available()
    file_path = [f for f in files_available if "tsp_5_1" in f][0]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_model_gpdp(tsp_model=tsp_model, compute_graph=True)
    gpdp = build_pruned_problem(gpdp, compute_graph=True)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    p = ParametersMilp.default()

    p.time_limit = 100
    res = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    sol: GPDPSolution = res.get_best_solution()
    plot_gpdp_solution(sol, gpdp)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_vrp():
    files_available = vrp_parser.get_data_available()
    file_path = [f for f in files_available if "vrp_16_3_1" in f][0]
    vrp_model = vrp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_vrp_model_to_gpdp(vrp_model=vrp_model, compute_graph=True)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    p = ParametersMilp.default()

    p.time_limit = 100
    res = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    sol: GPDPSolution = res.get_best_solution()
    plot_gpdp_solution(sol, gpdp)


@pytest.mark.skip(reason="build_pruned_problem() is buggy for now.")
@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_vrp_simplified():
    files_available = vrp_parser.get_data_available()
    file_path = [f for f in files_available if "vrp_16_3_1" in f][0]
    vrp_model = vrp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_vrp_model_to_gpdp(vrp_model=vrp_model, compute_graph=True)
    gpdp = build_pruned_problem(gpdp, compute_graph=True)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    p = ParametersMilp.default()
    p.time_limit = 100
    res = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    sol: GPDPSolution = res.get_best_solution()
    plot_gpdp_solution(sol, gpdp)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_selective_tsp():
    gpdp = create_selective_tsp(nb_nodes=20, nb_vehicles=1, nb_clusters=4)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=False,
        one_visit_per_cluster=True,
        include_capacity=False,
        include_time_evolution=False,
    )
    p = ParametersMilp.default()
    p.time_limit = 100
    res = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    sol: GPDPSolution = res.get_best_solution()
    plot_gpdp_solution(sol, gpdp)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_selective_vrp():
    gpdp = create_selective_tsp(nb_nodes=20, nb_vehicles=3, nb_clusters=4)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=False,
        one_visit_per_cluster=True,
        include_capacity=False,
        include_time_evolution=False,
    )
    p = ParametersMilp.default()
    p.time_limit = 100
    res = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    sol: GPDPSolution = res.get_best_solution()
    plot_gpdp_solution(sol, gpdp)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_selective_vrp_new_api_with_time():
    nb_nodes = 10
    nb_vehicles = 2
    nb_clusters = 4
    gpdp = create_selective_tsp(
        nb_nodes=nb_nodes, nb_vehicles=nb_vehicles, nb_clusters=nb_clusters
    )
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=False,
        one_visit_per_cluster=True,
        include_capacity=False,
        include_time_evolution=True,
    )
    p = ParametersMilp.default()
    p.time_limit = 100
    res = linear_flow_solver.solve(
        parameters_milp=p,
        do_lns=False,
        nb_iteration_max=20,
    )
    assert isinstance(res, ResultStorage)
    sol = res.get_best_solution()
    assert isinstance(sol, GPDPSolution)

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


if __name__ == "__main__":
    test_vrp()
