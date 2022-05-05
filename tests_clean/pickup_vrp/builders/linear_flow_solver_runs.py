import discrete_optimization.tsp.tsp_parser as tsp_parser
import discrete_optimization.vrp.vrp_parser as vrp_parser
from discrete_optimization.pickup_vrp.builders.instance_builders import (
    create_ortools_example,
    create_selective_tsp,
)
from discrete_optimization.pickup_vrp.gpdp import GPDP, ProxyClass, build_pruned_problem
from discrete_optimization.pickup_vrp.solver.lp_solver import (
    LinearFlowSolver,
    LinearFlowSolverLazyConstraint,
    ParametersMilp,
    plot_solution,
)


def run_on_tsp_vrp():
    vrp = True
    tsp = False
    if tsp:
        print(tsp_parser)
        print(tsp_parser.files_available)
        file_path = tsp_parser.files_available[2]
        file_path = [f for f in tsp_parser.files_available if "tsp_105_1" in f][0]
        tsp_model = tsp_parser.parse_file(file_path)
        gpdp = ProxyClass.from_tsp_model_gpdp(tsp_model=tsp_model)
    else:
        file_path = vrp_parser.files_available[3]
        vrp_model = vrp_parser.parse_file(file_path)
        gpdp = ProxyClass.from_vrp_model_to_gpdp(vrp_model=vrp_model)
    simplify = False
    if simplify:
        gpdp = build_pruned_problem(gpdp)
    print(gpdp.graph.get_nodes())
    print(len(gpdp.graph.get_nodes()))
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    # linear_flow_solver = LinearFlowSolverLazyConstraint(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True, include_capacity=False, include_time_evolution=False
    )
    p = ParametersMilp.default()

    p.TimeLimit = 100
    solutions = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    plot_solution(solutions[-1], gpdp)


def run_selective_tsp():
    gpdp = create_selective_tsp(nb_nodes=200, nb_vehicles=1, nb_clusters=50)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    # linear_flow_solver = LinearFlowSolverLazyConstraint(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=False,
        one_visit_per_cluster=True,
        include_capacity=False,
        include_time_evolution=False,
    )
    p = ParametersMilp.default()
    p.TimeLimit = 100
    solutions = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    plot_solution(solutions[-1], gpdp)


def run_selective_vrp():
    gpdp = create_selective_tsp(nb_nodes=200, nb_vehicles=3, nb_clusters=50)
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    # linear_flow_solver = LinearFlowSolverLazyConstraint(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=False,
        one_visit_per_cluster=True,
        include_capacity=False,
        include_time_evolution=False,
    )
    p = ParametersMilp.default()
    p.TimeLimit = 100
    solutions = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    plot_solution(solutions[-1], gpdp)


def run_ortools_example():
    gpdp = create_ortools_example()
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    # linear_flow_solver = LinearFlowSolverLazyConstraint(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True,
        one_visit_per_cluster=False,
        include_capacity=True,
        include_time_evolution=False,
    )
    p = ParametersMilp.default()
    p.TimeLimit = 100
    solutions = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    plot_solution(solutions[-1], gpdp)


if __name__ == "__main__":
    run_ortools_example()
