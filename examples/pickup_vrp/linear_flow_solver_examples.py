from discrete_optimization.pickup_vrp.builders.instance_builders import (
    create_ortools_example,
)
from discrete_optimization.pickup_vrp.solver.lp_solver import (
    LinearFlowSolver,
    ParametersMilp,
    plot_solution,
    plt,
)

try:
    import gurobipy as grb
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
try:
    import gurobipy as grb
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


def example_ortools_example():
    gpdp = create_ortools_example()
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
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
    plt.show()


if __name__ == "__main__":
    example_ortools_example()
