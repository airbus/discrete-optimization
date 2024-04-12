#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.pickup_vrp.builders.instance_builders import (
    create_ortools_example,
)
from discrete_optimization.pickup_vrp.gpdp import GPDPSolution
from discrete_optimization.pickup_vrp.plots.gpdp_plot_utils import plot_gpdp_solution
from discrete_optimization.pickup_vrp.solver.lp_solver import (
    LinearFlowSolver,
    ParametersMilp,
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
    p.time_limit = 100
    res = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=20, include_subtour=False
    )
    sol: GPDPSolution = res.get_best_solution()
    plot_gpdp_solution(sol, gpdp)
    plt.show()


if __name__ == "__main__":
    example_ortools_example()
