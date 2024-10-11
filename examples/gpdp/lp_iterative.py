#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt

from discrete_optimization.gpdp.builders.instance_builders import create_ortools_example
from discrete_optimization.gpdp.plot import plot_gpdp_solution
from discrete_optimization.gpdp.problem import GpdpSolution
from discrete_optimization.gpdp.solvers.lp_iterative import GurobiLinearFlowGpdpSolver

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
    linear_flow_solver = GurobiLinearFlowGpdpSolver(problem=gpdp)
    linear_flow_solver.init_model(
        one_visit_per_node=True,
        one_visit_per_cluster=False,
        include_capacity=True,
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
    plt.show()


if __name__ == "__main__":
    example_ortools_example()
