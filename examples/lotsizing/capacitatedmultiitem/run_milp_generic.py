"""Example script demonstrating the generic MILP solver for lot sizing.

This script shows how to use the GenericLotSizingMilp solver with the
MathOpt API (OR-Tools) to solve a capacitated multi-item lot sizing problem.
"""

import logging

from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.lp_tools import ParametersMilp, mathopt
from discrete_optimization.lotsizing.capacitatedmultiitem.parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.lotsizing.generic_solver.milp.generic_lotsizing_milp import (
    ChangeoverModel,
    MathOptGenericLotSizingMilp,
)
from discrete_optimization.lotsizing.utils import (
    plot_inventory_and_costs,
    plot_production_schedule,
    plot_solution_summary,
)

logging.basicConfig(level=logging.INFO)


def main():
    """Run the generic MILP solver on a lot sizing instance."""
    # Load a problem instance
    file = [f for f in get_data_available() if "PSP_100_1.psp" in f][0]
    problem = parse_file(file)

    # Create MILP solver
    solver = MathOptGenericLotSizingMilp(problem)

    # Initialize model with default hyperparameters
    # Default: FLOW_BASED (network flow formulation - best performance)
    # Other options: ChangeoverModel.BIG_M, ChangeoverModel.STATE_BASED
    solver.init_model(modeling_changeover=ChangeoverModel.FLOW_BASED)

    # Solve with time limit
    parameters_milp = ParametersMilp.default()
    res = solver.solve(
        mathopt_solver_type=mathopt.SolverType.CP_SAT,
        parameters_milp=parameters_milp,
        time_limit=200,
        mathopt_enable_output=True,
    )

    # Extract best solution
    sol = res[-1][0]

    # Display results
    print(f"Solution fitness: {solver.aggreg_from_sol(sol)}")
    if "known_bound" in problem.infos:
        print(f"Known bound: {problem.infos['known_bound']}")
    print(f"Evaluation: {problem.evaluate(sol)}")
    print(f"Feasible: {problem.satisfy(sol)}")

    # Visualize solution
    plot_solution_summary(problem, sol)
    plot_inventory_and_costs(problem, sol)
    plot_production_schedule(problem, sol)
    plt.show()


if __name__ == "__main__":
    main()
