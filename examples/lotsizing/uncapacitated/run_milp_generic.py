"""Example script demonstrating the generic MILP solver for uncapacitated lot sizing.

This script shows how to use the GenericLotSizingMilp solver to solve an
uncapacitated single-item lot sizing problem.
"""

import logging

from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.lotsizing.generic_solver.milp.generic_lotsizing_milp import (
    MathOptGenericLotSizingMilp,
)
from discrete_optimization.lotsizing.uncapacitatedsingleitem import (
    generate_random_instance,
)
from discrete_optimization.lotsizing.utils import (
    plot_inventory_and_costs,
    plot_production_schedule,
    plot_solution_summary,
)

logging.basicConfig(level=logging.INFO)


def main():
    """Run the generic MILP solver on an uncapacitated lot sizing instance."""
    # Load a problem instance
    problem = generate_random_instance(
        horizon=30,
        avg_demand=5,
        setup_cost=1,
        production_cost=5,
        inventory_cost=3,
        seed=42,
    )

    # Create MILP solver
    solver = MathOptGenericLotSizingMilp(problem)

    # Initialize model
    solver.init_model()

    # Solve with time limit
    parameters_milp = ParametersMilp.default()
    res = solver.solve(
        parameters_milp=parameters_milp,
        time_limit=30,
        mathopt_enable_output=True,
    )

    # Extract best solution
    sol = res[-1][0]

    # Display results
    print(f"Solution fitness: {solver.aggreg_from_sol(sol)}")
    print(f"Evaluation: {problem.evaluate(sol)}")
    print(f"Feasible: {problem.satisfy(sol)}")

    # Visualize solution
    plot_solution_summary(problem, sol)
    plot_inventory_and_costs(problem, sol)
    plot_production_schedule(problem, sol)
    plt.show()


if __name__ == "__main__":
    main()
