"""Example: MILP solver for lot sizing problem.

Implementation of the milp3 formulation from:
Ceschia, Di Gaspero, Schaerf (2017)
"""

import logging
import os

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.sequential_metasolver import (
    SequentialMetasolver,
    SubBrick,
)
from discrete_optimization.lotsizing.parser import get_data_available, parse_file
from discrete_optimization.lotsizing.solvers import (
    SimulatedAnnealingLotSizingSolverFast,
)
from discrete_optimization.lotsizing.solvers.lp_milp import MilpLotSizingSolver

logging.basicConfig(level=logging.INFO)


def run_milp():
    """Run MILP solver on a lot sizing instance."""
    instances = get_data_available()
    instance_sizes = [(inst, os.path.getsize(inst)) for inst in instances]
    instance_sizes.sort(key=lambda x: x[1])

    print(f"Available instances: {len(instances)}")

    # Use a medium-hard instance (not the largest to keep it manageable)
    instance_file = [inst for inst in instances if "PSP_100_1" in inst][0]
    print(instances)
    instance_file = [inst for inst in instances if "ps-400-10-80" in inst][0]

    print(f"\nInstance: {instance_file}")
    problem = parse_file(instance_file)

    print(f"Problem info:")
    print(f"  - Number of item types: {problem.nb_items_type}")
    print(f"  - Horizon: {problem.horizon}")
    print(f"  - Total demands: {sum(problem.total_demands_per_item.values())}")

    # Create solver
    solver = MilpLotSizingSolver(problem)
    solver.init_model(use_valid_inequalities=True)

    # Solve
    params = ParametersMilp.default()
    params.time_limit = 300  # 5 minutes
    params.mip_gap_abs = 0.0
    params.mip_gap = 0.01  # 1% gap

    print("\nSolving with MILP (milp3 formulation)...")
    result = solver.solve(
        time_limit=300,
        parameters_milp=params,
        gurobi_solver_kwargs={
            "NoRelHeurTime": 30,
            "Heuristics": 0.2,
            # "Threads": 1
        },
        callbacks=[ProblemEvaluateLogger(logging.INFO, logging.INFO)],
    )

    # Results
    sol = result[-1][0]
    fitness = result[-1][1]

    print(f"\nSolution found:")
    print(f"  - Fitness: {fitness}")
    print(f"  - Evaluation: {problem.evaluate(sol)}")
    print(f"  - Satisfies constraints: {problem.satisfy(sol)}")

    if problem.known_bound is not None:
        gap = (fitness - problem.known_bound) / problem.known_bound * 100
        print(f"  - Gap to known bound: {gap:.2f}%")

    return sol


def run_milp_ws():
    """Run MILP solver on a lot sizing instance."""
    instances = get_data_available()
    instance_sizes = [(inst, os.path.getsize(inst)) for inst in instances]
    instance_sizes.sort(key=lambda x: x[1])

    print(f"Available instances: {len(instances)}")

    # Use a medium-hard instance (not the largest to keep it manageable)
    # instance_file = [inst for inst in instances if "PSP_100_1" in inst][0]
    instance_file = [inst for inst in instances if "PSP_200_3" in inst][0]
    print(f"\nInstance: {instance_file}")
    problem = parse_file(instance_file)
    params = ParametersMilp.default()
    params.time_limit = 300  # 5 minutes
    params.mip_gap_abs = 0.0
    params.mip_gap = 0.01  # 1% gap
    seq_solver = SequentialMetasolver(
        problem=problem,
        list_subbricks=[
            SubBrick(
                SimulatedAnnealingLotSizingSolverFast,
                kwargs=dict(
                    T0=37.0,  # Initial temperature
                    alpha=0.999,  # Cooling rate
                    beta=0.7,  # Insert move probability
                    n_a=12049,  # Moves accepted at each temperature
                    n_s=60240,  # Moves sampled at each temperature
                    max_iterations=10**8,
                    restart_after_no_improvement=0,
                ),
            ),
            SubBrick(
                MilpLotSizingSolver,
                kwargs=dict(
                    parameters_milp=params,
                    time_limit=200,
                    gurobi_solver_kwargs={
                        "NoRelHeurTime": 30,
                        "Heuristics": 0.2,
                        "Threads": 1,
                    },
                ),
            ),
        ],
    )
    result = seq_solver.solve()
    # Results
    sol = result[-1][0]
    fitness = result[-1][1]

    print(f"\nSolution found:")
    print(f"  - Fitness: {fitness}")
    print(f"  - Evaluation: {problem.evaluate(sol)}")
    print(f"  - Satisfies constraints: {problem.satisfy(sol)}")

    if problem.known_bound is not None:
        gap = (fitness - problem.known_bound) / problem.known_bound * 100
        print(f"  - Gap to known bound: {gap:.2f}%")

    return sol


if __name__ == "__main__":
    run_milp()
