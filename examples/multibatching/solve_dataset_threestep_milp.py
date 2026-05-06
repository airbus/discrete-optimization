#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example solving a multibatching dataset with two-step approach (CPSat + Greedy packing)."""

import logging

from discrete_optimization.binpack.solvers.cpsat import (
    CpSatBinPackSolver,
    ModelingBinPack,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.multibatching.parser import get_data_available, parse_file
from discrete_optimization.multibatching.problem import MultibatchingSolution
from discrete_optimization.multibatching.solvers.lp import (
    GurobiMultibatchingSolver,
    MathOptMultibatchingSolverUnitFlow,
)
from discrete_optimization.multibatching.solvers.packing_subproblem import (
    PackingViaBinPacking,
)
from discrete_optimization.multibatching.solvers.two_steps import (
    TwoStepMultibatchingSolver,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("Multibatching Two-Step Solver Example (CPSat Flow + Greedy Packing)")
    print("=" * 80)

    # 1. Load dataset
    print("\n[1/5] Loading dataset...")
    try:
        datasets = get_data_available()
        if not datasets:
            print("No datasets found. Please run:")
            print(
                ">>> from discrete_optimization.datasets import fetch_data_from_multibatching"
            )
            print(">>> fetch_data_from_multibatching()")
            return
        dataset_path = datasets[0]
        print(f"      Using dataset: {dataset_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Parse the problem
    print("\n[2/5] Parsing problem...")
    problem = parse_file(
        dataset_path,
        scale_capacity=1.0 / 10**4,  # Scale down capacities
        scale_size=1.0 / 10**4,  # Scale down product sizes
        scale_co2=1.0 / 10**6,
    )
    print(f"      Problem loaded:")
    print(f"      - {problem.nb_locations} locations")
    print(f"      - {problem.nb_products} products")
    print(f"      - {problem.nb_transport_types} transport types")
    print(f"      - {problem.nb_transport_links} transport links")

    # 3. Configure solvers
    print("\n[3/5] Configuring two-step solver...")

    # Configure CP parameters for parallel solving
    parameters_cp = ParametersCp.default_cpsat()
    parameters_cp.nb_process = 8
    from discrete_optimization.generic_tools.callbacks.loggers import (
        ProblemEvaluateLogger,
    )

    # Configure CPSat flow solver with longer timeout and parallel workers
    flow_solver_config = SubBrick(
        cls=GurobiMultibatchingSolver,
        kwargs={
            "parameters_cp": parameters_cp,
            "callbacks": ProblemEvaluateLogger(logging.INFO, logging.INFO),
            "scaling_factor": 1000,
            "time_limit": 100,
        },
    )
    # Configure greedy packing solver
    packing_solver_config = SubBrick(
        cls=PackingViaBinPacking,
        kwargs={
            "bin_packing_solver": SubBrick(
                cls=CpSatBinPackSolver,
                kwargs=dict(
                    modeling=ModelingBinPack.SCHEDULING,
                    parameters_cp=ParametersCp.default_cpsat(),
                ),
            ),
            "time_limit_per_link": 2,
        },
    )
    print("      Flow solver: CPSat (FLOW modeling)")
    print(f"      - Time limit: {flow_solver_config.kwargs['time_limit']}s")
    print(f"      - Workers: {parameters_cp.nb_process}")
    print("      Packing solver: Cpsat")
    # 4. Solve
    print("\n[4/5] Solving...")
    print("      This may take several minutes...")
    two_step_config = SubBrick(
        TwoStepMultibatchingSolver,
        {"flow_solver": flow_solver_config, "packing_solver": packing_solver_config},
    )
    milp_no_delta = SubBrick(
        MathOptMultibatchingSolverUnitFlow,
        kwargs=dict(time_limit=int(100)),
        kwargs_from_solution={
            "max_trips_per_link": lambda sol: sol.problem.get_max_nb_trips(sol)
        },
    )
    from discrete_optimization.generic_tools.sequential_metasolver import (
        SequentialMetasolver,
    )

    seq_solver = SequentialMetasolver(
        problem, list_subbricks=[two_step_config, milp_no_delta]
    )
    result_storage = seq_solver.solve()
    # 5. Analyze results
    print("\n[5/5] Results:")
    print("=" * 80)
    if len(result_storage) == 0:
        print("No solution found within time limit.")
        return
    solution, fitness = result_storage[-1]
    solution: MultibatchingSolution
    print(f"\nBest solution found:")
    print(f"  - Objective value: {fitness}")
    print(f"  - Number of flows: {len(solution.list_flows)}")
    # Evaluate solution
    print("\nEvaluating solution...")
    evaluation = problem.evaluate(solution)
    print(evaluation)
    print(f"  - Transport cost: {evaluation.get('transport', 'N/A')}")
    print(f"  - Emission cost: {evaluation.get('emission', 'N/A')}")
    # Validate solution
    print("\nValidating solution...")
    is_feasible = problem.satisfy(solution)
    if is_feasible:
        print("  ✓ Solution is FEASIBLE")
        print("\nSUCCESS: Found a valid solution to the multibatching problem!")
    else:
        print("  ✗ Solution is NOT feasible")
        print("\nWARNING: Solution violates some constraints.")
    # Summary statistics
    print("\n" + "=" * 80)
    print(f"Total solutions explored: {len(result_storage)}")
    print(f"Best solution fitness: {fitness}")
    print(f"Feasible: {is_feasible}")
    print("=" * 80)


if __name__ == "__main__":
    main()
