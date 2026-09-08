#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example demonstrating CP-SAT solver on resource-dependent RCPSP problems.

This example shows:
1. Creating a resource-dependent problem manually
2. Generating problems from base RCPSP instances using the generator
3. Solving with different model configurations
4. Validating and visualizing solutions
"""

import matplotlib.pyplot as plt

from discrete_optimization.generic_tasks_tools.plot_utils import (
    plot_ressource_view,
    plot_task_gantt,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp_resource_dependent.generator import (
    DependencyStrategy,
    generate_resource_dependent_problem,
    validate_resource_dependent_problem,
)
from discrete_optimization.rcpsp_resource_dependent.problem import (
    RcpspResourceDependentProblem,
    RcpspResourceDependentSolution,
)
from discrete_optimization.rcpsp_resource_dependent.solvers.cpsat import (
    CpSatRcpspResourceDependentSolver,
)


def create_toy_model():
    resources = {"R1": 5, "R2": 8, "R3": 10, "N1": 20, "N2": 20}
    mode_details = {
        "source": {1: {"duration": 0}},
        "1": {
            1: {"R1": 1, "N1": 5, "duration": 2},
            2: {"R2": 4, "N2": 5, "duration": 1},
        },
        "2": {
            1: {
                "R1": {frozenset([("1", 1)]): 1, frozenset([("1", 2)]): 5},
                "N1": {frozenset([("1", 1)]): 20, frozenset([("1", 2)]): 10},
                "duration": 2,
            },
            2: {
                "R2": {frozenset([("1", 1)]): 1, frozenset([("1", 2)]): 5},
                "N2": 5,
                "duration": 1,
            },
        },
        "3": {
            1: {"R1": 1, "N1": 5, "duration": 3},
            2: {"R2": 4, "N2": 5, "duration": 2},
        },
        "4": {
            1: {"R1": 1, "N1": 5, "duration": 2},
            2: {"R2": 4, "N2": 5, "duration": 3},
        },
        "5": {
            1: {"R1": 1, "N1": 5, "duration": 4},
            2: {"R2": 4, "N2": 5, "duration": 1},
        },
        "6": {
            1: {"R1": 1, "N1": 5, "duration": 2},
            2: {"R2": 4, "N2": 5, "duration": 1},
        },
        "sink": {1: {"duration": 0}},
    }
    successors = {
        "source": ["1", "2"],
        "1": ["3", "4"],
        "2": ["5"],
        "3": ["4"],
        "4": ["6"],
        "5": ["sink"],
        "6": ["sink"],
        "sink": [],
    }
    problem = RcpspResourceDependentProblem(
        resources=resources,
        non_renewable_resources=["N1", "N2"],
        mode_details=mode_details,
        successors=successors,
        horizon=30,
        source_task="source",
        sink_task="sink",
    )
    solver = CpSatRcpspResourceDependentSolver(problem)
    solver.init_model(avoid_interval_optional=False)
    res = solver.solve(
        time_limit=10, ortools_cpsat_solver_kwargs={"log_search_progress": True}
    )
    sol: RcpspResourceDependentSolution = res[-1][0]
    resource_consumption = {}
    total_conso_nr = {r: 0 for r in problem.non_renewable_resources}
    for t in problem.tasks_list:
        for r in problem.cumulative_resources_list:
            resource_consumption[(t, r)] = sol.get_calendar_resource_consumption(r, t)
        for r in problem.non_renewable_resources_list:
            resource_consumption[(t, r)] = sol.get_non_renewable_resource_consumption(
                r, t
            )
            total_conso_nr[r] += resource_consumption[(t, r)]
    for t, r in solver.demands_cumulative_resource_vars:
        assert (
            solver.solver.Value(solver.demands_cumulative_resource_vars[t, r])
            == resource_consumption[(t, r)]
        )
    for r in problem.non_renewable_resources_list:
        assert total_conso_nr[r] <= problem.get_resource_max_capacity(r)
    for t in problem.tasks_list:
        for r in (
            problem.cumulative_resources_list + problem.non_renewable_resources_list
        ):
            print(t, r, ":", resource_consumption[(t, r)])
    print(total_conso_nr)
    print(sol.schedule, "\n", sol.modes)
    print(problem.evaluate(sol), problem.satisfy(sol))
    assert problem.satisfy(sol)

    plot_task_gantt(problem, sol)
    plot_ressource_view(problem, sol)
    plt.show()


def example_with_generator():
    """Example using the problem generator from base RCPSP instances."""
    print("\n" + "=" * 70)
    print("Example: Generating resource-dependent problems from base RCPSP")
    print("=" * 70)

    # Create a small base RCPSP problem
    resources = {"R1": 5, "R2": 8, "N1": 20}
    mode_details = {
        "source": {1: {"duration": 0}},
        "1": {
            1: {"R1": 2, "N1": 3, "duration": 3},
            2: {"R2": 3, "N1": 5, "duration": 2},
        },
        "2": {1: {"R1": 3, "R2": 1, "N1": 4, "duration": 4}},
        "3": {
            1: {"R1": 1, "N1": 2, "duration": 2},
            2: {"R2": 2, "N1": 3, "duration": 3},
        },
        "4": {1: {"R2": 2, "N1": 5, "duration": 2}},
        "sink": {1: {"duration": 0}},
    }
    successors = {
        "source": ["1", "2"],
        "1": ["3"],
        "2": ["4"],
        "3": ["sink"],
        "4": ["sink"],
        "sink": [],
    }
    base_problem = RcpspProblem(
        resources=resources,
        non_renewable_resources=["N1"],
        mode_details=mode_details,
        successors=successors,
        horizon=20,
        source_task="source",
        sink_task="sink",
    )

    print(f"\nBase problem: {len(base_problem.tasks_list)} tasks")
    print(f"Resources: {list(base_problem.resources.keys())}")

    # Test different dependency strategies
    strategies = [
        DependencyStrategy.PREDECESSOR_BASED,
        DependencyStrategy.RESOURCE_CONTENTION,
    ]

    for strategy in strategies:
        print(f"\n--- Dependency Strategy: {strategy.value} ---")

        # Generate resource-dependent variant
        rd_problem = generate_resource_dependent_problem(
            base_problem=base_problem,
            dependency_strategy=strategy,
            dependency_probability=0.4,
            variation_factor=0.3,
            seed=42,
        )

        # Show statistics
        stats = validate_resource_dependent_problem(rd_problem)
        print(f"Generated problem stats:")
        print(f"  - Tasks with dependencies: {stats['num_tasks_with_dependencies']}")
        print(f"  - Dependency ratio: {stats['dependency_ratio']:.1%}")
        print(
            f"  - Tasks having dependencies: {sorted(stats['tasks_with_dependencies'])}"
        )

        # Solve
        print(f"\nSolving with CP-SAT (10s time limit)...")
        solver = CpSatRcpspResourceDependentSolver(rd_problem)
        solver.init_model()
        res = solver.solve(time_limit=10)

        if len(res) > 0:
            sol: RcpspResourceDependentSolution = res[-1][0]
            makespan = rd_problem.evaluate(sol)["makespan"]
            is_feasible = rd_problem.satisfy(sol)
            print(f"  - Solution found: makespan={makespan}, feasible={is_feasible}")

            # Show schedule
            print("\n  Schedule:")
            for task in rd_problem.tasks_list:
                if task not in ["source", "sink"]:
                    start = sol.get_start_time(task)
                    end = sol.get_end_time(task)
                    mode = sol.get_mode(task)
                    print(f"    Task {task}: [{start:2d}, {end:2d}] mode={mode}")
        else:
            print("  - No solution found")


def example_from_psplib():
    """Example loading from PSPLIB and generating resource dependencies."""
    print("\n" + "=" * 70)
    print("Example: Loading from PSPLIB and adding dependencies")
    print("=" * 70)

    files = get_data_available()
    small_files = [f for f in files if "j301" in f]

    if not small_files:
        print("No PSPLIB j301 files found, skipping this example")
        return

    # Load a small instance
    base_problem = parse_file(small_files[0])
    print(f"\nLoaded: {small_files[0]}")
    print(f"  - Tasks: {len(base_problem.tasks_list)}")
    print(f"  - Resources: {list(base_problem.resources.keys())}")

    # Generate resource-dependent version
    rd_problem = generate_resource_dependent_problem(
        base_problem=base_problem,
        dependency_strategy=DependencyStrategy.MIXED,
        dependency_probability=0.25,
        variation_factor=0.4,
        seed=123,
    )

    stats = validate_resource_dependent_problem(rd_problem)
    print(f"\nGenerated resource-dependent variant:")
    print(f"  - Dependent consumptions: {stats['num_dependent_consumptions']}")
    print(f"  - Fixed consumptions: {stats['num_fixed_consumptions']}")
    print(f"  - Dependency ratio: {stats['dependency_ratio']:.1%}")

    # Solve
    print(f"\nSolving with CP-SAT (30s time limit)...")
    solver = CpSatRcpspResourceDependentSolver(rd_problem)
    solver.init_model(avoid_interval_optional_for_cumulative_resources=True)
    res = solver.solve(time_limit=30)

    if len(res) > 0:
        sol: RcpspResourceDependentSolution = res[-1][0]
        resource_consumption = {}
        total_conso_nr = {r: 0 for r in rd_problem.non_renewable_resources}
        for t in rd_problem.tasks_list:
            for r in rd_problem.cumulative_resources_list:
                resource_consumption[(t, r)] = sol.get_calendar_resource_consumption(
                    r, t
                )
            for r in rd_problem.non_renewable_resources_list:
                resource_consumption[(t, r)] = (
                    sol.get_non_renewable_resource_consumption(r, t)
                )
                total_conso_nr[r] += resource_consumption[(t, r)]
        for t, r in solver.demands_cumulative_resource_vars:
            assert (
                solver.solver.Value(solver.demands_cumulative_resource_vars[t, r])
                == resource_consumption[(t, r)]
            )

        print(f"  - Makespan: {rd_problem.evaluate(sol)['makespan']}")
        print(f"  - Feasible: {rd_problem.satisfy(sol)}")
        plot_task_gantt(rd_problem, sol)
        plot_ressource_view(rd_problem, sol)
        plt.show()
    else:
        print("  - No solution found")


if __name__ == "__main__":
    # Run the original toy example
    print("Running manual toy model example...")
    create_toy_model()

    # Run generator examples
    example_with_generator()
    example_from_psplib()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
