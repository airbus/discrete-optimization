"""
Example: Solving RC-ALBP with CP-SAT

This example demonstrates:
1. Loading an RCPSP instance and converting to RC-ALBP
2. Solving with both FOLDED and CALENDAR modeling approaches
3. Comparing results and visualizing solutions
"""

from discrete_optimization.alb.rcalbp.problem import RCALBPSolution
from discrete_optimization.alb.rcalbp.solvers.cpsat import (
    CpSatRcAlbpSolver,
    ModelingShared,
)
from discrete_optimization.alb.rcalbp.utils import load_rcpsp_as_albp
from discrete_optimization.generic_tools.cp_tools import ParametersCp


def main():
    # Load problem from RCPSP instance
    print("Loading RC-ALBP problem from RCPSP instance...")
    problem = load_rcpsp_as_albp(instance_name="j301_1", nb_stations=3, seed=42)

    print(f"Problem: {problem.nb_tasks} tasks, {problem.nb_stations} stations")
    print(
        f"Resources: {len(problem.resources)} station-specific, "
        f"{len(problem.shared_resources)} shared"
    )
    print()

    # Solve with FOLDED modeling
    print("=" * 60)
    print("Solving with FOLDED modeling...")
    print("=" * 60)

    solver = CpSatRcAlbpSolver(problem)
    params_cp = ParametersCp.default_cpsat()

    result_storage = solver.solve(
        modeling=ModelingShared.FOLDED,
        parameters_cp=params_cp,
        time_limit=30,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )

    if len(result_storage) > 0:
        solution: RCALBPSolution = result_storage.get_best_solution()
        print(f"\nBest solution found:")
        print(f"  Cycle time: {solution.cycle_time}")
        print(f"  Valid: {problem.satisfy(solution)}")

        # Show task assignments
        print(f"\nTask assignments:")
        for station in problem.stations:
            tasks_on_station = [
                t for t in problem.tasks if solution.task_assignment[t] == station
            ]
            print(f"  {station}: {tasks_on_station}")
    else:
        print("No solution found")

    print()

    # Solve with CALENDAR modeling
    print("=" * 60)
    print("Solving with CALENDAR modeling...")
    print("=" * 60)

    solver = CpSatRcAlbpSolver(problem)

    result_storage = solver.solve(
        modeling=ModelingShared.CALENDAR,
        parameters_cp=params_cp,
        time_limit=30,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )

    if len(result_storage) > 0:
        solution = result_storage.get_best_solution()
        print(f"\nBest solution found:")
        print(f"  Cycle time: {solution.cycle_time}")
        print(f"  Valid: {problem.satisfy(solution)}")

        # Show task assignments
        print(f"\nTask assignments:")
        for station in problem.stations:
            tasks_on_station = [
                t for t in problem.tasks if solution.task_assignment[t] == station
            ]
            print(f"  {station}: {tasks_on_station}")
    else:
        print("No solution found")


if __name__ == "__main__":
    main()
