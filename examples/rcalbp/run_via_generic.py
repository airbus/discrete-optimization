"""
Example: Solving RC-ALBP with generic scheduling solver

This example demonstrates:
1. Loading an RCPSP instance and converting to RC-ALBP
2. Solving with both FOLDED and CALENDAR modeling approaches
3. Comparing results and visualizing solutions
"""

from discrete_optimization.alb.rcalbp.problem import RCALBPSolution
from discrete_optimization.alb.rcalbp.transformations.to_generic import (
    RCALBPToGenericSchedulingTransformation,
)
from discrete_optimization.alb.rcalbp.utils import (
    load_rcpsp_as_albp,
    visualize_interactive_flow,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    ObjectiveGapStopper,
)
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver


def main():
    # Load problem from RCPSP instance
    print("Loading RC-ALBP problem from RCPSP instance...")
    problem = load_rcpsp_as_albp(instance_name="j301_1", nb_stations=3, seed=42)

    print(f"Problem: {problem.nb_tasks} tasks, {problem.nb_stations} stations")
    print(
        f"Resources: {len(problem.resources)} station-specific, "
        f"{len(problem.shared_resources)} shared"
    )

    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    solver = TransformationSolver(
        transformation=RCALBPToGenericSchedulingTransformation(),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            {
                "time_limit": 100,
                "parameters_cp": p,
                "ortools_cpsat_solver_kwargs": {"log_search_progress": True},
            },
        ),
        source_problem=problem,
    )

    res = solver.solve(
        callbacks=[ObjectiveGapStopper(0, 0), BasicStatsCallback()],
    )

    if len(res) > 0:
        solution: RCALBPSolution = res.get_best_solution()
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
        visualize_interactive_flow(problem, solution)
    else:
        print("No solution found")


if __name__ == "__main__":
    main()
