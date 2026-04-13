"""Example 1: Simple Problem Transformation.

This example demonstrates:
- Transforming RCPSP to RCPSPMultiskill
- Solving via TransformationSolver
- Automatic back-transformation to original problem space
"""

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem import RcpspSolution
from discrete_optimization.rcpsp.transformations import RcpspToMultiskillTransformation
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)


def main():
    """Run simple transformation example."""
    print("=" * 80)
    print("Example 1: Simple Problem Transformation (RCPSP → RCPSPMultiskill)")
    print("=" * 80)

    # Load a small RCPSP instance
    files = get_data_available()
    file = [f for f in files if "j301_1" in f][0]
    rcpsp_problem = parse_file(file)  # Small instance

    print(f"\nOriginal problem: {type(rcpsp_problem).__name__}")
    print(f"  - Tasks: {rcpsp_problem.n_jobs}")
    print(f"  - Resources: {rcpsp_problem.resources_list}")
    print(f"  - Horizon: {rcpsp_problem.horizon}")

    # Create transformation
    print("\nCreating transformation...")
    transformation = RcpspToMultiskillTransformation()

    # Transform problem (for inspection)
    multiskill_problem = transformation.transform_problem(rcpsp_problem)
    print(f"\nTransformed problem: {type(multiskill_problem).__name__}")
    print(f"  - Tasks: {multiskill_problem.n_jobs}")
    print(f"  - Skills: {multiskill_problem.skills_set}")
    print(f"  - Employees: {len(multiskill_problem.employees)}")

    # Create TransformationSolver
    print("\nCreating TransformationSolver...")
    from discrete_optimization.generic_tools.cp_tools import ParametersCp

    p = ParametersCp.default_cpsat()
    solver_brick = SubBrick(
        cls=CpSatMultiskillRcpspSolver, kwargs={"time_limit": 10, "parameters_cp": p}
    )
    solver_name = "CPSatMSRcpspSolver"

    solver = TransformationSolver(
        transformation=transformation,
        source_problem=rcpsp_problem,
        solver_brick=solver_brick,
    )

    print(f"  - Wrapped solver: {solver_name}")
    print(f"  - Transformation: {transformation}")

    # Solve
    print("\nSolving...")
    solver.init_model()
    result = solver.solve()

    print(f"  - Found {len(result)} solutions")
    print(f"  - Solver status: {solver.status_solver}")

    # Check result
    if len(result) > 0:
        best_solution: RcpspSolution = result.get_best_solution()
        best_fit = result.get_best_solution_fit()[1]

        print(f"\nBest solution:")
        print(f"  - Type: {type(best_solution).__name__}")  # Should be RcpspSolution!
        print(f"  - Fitness: {best_fit}")
        print(f"  - Feasible: {rcpsp_problem.satisfy(best_solution)}")

        # Show makespan
        makespan = rcpsp_problem.evaluate(best_solution)["makespan"]
        print(f"  - Makespan: {makespan}")
        # Verify solution is in original problem space
        assert best_solution.problem == rcpsp_problem, (
            "Solution should reference original problem!"
        )
        print("\n✓ Solution successfully back-transformed to original problem space")
    else:
        print("\n✗ No solution found")

    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("1. TransformationSolver automatically transforms the problem")
    print("2. Solutions are back-transformed to original problem type (RcpspSolution)")
    print("3. You can use any solver for the transformed problem")
    print("4. The transformation is reusable across different problem instances")
    print("=" * 80)


if __name__ == "__main__":
    main()
