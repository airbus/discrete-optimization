"""Example 2: Composite Transformation (Chaining).

This example demonstrates:
- Chaining multiple transformations
- Using CompositeTransformation
- Automatic reverse-order back-transformation

Note: This is a conceptual example. In practice, you would chain different
transformations like RCPSP → Multiskill → Preemptive, but we only have
RCPSP → Multiskill implemented currently.
"""

from discrete_optimization.datasets import get_data_home
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import (
    CompositeTransformation,
    TransformationSolver,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.transformations import RcpspToMultiskillTransformation
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)


def main():
    """Run composite transformation example."""
    print("=" * 80)
    print("Example 2: Composite Transformation (Chaining)")
    print("=" * 80)

    # Load RCPSP instance
    data_folder = f"{get_data_home()}/rcpsp/BL"
    files = get_data_available(data_folder)
    rcpsp_problem = parse_file(files[0])

    print(f"\nOriginal problem: {type(rcpsp_problem).__name__}")
    print(f"  - Tasks: {rcpsp_problem.n_jobs}")

    # Example 1: Single transformation (for comparison)
    print("\n" + "-" * 80)
    print("Approach 1: Single Transformation")
    print("-" * 80)

    t1 = RcpspToMultiskillTransformation()
    print(f"Transformation: {t1}")

    # Show transformed problem
    ms_problem = t1.transform_problem(rcpsp_problem)
    print(f"Result: {type(ms_problem).__name__}")

    # Example 2: Composite transformation (conceptual)
    print("\n" + "-" * 80)
    print("Approach 2: Composite Transformation (Conceptual)")
    print("-" * 80)

    print("\nIn a complete implementation, you would chain:")
    print("  T1: RCPSP → Multiskill")
    print("  T2: Multiskill → Preemptive")
    print("  T3: Preemptive → Multiskill (round-trip!)")
    print("\nUsage:")
    print("  composite = chain_transformations(t1, t2, t3)")
    print("  final_problem = composite.transform_problem(rcpsp_problem)")
    print("\nBack-transformation happens automatically in reverse:")
    print(
        "  S_final → T3⁻¹ → S_intermediate2 → T2⁻¹ → S_intermediate1 → T1⁻¹ → S_original"
    )

    # Example 3: Using composite with solver
    print("\n" + "-" * 80)
    print("Approach 3: Solving with Composite Transformation")
    print("-" * 80)

    # For now, just use single transformation
    composite = CompositeTransformation([t1])
    print(f"\nComposite: {composite}")

    solver_brick = SubBrick(cls=CpSatMultiskillRcpspSolver, kwargs={"time_limit": 10})
    solver_name = "CPSatMSRcpspSolver"
    solver = TransformationSolver(
        transformation=composite,  # Use composite transformation!
        source_problem=rcpsp_problem,
        solver_brick=solver_brick,
    )

    print(f"Wrapped solver: {solver_name}")
    print("\nSolving...")
    solver.init_model()
    result = solver.solve()

    if len(result) > 0:
        best_solution = result.get_best_solution()
        print(f"\n✓ Found solution via composite transformation")
        print(f"  - Solution type: {type(best_solution).__name__}")
        print(f"  - Feasible: {rcpsp_problem.satisfy(best_solution)}")

        makespan = max(d["end_time"] for d in best_solution.rcpsp_schedule.values())
        print(f"  - Makespan: {makespan}")
    else:
        print("\n✗ No solution found")

    # Show how to add more transformations (when available)
    print("\n" + "=" * 80)
    print("How to extend with more transformations:")
    print("=" * 80)
    print("""
# When you have more transformations implemented:
t1 = RcpspToMultiskillTransformation()
t2 = MultiskillToPreemptiveTransformation()
t3 = PreemptiveToRelaxedTransformation()

# Chain them
composite = chain_transformations(t1, t2, t3)

# Or use list syntax
composite = CompositeTransformation([t1, t2, t3])

# Use with solver
solver = TransformationSolver(
    transformation=composite,
    source_problem=original_problem,
    solver_brick=SubBrick(cls=YourSolver, kwargs={...})
)

# Solutions automatically back-transform through the entire chain!
result = solver.solve()
    """)

    print("=" * 80)
    print("Key Takeaways:")
    print("1. CompositeTransformation chains multiple transformations")
    print("2. Back-transformation happens automatically in reverse order")
    print("3. Useful for complex transformation pipelines")
    print("4. Can create round-trip transformations (P1→P2→P3→P2→P1)")
    print("=" * 80)


if __name__ == "__main__":
    main()
