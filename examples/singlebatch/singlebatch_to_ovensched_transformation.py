#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example: Transform SingleBatch to OvenSched and solve.

This example demonstrates how to:
1. Create a SingleBatch problem
2. Transform it to OvenSched
3. Solve using OvenSched solvers
4. Transform the solution back to SingleBatch
"""

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    TransformationSolver,
)
from discrete_optimization.singlebatch.problem import Job, SingleBatchProcessingProblem
from discrete_optimization.singlebatch.transformations.to_ovensched import (
    SinglebatchToOvenschedTransformation,
)


def create_example_singlebatch_problem() -> SingleBatchProcessingProblem:
    """Create a small SingleBatch problem for demonstration."""
    jobs = [
        Job(job_id=0, processing_time=5, size=3),
        Job(job_id=1, processing_time=4, size=2),
        Job(job_id=2, processing_time=6, size=4),
        Job(job_id=3, processing_time=3, size=2),
        Job(job_id=4, processing_time=7, size=3),
        Job(job_id=5, processing_time=2, size=1),
    ]

    capacity = 6  # Batch capacity

    return SingleBatchProcessingProblem(jobs=jobs, capacity=capacity)


def example_direct_transformation():
    """Example 1: Direct transformation without solver."""
    print("=" * 80)
    print("Example 1: Direct Problem Transformation")
    print("=" * 80)

    # Create problem
    singlebatch_problem = create_example_singlebatch_problem()

    print(f"\nOriginal SingleBatch Problem:")
    print(f"  Number of jobs: {singlebatch_problem.nb_jobs}")
    print(f"  Capacity: {singlebatch_problem.capacity}")
    print(f"  Jobs:")
    for job in singlebatch_problem.jobs:
        print(f"    {job}")

    # Create transformation
    transformation = SinglebatchToOvenschedTransformation()

    # Transform problem
    ovensched_problem = transformation.transform_problem(singlebatch_problem)

    print(f"\nTransformed OvenSched Problem:")
    print(f"  Number of jobs: {ovensched_problem.n_jobs}")
    print(f"  Number of machines: {ovensched_problem.n_machines}")
    print(f"  Machine capacity: {ovensched_problem.machines_data[0].capacity}")
    print(f"  Tasks:")
    for i, task in enumerate(ovensched_problem.tasks_data):
        print(
            f"    Task {i}: attr={task.attribute}, duration={task.min_duration}, size={task.size}"
        )

    # Create a dummy solution in SingleBatch
    singlebatch_solution = singlebatch_problem.get_dummy_solution()

    print(f"\nSingleBatch Dummy Solution:")
    print(f"  Job-to-batch mapping: {singlebatch_solution.job_to_batch}")
    print(f"  Schedule: {singlebatch_solution.schedule_batch}")

    # Forward transform solution
    ovensched_solution = transformation.forward_transform_solution(
        singlebatch_solution, ovensched_problem
    )

    print(f"\nForward-transformed OvenSched Solution:")
    for machine, batches in ovensched_solution.schedule_per_machine.items():
        print(f"  Machine {machine}: {len(batches)} batches")
        for i, batch in enumerate(batches):
            print(
                f"    Batch {i}: tasks={sorted(batch.tasks)}, "
                f"time=[{batch.start_time}, {batch.end_time}]"
            )

    # Back transform solution
    recovered_solution = transformation.back_transform_solution(
        ovensched_solution, singlebatch_problem
    )

    print(f"\nBack-transformed SingleBatch Solution:")
    print(f"  Job-to-batch mapping: {recovered_solution.job_to_batch}")
    print(f"  Schedule: {recovered_solution.schedule_batch}")

    # Verify round-trip
    print(f"\nRound-trip verification:")
    print(
        f"  Original mapping == Recovered mapping: "
        f"{singlebatch_solution.job_to_batch == recovered_solution.job_to_batch}"
    )


def example_transformation_solver():
    """Example 2: Solve SingleBatch using OvenSched solver via transformation."""
    print("\n" + "=" * 80)
    print("Example 2: Solving SingleBatch via OvenSched Transformation")
    print("=" * 80)

    # Create problem
    singlebatch_problem = create_example_singlebatch_problem()

    print(f"\nSingleBatch Problem:")
    print(f"  Jobs: {singlebatch_problem.nb_jobs}")
    print(f"  Capacity: {singlebatch_problem.capacity}")

    # Create transformation
    transformation = SinglebatchToOvenschedTransformation()

    # Get metadata
    metadata = transformation.get_forward_metadata()
    print(f"\nTransformation Metadata:")
    print(f"  Completeness: {metadata.completeness.value}")
    print(f"  Is exact: {metadata.is_exact()}")
    print(f"  Use cases:")
    for use_case in metadata.use_cases:
        print(f"    - {use_case}")

    # Try to use an OvenSched solver
    try:
        from discrete_optimization.ovensched.solvers.greedy import (
            GreedyOvenSchedulingSolver,
        )

        print(f"\nSolving with OvenSched Greedy Solver...")

        # Create TransformationSolver
        solver = TransformationSolver(
            problem=singlebatch_problem,
            transformation=transformation,
            solver_class=GreedyOvenSchedulingSolver,
        )

        # Solve
        result_storage = solver.solve(
            callbacks=[TimerStopper(total_seconds=5)],
        )

        # Get best solution
        best_solution = result_storage.get_best_solution()

        if best_solution is not None:
            print(f"\nSolution found!")
            print(f"  Job-to-batch mapping: {best_solution.job_to_batch}")
            print(f"  Number of batches: {max(best_solution.job_to_batch) + 1}")

            # Evaluate
            evaluation = singlebatch_problem.evaluate(best_solution)
            print(f"  Makespan: {evaluation['makespan']}")
            print(f"  Violation: {evaluation['violation']}")
            print(f"  Feasible: {singlebatch_problem.satisfy(best_solution)}")

    except ImportError as e:
        print(f"\nOvenSched solver not available: {e}")
        print("Install required dependencies or check OvenSched solver implementation")


def example_comparison():
    """Example 3: Compare direct vs transformation approach."""
    print("\n" + "=" * 80)
    print("Example 3: Benefits of Transformation Approach")
    print("=" * 80)

    singlebatch_problem = create_example_singlebatch_problem()

    print(f"\nSingleBatch Problem Size:")
    print(f"  Jobs: {singlebatch_problem.nb_jobs}")

    # Show what solvers become available
    print(f"\nDirect SingleBatch Solvers:")
    print(f"  - Arc flow LP formulation")
    print(f"  - Custom heuristics (if implemented)")

    print(f"\nOvenSched Solvers Available via Transformation:")
    print(f"  - Greedy solver")
    print(f"  - CP-SAT solver")
    print(f"  - Dynamic programming")
    print(f"  - OptalCP solver")
    print(f"  - Metaheuristics (SA, GA, etc.)")

    print(f"\nKey Advantage:")
    print(f"  SingleBatch gets access to the entire OvenSched solver ecosystem!")
    print(f"  This is an EXACT transformation, so solutions are guaranteed valid.")


def main():
    """Run all examples."""
    print("=" * 80)
    print("SingleBatch → OvenSched Transformation Examples")
    print("=" * 80)
    print(
        "\nSingleBatch is a special case of OvenSched (1 machine, 1 attribute, no setup)."
    )
    print("This transformation allows using OvenSched solvers on SingleBatch problems.")

    example_direct_transformation()
    example_transformation_solver()
    example_comparison()

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
The SingleBatch → OvenSched transformation is EXACT:
- Every SingleBatch problem maps to a valid OvenSched problem
- Solutions transform bidirectionally without information loss
- Access to rich OvenSched solver ecosystem

Use this when:
- You have a SingleBatch problem
- You want to use advanced OvenSched solvers
- You need guaranteed exact solutions

The transformation handles:
- Mapping to 1 machine with 1 attribute
- Setting up no-setup costs/times (all zeros)
- Creating unbounded time windows
- Preserving capacity constraints
    """)


if __name__ == "__main__":
    main()
