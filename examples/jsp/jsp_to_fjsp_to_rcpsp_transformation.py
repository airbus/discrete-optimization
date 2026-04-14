#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example: Solve JobShop via FlexibleJobShop and RCPSP transformations.

This example demonstrates:
- Transforming JobShop → FlexibleJobShop → RCPSP
- Composing transformations for multi-step conversion
- Solving with RCPSP solvers
- Automatic back-transformation through the chain
"""

from discrete_optimization.fjsp.transformations import FjspToRcpspTransformation
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import (
    TransformationSolver,
    chain_transformations,
)
from discrete_optimization.jsp.parser import get_data_available, parse_file
from discrete_optimization.jsp.transformations import JspToFjspTransformation
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver


def main():
    """Run JobShop → FlexibleJobShop → RCPSP transformation example."""
    print("=" * 80)
    print("Example: Solve JobShop via FJSP and RCPSP Transformations")
    print("=" * 80)
    # Load a small JobShop instance
    files = get_data_available()
    jsp_problem = parse_file(files[0])

    print(f"\nOriginal JobShop problem:")
    print(f"  - Jobs: {jsp_problem.n_jobs}")
    print(f"  - Machines: {jsp_problem.n_machines}")
    print(f"  - Total operations: {jsp_problem.n_all_jobs}")
    print(f"  - Horizon: {jsp_problem.horizon}")

    # Create transformations
    print("\n" + "=" * 80)
    print("Building Transformation Chain")
    print("=" * 80)

    jsp_to_fjsp = JspToFjspTransformation()
    fjsp_to_rcpsp = FjspToRcpspTransformation()

    # Compose transformations: JSP → FJSP → RCPSP
    composite = chain_transformations(jsp_to_fjsp, fjsp_to_rcpsp)

    print("\nTransformation chain:")
    print("  JobShop → FlexibleJobShop → RCPSP")

    # Transform to inspect intermediate representations
    print("\n" + "=" * 80)
    print("Intermediate Transformations")
    print("=" * 80)

    # Step 1: JSP → FJSP
    fjsp_problem = jsp_to_fjsp.transform_problem(jsp_problem)
    print(f"\nFlexibleJobShop problem:")
    print(f"  - Jobs: {fjsp_problem.n_jobs}")
    print(f"  - Machines: {fjsp_problem.n_machines}")
    print(f"  - Total operations: {fjsp_problem.n_all_jobs}")

    # Step 2: FJSP → RCPSP
    rcpsp_problem = fjsp_to_rcpsp.transform_problem(fjsp_problem)
    print(f"\nRCPSP problem:")
    print(f"  - Tasks: {rcpsp_problem.n_jobs} (including source/sink)")
    print(f"  - Real tasks: {rcpsp_problem.n_jobs_non_dummy}")
    print(f"  - Resources: {list(rcpsp_problem.resources.keys())}")
    print(f"  - Horizon: {rcpsp_problem.horizon}")

    # Solve using composite transformation
    print("\n" + "=" * 80)
    print("Solving with RCPSP CP-SAT Solver")
    print("=" * 80)

    solver = TransformationSolver(
        transformation=composite,  # Composite transformation!
        source_problem=jsp_problem,
        solver_brick=SubBrick(cls=CpSatRcpspSolver, kwargs={"time_limit": 10}),
    )

    result = solver.solve()

    # Analyze results
    print(f"\nResults:")
    print(f"  - Solutions found: {len(result)}")

    if len(result) > 0:
        best_solution = result.get_best_solution()
        best_fit = result.get_best_solution_fit()[1]

        print(f"\nBest solution:")
        print(f"  - Fitness: {best_fit}")
        print(f"  - Solution type: {type(best_solution).__name__}")
        print(f"  - Feasible: {jsp_problem.satisfy(best_solution)}")

        # Evaluate in original JobShop space
        eval_result = jsp_problem.evaluate(best_solution)
        print(f"  - Makespan: {eval_result['makespan']}")

        # Show schedule for first job
        print(f"\nSchedule for Job 0:")
        for subjob_idx, (start, end) in enumerate(best_solution.schedule[0]):
            machine = jsp_problem.list_jobs[0][subjob_idx].machine_id
            duration = end - start
            print(
                f"  - Subjob {subjob_idx}: Machine {machine}, Start={start}, End={end}, Duration={duration}"
            )

    print("\n" + "=" * 80)
    print("How It Works")
    print("=" * 80)
    print("""
1. JobShop → FlexibleJobShop:
   - Each operation with fixed machine → operation with 1 machine option
   - JSP is a special case of FJSP

2. FlexibleJobShop → RCPSP:
   - Each operation → task
   - Each machine option → mode
   - Each machine → renewable resource (capacity 1)
   - Job precedence → task successors

3. Composite Transformation:
   - chain_transformations(JSP→FJSP, FJSP→RCPSP) = JSP→RCPSP
   - Automatic back-transformation through both steps!

4. Solve with RCPSP solver

5. Solution automatically back-transformed:
   - RCPSP → FJSP → JSP

6. All solutions are valid JobShop solutions!
    """)

    print("=" * 80)
    print("Key Insight:")
    print("By composing transformations, we can solve JobShop using RCPSP solvers!")
    print("This demonstrates the power of transformation composition:")
    print("  JSP ⊂ FJSP ⊂ RCPSP (each generalizes the previous)")
    print("=" * 80)


if __name__ == "__main__":
    main()
