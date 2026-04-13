"""Basic test of transformation functionality (no datasets required)."""

from discrete_optimization.generic_tools.transformation import (
    CompositeTransformation,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.transformations import RcpspToMultiskillTransformation


def create_simple_rcpsp():
    """Create a minimal RCPSP instance for testing."""
    resources = {"R1": 2, "R2": 1}
    mode_details = {
        1: {1: {"duration": 0, "R1": 0, "R2": 0}},  # Source
        2: {1: {"duration": 3, "R1": 1, "R2": 0}},  # Task A
        3: {1: {"duration": 2, "R1": 0, "R2": 1}},  # Task B
        4: {1: {"duration": 0, "R1": 0, "R2": 0}},  # Sink
    }
    successors = {
        1: [2, 3],  # Source → A, B
        2: [4],  # A → Sink
        3: [4],  # B → Sink
        4: [],  # Sink
    }

    return RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=20,
        source_task=1,
        sink_task=4,
    )


def main():
    """Test basic transformation functionality."""
    print("Testing problem transformation framework...")
    print("=" * 60)

    # Create simple problem
    rcpsp = create_simple_rcpsp()
    print(f"\n✓ Created RCPSP problem: {rcpsp.n_jobs} tasks")

    # Test 1: Transformation
    print("\n[Test 1] Creating transformation...")
    transformation = RcpspToMultiskillTransformation()
    ms_problem = transformation.transform_problem(rcpsp)
    print(f"  ✓ Transformed to: {type(ms_problem).__name__}")
    print(f"    - Resources: {ms_problem.resources_set}")
    print(f"    - Skills: {ms_problem.skills_set}")
    print(f"    - Employees: {len(ms_problem.employees)}")

    # Test 2: Composite transformation
    print("\n[Test 2] Creating composite transformation...")
    composite = CompositeTransformation([transformation])
    print(f"  ✓ Composite: {composite}")

    print("\n" + "=" * 60)
    print("All tests passed!")


if __name__ == "__main__":
    main()
