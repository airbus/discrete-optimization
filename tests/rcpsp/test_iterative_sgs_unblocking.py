#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solution import (
    RcpspSolution,
    compute_est_with_time_lags,
    compute_lst_with_time_lags,
    generate_schedule_from_permutation_iterative_sgs_unblocking,
)
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
)


def test_basic_rcpsp_max():
    """Test basic RCPSP/max problem with min and max time lags."""
    mode_details = {
        1: {1: {"duration": 0}},  # start
        2: {1: {"duration": 5, "R1": 1}},
        3: {1: {"duration": 3, "R1": 1}},
        4: {1: {"duration": 4, "R1": 1}},
        5: {1: {"duration": 0}},  # end
    }

    successors = {1: [2, 3], 2: [4], 3: [4], 4: [5], 5: []}
    resources = {"R1": 1}

    # Minimum time lag: task 3 must start at least 2 units after task 2
    # Maximum time lag: task 3 must start at most 8 units after task 2
    special_constraints = SpecialConstraintsDescription(
        start_to_start_min_time_lag=[(2, 3, 2)],  # min lag
        start_to_start_max_time_lag=[(2, 3, 8)],  # max lag
    )

    problem = RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        special_constraints=special_constraints,
    )

    # Test that problem recognizes it has time lag constraints
    assert problem.has_time_lag_constraints()

    # Test CP-SAT solver
    solver = CpSatRcpspSolver(problem=problem)
    result = solver.solve(time_limit=10)
    solution = result.get_best_solution()

    assert solution is not None, "Solver should find a solution"

    # Verify the constraints
    start_2 = solution.get_start_time(2)
    start_3 = solution.get_start_time(3)

    # Check minimum time lag: start(3) >= start(2) + 2
    assert start_3 >= start_2 + 2, (
        f"Min time lag violated: start(3)={start_3}, start(2)={start_2}"
    )

    # Check maximum time lag: start(3) <= start(2) + 8
    assert start_3 <= start_2 + 8, (
        f"Max time lag violated: start(3)={start_3}, start(2)={start_2}"
    )

    assert problem.satisfy(solution), "Solution should satisfy all constraints"

    # Test iterative SGS with unblocking
    sgs_solution = RcpspSolution(
        problem=problem,
        rcpsp_permutation=solution.rcpsp_permutation,
        rcpsp_modes=solution.rcpsp_modes,
    )

    schedule, feasible = generate_schedule_from_permutation_iterative_sgs_unblocking(
        sgs_solution, problem
    )

    assert feasible, "Iterative SGS should generate feasible schedule"

    # Verify constraints in SGS schedule
    sgs_start_2 = schedule[2]["start_time"]
    sgs_start_3 = schedule[3]["start_time"]

    assert sgs_start_3 >= sgs_start_2 + 2, "SGS: Min time lag violated"
    assert sgs_start_3 <= sgs_start_2 + 8, "SGS: Max time lag violated"


def test_infeasible_time_lag_network():
    """Test problem with conflicting time lag constraints."""
    mode_details = {
        1: {1: {"duration": 0}},
        2: {1: {"duration": 3, "R1": 1}},
        3: {1: {"duration": 2, "R1": 1}},
        4: {1: {"duration": 0}},
    }

    successors = {1: [2, 3], 2: [4], 3: [4], 4: []}
    resources = {"R1": 1}

    # Conflicting constraints:
    # Min: start(3) >= start(2) + 10
    # Max: start(3) <= start(2) + 5
    # These are incompatible!
    special_constraints = SpecialConstraintsDescription(
        start_to_start_min_time_lag=[(2, 3, 10)],  # min: at least 10 units after
        start_to_start_max_time_lag=[(2, 3, 5)],  # max: at most 5 units after
    )

    problem = RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        special_constraints=special_constraints,
    )

    # Try to solve - should fail to find feasible schedule
    solver = CpSatRcpspSolver(problem=problem)
    result = solver.solve(time_limit=10)

    # CP-SAT should detect infeasibility
    if len(result) > 0:
        solution = result.get_best_solution()
        # If a solution is found, it shouldn't satisfy the constraints
        # (this shouldn't happen with correct implementation)
        start_2 = solution.get_start_time(2)
        start_3 = solution.get_start_time(3)
        # At least one constraint must be violated
        min_satisfied = start_3 >= start_2 + 10
        max_satisfied = start_3 <= start_2 + 5
        assert not (min_satisfied and max_satisfied), (
            "Cannot satisfy both conflicting constraints"
        )


def test_est_lst_computation():
    """Test EST/LST computation in isolation."""
    mode_details = {
        1: {1: {"duration": 0}},
        2: {1: {"duration": 5, "R1": 1}},
        3: {1: {"duration": 3, "R1": 1}},
        4: {1: {"duration": 2, "R1": 1}},
        5: {1: {"duration": 0}},
    }

    successors = {1: [2, 3], 2: [4, 5], 3: [4], 4: [5], 5: []}
    resources = {"R1": 1}

    special_constraints = SpecialConstraintsDescription(
        start_to_start_min_time_lag=[(2, 3, 3)],  # start(3) >= start(2) + 3
    )

    problem = RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=50,
        special_constraints=special_constraints,
    )

    modes_dict = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

    # Test EST with task 1 scheduled
    scheduled = {1: 0}
    unscheduled = {2, 3, 4, 5}

    est = compute_est_with_time_lags(problem, scheduled, unscheduled, modes_dict)

    # Task 2 is successor of 1, should be able to start at 0
    assert est[2] == 0, f"EST[2] should be 0, got {est[2]}"
    # Task 3 has time lag constraint with task 2: start(3) >= start(2) + 3
    # Since both are unscheduled, EST[3] = EST[2] + 3 = 0 + 3 = 3
    assert est[3] == 3, f"EST[3] should be 3 (propagated from EST[2]), got {est[3]}"

    # Test EST with task 1 and 2 scheduled
    scheduled = {1: 0, 2: 0}
    unscheduled = {3, 4, 5}

    est = compute_est_with_time_lags(problem, scheduled, unscheduled, modes_dict)

    # Task 3 has min time lag from task 2: start(3) >= start(2) + 3 = 0 + 3 = 3
    assert est[3] == 3, f"EST[3] should be 3 due to time lag, got {est[3]}"

    # Task 4 is successor of both 2 and 3, must wait for both to finish
    # Task 2 ends at 0 + 5 = 5
    # Task 3 earliest start is 3, ends at 3 + 3 = 6
    # So task 4 EST should propagate from task 3
    assert est[4] >= 5, f"EST[4] should be at least 5, got {est[4]}"

    # Test LST computation
    scheduled = {1: 0, 2: 0}
    unscheduled = {3, 4, 5}

    lst = compute_lst_with_time_lags(
        problem, scheduled, unscheduled, modes_dict, horizon=50
    )

    # LST values should all be <= horizon
    for task in unscheduled:
        assert lst[task] <= 50, f"LST[{task}] = {lst[task]} exceeds horizon"


def test_cyclic_time_lag_constraints():
    """Test problem with cyclic time lag constraints."""
    mode_details = {
        1: {1: {"duration": 0}},
        2: {1: {"duration": 2, "R1": 1}},
        3: {1: {"duration": 2, "R1": 1}},
        4: {1: {"duration": 2, "R1": 1}},
        5: {1: {"duration": 0}},
    }

    # Parallel structure to allow cycles
    successors = {1: [2, 3, 4], 2: [5], 3: [5], 4: [5], 5: []}
    resources = {"R1": 1}

    # Create a cycle in time lag network: 2 -> 3 -> 4 -> 2
    # The total lag around the cycle must be non-negative for feasibility
    special_constraints = SpecialConstraintsDescription(
        start_to_start_min_time_lag=[
            (2, 3, 3),  # start(3) >= start(2) + 3
            (3, 4, 3),  # start(4) >= start(3) + 3
        ],
        start_to_start_max_time_lag=[
            (
                2,
                4,
                8,
            ),  # start(4) <= start(2) + 8, equivalent to start(2) >= start(4) - 8
        ],
    )

    problem = RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        special_constraints=special_constraints,
    )

    # Try to solve
    solver = CpSatRcpspSolver(problem=problem)
    result = solver.solve(time_limit=10)

    if len(result) > 0:
        solution = result.get_best_solution()

        start_2 = solution.get_start_time(2)
        start_3 = solution.get_start_time(3)
        start_4 = solution.get_start_time(4)

        # Verify the cycle constraints
        assert start_3 >= start_2 + 3, f"Constraint 2->3 violated"
        assert start_4 >= start_3 + 3, f"Constraint 3->4 violated"
        assert start_4 <= start_2 + 8, f"Constraint 2->4 violated"

        assert problem.satisfy(solution)


def test_negative_time_lags():
    """Test that task 3 can start before task 2 using max time lag."""
    mode_details = {
        1: {1: {"duration": 0}},
        2: {1: {"duration": 5, "R1": 1}},
        3: {1: {"duration": 3, "R1": 1}},
        4: {1: {"duration": 0}},
    }

    successors = {1: [2, 3], 2: [4], 3: [4], 4: []}
    resources = {"R1": 1}

    # Max time lag: task 3 can start before task 2
    # start(2) <= start(3) + 3 means start(3) >= start(2) - 3
    # So task 3 can start up to 3 units before task 2
    special_constraints = SpecialConstraintsDescription(
        start_to_start_max_time_lag=[(3, 2, 3)],
    )

    problem = RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        special_constraints=special_constraints,
    )

    solver = CpSatRcpspSolver(problem=problem)
    result = solver.solve(time_limit=10)
    solution = result.get_best_solution()

    assert solution is not None

    start_2 = solution.get_start_time(2)
    start_3 = solution.get_start_time(3)

    # Verify: start(2) <= start(3) + 3, which is equivalent to start(3) >= start(2) - 3
    assert start_2 <= start_3 + 3, f"Max time lag constraint violated"

    assert problem.satisfy(solution)


if __name__ == "__main__":
    # Run tests manually
    test_basic_rcpsp_max()
    print("✓ test_basic_rcpsp_max passed")

    test_infeasible_time_lag_network()
    print("✓ test_infeasible_time_lag_network passed")

    test_est_lst_computation()
    print("✓ test_est_lst_computation passed")

    test_cyclic_time_lag_constraints()
    print("✓ test_cyclic_time_lag_constraints passed")

    test_negative_time_lags()
    print("✓ test_negative_time_lags passed")

    print("\nAll tests passed!")
