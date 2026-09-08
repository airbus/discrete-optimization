#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.rcpsp_resource_dependent.problem import (
    RcpspResourceDependentProblem,
    RcpspResourceDependentSolution,
)
from discrete_optimization.rcpsp_resource_dependent.solvers.cpsat import (
    CpSatRcpspResourceDependentSolver,
)


@pytest.fixture
def toy_problem():
    """Fixture providing a toy RCPSP resource-dependent problem instance."""
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
    return RcpspResourceDependentProblem(
        resources=resources,
        non_renewable_resources=["N1", "N2"],
        mode_details=mode_details,
        successors=successors,
        horizon=30,
        source_task="source",
        sink_task="sink",
    )


@pytest.mark.parametrize(
    "avoid_interval_optional_for_unary_resources",
    [True, False],
)
@pytest.mark.parametrize(
    "avoid_interval_optional_for_cumulative_resources",
    [True, False],
)
@pytest.mark.parametrize(
    "use_demand_variables_for_non_renewable_resources",
    [True, False],
)
def test_cpsat_with_different_model_options(
    toy_problem,
    avoid_interval_optional_for_unary_resources,
    avoid_interval_optional_for_cumulative_resources,
    use_demand_variables_for_non_renewable_resources,
):
    """Test CP-SAT solver with different model configuration options."""
    solver = CpSatRcpspResourceDependentSolver(toy_problem)
    solver.init_model(
        avoid_interval_optional_for_unary_resources=avoid_interval_optional_for_unary_resources,
        avoid_interval_optional_for_cumulative_resources=avoid_interval_optional_for_cumulative_resources,
        use_demand_variables_for_non_renewable_resources=use_demand_variables_for_non_renewable_resources,
    )

    res = solver.solve(time_limit=10)
    sol: RcpspResourceDependentSolution = res[-1][0]

    # Verify resource consumption matches solver variables
    resource_consumption = {}
    total_conso_nr = {r: 0 for r in toy_problem.non_renewable_resources}

    for t in toy_problem.tasks_list:
        for r in toy_problem.cumulative_resources_list:
            resource_consumption[(t, r)] = sol.get_calendar_resource_consumption(r, t)
        for r in toy_problem.non_renewable_resources_list:
            consumption = sol.get_non_renewable_resource_consumption(r, t)
            resource_consumption[(t, r)] = consumption
            total_conso_nr[r] += consumption

    # Verify cumulative resource demands match solver variables
    for t, r in solver.demands_cumulative_resource_vars:
        solver_value = solver.solver.Value(
            solver.demands_cumulative_resource_vars[t, r]
        )
        assert solver_value == resource_consumption[(t, r)]

    # Verify non-renewable resource constraints
    for r in toy_problem.non_renewable_resources_list:
        assert total_conso_nr[r] <= toy_problem.get_resource_max_capacity(r)

    # Verify solution satisfies all constraints
    assert toy_problem.satisfy(sol)
