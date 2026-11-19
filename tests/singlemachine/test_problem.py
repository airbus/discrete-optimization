#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tasks_tools.enums import AbsentValue
from discrete_optimization.singlemachine.problem import (
    WeightedTardinessProblem,
    WTSolution,
)


def test_wtsolution_from_permutation():
    p = WeightedTardinessProblem(
        num_jobs=3,
        processing_times=[2, 4, 1],
        weights=[1, 2, 3],
        due_dates=[3, 5, 2],
    )
    sol = WTSolution(problem=p, permutation=[2, 0, 1])
    assert sol.permutation == [2, 0, 1]
    assert sol.schedule == [(1, 3), (3, 7), (0, 1)]
    assert sol.is_present(0) is True
    assert sol.is_present(1) is True
    assert sol.is_present(2) is True

    # Copying should preserve permutation and schedule
    sol_copy = sol.copy()
    assert sol_copy.permutation == [2, 0, 1]
    assert sol_copy.schedule == [(1, 3), (3, 7), (0, 1)]


def test_wtsolution_from_schedule():
    p = WeightedTardinessProblem(
        num_jobs=3,
        processing_times=[2, 4, 1],
        weights=[1, 2, 3],
        due_dates=[3, 5, 2],
    )
    # Solution constructed directly from schedule (e.g. from CP-SAT / Generic transformations)
    sol = WTSolution(problem=p, schedule=[(1, 3), (3, 7), (0, 1)])
    assert sol.permutation == [2, 0, 1]
    assert sol.is_present(0) is True
    assert sol.is_present(1) is True
    assert sol.is_present(2) is True

    # Copying must not raise AssertionError when permutation was not originally given
    sol_copy = sol.copy()
    assert sol_copy.schedule == [(1, 3), (3, 7), (0, 1)]
    assert sol_copy.permutation == [2, 0, 1]

    # Lazy copy
    sol_lazy = sol.lazy_copy()
    assert sol_lazy.schedule == [(1, 3), (3, 7), (0, 1)]
    assert sol_lazy.permutation == [2, 0, 1]


def test_wtsolution_with_absent_tasks():
    p = WeightedTardinessProblem(
        num_jobs=3,
        processing_times=[2, 4, 1],
        weights=[1, 2, 3],
        due_dates=[3, 5, 2],
    )
    # Task 0 is absent
    sol = WTSolution(
        problem=p,
        schedule=[(AbsentValue.ABSENT, AbsentValue.ABSENT), (3, 7), (0, 1)],
    )
    assert sol.is_present(0) is False
    assert sol.is_present(1) is True
    assert sol.is_present(2) is True
    assert sol.permutation == [2, 1]

    sol_copy = sol.copy()
    assert sol_copy.is_present(0) is False
    assert sol_copy.permutation == [2, 1]


def test_wtsolution_change_problem():
    p1 = WeightedTardinessProblem(
        num_jobs=3,
        processing_times=[2, 4, 1],
        weights=[1, 2, 3],
        due_dates=[3, 5, 2],
    )
    sol = WTSolution(problem=p1, schedule=[(1, 3), (3, 7), (0, 1)])
    p2 = WeightedTardinessProblem(
        num_jobs=3,
        processing_times=[3, 3, 3],
        weights=[1, 1, 1],
        due_dates=[5, 5, 5],
    )
    sol.change_problem(p2)
    assert sol.schedule == [(3, 6), (6, 9), (0, 3)]
