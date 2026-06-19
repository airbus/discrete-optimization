#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging

from pytest import fixture

from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    Objective,
    RawSolution,
    TaskVariable,
)


@fixture
def problem_wo_skills():
    def custom_evaluate_fn(variable: GenericSchedulingImplSolution):
        return variable.compute_nb_tasks_done() - variable.get_max_end_time()

    return GenericSchedulingImplProblem(
        horizon=10,
        durations_per_mode={
            "task-1": {
                0: 1,
                1: 3,
            },
            "task-2": {
                0: 4,
            },
        },
        resource_consumptions={
            "task-1": {
                0: {
                    "non_renewable_resource": 2,
                },
                1: {
                    "non_renewable_resource": 1,
                },
            },
            "task-2": {
                0: {
                    "cumulative_resource": 2,
                },
            },
        },
        successors={"task-1": ["task-2"]},
        unary_resources={"worker1", "worker2"},
        unary_resources_availabilities={
            "worker1": [(1, 4)],
            "worker2": [(3, 18)],
        },
        non_skill_cumulative_resources={
            "cumulative_resource": [
                (3, 5, 1),
                (5, 10, 2),
            ],
        },
        non_renewable_resources={
            "non_renewable_resource": 1,
        },
        objective=[(Objective.MAKESPAN, -1)]
        + [(obj, 0) for obj in Objective if obj not in [Objective.MAKESPAN]],
        custom_evaluate_fn=custom_evaluate_fn,
    )


def test_problem(problem_wo_skills, caplog):
    problem = problem_wo_skills
    sol = GenericSchedulingImplSolution(
        problem=problem,
        raw_sol=RawSolution(
            task_variables={
                "task-1": TaskVariable(
                    start=1, end=4, mode=1, allocated={"worker1": set()}
                ),
                "task-2": TaskVariable(
                    start=5, end=9, mode=0, allocated={"worker2": set()}
                ),
            }
        ),
    )
    problem.satisfy(sol)
    d = problem.evaluate(sol)
    print(d)
    assert d["makespan"] == 9
    assert d["nb_tasks_done"] == 2
    assert d["nb_unary_resources_used"] == 2
    assert d["nb_resources_used"] == 4
    assert d["resources_levels"] == 5
    assert d["custom_objective"] == -7

    sol = GenericSchedulingImplSolution(
        problem=problem,
        raw_sol=RawSolution(
            task_variables={
                "task-1": TaskVariable(
                    start=3, end=6, mode=1, allocated={"worker2": set()}
                ),
                "task-2": TaskVariable(
                    start=6, end=10, mode=0, allocated={"worker2": set()}
                ),
            }
        ),
    )
    problem.satisfy(sol)
    d = problem.evaluate(sol)
    assert d["makespan"] == 10
    assert d["nb_tasks_done"] == 2
    assert d["nb_unary_resources_used"] == 1

    sol = GenericSchedulingImplSolution(
        problem=problem,
        raw_sol=RawSolution(
            task_variables={
                "task-1": TaskVariable(
                    start=3,
                    end=6,
                    mode=1,
                ),
                "task-2": TaskVariable(
                    start=6,
                    end=10,
                    mode=0,
                ),
            }
        ),
    )
    problem.satisfy(sol)
    d = problem.evaluate(sol)
    assert d["makespan"] == 10
    assert d["nb_tasks_done"] == 0
    assert d["nb_unary_resources_used"] == 0

    # nok: worker not available
    sol = GenericSchedulingImplSolution(
        problem=problem,
        raw_sol=RawSolution(
            task_variables={
                "task-1": TaskVariable(
                    start=3, end=6, mode=1, allocated={"worker1": set()}
                ),
                "task-2": TaskVariable(
                    start=6, end=10, mode=0, allocated={"worker1": set()}
                ),
            }
        ),
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    assert "worker1" in caplog.text

    # nok: cumulative resource not available
    sol = GenericSchedulingImplSolution(
        problem=problem,
        raw_sol=RawSolution(
            task_variables={
                "task-1": TaskVariable(
                    start=1, end=4, mode=1, allocated={"worker1": set()}
                ),
                "task-2": TaskVariable(
                    start=4, end=8, mode=0, allocated={"worker2": set()}
                ),
            }
        ),
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    assert "cumulative_resource" in caplog.text

    # nok: duration not correct
    sol = GenericSchedulingImplSolution(
        problem=problem,
        raw_sol=RawSolution(
            task_variables={
                "task-1": TaskVariable(
                    start=1, end=2, mode=1, allocated={"worker1": set()}
                ),
                "task-2": TaskVariable(
                    start=5, end=9, mode=0, allocated={"worker2": set()}
                ),
            }
        ),
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    assert "Duration" in caplog.text

    # lower cumulative_resource requirement => better sol
    problem.resource_consumptions["task-2"][0]["cumulative_resource"] = 1
    problem.update_problem()
    sol = GenericSchedulingImplSolution(
        problem=problem,
        raw_sol=RawSolution(
            task_variables={
                "task-1": TaskVariable(
                    start=1, end=4, mode=1, allocated={"worker1": set()}
                ),
                "task-2": TaskVariable(
                    start=4, end=8, mode=0, allocated={"worker2": set()}
                ),
            }
        ),
    )
    assert problem.satisfy(sol)


def test_task_bounds_simple(problem_wo_skills):
    assert problem_wo_skills.compute_tighter_task_bounds(horizon=8) == {
        "task-1": (0, 1, 7, 8),
        "task-2": (0, 4, 4, 8),
    }


def test_task_bounds_cpm(problem_wo_skills):
    assert problem_wo_skills.compute_tighter_task_bounds(horizon=8, use_cpm=True) == {
        "task-1": (0, 1, 3, 4),
        "task-2": (1, 5, 4, 8),
    }
