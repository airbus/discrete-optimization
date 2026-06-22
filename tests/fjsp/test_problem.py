#  Copyright (c) 2024-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from pytest import fixture

from discrete_optimization.shop.base import Job, Subjob, SubjobRecipe
from discrete_optimization.shop.fjsp.problem import (
    FJobShopProblem,
    FJobShopSolution,
)

logging.basicConfig(level=logging.INFO)


@fixture
def problem():
    job_0 = Job(
        job_index=0,
        subjobs=[
            Subjob(0, 0, recipes=[SubjobRecipe(0, 1), SubjobRecipe(1, 2)]),
            Subjob(1, 0, [SubjobRecipe(3, 1), SubjobRecipe(4, 2)]),
            Subjob(2, 0, [SubjobRecipe(1, 1), SubjobRecipe(2, 2)]),
        ],
    )
    job_1 = Job(
        job_index=1,
        subjobs=[
            Subjob(0, 1, [SubjobRecipe(0, 1), SubjobRecipe(1, 2)]),
            Subjob(1, 1, [SubjobRecipe(3, 1), SubjobRecipe(4, 2)]),
            Subjob(2, 1, [SubjobRecipe(1, 4), SubjobRecipe(2, 2)]),
        ],
    )
    return FJobShopProblem(list_jobs=[job_0, job_1], n_jobs=2, n_machines=5, horizon=30)


def test_fjsp_satisfy(problem):
    sol = FJobShopSolution(
        problem=problem,
        schedule=[
            [(0, 1), (1, 2), (2, 4)],
            [(0, 2), (2, 4), (4, 6)],
        ],
        machine_index=[[0, 3, 2], [1, 4, 2]],
    )
    assert problem.satisfy(sol)


def test_fjsp_satisfy_nok_unallowed_machine(problem, caplog):
    sol = FJobShopSolution(
        problem=problem,
        schedule=[
            [(0, 1), (1, 2), (2, 4)],
            [(0, 2), (2, 4), (4, 6)],
        ],
        machine_index=[[0, 3, 2], [5, 4, 2]],
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    # assert re.search("Unallowed machine.*(1, 0)", caplog.text)


def test_fjsp_satisfy_nok_overlap_machine(problem, caplog):
    sol = FJobShopSolution(
        problem=problem,
        schedule=[
            [
                (0, 1),
                (
                    1,
                    2,
                ),
                (2, 4),
            ],
            [(0, 1), (2, 4), (4, 6)],
        ],
        machine_index=[[0, 3, 2], [0, 4, 2]],
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    # assert re.search("Overlapping.*0", caplog.text)


def test_fjsp_satisfy_nok_mode_machine(problem, caplog):
    sol = FJobShopSolution(
        problem=problem,
        schedule=[
            [(0, 1), (1, 2), (2, 4)],
            [(0, 2), (2, 4), (4, 6)],
        ],
        machine_index=[[0, 3, 2], [1, 4, 2]],
        recipe_index=[[0, 0, 1], [0, 1, 1]],
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    # assert re.search("not match.*(1, 0)", caplog.text)


def test_fjsp_satisfy_nok_precedence(problem, caplog):
    sol = FJobShopSolution(
        problem=problem,
        schedule=[
            [(0, 1), (1, 2), (2, 4)],
            [(0, 2), (6, 8), (4, 6)],
        ],
        machine_index=[[0, 3, 2], [1, 4, 2]],
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    # assert re.search("Precedence.*(1, 1).*(1, 2)", caplog.text)


def test_fjsp_satisfy_nok_duration(problem, caplog):
    sol = FJobShopSolution(
        problem=problem,
        schedule=[
            [(0, 1), (1, 2), (2, 4)],
            [(0, 1), (2, 4), (4, 6)],
        ],
        machine_index=[[0, 3, 2], [1, 4, 2]],
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    # assert re.search("Duration.*(1, 0)", caplog.text)
