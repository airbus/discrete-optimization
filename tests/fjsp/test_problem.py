#  Copyright (c) 2024-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import re

from pytest import fixture

from discrete_optimization.fjsp.problem import (
    FJobShopProblem,
    FJobShopSolution,
    Job,
    Subjob,
)

logging.basicConfig(level=logging.INFO)


@fixture
def problem():
    job_0 = Job(
        job_id=0,
        sub_jobs=[
            [Subjob(0, 1), Subjob(1, 2)],
            [Subjob(3, 1), Subjob(4, 2)],
            [Subjob(1, 1), Subjob(2, 2)],
        ],
    )
    job_1 = Job(
        job_id=1,
        sub_jobs=[
            [Subjob(0, 1), Subjob(1, 2)],
            [Subjob(3, 1), Subjob(4, 2)],
            [Subjob(1, 4), Subjob(2, 2)],
        ],
    )
    return FJobShopProblem(list_jobs=[job_0, job_1], n_jobs=2, n_machines=5, horizon=30)


def test_fjsp_satisfy(problem):
    sol = FJobShopSolution(
        problem=problem,
        schedule=[
            [(0, 1, 0, 0), (1, 2, 3, 0), (2, 4, 2, 1)],
            [(0, 2, 1, 1), (2, 4, 4, 1), (4, 6, 2, 1)],
        ],
    )
    assert problem.satisfy(sol)


def test_fjsp_satisfy_nok_unallowed_machine(problem, caplog):
    sol = FJobShopSolution(
        problem=problem,
        schedule=[
            [(0, 1, 0, 0), (1, 2, 3, 0), (2, 4, 2, 1)],
            [(0, 2, 5, 0), (2, 4, 4, 1), (4, 6, 2, 1)],
        ],
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    assert re.search("Unallowed machine.*(1, 0)", caplog.text)


def test_fjsp_satisfy_nok_overlap_machine(problem, caplog):
    sol = FJobShopSolution(
        problem=problem,
        schedule=[
            [(0, 1, 0, 0), (1, 2, 3, 0), (2, 4, 2, 1)],
            [(0, 1, 0, 0), (2, 4, 4, 1), (4, 6, 2, 1)],
        ],
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    assert re.search("Overlapping.*0", caplog.text)


def test_fjsp_satisfy_nok_mode_machine(problem, caplog):
    sol = FJobShopSolution(
        problem=problem,
        schedule=[
            [(0, 1, 0, 0), (1, 2, 3, 0), (2, 4, 2, 1)],
            [(0, 2, 1, 0), (2, 4, 4, 1), (4, 6, 2, 1)],
        ],
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    assert re.search("not match.*(1, 0)", caplog.text)


def test_fjsp_satisfy_nok_precedence(problem, caplog):
    sol = FJobShopSolution(
        problem=problem,
        schedule=[
            [(0, 1, 0, 0), (1, 2, 3, 0), (2, 4, 2, 1)],
            [(0, 2, 1, 1), (6, 8, 4, 1), (4, 6, 2, 1)],
        ],
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    assert re.search("Precedence.*(1, 1).*(1, 2)", caplog.text)


def test_fjsp_satisfy_nok_duration(problem, caplog):
    sol = FJobShopSolution(
        problem=problem,
        schedule=[
            [(0, 1, 0, 0), (1, 2, 3, 0), (2, 4, 2, 1)],
            [(0, 1, 1, 1), (2, 4, 4, 1), (4, 6, 2, 1)],
        ],
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    assert re.search("Duration.*(1, 0)", caplog.text)
