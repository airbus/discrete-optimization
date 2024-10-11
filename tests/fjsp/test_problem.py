#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.fjsp.problem import (
    FJobShopProblem,
    FJobShopSolution,
    Job,
    Subjob,
)

logging.basicConfig(level=logging.INFO)


def create_dummy_fjsp_and_sol():
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
    problem = FJobShopProblem(
        list_jobs=[job_0, job_1], n_jobs=2, n_machines=5, horizon=30
    )
    sol = FJobShopSolution(
        problem=problem,
        schedule=[[(0, 1, 0), (1, 2, 3), (2, 4, 2)], [(0, 2, 1), (2, 4, 4), (4, 6, 2)]],
    )
    return problem, sol


def test_fjsp_satisfy():
    problem, sol = create_dummy_fjsp_and_sol()
    assert problem.satisfy(sol)
    # Overlap of machine 0 at time 0 !
    sol = FJobShopSolution(
        problem=problem,
        schedule=[[(0, 1, 0), (1, 2, 3), (2, 4, 2)], [(0, 1, 0), (2, 4, 4), (4, 6, 2)]],
    )
    assert not problem.satisfy(sol)

    # Wrong machine on job (1, 0)

    sol = FJobShopSolution(
        problem=problem,
        schedule=[[(0, 1, 0), (1, 2, 3), (2, 4, 2)], [(0, 2, 5), (2, 4, 4), (4, 6, 2)]],
    )
    assert not problem.satisfy(sol)

    # Precedence constraint broken between (1, 2) and (1, 1)
    sol = FJobShopSolution(
        problem=problem,
        schedule=[[(0, 1, 0), (1, 2, 3), (2, 4, 2)], [(0, 2, 1), (4, 6, 4), (2, 4, 2)]],
    )
    assert not problem.satisfy(sol)
