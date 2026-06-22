#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from discrete_optimization.generic_tasks_tools.solvers.cpm import Cpm
from discrete_optimization.shop.base import Job, Subjob, SubjobRecipe
from discrete_optimization.shop.fjsp.problem import (
    FJobShopProblem,
)


def test_cpm_1_critical_path():
    problem = FJobShopProblem(
        list_jobs=[
            Job(
                job_index=0,
                subjobs=[
                    Subjob(
                        0,
                        0,
                        [
                            SubjobRecipe(machine_index=0, processing_time=1),
                            SubjobRecipe(machine_index=1, processing_time=2),
                        ],
                    ),
                    Subjob(
                        1,
                        0,
                        [
                            SubjobRecipe(machine_index=0, processing_time=2),
                            SubjobRecipe(machine_index=1, processing_time=1),
                        ],
                    ),
                ],
            ),
            Job(
                job_index=1,
                subjobs=[
                    Subjob(
                        0,
                        1,
                        [
                            SubjobRecipe(machine_index=0, processing_time=1),
                            SubjobRecipe(machine_index=1, processing_time=2),
                        ],
                    ),
                    Subjob(
                        1,
                        1,
                        [
                            SubjobRecipe(machine_index=0, processing_time=2),
                            SubjobRecipe(machine_index=1, processing_time=3),
                        ],
                    ),
                ],
            ),
        ]
    )
    cpm = Cpm(problem=problem)
    assert len(cpm.get_a_critical_path()) == 2
    assert len(cpm.get_critical_subgraph().nodes) == 2


def test_cpm_2_critical_paths():
    problem = FJobShopProblem(
        list_jobs=[
            Job(
                job_index=0,
                subjobs=[
                    Subjob(
                        0,
                        0,
                        [
                            SubjobRecipe(machine_index=0, processing_time=1),
                            SubjobRecipe(machine_index=1, processing_time=2),
                        ],
                    ),
                    Subjob(
                        1,
                        0,
                        [
                            SubjobRecipe(machine_index=0, processing_time=2),
                            SubjobRecipe(machine_index=1, processing_time=1),
                        ],
                    ),
                ],
            ),
            Job(
                job_index=1,
                subjobs=[
                    Subjob(
                        0,
                        1,
                        [
                            SubjobRecipe(machine_index=0, processing_time=1),
                            SubjobRecipe(machine_index=1, processing_time=2),
                        ],
                    ),
                    Subjob(
                        1,
                        1,
                        [
                            SubjobRecipe(machine_index=0, processing_time=1),
                            SubjobRecipe(machine_index=1, processing_time=3),
                        ],
                    ),
                ],
            ),
        ]
    )
    cpm = Cpm(problem=problem)
    assert len(cpm.get_a_critical_path()) == 2
    assert len(cpm.get_critical_subgraph().nodes) == 4
