#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from discrete_optimization.fjsp.problem import Job
from discrete_optimization.fjsp.solvers.cpsat_auto import (
    FJobShopProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.cpm import Cpm
from discrete_optimization.jsp.problem import Subjob


def test_cpm_1_critical_path():
    problem = FJobShopProblem(
        list_jobs=[
            Job(
                job_id=0,
                sub_jobs=[
                    [
                        Subjob(machine_id=0, processing_time=1),
                        Subjob(machine_id=1, processing_time=2),
                    ],
                    [
                        Subjob(machine_id=0, processing_time=2),
                        Subjob(machine_id=1, processing_time=1),
                    ],
                ],
            ),
            Job(
                job_id=1,
                sub_jobs=[
                    [
                        Subjob(machine_id=0, processing_time=1),
                        Subjob(machine_id=1, processing_time=2),
                    ],
                    [
                        Subjob(machine_id=0, processing_time=2),
                        Subjob(machine_id=1, processing_time=3),
                    ],
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
                job_id=0,
                sub_jobs=[
                    [
                        Subjob(machine_id=0, processing_time=1),
                        Subjob(machine_id=1, processing_time=2),
                    ],
                    [
                        Subjob(machine_id=0, processing_time=2),
                        Subjob(machine_id=1, processing_time=1),
                    ],
                ],
            ),
            Job(
                job_id=1,
                sub_jobs=[
                    [
                        Subjob(machine_id=0, processing_time=1),
                        Subjob(machine_id=1, processing_time=2),
                    ],
                    [
                        Subjob(machine_id=0, processing_time=1),
                        Subjob(machine_id=1, processing_time=3),
                    ],
                ],
            ),
        ]
    )
    cpm = Cpm(problem=problem)
    assert len(cpm.get_a_critical_path()) == 2
    assert len(cpm.get_critical_subgraph().nodes) == 4
