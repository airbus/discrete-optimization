#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pytest_cases import fixture

from discrete_optimization.singlebatch.problem import (
    Job,
    SingleBatchProcessingProblem,
)


@fixture
def tiny_problem():
    """Create a tiny problem (5 jobs) for fast testing."""
    jobs = [
        Job(job_id=0, processing_time=3, size=2),
        Job(job_id=1, processing_time=2, size=1),
        Job(job_id=2, processing_time=4, size=2),
        Job(job_id=3, processing_time=2, size=1),
        Job(job_id=4, processing_time=3, size=1),
    ]
    return SingleBatchProcessingProblem(jobs=jobs, capacity=5)


@fixture
def small_problem():
    """Create a small problem (10 jobs) for testing."""
    jobs = [
        Job(job_id=i, processing_time=(i % 4) + 2, size=(i % 3) + 1) for i in range(10)
    ]
    return SingleBatchProcessingProblem(jobs=jobs, capacity=8)
