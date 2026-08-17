#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Test solving shop problems (JSP, FJSP, OSP) via cpmpy."""

import pytest

from discrete_optimization.shop.base import Job, Subjob, SubjobRecipe
from discrete_optimization.shop.fjsp.problem import FJobShopProblem
from discrete_optimization.shop.jsp.problem import JobShopProblem
from discrete_optimization.shop.osp.problem import OpenShopProblem
from discrete_optimization.shop.solvers.cpmpy import CpmpyShopSolver


@pytest.fixture
def simple_jsp_problem():
    """Create a simple JSP instance."""
    job_0 = Job(
        job_index=0,
        subjobs=[
            Subjob(0, 0, recipes=[SubjobRecipe(0, 3)]),
            Subjob(1, 0, recipes=[SubjobRecipe(1, 2)]),
            Subjob(2, 0, recipes=[SubjobRecipe(2, 4)]),
        ],
    )
    job_1 = Job(
        job_index=1,
        subjobs=[
            Subjob(0, 1, recipes=[SubjobRecipe(0, 2)]),
            Subjob(1, 1, recipes=[SubjobRecipe(2, 3)]),
            Subjob(2, 1, recipes=[SubjobRecipe(1, 1)]),
        ],
    )
    return JobShopProblem(list_jobs=[job_0, job_1], n_jobs=2, n_machines=3, horizon=20)


@pytest.fixture
def simple_fjsp_problem():
    """Create a simple FJSP instance with multiple machine options."""
    job_0 = Job(
        job_index=0,
        subjobs=[
            Subjob(0, 0, recipes=[SubjobRecipe(0, 3), SubjobRecipe(1, 4)]),
            Subjob(1, 0, recipes=[SubjobRecipe(1, 2), SubjobRecipe(2, 3)]),
            Subjob(2, 0, recipes=[SubjobRecipe(2, 4)]),
        ],
    )
    job_1 = Job(
        job_index=1,
        subjobs=[
            Subjob(0, 1, recipes=[SubjobRecipe(0, 2), SubjobRecipe(1, 3)]),
            Subjob(1, 1, recipes=[SubjobRecipe(2, 3)]),
            Subjob(2, 1, recipes=[SubjobRecipe(1, 1), SubjobRecipe(2, 2)]),
        ],
    )
    return FJobShopProblem(list_jobs=[job_0, job_1], n_jobs=2, n_machines=3, horizon=25)


@pytest.fixture
def simple_osp_problem():
    """Create a simple OSP instance."""
    job_0 = Job(
        job_index=0,
        subjobs=[
            Subjob(0, 0, recipes=[SubjobRecipe(0, 3)]),
            Subjob(1, 0, recipes=[SubjobRecipe(1, 2)]),
            Subjob(2, 0, recipes=[SubjobRecipe(2, 4)]),
        ],
    )
    job_1 = Job(
        job_index=1,
        subjobs=[
            Subjob(0, 1, recipes=[SubjobRecipe(0, 2)]),
            Subjob(1, 1, recipes=[SubjobRecipe(1, 3)]),
            Subjob(2, 1, recipes=[SubjobRecipe(2, 1)]),
        ],
    )
    return OpenShopProblem(list_jobs=[job_0, job_1], n_jobs=2, n_machines=3, horizon=20)


# ============================================================================
# JSP Tests
# ============================================================================
def test_jsp_via_rcpsp_pile(simple_jsp_problem):
    """Solve JSP via RCPSP transformation using Pile solver."""
    solver = CpmpyShopSolver(simple_jsp_problem)
    res = solver.solve()
    sol = res[-1][0]
    assert simple_jsp_problem.satisfy(sol)


# ============================================================================
# FJSP Tests
# ============================================================================
def test_fjsp_cpmpy(simple_fjsp_problem):
    """Solve FJSP via RCPSP transformation using Pile solver."""
    solver = CpmpyShopSolver(simple_fjsp_problem)
    res = solver.solve()
    sol = res[-1][0]
    assert simple_fjsp_problem.satisfy(sol)


# ============================================================================
# OSP Tests
# ============================================================================


def test_osp_cpmpy(simple_osp_problem):
    """Solve OSP via RCPSP transformation using Pile solver."""
    solver = CpmpyShopSolver(simple_osp_problem)
    res = solver.solve()
    sol = res[-1][0]
    assert simple_osp_problem.satisfy(sol)
