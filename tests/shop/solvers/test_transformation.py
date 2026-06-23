#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Test solving shop problems (JSP, FJSP, OSP) via transformations."""

import pytest

from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    TransformationSolver,
)
from discrete_optimization.rcpsp.solvers.cpsat_auto import CpSatAutoRcpspSolver
from discrete_optimization.rcpsp.solvers.pile import PileRcpspSolver
from discrete_optimization.shop.base import Job, Subjob, SubjobRecipe
from discrete_optimization.shop.fjsp.parser import get_data_available as get_fjsp_data
from discrete_optimization.shop.fjsp.parser import parse_file as parse_fjsp
from discrete_optimization.shop.fjsp.problem import FJobShopProblem
from discrete_optimization.shop.jsp.parser import get_data_available as get_jsp_data
from discrete_optimization.shop.jsp.parser import parse_file as parse_jsp
from discrete_optimization.shop.jsp.problem import JobShopProblem
from discrete_optimization.shop.osp.problem import OpenShopProblem
from discrete_optimization.shop.transformations.to_generic_scheduling import (
    ShopToGenericSchedulingTransformation,
)
from discrete_optimization.shop.transformations.to_rcpsp_multimode import (
    ShopToRcpspMultimodeTransformation,
)


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
    solver = TransformationSolver(
        transformation=ShopToRcpspMultimodeTransformation(),
        source_problem=simple_jsp_problem,
        solver_brick=SubBrick(cls=PileRcpspSolver, kwargs={}),
    )
    res = solver.solve()
    sol = res[-1][0]
    assert simple_jsp_problem.satisfy(sol)
    print(f"JSP via RCPSP Pile: {simple_jsp_problem.evaluate(sol)}")


def test_jsp_via_rcpsp_cpsat(simple_jsp_problem):
    """Solve JSP via RCPSP transformation using CP-SAT solver."""
    solver = TransformationSolver(
        transformation=ShopToRcpspMultimodeTransformation(),
        source_problem=simple_jsp_problem,
        solver_brick=SubBrick(cls=CpSatAutoRcpspSolver, kwargs={}),
    )
    solver.init_model()
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    )
    sol = res[-1][0]
    assert simple_jsp_problem.satisfy(sol)
    print(f"JSP via RCPSP CP-SAT: {simple_jsp_problem.evaluate(sol)}")


def test_jsp_via_generic_scheduling(simple_jsp_problem):
    """Solve JSP via GenericScheduling transformation using CP-SAT solver."""
    solver = TransformationSolver(
        transformation=ShopToGenericSchedulingTransformation(),
        source_problem=simple_jsp_problem,
        solver_brick=SubBrick(cls=GenericSchedulingAutoCpSatImplSolver, kwargs={}),
    )
    solver.init_model()
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    )
    sol = res[-1][0]
    assert simple_jsp_problem.satisfy(sol)
    print(f"JSP via GenericScheduling: {simple_jsp_problem.evaluate(sol)}")


def test_jsp_from_file_via_rcpsp():
    """Solve JSP from data file via RCPSP transformation."""
    files = get_jsp_data()
    file = [f for f in files if "la02" in f][0]
    problem = parse_jsp(file)

    solver = TransformationSolver(
        transformation=ShopToRcpspMultimodeTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(cls=PileRcpspSolver, kwargs={}),
    )
    res = solver.solve()
    sol = res[-1][0]
    assert problem.satisfy(sol)


# ============================================================================
# FJSP Tests
# ============================================================================


def test_fjsp_via_rcpsp_pile(simple_fjsp_problem):
    """Solve FJSP via RCPSP transformation using Pile solver."""
    solver = TransformationSolver(
        transformation=ShopToRcpspMultimodeTransformation(),
        source_problem=simple_fjsp_problem,
        solver_brick=SubBrick(cls=PileRcpspSolver, kwargs={}),
    )
    res = solver.solve()
    sol = res[-1][0]
    assert simple_fjsp_problem.satisfy(sol)
    print(f"FJSP via RCPSP Pile: {simple_fjsp_problem.evaluate(sol)}")


def test_fjsp_via_rcpsp_cpsat(simple_fjsp_problem):
    """Solve FJSP via RCPSP transformation using CP-SAT solver."""
    solver = TransformationSolver(
        transformation=ShopToRcpspMultimodeTransformation(),
        source_problem=simple_fjsp_problem,
        solver_brick=SubBrick(cls=CpSatAutoRcpspSolver, kwargs={}),
    )
    solver.init_model()
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    )
    sol = res[-1][0]
    assert simple_fjsp_problem.satisfy(sol)
    print(f"FJSP via RCPSP CP-SAT: {simple_fjsp_problem.evaluate(sol)}")


def test_fjsp_via_generic_scheduling(simple_fjsp_problem):
    """Solve FJSP via GenericScheduling transformation using CP-SAT solver."""
    solver = TransformationSolver(
        transformation=ShopToGenericSchedulingTransformation(),
        source_problem=simple_fjsp_problem,
        solver_brick=SubBrick(cls=GenericSchedulingAutoCpSatImplSolver, kwargs={}),
    )
    solver.init_model()
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    )
    sol = res[-1][0]
    assert simple_fjsp_problem.satisfy(sol)
    print(f"FJSP via GenericScheduling: {simple_fjsp_problem.evaluate(sol)}")


def test_fjsp_from_file_via_rcpsp():
    """Solve FJSP from data file via RCPSP transformation."""
    files = get_fjsp_data()
    file = [f for f in files if "Behnke1.fjs" in f][0]
    problem = parse_fjsp(file)

    solver = TransformationSolver(
        transformation=ShopToRcpspMultimodeTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(cls=PileRcpspSolver, kwargs={}),
    )
    res = solver.solve()
    sol = res[-1][0]
    assert problem.satisfy(sol)


# ============================================================================
# OSP Tests
# ============================================================================


def test_osp_via_rcpsp_pile(simple_osp_problem):
    """Solve OSP via RCPSP transformation using Pile solver."""
    solver = TransformationSolver(
        transformation=ShopToRcpspMultimodeTransformation(),
        source_problem=simple_osp_problem,
        solver_brick=SubBrick(cls=PileRcpspSolver, kwargs={}),
    )
    res = solver.solve()
    sol = res[-1][0]
    assert simple_osp_problem.satisfy(sol)
    print(f"OSP via RCPSP Pile: {simple_osp_problem.evaluate(sol)}")


def test_osp_via_rcpsp_cpsat(simple_osp_problem):
    """Solve OSP via RCPSP transformation using CP-SAT solver."""
    solver = TransformationSolver(
        transformation=ShopToRcpspMultimodeTransformation(),
        source_problem=simple_osp_problem,
        solver_brick=SubBrick(cls=CpSatAutoRcpspSolver, kwargs={}),
    )
    solver.init_model()
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    )
    sol = res[-1][0]
    assert simple_osp_problem.satisfy(sol)
    print(f"OSP via RCPSP CP-SAT: {simple_osp_problem.evaluate(sol)}")


def test_osp_via_generic_scheduling(simple_osp_problem):
    """Solve OSP via GenericScheduling transformation using CP-SAT solver."""
    solver = TransformationSolver(
        transformation=ShopToGenericSchedulingTransformation(),
        source_problem=simple_osp_problem,
        solver_brick=SubBrick(cls=GenericSchedulingAutoCpSatImplSolver, kwargs={}),
    )
    solver.init_model()
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    )
    sol = res[-1][0]
    assert simple_osp_problem.satisfy(sol)
    print(f"OSP via GenericScheduling: {simple_osp_problem.evaluate(sol)}")


def test_osp_from_jsp_file_via_rcpsp():
    """Solve OSP created from JSP data file via RCPSP transformation."""
    files = get_jsp_data()
    file = [f for f in files if "la02" in f][0]
    jsp_problem = parse_jsp(file)

    # Convert to OSP (no precedence)
    osp_problem = OpenShopProblem(
        list_jobs=jsp_problem.list_jobs,
        n_jobs=jsp_problem.n_jobs,
        n_machines=jsp_problem.n_machines,
        horizon=jsp_problem.horizon,
    )

    solver = TransformationSolver(
        transformation=ShopToRcpspMultimodeTransformation(),
        source_problem=osp_problem,
        solver_brick=SubBrick(cls=PileRcpspSolver, kwargs={}),
    )
    res = solver.solve()
    sol = res[-1][0]
    assert osp_problem.satisfy(sol)
