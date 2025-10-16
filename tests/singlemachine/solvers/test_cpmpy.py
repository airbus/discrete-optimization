#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cpmpy_tools import (
    CpmpyExplainUnsatMethod,
)
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.singlemachine.solvers.cpmpy_solver import (
    CpmpySingleMachineSolver,
    SingleMachineModel,
)


@pytest.fixture(params=list(SingleMachineModel))
def model_type(request):
    return request.param


def test_cpmpy(problem, model_type):
    solver = CpmpySingleMachineSolver(problem)
    solver.init_model(model_type=model_type)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert problem.satisfy(sol)


def test_explanation_meta(problem, model_type):
    solver = CpmpySingleMachineSolver(problem)
    solver.init_model(model_type=model_type, add_impossible_constraints=True)
    res = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])
    assert len(res) == 0
    assert solver.status_solver == StatusSolver.UNSATISFIABLE
    soft_constraints = solver.get_soft_meta_constraints()
    explanation = solver.explain_unsat_meta(
        soft=soft_constraints,
        hard=solver.get_hard_meta_constraints(),
        cpmpy_method=CpmpyExplainUnsatMethod.mus,
    )
    assert len(explanation) > 0

    if model_type == SingleMachineModel.LP:
        soft_constraints = []
        soft_constraints.append(solver.meta_constraints["impossible_deadline"])
        for i in range(problem.num_jobs):
            soft_constraints.append(solver.meta_constraints[f"completion_time_{i}"])
        explanation = solver.explain_unsat_meta(
            soft=soft_constraints,
            hard=solver.get_hard_meta_constraints(),
            cpmpy_method=CpmpyExplainUnsatMethod.mus,
        )
        assert len(explanation) > 0


def test_explanation_fine(problem, model_type):
    solver = CpmpySingleMachineSolver(problem)
    solver.init_model(model_type=model_type, add_impossible_constraints=True)
    res = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])
    assert len(res) == 0
    assert solver.status_solver == StatusSolver.UNSATISFIABLE
    explanation = solver.explain_unsat_fine(
        cpmpy_method=CpmpyExplainUnsatMethod.mus,
    )
    assert len(explanation) > 0
