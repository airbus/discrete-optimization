import logging

import pytest

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.maximum_independent_set.mis_model import MisProblem
from discrete_optimization.maximum_independent_set.mis_parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.solvers.mis_decomposition import (
    MisDecomposedSolver,
)
from discrete_optimization.maximum_independent_set.solvers.mis_ortools import (
    MisOrtoolsSolver,
)


def test_decomposition_ortools():
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = MisDecomposedSolver(problem=mis_model)
    p = ParametersCP.default_cpsat()
    p.time_limit = 5
    res = solver.solve(
        initial_solver=SubBrick(cls=MisOrtoolsSolver, kwargs={"parameters_cp": p}),
        root_solver=SubBrick(cls=MisOrtoolsSolver, kwargs={"parameters_cp": p}),
        proportion_to_remove=0.6,
        nb_iteration=10,
    )
    solution, fit = res.get_best_solution_fit()
    print(fit)
    assert mis_model.satisfy(solution)
