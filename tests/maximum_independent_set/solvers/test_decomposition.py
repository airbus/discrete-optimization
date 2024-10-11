import logging

import pytest

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.maximum_independent_set.parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.problem import MisProblem
from discrete_optimization.maximum_independent_set.solvers.cpsat import CpSatMisSolver
from discrete_optimization.maximum_independent_set.solvers.decomposition import (
    DecomposedMisSolver,
)


def test_decomposition_ortools():
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = DecomposedMisSolver(problem=mis_model)
    p = ParametersCp.default_cpsat()
    res = solver.solve(
        initial_solver=SubBrick(
            cls=CpSatMisSolver, kwargs={"parameters_cp": p, "time_limit": 5}
        ),
        root_solver=SubBrick(
            cls=CpSatMisSolver, kwargs={"parameters_cp": p, "time_limit": 5}
        ),
        proportion_to_remove=0.6,
        nb_iteration=10,
    )
    solution, fit = res.get_best_solution_fit()
    print(fit)
    assert mis_model.satisfy(solution)


def test_decomposition_warm_start():
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = DecomposedMisSolver(problem=mis_model)
    p = ParametersCp.default_cpsat()
    res = solver.solve(
        root_solver=SubBrick(
            cls=CpSatMisSolver, kwargs={"parameters_cp": p, "time_limit": 5}
        ),
        proportion_to_remove=0.6,
        nb_iteration=2,
    )

    # test warm start
    start_solution = mis_model.get_dummy_solution()
    assert res[0][0].chosen != start_solution.chosen
    solver.set_warm_start(start_solution)
    res2 = solver.solve(
        root_solver=SubBrick(cls=CpSatMisSolver, kwargs={"parameters_cp": p}),
        proportion_to_remove=0.6,
        nb_iteration=2,
    )

    assert res2[0][0].chosen == start_solution.chosen
