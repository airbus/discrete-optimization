#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from discrete_optimization.fjsp.parser import get_data_available, parse_file
from discrete_optimization.fjsp.solvers.cpsat import CpSatFjspSolver
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    ConstraintExtractorList,
    MultimodeConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    TasksConstraintHandler,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.neighbor_tools import (
    NeighborBuilderMix,
    NeighborBuilderSubPart,
    NeighborRandom,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat


def test_lnscpsat_fjsp():
    files = get_data_available()
    file = [f for f in files if "Behnke1.fjs" in f][0]
    print(file)
    problem = parse_file(file)
    solver = CpSatFjspSolver(problem=problem)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    constraint_handler = TasksConstraintHandler(
        problem=problem,
    )
    assert isinstance(constraint_handler.constraints_extractor, ConstraintExtractorList)
    assert any(
        isinstance(extractor, MultimodeConstraintExtractor)
        for extractor in constraint_handler.constraints_extractor.extractors
    )

    lns_solver = LnsOrtoolsCpSat(
        problem=problem, subsolver=solver, constraint_handler=constraint_handler
    )
    res = lns_solver.solve(
        skip_initial_solution_provider=True,
        nb_iteration_lns=20,
        parameters_cp=p,
        time_limit_subsolver_iter0=1,
        time_limit_subsolver=2,
    )
    sol, fit = res.get_best_solution_fit()
    assert problem.satisfy(sol)
