#  Copyright (c) 2024-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from discrete_optimization.fjsp.parser import get_data_available, parse_file
from discrete_optimization.fjsp.solvers.cpsat import CpSatFjspSolver
from discrete_optimization.fjsp.solvers.lns_cpsat import (
    FjspConstraintHandler,
    NeighborBuilderSubPart,
    NeighFjspConstraintHandler,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import (
    ConstraintHandlerMix,
    TrivialInitialSolution,
)


def test_lnscpsat_fjsp():
    files = get_data_available()
    file = [f for f in files if "Behnke1.fjs" in f][0]
    print(file)
    problem = parse_file(file)
    solver = CpSatFjspSolver(problem=problem)
    start_solution = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )
    initial_solution_provider = TrivialInitialSolution(solution=start_solution)
    lns_solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=solver,
        constraint_handler=ConstraintHandlerMix(
            problem=problem,
            list_constraints_handler=[
                FjspConstraintHandler(problem=problem, fraction_segment_to_fix=0.65),
                NeighFjspConstraintHandler(
                    problem=problem,
                    neighbor_builder=NeighborBuilderSubPart(
                        problem=problem, nb_cut_part=8
                    ),
                ),
            ],
            tag_constraint_handler=["random", "cut"],
            list_proba=[0.5, 0.5],
        ),
        initial_solution_provider=initial_solution_provider,
    )
    res = lns_solver.solve(
        nb_iteration_lns=2,
        time_limit_subsolver=1,
        subsolver_kwargs_factory=lambda: dict(callbacks=[NbIterationStopper(1)]),
    )
    sol, fit = res.get_best_solution_fit()
    assert problem.satisfy(sol)
