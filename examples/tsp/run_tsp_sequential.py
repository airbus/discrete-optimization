#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import numpy as np
from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.lns_cp import LNS_OrtoolsCPSat
from discrete_optimization.generic_tools.lns_tools import ConstraintHandlerMix
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandlerLimit,
)
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)
from discrete_optimization.generic_tools.sequential_metasolver import (
    SequentialMetasolver,
)
from discrete_optimization.tsp.mutation.mutation_tsp import (
    Mutation2Opt,
    MutationSwapTSP,
)
from discrete_optimization.tsp.plots.plot_tsp import plot_tsp_solution
from discrete_optimization.tsp.solver.tsp_cpsat_lns import (
    ConstraintHandlerSubpathTSP,
    ConstraintHandlerTSP,
)
from discrete_optimization.tsp.solver.tsp_cpsat_solver import CpSatTspSolver
from discrete_optimization.tsp.solver.tsp_gpdp_solver import SolverGpdpBased
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file

logging.basicConfig(level=logging.INFO)


def run_seq():
    files = get_data_available()
    files = [f for f in files if "tsp_574_1" in f]
    model = parse_file(files[0])
    params_objective_function = get_default_objective_setup(problem=model)
    solution = model.get_dummy_solution()
    _, list_mutation = get_available_mutations(model, solution)
    res = RestartHandlerLimit(3000)
    list_mutation = [
        mutate[0].build(model, solution, attribute="permutation", **mutate[1])
        for mutate in list_mutation
        if mutate[0] in [MutationSwapTSP, Mutation2Opt]
    ]
    weight = np.ones(len(list_mutation))
    mutate_portfolio = BasicPortfolioMutation(list_mutation, weight)
    solver = CpSatTspSolver(model, params_objective_function=params_objective_function)
    solv = SequentialMetasolver(
        problem=model,
        list_subbricks=[
            SubBrick(SolverGpdpBased, dict(time_limit=40)),
            SubBrick(
                SimulatedAnnealing,
                dict(
                    mutator=mutate_portfolio,
                    restart_handler=res,
                    temperature_handler=TemperatureSchedulingFactor(
                        temperature=100, restart_handler=res, coefficient=0.99999
                    ),
                    mode_mutation=ModeMutation.MUTATE_AND_EVALUATE,
                    nb_iteration_max=20000,
                ),
            ),
            SubBrick(SolverGpdpBased, dict(time_limit=20)),
            SubBrick(
                LNS_OrtoolsCPSat,
                dict(
                    subsolver=solver,
                    constraint_handler=ConstraintHandlerMix(
                        problem=model,
                        list_constraints_handler=[
                            ConstraintHandlerSubpathTSP(
                                problem=model, fraction_segment_to_fix=0.7
                            ),
                            ConstraintHandlerTSP(
                                problem=model, fraction_segment_to_fix=0.7
                            ),
                        ],
                        list_proba=[0.5, 0.5],
                    ),
                    nb_iteration_lns=1000,
                    callbacks=[TimerStopper(total_seconds=100)],
                    time_limit_subsolver=10,
                    time_limit_subsolver_iter0=10,
                ),
            ),
            SubBrick(
                CpSatTspSolver,
                dict(parameters_cp=ParametersCP.default_cpsat(), time_limit=100),
            ),
            SubBrick(SolverGpdpBased, dict(time_limit=20)),
        ],
    )
    res = solv.solve(
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ]
    )
    fig, ax = plt.subplots(1)
    list_solution_fit = sorted(res.list_solution_fits, key=lambda x: x[1], reverse=True)
    for sol, fit in list_solution_fit:
        ax.clear()
        plot_tsp_solution(tsp_model=model, solution=sol, ax=ax)
        ax.set_title(f"Length ={fit}")
        plt.pause(0.05)
    plt.show()


if __name__ == "__main__":
    run_seq()
