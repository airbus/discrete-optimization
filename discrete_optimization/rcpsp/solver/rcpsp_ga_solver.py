#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Optional

from discrete_optimization.generic_tools.ea.alternating_ga import AlternatingGa
from discrete_optimization.generic_tools.ea.ga import Ga
from discrete_optimization.generic_tools.ea.ga_tools import (
    ParametersAltGa,
    ParametersGa,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.solver.rcpsp_solver import SolverRCPSP


class GA_RCPSP_Solver(SolverRCPSP):
    problem: RCPSPModel
    hyperparameters = Ga.hyperparameters

    def solve(self, parameters_ga: Optional[ParametersGa] = None, **args):
        if parameters_ga is None:
            parameters_ga = ParametersGa.default_rcpsp()
            args = self.complete_with_default_hyperparameters(args)
        for key in args:
            setattr(parameters_ga, key, args[key])
        ga_solver = Ga(
            problem=self.problem,
            encoding=parameters_ga.encoding,
            objective_handling=parameters_ga.objective_handling,
            objectives=parameters_ga.objectives,
            objective_weights=parameters_ga.objective_weights,
            mutation=parameters_ga.mutation,
            max_evals=parameters_ga.max_evals,
            crossover=parameters_ga.crossover,
            selection=parameters_ga.selection,
            pop_size=parameters_ga.pop_size,
            mut_rate=parameters_ga.mut_rate,
            crossover_rate=parameters_ga.crossover_rate,
            tournament_size=parameters_ga.tournament_size,
            deap_verbose=parameters_ga.deap_verbose,
        )
        return ga_solver.solve()


class GA_MRCPSP_Solver(SolverRCPSP):
    problem: RCPSPModel
    hyperparameters = Ga.hyperparameters

    def solve(
        self, parameters_ga: ParametersAltGa = ParametersAltGa.default_mrcpsp(), **args
    ):
        if parameters_ga is None:
            parameters_ga = ParametersAltGa.default_mrcpsp()
            args = self.complete_with_default_hyperparameters(args)
        for key in args:
            setattr(parameters_ga, key, args[key])
        ga_solver = AlternatingGa(
            problem=self.problem,
            encodings=parameters_ga.encodings,
            objective_handling=parameters_ga.objective_handling,
            objectives=parameters_ga.objectives,
            objective_weights=parameters_ga.objective_weights,
            mutations=parameters_ga.mutations,
            selections=parameters_ga.selections,
            crossovers=parameters_ga.crossovers,
            max_evals=parameters_ga.max_evals,
            sub_evals=parameters_ga.sub_evals,
            pop_size=parameters_ga.pop_size,
            mut_rate=parameters_ga.mut_rate,
            crossover_rate=parameters_ga.crossover_rate,
            tournament_size=parameters_ga.tournament_size,
            deap_verbose=parameters_ga.deap_verbose,
        )
        return ga_solver.solve()
