#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.ea.alternating_ga import AlternatingGa
from discrete_optimization.generic_tools.ea.ga_tools import ParametersAltGa
from discrete_optimization.rcpsp.problem import RcpspProblem


class GaMultiskillRcpspSolver(SolverDO):
    problem: RcpspProblem

    def solve(self, parameters_ga: ParametersAltGa = None, **args):
        if parameters_ga is None:
            parameters_ga = ParametersAltGa.default_msrcpsp()
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
