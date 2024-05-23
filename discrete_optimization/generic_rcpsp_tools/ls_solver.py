#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import List, Optional

import numpy as np

from discrete_optimization.generic_rcpsp_tools.generic_rcpsp_solver import (
    SolverGenericRCPSP,
)
from discrete_optimization.generic_rcpsp_tools.typing import ANY_RCPSP
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.ls.hill_climber import HillClimber
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
from discrete_optimization.rcpsp.mutations.mutation_rcpsp import (
    PermutationMutationRCPSP,
)
from discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import InitialMethodRCPSP
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPModel
from discrete_optimization.rcpsp_multiskill.solvers.ms_rcpsp_lp_lns_solver import (
    InitialSolutionMS_RCPSP,
)

logger = logging.getLogger(__name__)


class LS_SOLVER(Enum):
    SA = 0
    HC = 1


class LS_RCPSP_Solver(SolverGenericRCPSP):
    hyperparameters = [
        CategoricalHyperparameter(
            name="init_solution_process", choices=[True, False], default=False
        ),
        EnumHyperparameter(name="ls_solver", enum=LS_SOLVER, default=LS_SOLVER.SA),
        FloatHyperparameter(name="temperature", low=0.01, high=10, default=3),
        IntegerHyperparameter(
            name="nb_iteration_no_improvement", low=10, high=2000, default=200
        ),
    ]

    def __init__(
        self,
        problem: ANY_RCPSP,
        params_objective_function: ParamsObjectiveFunction = None,
        ls_solver: LS_SOLVER = LS_SOLVER.SA,
        **args
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.ls_solver = ls_solver

    def solve(self, callbacks: Optional[List[Callback]] = None, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        model = self.problem
        dummy = kwargs.get("starting_point", model.get_dummy_solution())
        find_better_starting_solution = kwargs.get("init_solution_process", False)
        if isinstance(model, MS_RCPSPModel) and find_better_starting_solution:
            init = InitialSolutionMS_RCPSP(
                problem=self.problem,
                initial_method=InitialMethodRCPSP.PILE_CALENDAR,
                params_objective_function=self.params_objective_function,
            )
            sol = init.get_starting_solution()
            dummy = sol.get_best_solution()
        dummy = kwargs.get("init_solution", dummy)
        _, mutations = get_available_mutations(model, dummy)
        logger.debug(mutations)
        list_mutation = [
            mutate[0].build(model, dummy, **mutate[1])
            for mutate in mutations
            if mutate[0] == PermutationMutationRCPSP
        ]
        mixed_mutation = BasicPortfolioMutation(
            list_mutation, np.ones((len(list_mutation)))
        )
        res = RestartHandlerLimit(
            nb_iteration_no_improvement=kwargs["nb_iteration_no_improvement"],
        )
        ls = None
        if self.ls_solver == LS_SOLVER.SA:
            ls = SimulatedAnnealing(
                problem=model,
                mutator=mixed_mutation,
                restart_handler=res,
                temperature_handler=TemperatureSchedulingFactor(
                    temperature=kwargs["temperature"],
                    restart_handler=res,
                    coefficient=kwargs.get("decay_temperature", 0.9999),
                ),
                mode_mutation=ModeMutation.MUTATE,
                params_objective_function=self.params_objective_function,
                store_solution=False,
            )
        elif self.ls_solver == LS_SOLVER.HC:
            ls = HillClimber(
                problem=model,
                mutator=mixed_mutation,
                restart_handler=res,
                mode_mutation=ModeMutation.MUTATE,
                params_objective_function=self.params_objective_function,
                store_solution=True,
            )
        result_sa = ls.solve(
            dummy,
            callbacks=callbacks,
            nb_iteration_max=kwargs.get("nb_iteration_max", 2000),
        )
        return result_sa
