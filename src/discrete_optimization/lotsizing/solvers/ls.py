#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from enum import Enum

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import SolverDO, WarmstartMixin
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
    Ga,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
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
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    create_mutations_portfolio_from_problem,
)
from discrete_optimization.lotsizing.problem import LotSizingProblem
from discrete_optimization.lotsizing.solvers.greedy import (
    GreedyLotSizingSolver,
    GreedyStrategy,
)


class LocalSearchAlgo(Enum):
    HC = 0
    SA = 1
    GA = 2


class LSLotSizingSolver(SolverDO, WarmstartMixin):
    warm_start: Solution

    def set_warm_start(self, solution: Solution) -> None:
        self.warm_start = solution

    problem: LotSizingProblem
    hyperparameters = [
        EnumHyperparameter("solver", enum=LocalSearchAlgo, default=LocalSearchAlgo.SA)
    ]

    def solve(self, callbacks: list[Callback] = None, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        mutation = create_mutations_portfolio_from_problem(self.problem)
        solver = None
        if kwargs["solver"] == LocalSearchAlgo.HC:
            solver = HillClimber(
                problem=self.problem,
                mutator=mutation,
                restart_handler=RestartHandlerLimit(100),
                mode_mutation=ModeMutation.MUTATE,
                params_objective_function=self.params_objective_function,
                store_solution=False,
            )
        if kwargs["solver"] == LocalSearchAlgo.SA:
            rs = RestartHandlerLimit(100)
            solver = SimulatedAnnealing(
                problem=self.problem,
                mutator=mutation,
                restart_handler=rs,
                mode_mutation=ModeMutation.MUTATE,
                params_objective_function=self.params_objective_function,
                temperature_handler=TemperatureSchedulingFactor(
                    temperature=100, restart_handler=rs, coefficient=0.99999
                ),
                store_solution=False,
            )
        if kwargs["solver"] == LocalSearchAlgo.GA:
            solver = Ga(
                problem=self.problem,
                mutation=DeapMutation.MUT_SHUFFLE_INDEXES,
                crossover=DeapCrossover.CX_UNIFORM,
                selection=DeapSelection.SEL_TOURNAMENT,
                encoding="list_item_per_time",
                pop_size=50,
                max_evals=kwargs.get("nb_iteration_max", 10000),
            )
        # Use greedy solver for initial solution
        if not hasattr(self, "warm_start"):
            greedy_strategy = kwargs.get(
                "greedy_strategy", GreedyStrategy.EARLIEST_DEMAND_FIRST
            )
            greedy_solver = GreedyLotSizingSolver(
                problem=self.problem,
                params_objective_function=self.params_objective_function,
            )
            greedy_result = greedy_solver.solve(strategy=greedy_strategy)
            init_sol = greedy_result[0][0]
        else:
            init_sol = self.warm_start
        if kwargs["solver"] == LocalSearchAlgo.GA:
            solver: Ga
            solver.set_warm_start(init_sol)
            res = solver.solve(callbacks=callbacks)
        else:
            res = solver.solve(
                initial_variable=init_sol,
                nb_iteration_max=kwargs.get("nb_iteration_max", 10000),
                callbacks=callbacks,
            )
        return res
