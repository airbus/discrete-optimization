#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import List, Union

from discrete_optimization.generic_tools.do_mutation import Mutation
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
    ObjectiveHandling,
)


class ParametersGa:
    def __init__(
        self,
        mutation: Union[Mutation, DeapMutation],
        crossover: DeapCrossover,
        selection: DeapSelection,
        encoding: str,
        objective_handling: ObjectiveHandling,
        objectives: Union[str, List[str]],
        objective_weights: List[float],
        pop_size: int,
        max_evals: int,
        mut_rate: float,
        crossover_rate: float,
        tournament_size: float,
        deap_verbose: bool,
    ):
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection
        self.encoding = encoding
        self.objective_handling = objective_handling
        self.objectives = objectives
        self.objective_weights = objective_weights
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.mut_rate = mut_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.deap_verbose = deap_verbose

    @staticmethod
    def default_rcpsp() -> "ParametersGa":
        return ParametersGa(
            mutation=DeapMutation.MUT_SHUFFLE_INDEXES,
            crossover=DeapCrossover.CX_UNIFORM_PARTIALY_MATCHED,
            selection=DeapSelection.SEL_TOURNAMENT,
            encoding="rcpsp_permutation",
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["makespan"],
            objective_weights=[-1],
            pop_size=100,
            max_evals=10000,
            mut_rate=0.1,
            crossover_rate=0.9,
            tournament_size=0.2,
            deap_verbose=True,
        )


class ParametersAltGa:
    def __init__(
        self,
        mutations: List[Union[Mutation, DeapMutation]],
        crossovers: List[DeapCrossover],
        selections: List[DeapSelection],
        encodings: List[str],
        objective_handling: ObjectiveHandling,
        objectives: Union[str, List[str]],
        objective_weights: List[float],
        pop_size: int,
        max_evals: int,
        mut_rate: float,
        crossover_rate: float,
        tournament_size: float,
        deap_verbose: bool,
        sub_evals: List[int],
    ):
        self.mutations = mutations
        self.crossovers = crossovers
        self.selections = selections
        self.encodings = encodings
        self.objective_handling = objective_handling
        self.objectives = objectives
        self.objective_weights = objective_weights
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.mut_rate = mut_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.deap_verbose = deap_verbose
        self.sub_evals = sub_evals

    @staticmethod
    def default_mrcpsp() -> "ParametersAltGa":
        return ParametersAltGa(
            mutations=[DeapMutation.MUT_UNIFORM_INT, DeapMutation.MUT_SHUFFLE_INDEXES],
            crossovers=[
                DeapCrossover.CX_ONE_POINT,
                DeapCrossover.CX_UNIFORM_PARTIALY_MATCHED,
            ],
            selections=[DeapSelection.SEL_TOURNAMENT, DeapSelection.SEL_TOURNAMENT],
            encodings=["rcpsp_modes_arity_fix", "rcpsp_permutation"],
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["makespan"],
            objective_weights=[-1],
            pop_size=100,
            max_evals=10000,
            mut_rate=0.1,
            crossover_rate=0.9,
            tournament_size=0.2,
            deap_verbose=True,
            sub_evals=[1000, 1000],
        )

    @staticmethod
    def default_msrcpsp() -> "ParametersAltGa":
        return ParametersAltGa(
            mutations=[
                DeapMutation.MUT_UNIFORM_INT,
                DeapMutation.MUT_SHUFFLE_INDEXES,
                DeapMutation.MUT_SHUFFLE_INDEXES,
            ],
            crossovers=[
                DeapCrossover.CX_ONE_POINT,
                DeapCrossover.CX_UNIFORM_PARTIALY_MATCHED,
                DeapCrossover.CX_UNIFORM_PARTIALY_MATCHED,
            ],
            selections=[
                DeapSelection.SEL_TOURNAMENT,
                DeapSelection.SEL_TOURNAMENT,
                DeapSelection.SEL_TOURNAMENT,
            ],
            encodings=[
                "modes_arity_fix_from_0",
                "priority_list_task",
                "priority_worker_per_task_perm",
            ],
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["makespan"],
            objective_weights=[-1],
            pop_size=100,
            max_evals=10000,
            mut_rate=0.1,
            crossover_rate=0.9,
            tournament_size=0.2,
            deap_verbose=True,
            sub_evals=[500, 500, 500],
        )
