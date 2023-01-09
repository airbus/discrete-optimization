#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from deap import algorithms, base, creator, tools

from discrete_optimization.generic_tools.do_mutation import Mutation
from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ObjectiveHandling,
    Problem,
    TypeAttribute,
    build_aggreg_function_and_params_objective,
    lower_bound_vector_encoding_from_dict,
    upper_bound_vector_encoding_from_dict,
)
from discrete_optimization.generic_tools.ea.deap_wrappers import generic_mutate_wrapper
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class DeapSelection(Enum):
    SEL_TOURNAMENT = 0
    SEL_RANDOM = 1
    SEL_BEST = 2
    SEL_ROULETTE = 4
    SEL_WORST = 5
    SEL_STOCHASTIC_UNIVERSAL_SAMPLING = 6


class DeapMutation(Enum):
    MUT_FLIP_BIT = 0  # bit
    MUT_SHUFFLE_INDEXES = 1  # perm
    MUT_UNIFORM_INT = 2  # int


class DeapCrossover(Enum):
    CX_UNIFORM = 0  # bit, int
    CX_UNIFORM_PARTIALY_MATCHED = 1  # perm
    CX_ORDERED = 2  # perm
    CX_ONE_POINT = 3  # bit, int
    CX_TWO_POINT = 4  # bit, int
    CX_PARTIALY_MATCHED = 5  # perm


_default_crossovers = {
    TypeAttribute.LIST_BOOLEAN: DeapCrossover.CX_UNIFORM,
    TypeAttribute.LIST_INTEGER: DeapCrossover.CX_ONE_POINT,
    TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY: DeapCrossover.CX_ONE_POINT,
    TypeAttribute.PERMUTATION: DeapCrossover.CX_UNIFORM_PARTIALY_MATCHED,
}

_default_mutations = {
    TypeAttribute.LIST_BOOLEAN: DeapMutation.MUT_FLIP_BIT,
    TypeAttribute.LIST_INTEGER: DeapMutation.MUT_UNIFORM_INT,
    TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY: DeapMutation.MUT_UNIFORM_INT,
    TypeAttribute.PERMUTATION: DeapMutation.MUT_SHUFFLE_INDEXES,
}


class Ga:
    """Single objective GA

    Args:
        problem:
            the problem to solve
        encoding:
            name (str) of an encoding registered in the register solution of Problem
            or a dictionary of the form {'type': TypeAttribute, 'n': int} where type refers to a TypeAttribute and n
             to the dimension of the problem in this encoding (e.g. length of the vector)
            by default, the first encoding in the problem register_solution will be used.

    """

    def __init__(
        self,
        problem: Problem,
        objectives: Union[str, List[str]],
        mutation: Optional[Union[Mutation, DeapMutation]] = None,
        crossover: Optional[DeapCrossover] = None,
        selection: DeapSelection = DeapSelection.SEL_TOURNAMENT,
        encoding: Optional[Union[str, Dict[str, Any]]] = None,
        objective_handling: ObjectiveHandling = ObjectiveHandling.SINGLE,
        objective_weights: Optional[List[float]] = None,
        pop_size: int = 100,
        max_evals: Optional[int] = None,
        mut_rate: float = 0.1,
        crossover_rate: float = 0.9,
        tournament_size: float = 0.2,  # as a percentage of the population
        deap_verbose: bool = True,
        initial_population: Optional[List[List[Any]]] = None,
    ):
        self.problem = problem
        if not hasattr(self.problem, "evaluate_from_encoding"):
            raise ValueError("self.problem shoud define an evaluate_from_encoding()")
            # self.problem.evaluate_from_encoding: Callable[[List[int], str], Dict[str, float]]

        self._pop_size = pop_size
        if max_evals is not None:
            self._max_evals = max_evals
        else:
            self._max_evals = 100 * self._pop_size
            logger.warning(
                "No value specified for max_evals. Using the default 10*pop_size - This should really be set carefully"
            )
        self._mut_rate = mut_rate
        self._crossover_rate = crossover_rate
        self._tournament_size = tournament_size
        self._deap_verbose = deap_verbose

        self.initial_population = initial_population

        # set encoding
        register_solution: EncodingRegister = problem.get_attribute_register()
        encoding_name: Optional[str] = None
        if encoding is not None and isinstance(encoding, str):
            # check name specified is in problem register
            if encoding in register_solution.dict_attribute_to_type.keys():
                encoding_name = encoding
                self._encoding_variable_name = register_solution.dict_attribute_to_type[
                    encoding_name
                ]["name"]
                self._encoding_type = register_solution.dict_attribute_to_type[
                    encoding_name
                ]["type"][0]
                self.n = register_solution.dict_attribute_to_type[encoding_name]["n"]
                if self._encoding_type in {
                    TypeAttribute.LIST_INTEGER,
                    TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY,
                }:
                    self.lows = lower_bound_vector_encoding_from_dict(
                        register_solution.dict_attribute_to_type[encoding_name]
                    )
                    self.ups = upper_bound_vector_encoding_from_dict(
                        register_solution.dict_attribute_to_type[encoding_name]
                    )

        elif encoding is not None and isinstance(encoding, Dict):
            # check there is a type key and a n key
            if (
                "name" in encoding.keys()
                and "type" in encoding.keys()
                and "n" in encoding.keys()
            ):
                encoding_name = "custom"
                self._encoding_variable_name = encoding["name"]
                self._encoding_type = encoding["type"][0]
                self.n = encoding["n"]
                if self._encoding_type in {
                    TypeAttribute.LIST_INTEGER,
                    TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY,
                }:
                    self.lows = lower_bound_vector_encoding_from_dict(encoding)
                    self.ups = upper_bound_vector_encoding_from_dict(encoding)
            else:
                logger.warning(
                    "Erroneous encoding provided as input (encoding name not matching encoding of problem or custom "
                    "definition not respecting encoding dict entry format, trying to use default one instead"
                )

        if encoding_name is None:
            if len(register_solution.dict_attribute_to_type.keys()) == 0:
                raise Exception(
                    "An encoding of type TypeAttribute should be specified or at least 1 TypeAttribute "
                    "should be defined in the RegisterSolution of your Problem"
                )
            encoding_name = list(register_solution.dict_attribute_to_type.keys())[0]
            self._encoding_variable_name = register_solution.dict_attribute_to_type[
                encoding_name
            ]["name"]
            self._encoding_type = register_solution.dict_attribute_to_type[
                encoding_name
            ]["type"][0]
            self.n = register_solution.dict_attribute_to_type[encoding_name]["n"]

            dict_register = register_solution.dict_attribute_to_type
            if self._encoding_type in {
                TypeAttribute.LIST_INTEGER,
                TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY,
            }:
                self.lows = lower_bound_vector_encoding_from_dict(
                    dict_register[encoding_name]
                )
                self.ups = upper_bound_vector_encoding_from_dict(
                    dict_register[encoding_name]
                )

        if self._encoding_type == TypeAttribute.LIST_BOOLEAN:
            self.lows = [0 for i in range(self.n)]
            self.ups = [1 for i in range(self.n)]

        self._encoding_name: str = encoding_name

        logger.debug(
            f"Encoding used by the GA: {self._encoding_name}: {self._encoding_type} of length {self.n}"
        )

        # set objective handling stuff
        self._objective_handling: ObjectiveHandling
        if objective_handling is None:
            self._objective_handling = ObjectiveHandling.SINGLE
        else:
            self._objective_handling = objective_handling

        if isinstance(objectives, str):
            self._objectives = [objectives]
        else:
            self._objectives = objectives
        if (
            len(self._objectives) > 1
            and self._objective_handling == ObjectiveHandling.SINGLE
        ):
            logger.warning(
                "Many objectives specified but single objective handling, using the first objective in the dictionary"
            )

        self._objective_weights: List[float]
        if (objective_weights is None) or (
            objective_weights is not None
            and (
                (
                    len(objective_weights) != len(self._objectives)
                    and self._objective_handling == ObjectiveHandling.AGGREGATE
                )
            )
        ):
            logger.warning(
                "Objective weight issue: no weight given or size of weights and objectives lists mismatch. "
                "Setting all weights to default 1 value."
            )
            self._objective_weights = [1 for i in range(len(self._objectives))]
        else:
            self._objective_weights = objective_weights

        self._selection_type = selection

        # DEAP toolbox setup
        self._toolbox = base.Toolbox()

        # Define representation
        creator.create(
            "fitness",
            base.Fitness,
            weights=(
                1.0,
            ),  # we keep this to 1 and let the user provides the weights for each subobjective
        )  # (a negative weight defines the objective as a minimisation)
        creator.create(
            "individual", list, fitness=creator.fitness
        )  # associate the fitness function to the individual type

        # Create the individuals required by the encoding
        if self._encoding_type == TypeAttribute.LIST_BOOLEAN:
            self._toolbox.register(
                "bit", random.randint, 0, 1
            )  # Each element of a solution is a bit (i.e. an int between 0 and 1 incl.)

            self._toolbox.register(
                "individual",
                tools.initRepeat,
                creator.individual,
                self._toolbox.bit,
                n=self.n,
            )  # An individual (aka solution) contains n bits
        elif self._encoding_type == TypeAttribute.PERMUTATION:
            self._toolbox.register(
                "permutation_indices", random.sample, range(self.n), self.n
            )
            self._toolbox.register(
                "individual",
                tools.initIterate,
                creator.individual,
                self._toolbox.permutation_indices,
            )
        elif self._encoding_type == TypeAttribute.LIST_INTEGER:
            self._toolbox.register("int_val", random.randint, self.lows[0], self.ups[0])
            self._toolbox.register(
                "individual",
                tools.initRepeat,
                creator.individual,
                self._toolbox.int_val,
                n=self.n,
            )
        elif self._encoding_type == TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY:
            gen_idx = lambda: [
                random.randint(low, up) for low, up in zip(self.lows, self.ups)
            ]
            self._toolbox.register(
                "individual", tools.initIterate, creator.individual, gen_idx
            )
        elif self._encoding_type == TypeAttribute.LIST_FLOATS:
            gen_idx = lambda: [
                random.randrange(low, up) for low, up in zip(self.lows, self.ups)
            ]
            self._toolbox.register(
                "individual", tools.initIterate, creator.individual, gen_idx
            )

        self._toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self._toolbox.individual,
            n=self._pop_size,
        )  # A population is made of pop_size individuals

        # Define objective function
        self._toolbox.register(
            "evaluate",
            self.evaluate_problem,
        )

        # Define crossover
        if crossover is None:
            self._crossover = _default_crossovers[self._encoding_type]
        else:
            self._crossover = crossover

        if self._crossover == DeapCrossover.CX_UNIFORM:
            self._toolbox.register("mate", tools.cxUniform, indpb=self._crossover_rate)
        elif self._crossover == DeapCrossover.CX_ONE_POINT:
            self._toolbox.register("mate", tools.cxOnePoint)
        elif self._crossover == DeapCrossover.CX_TWO_POINT:
            self._toolbox.register("mate", tools.cxTwoPoint)
        elif self._crossover == DeapCrossover.CX_UNIFORM_PARTIALY_MATCHED:
            self._toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=0.5)
        elif self._crossover == DeapCrossover.CX_ORDERED:
            self._toolbox.register("mate", tools.cxOrdered)
        elif self._crossover == DeapCrossover.CX_PARTIALY_MATCHED:
            self._toolbox.register("mate", tools.cxPartialyMatched)
        else:
            logger.warning("Crossover of specified type not handled!")

        # Define mutation
        self._mutation: Union[Mutation, DeapMutation]
        if mutation is None:
            self._mutation = _default_mutations[self._encoding_type]
        else:
            self._mutation = mutation

        if isinstance(self._mutation, Mutation):
            self._toolbox.register(
                "mutate",
                generic_mutate_wrapper,
                problem=self.problem,
                encoding_name=self._encoding_variable_name,
                indpb=self._mut_rate,
                solution_fn=self.problem.get_solution_type(),
                custom_mutation=mutation,
            )
        elif isinstance(self._mutation, DeapMutation):
            if self._mutation == DeapMutation.MUT_FLIP_BIT:
                self._toolbox.register(
                    "mutate", tools.mutFlipBit, indpb=self._mut_rate
                )  # Choice of mutation operator
            elif self._mutation == DeapMutation.MUT_SHUFFLE_INDEXES:
                self._toolbox.register(
                    "mutate", tools.mutShuffleIndexes, indpb=self._mut_rate
                )  # Choice of mutation operator
            elif self._mutation == DeapMutation.MUT_UNIFORM_INT:
                self._toolbox.register(
                    "mutate",
                    tools.mutUniformInt,
                    low=self.lows,
                    up=self.ups,
                    indpb=self._mut_rate,
                )

        # Choice of selection
        if self._selection_type == DeapSelection.SEL_TOURNAMENT:
            self._toolbox.register(
                "select",
                tools.selTournament,
                tournsize=int(self._tournament_size * self._pop_size),
            )
        elif self._selection_type == DeapSelection.SEL_RANDOM:
            self._toolbox.register("select", tools.selRandom)
        elif self._selection_type == DeapSelection.SEL_BEST:
            self._toolbox.register("select", tools.selBest)
        elif self._selection_type == DeapSelection.SEL_ROULETTE:
            self._toolbox.register("select", tools.selRoulette)
        elif self._selection_type == DeapSelection.SEL_WORST:
            self._toolbox.register("select", tools.selWorst)
        elif self._selection_type == DeapSelection.SEL_STOCHASTIC_UNIVERSAL_SAMPLING:
            self._toolbox.register("select", tools.selStochasticUniversalSampling)

        (
            self.aggreg_from_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem, params_objective_function=None
        )

    def evaluate_problem(self, int_vector: List[int]) -> Tuple[float]:
        objective_values: Dict[str, float] = self.problem.evaluate_from_encoding(  # type: ignore
            int_vector, self._encoding_variable_name
        )
        if self._objective_handling == ObjectiveHandling.SINGLE:
            val = objective_values[self._objectives[0]]
        elif self._objective_handling == ObjectiveHandling.AGGREGATE:
            val = sum(
                [
                    objective_values[self._objectives[i]] * self._objective_weights[i]
                    for i in range(len(self._objectives))
                ]
            )
        else:  # ObjectiveHandling.MULTI_OBJ
            raise NotImplementedError(
                "objective_handling can only be SINGLE or AGGREGATE"
            )
        return (val,)

    def generate_custom_population(self) -> List[Any]:
        if self.initial_population is None:
            raise RuntimeError(
                "self.initial_population cannot be None when calling generate_custom_population()."
            )
        pop = []
        for ind in self.initial_population:
            newind = self._toolbox.individual()
            for j in range(len(ind)):
                newind[j] = ind[j]
            pop.append(newind)
        return pop

    def solve(self, **kwargs: Any) -> ResultStorage:
        if self.initial_population is None:
            # Initialise the population (here at random)
            population = self._toolbox.population()
        else:
            population = self.generate_custom_population()
            self._pop_size = len(population)

        fits = self._toolbox.map(self._toolbox.evaluate, population)
        for fit, ind in zip(fits, population):
            ind.fitness.values = fit

        #  Define the statistics to collect at each generation
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run the GA: final population and statistics logbook are created
        pop_vector, logbook = algorithms.eaSimple(
            population=population,
            toolbox=self._toolbox,
            cxpb=self._crossover_rate,
            mutpb=self._mut_rate,
            ngen=int(self._max_evals / self._pop_size),
            stats=stats,
            halloffame=hof,
            verbose=self._deap_verbose,
        )

        best_vector = hof[0]

        s_pure_int = [i for i in best_vector]
        kwargs = {self._encoding_variable_name: s_pure_int, "problem": self.problem}
        problem_sol = self.problem.get_solution_type()(**kwargs)

        result_storage = ResultStorage(
            list_solution_fits=[(problem_sol, self.aggreg_from_sol(problem_sol))],
            best_solution=problem_sol,
            mode_optim=self.params_objective_function.sense_function,
        )
        return result_storage
