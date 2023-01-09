#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from deap import algorithms, base, creator, tools

from discrete_optimization.generic_tools.do_mutation import Mutation
from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
    Problem,
    Solution,
    TypeAttribute,
    build_evaluate_function_aggregated,
)
from discrete_optimization.generic_tools.ea.deap_wrappers import generic_mutate_wrapper
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    TupleFitness,
)

logger = logging.getLogger(__name__)

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


class Nsga:
    """NSGA

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
        encoding: Optional[Union[str, Dict[str, Any]]] = None,
        objective_weights: Optional[List[float]] = None,
        pop_size: int = 100,
        max_evals: Optional[int] = None,
        mut_rate: float = 0.1,
        crossover_rate: float = 0.9,
        deap_verbose: bool = True,
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
        self._deap_verbose = deap_verbose

        # set encoding
        register_solution: EncodingRegister = problem.get_attribute_register()
        encoding_name = None
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

                if self._encoding_type == TypeAttribute.LIST_INTEGER:
                    self.arity = register_solution.dict_attribute_to_type[
                        encoding_name
                    ]["arity"]
                    self.arities = [self.arity for i in range(self.n)]
                else:
                    self.arity = None
                if self._encoding_type == TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY:
                    self.arities = register_solution.dict_attribute_to_type[
                        encoding_name
                    ]["arities"]

        if encoding is not None and isinstance(encoding, Dict):
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
                if "arity" in encoding.keys():
                    self.arity = encoding["arity"]
                    self.arities = [self.arity for i in range(self.n)]
                if "arities" in encoding.keys():
                    self.arities = register_solution.dict_attribute_to_type[
                        encoding_name
                    ]["arities"]
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

            if self._encoding_type == TypeAttribute.LIST_INTEGER:
                self.arity = register_solution.dict_attribute_to_type[encoding_name][
                    "arity"
                ]
                self.arities = [self.arity for i in range(self.n)]
            else:
                self.arity = None
            if self._encoding_type == TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY:
                self.arities = register_solution.dict_attribute_to_type[encoding_name][
                    "arities"
                ]

        self._encoding_name = encoding_name

        if self._encoding_type == TypeAttribute.LIST_BOOLEAN:
            self.arity = 2
            self.arities = [2 for i in range(self.n)]

        logger.debug(
            f"Encoding used by the GA: {self._encoding_name}: {self._encoding_type} of length {self.n}"
        )

        if isinstance(objectives, str):
            self._objectives = [objectives]
        else:
            self._objectives = objectives
        self._objective_weights: List[float]
        if (objective_weights is None) or (
            objective_weights is not None
            and (len(objective_weights) != len(self._objectives))
        ):
            logger.warning(
                "Objective weight issue: no weight given or size of weights and objectives lists mismatch. "
                "Setting all weights to default 1 value."
            )
            self._objective_weights = [1 for i in range(len(self._objectives))]
        else:
            self._objective_weights = objective_weights
        self.params_objective_function = ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.MULTI_OBJ,
            objectives=self._objectives,
            weights=self._objective_weights,
            sense_function=ModeOptim.MAXIMIZATION,
        )
        self.evaluate_sol: Union[
            Callable[[Solution], float], Callable[[Solution], TupleFitness]
        ]
        self.evaluate_sol, _ = build_evaluate_function_aggregated(
            problem=problem, params_objective_function=self.params_objective_function
        )

        nobj = len(self._objectives)
        ref_points = tools.uniform_reference_points(nobj=nobj)

        # DEAP toolbox setup
        self._toolbox = base.Toolbox()

        # Define representation
        creator.create("fitness", base.Fitness, weights=tuple(self._objective_weights))
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
            self._toolbox.register("int_val", random.randint, 0, self.arity - 1)
            self._toolbox.register(
                "individual",
                tools.initRepeat,
                creator.individual,
                self._toolbox.int_val,
                n=self.n,
            )
        elif self._encoding_type == TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY:
            gen_idx = lambda: [random.randint(0, arity - 1) for arity in self.arities]
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
                    low=0,
                    up=self.arities,
                    indpb=self._mut_rate,
                )

        # No choice of selection: In NSGA, only 1 selection: Non Dominated Sorted Selection
        self._toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    def evaluate_problem(self, int_vector: List[int]) -> Tuple[float, ...]:
        objective_values = self.problem.evaluate_from_encoding(  # type: ignore
            int_vector, self._encoding_variable_name
        )
        val = tuple([objective_values[obj_name] for obj_name in self._objectives])

        return val

    def solve(self, **kwargs: Any) -> ResultStorage:

        #  Define the statistics to collect at each generation
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        # Initialise the population (here at random)
        pop = self._toolbox.population()

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self._toolbox.map(self._toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Compile statistics about the population
        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        logger.debug(logbook.stream)

        # Begin the generational process
        ngen = int(self._max_evals / self._pop_size)
        logger.debug(f"ngen: {ngen}")
        for gen in range(1, ngen):
            offspring = algorithms.varAnd(
                pop, self._toolbox, self._crossover_rate, self._mut_rate
            )

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self._toolbox.map(self._toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population from parents and offspring
            pop = self._toolbox.select(pop + offspring, self._pop_size)

            # Compile statistics about the new population
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            logger.debug(logbook.stream)

        sols = []
        for s in pop:
            s_pure_int = [i for i in s]
            kwargs = {self._encoding_variable_name: s_pure_int, "problem": self.problem}
            problem_sol = self.problem.get_solution_type()(**kwargs)
            fits = self.evaluate_sol(problem_sol)
            sols.append((problem_sol, fits))
        rs = ResultStorage(
            list_solution_fits=sols,
            best_solution=None,
            mode_optim=self.params_objective_function.sense_function,
        )
        return rs
