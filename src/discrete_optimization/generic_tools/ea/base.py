#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


import logging
import random
from enum import Enum
from typing import Any, Optional, Union

from deap import base, creator, tools

from discrete_optimization.generic_tools.do_mutation import Mutation
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
    Problem,
    Solution,
    build_evaluate_function_aggregated,
)
from discrete_optimization.generic_tools.do_solver import SolverDO, WarmstartMixin
from discrete_optimization.generic_tools.ea.deap_wrappers import generic_mutate_wrapper
from discrete_optimization.generic_tools.encoding_register import (
    AttributeType,
    EncodingRegister,
    ListBoolean,
    ListInteger,
    Permutation,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)

logger = logging.getLogger(__name__)


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


def get_default_crossovers(attribute_type: AttributeType) -> DeapCrossover:
    match attribute_type:
        case ListBoolean():  # before ListInteger as subclass of it
            return DeapCrossover.CX_UNIFORM
        case ListInteger():
            return DeapCrossover.CX_ONE_POINT
        case Permutation():
            return DeapCrossover.CX_UNIFORM_PARTIALY_MATCHED
        case _:
            raise NotImplementedError()


def get_default_mutations(attribute_type: AttributeType) -> DeapMutation:
    match attribute_type:
        case ListBoolean():  # before ListInteger as subclass of it
            return DeapMutation.MUT_FLIP_BIT
        case ListInteger():
            return DeapMutation.MUT_UNIFORM_INT
        case Permutation():
            return DeapMutation.MUT_SHUFFLE_INDEXES
        case _:
            raise NotImplementedError()


class BaseGa(SolverDO, WarmstartMixin):
    """Base class for genetic algorithms.

    Shared code for Ga (single or aggregated objectives) and Nsga (multi-objective).

    Notes:
    - registering selection left to child classes

    """

    hyperparameters = [
        EnumHyperparameter(name="crossover", enum=DeapCrossover, default=None),
        IntegerHyperparameter(name="pop_size", low=1, high=1000, default=100),
        FloatHyperparameter(name="mut_rate", low=0, high=0.9, default=0.1),
        FloatHyperparameter(name="crossover_rate", low=0, high=1, default=0.9),
    ]

    initial_solution: Optional[Solution] = None
    """Initial solution used for warm start."""

    allowed_objective_handling = list(ObjectiveHandling)

    def __init__(
        self,
        problem: Problem,
        mutation: Mutation | DeapMutation | None = None,
        crossover: DeapCrossover | None = None,
        encoding: str | tuple[str, AttributeType] | None = None,
        pop_size: int = 100,
        max_evals: int | None = None,
        mut_rate: float = 0.1,
        crossover_rate: float = 0.9,
        deap_verbose: bool = True,
        initial_population: list[list[Any]] | None = None,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs: Any,
    ):
        # Get default params_objective_function if not specified
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        # Update params_objective_function objectives, objective_weights, and objective_handling with GA conventions
        self._check_params_objective_function()
        # update aggreg_sol & co as objective_handling can change during previous call
        (
            self.aggreg_from_sol,
            self.aggreg_from_dict,
        ) = build_evaluate_function_aggregated(
            problem=self.problem,
            params_objective_function=self.params_objective_function,
        )

        self._pop_size = pop_size
        if max_evals is not None:
            self._max_evals = max_evals
        else:
            self._max_evals = 100 * self._pop_size
            logger.warning(
                "No value specified for max_evals. Using the default 100*pop_size - This should really be set carefully"
            )
        self._mut_rate = mut_rate
        self._crossover_rate = crossover_rate
        self._deap_verbose = deap_verbose

        self.initial_population = initial_population

        # set encoding
        register_solution: EncodingRegister = problem.get_attribute_register()
        if encoding is None:
            # take the first attribute of the register
            if len(register_solution) == 0:
                raise Exception(
                    "No encoding defined in the EncodingRegister of your problem."
                    "Please specify a tuple `(attribute_name, attribute_type)` as encoding argument."
                )
            attribute_name, attribute_type = next(iter(register_solution.items()))
        elif isinstance(encoding, str):
            attribute_name = encoding
            try:
                attribute_type = register_solution[encoding]
            except KeyError:
                raise ValueError(
                    f"{encoding} is not in the attribute register of the problem."
                )
        else:
            attribute_name, attribute_type = encoding

        logger.debug(f"Encoding used: {attribute_name}: {attribute_type}")

        # DEAP toolbox setup
        self._toolbox = base.Toolbox()

        # Define representation
        if self._objective_handling == ObjectiveHandling.MULTI_OBJ:
            creator.create(
                "fitness", base.Fitness, weights=tuple(self._objective_weights)
            )
        else:
            # handle weights in evaluate_problem()
            creator.create(
                "fitness",
                base.Fitness,
                weights=(1.0,),
            )
        creator.create(
            "individual", list, fitness=creator.fitness
        )  # associate the fitness function to the individual type

        # Create the individuals required by the encoding
        n = attribute_type.length
        if isinstance(attribute_type, ListBoolean):
            self._toolbox.register(
                "bit", random.randint, 0, 1
            )  # Each element of a solution is a bit (i.e. an int between 0 and 1 incl.)

            self._toolbox.register(
                "individual",
                tools.initRepeat,
                creator.individual,
                self._toolbox.bit,
                n=n,
            )  # An individual (aka solution) contains n bits
        elif isinstance(attribute_type, Permutation):
            self._toolbox.register(
                "permutation_indices", random.sample, attribute_type.range, n
            )
            self._toolbox.register(
                "individual",
                tools.initIterate,
                creator.individual,
                self._toolbox.permutation_indices,
            )
        elif isinstance(attribute_type, ListInteger):
            gen_idx = lambda: [
                random.randint(low, up)
                for low, up in zip(attribute_type.lows, attribute_type.ups)
            ]
            self._toolbox.register(
                "individual", tools.initIterate, creator.individual, gen_idx
            )
        else:
            raise NotImplementedError()

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
            self._crossover = get_default_crossovers(attribute_type)
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
            self._mutation = get_default_mutations(attribute_type)
        else:
            self._mutation = mutation

        if isinstance(self._mutation, Mutation):
            self._toolbox.register(
                "mutate",
                generic_mutate_wrapper,
                problem=self.problem,
                attribute_name=attribute_name,
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
                if isinstance(attribute_type, ListInteger):
                    self._toolbox.register(
                        "mutate",
                        tools.mutUniformInt,
                        low=attribute_type.lows,
                        up=attribute_type.ups,
                        indpb=self._mut_rate,
                    )
                else:
                    raise NotImplementedError()

        self._attribute_type = attribute_type
        self._attribute_name = attribute_name

    def _check_params_objective_function(
        self,
    ) -> None:
        # Use params_objective_function for objectives and weights (with maximization convention)
        objectives = self.params_objective_function.objectives
        objective_weights = self.params_objective_function.weights
        objective_handling = self.params_objective_function.objective_handling
        if self.params_objective_function.sense_function == ModeOptim.MINIMIZATION:
            # GA take the convention of maximizing => - weights
            objective_weights = [-w for w in objective_weights]

        # Check allowed objective handling
        if objective_handling not in self.allowed_objective_handling:
            new_objective_handling = self.allowed_objective_handling[0]
            logger.warning(
                f"Objective handling {objective_handling} not allowed for this algorithm. "
                f"Switching to {new_objective_handling}."
            )
            objective_handling = new_objective_handling
        # Get objectives as a list
        if isinstance(objectives, str):
            objectives = [objectives]
        # Check objective_weights
        if objective_weights is None:
            objective_weights = [1.0] * len(objectives)
        elif len(objective_weights) != len(objectives):
            logger.warning(
                "Objective weight issue: size of weights and objectives lists mismatch. "
                "Setting all weights to default 1 value."
            )
            objective_weights = [1.0] * len(objectives)
        # Check objective vs objective_handling
        if len(objectives) == 0:
            raise ValueError("You cannot specify an empty list of objectives.")
        if objective_handling == ObjectiveHandling.SINGLE:
            if len(objectives) > 1:
                logger.warning(
                    "Many objectives specified but single objective handling, using the first objective in the dictionary"
                )
            objectives = objectives[:1]
        # store final objectives, weights, and handling
        self.params_objective_function.objective_handling = objective_handling
        self.params_objective_function.objectives = objectives
        self.params_objective_function.weights = objective_weights
        self.params_objective_function.sense_function = ModeOptim.MAXIMIZATION
        self._objectives = objectives
        self._objective_weights = objective_weights
        self._objective_handling = objective_handling

    def evaluate_problem(self, int_vector: list[int]) -> tuple[float, ...]:
        objective_values = self.problem.evaluate_from_encoding(
            int_vector=int_vector, encoding_name=self._attribute_name
        )
        if self._objective_handling == ObjectiveHandling.MULTI_OBJ:
            # NB: weights managed in registered fitness
            return tuple([objective_values[obj_name] for obj_name in self._objectives])
        else:
            if self._objective_handling == ObjectiveHandling.SINGLE:
                # take into account weight in case of minimization
                val = objective_values[self._objectives[0]] * self._objective_weights[0]
            else:  # self._objective_handling == ObjectiveHandling.AGGREGATE:
                val = sum(
                    [
                        objective_values[self._objectives[i]]
                        * self._objective_weights[i]
                        for i in range(len(self._objectives))
                    ]
                )

            return (val,)

    def generate_initial_population(self) -> list[Any]:
        if self.initial_population is None:
            # Initialise the population (here at random)
            population = self._toolbox.population()
        else:
            population = self.generate_custom_population()
            self._pop_size = len(population)

        # manage warm start: set 1 element of the population to the initial_solution
        if self.initial_solution is not None:
            population[0] = self.create_individual_from_solution(self.initial_solution)

        return population

    def generate_custom_population(self) -> list[Any]:
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

    def create_individual_from_solution(self, solution: Solution) -> Any:
        ind = getattr(solution, self._attribute_name)
        newind = self._toolbox.individual()
        for j in range(len(ind)):
            newind[j] = ind[j]
        return newind

    def set_warm_start(self, solution: Solution) -> None:
        """Make the solver warm start from the given solution.

        Will be ignored if arg `initial_variable` is set and not None in call to `solve()`.

        """
        self.initial_solution = solution
