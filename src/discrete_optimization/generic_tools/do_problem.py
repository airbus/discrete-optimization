"""Base module for the problem implementation in discrete-optimization library."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from discrete_optimization.generic_tools.encoding_register import EncodingRegister
from discrete_optimization.generic_tools.result_storage.multiobj_utils import (
    TupleFitness,
)

logger = logging.getLogger(__name__)


class ModeOptim(Enum):
    """Enum class to specify minimization or maximization problems."""

    MAXIMIZATION = 0
    MINIMIZATION = 1


def lower_bound_vector_encoding_from_dict(dict_encoding: dict[str, Any]) -> list[int]:
    length_encoding = dict_encoding["n"]
    if "low" in dict_encoding:
        low_value: Union[int, Iterable[int]] = dict_encoding["low"]
        if isinstance(low_value, int):
            return [low_value for i in range(length_encoding)]
        else:
            return list(low_value)
    else:
        return [0 for i in range(length_encoding)]  # By default we start at zero.


def upper_bound_vector_encoding_from_dict(dict_encoding: dict[str, Any]) -> list[int]:
    """Return for an encoding that is of type LIST_INTEGER or associated, the upper bound vector.

    Examples: if the vector should contains value higher or equal to 1, the function will return a list full of 1.
    """
    length_encoding = dict_encoding["n"]
    if "up" in dict_encoding:
        up_value: Union[int, Iterable[int]] = dict_encoding["up"]
        if isinstance(up_value, int):
            return [up_value for i in range(length_encoding)]
        else:
            return list(up_value)
    else:
        low = lower_bound_vector_encoding_from_dict(dict_encoding)
        up_value_vector = None
        if "arity" in dict_encoding:
            arity = dict_encoding["arity"]  # number of possible value.
            return [l + arity - 1 for l in low]
        elif "arities" in dict_encoding:
            arities = dict_encoding["arities"]
            return [l + arr - 1 for l, arr in zip(low, arities)]
        else:
            raise ValueError(
                "dict_encoding must either have 'up', 'arity' or 'arities' as a key."
            )


class ObjectiveHandling(Enum):
    """Enum class specifying how should be built the objective criteria.

    When SINGLE, it means the problem only returns one KPI to be minimize/maximize
    When AGGREGATE, the problems has several KPI to combine with different ponderations.
    When MULTI_OBJ, pareto optimisation will be done if possible.
    """

    SINGLE = 0
    AGGREGATE = 1
    MULTI_OBJ = 2


class TypeObjective(Enum):
    """Enum class to specify what should each KPI be."""

    OBJECTIVE = 0
    PENALTY = 1


@dataclass(frozen=True)
class ObjectiveDoc:
    type: TypeObjective
    default_weight: float


class ObjectiveRegister:
    """Store all the specification concerning the objective criteria.

    To specify the objective criteria, you're invited to choose the objective_sense (ModeOptim),
    how the criteria is computed (ObjectiveHandling) and how are defined each KPI that are returned by the problem.evaluate() function.

    Even though the dict_objective is not strictly typed, it should contain as key the same key as the
    problem.evaluate(sol) function as you see in the examples. As value the dictionnary contains a type of the
    corresponding KPI (one TypeObjective value), and a default weight of the KPI to take into account (for SINGLE,
    and AGGREGATE ObjectiveHandling). The weight should be coherent with the ModeOptim chosen.

    Examples:
        In ColoringProblem implementation.
        dict_objective = {
            "nb_colors": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=-1),
            "nb_violations": ObjectiveDoc(type=TypeObjective.PENALTY, default_weight=-100),
        }

    Attributes
        objective_sense (ModeOptim): min or max problem
        objective_handling (ObjectiveHandling): specify how the different kpi are transformed into an optimization criteria
        dict_objective_to_doc: for each kpi, gives their default weight and their TypeObjective.


    """

    objective_sense: ModeOptim
    objective_handling: ObjectiveHandling
    dict_objective_to_doc: dict[str, ObjectiveDoc]

    def __init__(
        self,
        objective_sense: ModeOptim,
        objective_handling: ObjectiveHandling,
        dict_objective_to_doc: dict[str, ObjectiveDoc],
    ):
        self.objective_sense = objective_sense
        self.objective_handling = objective_handling
        self.dict_objective_to_doc = dict_objective_to_doc

    def get_objective_names(self) -> list[str]:
        return sorted(self.dict_objective_to_doc)

    def get_list_objective_and_default_weight(self) -> tuple[list[str], list[float]]:
        """Flatten the list of kpi names and default weight.

        Returns: list of kpi names, list of default weight for the aggregated objective function.
        """
        d = [
            (k, self.dict_objective_to_doc[k].default_weight)
            for k in self.get_objective_names()
        ]
        return [s[0] for s in d], [s[1] for s in d]

    def __str__(self) -> str:
        s = "Objective Register :\n"
        s += "Obj sense : " + str(self.objective_sense) + "\n"
        s += "Obj handling : " + str(self.objective_handling) + "\n"
        s += "detail : " + str(self.dict_objective_to_doc)
        return s


class Solution(ABC):
    """Base class for a solution to a Problem."""

    def __init__(self, problem: Problem):
        self.problem = problem

    @abstractmethod
    def copy(self) -> Solution:
        """Deep copy of the solution.

        The copy() function should return a new object containing the same input as the current object,
        that respects the following expected behaviour:
        -y = x.copy()
        -if do some inplace change of y, the changes are not done in x.

        Returns: a new object from which you can manipulate attributes without changing the original object.
        """
        ...

    def lazy_copy(self) -> Solution:
        """This function should return a new object but possibly with mutable attributes from the original objects.

        A typical use of lazy copy is in evolutionary algorithms or genetic algorithm where the use of
        local move don't need to do a possibly costly deepcopy.

        Returns (Solution): copy (possibly shallow) of the Solution

        """
        return self.copy()

    def change_problem(self, new_problem: Problem) -> None:
        """If relevant to the optimisation problem, change the underlying problem instance for the solution.

        This method can be used to evaluate a solution for different instance of problems.
        It should be implemented in child classes when caching subresults depending on the problem.

        Args:
            new_problem (Problem): another problem instance from which the solution can be evaluated

        Returns: None

        """
        self.problem = new_problem


class Problem(ABC):
    """Base class for a discrete optimization problem."""

    @abstractmethod
    def evaluate(self, variable: Solution) -> dict[str, float]:
        """Evaluate a given solution object for the given problem.

        This method should return a dictionnary of KPI, that can be then used for mono or multiobjective optimization.

        Args:
            variable (Solution): the Solution object to evaluate.

        Returns: dictionnary of float kpi for the solution.

        """
        ...

    def evaluate_mobj(self, variable: Solution) -> TupleFitness:
        """Default implementation of multiobjective evaluation.

        It consists in flattening the evaluate() function and put in an array.
        User should probably custom this to be more efficient.

        Args:
            variable (Solution): the Solution object to evaluate.

        Returns (TupleFitness): a flattened tuple fitness object representing the multi-objective criteria.

        """
        keys = self.get_objective_names()
        dict_values = self.evaluate(variable)
        return TupleFitness(np.array([dict_values[k] for k in keys]), len(keys))

    def evaluate_mobj_from_dict(self, dict_values: dict[str, float]) -> TupleFitness:
        """Return an multiobjective fitness from a dictionnary of kpi (output of evaluate function).

        It consists in flattening the evaluate() function and put in an array.
        User should probably custom this to be more efficient.

        Args:
            dict_values: output of evaluate() function

        Returns (TupleFitness): a flattened tuple fitness object representing the multi-objective criteria.

        """
        # output of evaluate(solution) typically
        keys = self.get_objective_names()
        return TupleFitness(np.array([dict_values[k] for k in keys]), len(keys))

    def evaluate_from_encoding(
        self, int_vector: list[int], encoding_name: str
    ) -> dict[str, float]:
        """Evaluate a solution represented by its encoding.

        Necessary to apply a genetic algorithm.

        Args:
            int_vector: solution attribute seen as a list of integer
            encoding_name: name of the solution attribute used.
                Should be an entry of `self.get_attribute_register()`.

        Returns:

        """
        return self.evaluate(
            self.build_solution_from_encoding(int_vector, encoding_name)
        )

    def build_solution_from_encoding(
        self, int_vector: list[int], encoding_name: str
    ) -> Solution:
        """Build a solution from its encoding.

        Used by genetic algorithms.

        """
        return self.get_solution_type()(problem=self, **{encoding_name: int_vector})

    def set_fixed_attributes(self, attribute_name: str, solution: Solution) -> None:
        """Fix some solution attribute.

        Useful when applying successively GA on different attribute of the solution, fixing the others.

        Should be implemented at least for attributes described by attribute_register.

        Args:
            attribute_name: an attribute name
            solution:

        Returns:

        """
        raise NotImplementedError()

    @abstractmethod
    def satisfy(self, variable: Solution) -> bool:
        """Computes if a solution satisfies or not the constraints of the problem.

        Args:
            variable: the Solution object to check satisfability

        Returns (bool): boolean true if the constraints are fulfilled, false elsewhere.

        """
        ...

    def get_attribute_register(self) -> EncodingRegister:
        """Returns how the Solution should be encoded.

        Useful to find automatically available mutations for local search.
        Used by genetic algorithms Ga and Nsga.

        This needs only to be implemented in child classes when GA or LS solvers are to be used.

        Returns (EncodingRegister): content of the encoding of the solution
        """
        raise NotImplementedError()

    @abstractmethod
    def get_solution_type(self) -> type[Solution]:
        """Returns the class implementation of a Solution.

        Returns (class): class object of the given Problem.
        """
        ...

    @abstractmethod
    def get_objective_register(self) -> ObjectiveRegister:
        """Returns the objective definition.

        Returns (ObjectiveRegister): object defining the objective criteria.

        """
        ...

    def get_optuna_study_direction(self) -> str:
        """Convert the objective sense into the expected string by Optuna."""
        objective_register = self.get_objective_register()
        if objective_register.objective_sense == ModeOptim.MINIMIZATION:
            direction = "minimize"
        else:
            direction = "maximize"
        return direction

    def get_objective_names(self) -> list[str]:
        return self.get_objective_register().get_objective_names()

    def get_dummy_solution(self) -> Solution:
        """Create a trivial solution for the problem.

        Should satisfy the problem ideally.
        Does not exist for all kind of problems.

        """
        raise NotImplementedError()


class BaseMethodAggregating(Enum):
    """Enum class used to specify how an evaluation of a multiscenario problem should be aggregated."""

    MEAN = 0
    """averaging over scenarios"""
    MEDIAN = 1
    """taking the median over scenarios"""
    PERCENTILE = 2
    """take a given percentile over scenario (the percentile value is given as additional parameter
    in MethodAggregating object"""
    PONDERATION = 3
    """ponderate the different scenario with different weights.
    (MEAN would be equivalent with equal ponderation for example) """
    MIN = 4
    """Take the min value over the scenarios"""
    MAX = 5
    """Take the max value over the scenarios"""


class MethodAggregating:
    """Specifies how the evaluation on a RobustProblem (i.e a multiscenario problem) should be aggregated in an objective criteria.

    Attributes:
        base_method_aggregating (BaseMethodAggregating): the base method for aggregation of evaluation
        percentile (float): if base_method_aggregating==BaseMethodAggregating.PERCENTILE, then the percentile value used will be this one.
        ponderation (np.array): if base_method_aggregating==BaseMethodAggregating.PONDERATION, then the ponderation value used will be this one.
         It should be the same size as the number of scenario in the RobustProblem
    """

    def __init__(
        self,
        base_method_aggregating: BaseMethodAggregating,
        percentile: float = 90.0,
        ponderation: Optional[np.ndarray] = None,
    ):
        self.base_method_aggregating = base_method_aggregating
        self.percentile = percentile
        self.ponderation = ponderation


class RobustProblem(Problem):
    """Problem built from a list of other problem (that should be considered as "scenario" optimisation problems).

    Attributes:
        list_problem: List of Problems corresponding to different scenarios.
        method_aggregating: specifies how the evaluation on each scenario should be merged

    """

    def __init__(
        self, list_problem: Sequence[Problem], method_aggregating: MethodAggregating
    ):
        self.list_problem = list_problem
        self.method_aggregating = method_aggregating
        self.nb_problem = len(self.list_problem)
        self.agg_vec = self.aggregate_vector()

    def aggregate_vector(self) -> Callable[[npt.ArrayLike], float]:
        """Returns the aggregation function coherent with the method_aggregating attribute.

        Returns: aggregation function

        """
        func: Callable[[npt.ArrayLike], float]
        if (
            self.method_aggregating.base_method_aggregating
            == BaseMethodAggregating.MEAN
        ):
            func = np.mean
        elif (
            self.method_aggregating.base_method_aggregating
            == BaseMethodAggregating.MEDIAN
        ):
            func = np.median  # type: ignore
        elif (
            self.method_aggregating.base_method_aggregating
            == BaseMethodAggregating.PERCENTILE
        ):

            def func(x: npt.ArrayLike) -> float:
                return np.percentile(x, q=[self.method_aggregating.percentile])[0]  # type: ignore

        elif (
            self.method_aggregating.base_method_aggregating
            == BaseMethodAggregating.PONDERATION
        ):

            def func(x: npt.ArrayLike) -> float:
                return np.dot(x, self.method_aggregating.ponderation)  # type: ignore

        elif (
            self.method_aggregating.base_method_aggregating == BaseMethodAggregating.MIN
        ):
            func = np.min
        elif (
            self.method_aggregating.base_method_aggregating == BaseMethodAggregating.MAX
        ):
            func = np.max
        else:
            raise ValueError(
                f"Unknown aggregating method {self.method_aggregating.base_method_aggregating}"
            )

        return func

    def evaluate(self, variable: Solution) -> dict[str, float]:
        """Aggregated evaluate function.

        Args:
            variable (Solution): Solution to evaluate on the different scenarios.

        Returns (dict[str,float]): aggregated kpi on different scenarios.

        """
        fits = [self.list_problem[i].evaluate(variable) for i in range(self.nb_problem)]
        keys = fits[0].keys()
        aggreg = {}
        for k in keys:
            vals = np.array([fit[k] for fit in fits])
            aggreg[k] = self.agg_vec(vals)
        return aggreg

    def satisfy(self, variable: Solution) -> bool:
        """Computes if a solution satisfies or not the constraints of the problem.

        Warnings:
            For RobustProblem, we consider than checking the satisfiability on the first scenario is enough.
            It is not necessarly correct

        Args:
            variable: the Solution object to check satisfability

        Returns (bool): boolean true if the constraints are fulfilled, false elsewhere.

        """
        return self.list_problem[0].satisfy(variable)

    def get_attribute_register(self) -> EncodingRegister:
        """See ```Problem.get_attribute_register``` doc."""
        return self.list_problem[0].get_attribute_register()

    def get_solution_type(self) -> type[Solution]:
        """See ```Problem.get_solution_type``` doc."""
        return self.list_problem[0].get_solution_type()

    def get_objective_register(self) -> ObjectiveRegister:
        """See ```Problem.get_objective_register``` doc."""
        return self.list_problem[0].get_objective_register()


class ParamsObjectiveFunction:
    """Alternative of Objective Register, but with the same idea of storing the objective handling, ponderation and sense of optimization.

    This class has been implemented after ObjectiveRegister to be able to call solvers and use user choice optimization.
    """

    objective_handling: ObjectiveHandling
    objectives: list[str]
    weights: list[float]
    sense_function: ModeOptim

    def __init__(
        self,
        objective_handling: ObjectiveHandling,
        objectives: list[str],
        weights: list[float],
        sense_function: ModeOptim,
    ):
        self.objective_handling = objective_handling
        self.objectives = objectives
        self.weights = weights
        self.sense_function = sense_function

    def __str__(self) -> str:
        s = "Params objective function :  \n"
        s += "Sense : " + str(self.sense_function) + "\n"
        s += "Objective handling " + str(self.objective_handling) + "\n"
        s += "Objectives " + str(self.objectives) + "\n"
        s += "weights : " + str(self.weights)
        return s


def get_default_objective_setup(problem: Problem) -> ParamsObjectiveFunction:
    """Build ParamsObjectiveFunction from the ObjectiveRegister returned by the problem.

    Args:
        problem (Problem): problem to build objective setup

    Returns: default ParamsObjectiveFunction of the problem.

    """
    register_objective = problem.get_objective_register()
    objs, weights = register_objective.get_list_objective_and_default_weight()
    sense = register_objective.objective_sense
    logger.debug((sense, register_objective.objective_handling, objs, weights))
    return ParamsObjectiveFunction(
        objective_handling=register_objective.objective_handling,
        objectives=objs,
        weights=weights,
        sense_function=sense,
    )


def build_aggreg_function_and_params_objective(
    problem: Problem,
    params_objective_function: Optional[ParamsObjectiveFunction] = None,
) -> tuple[
    Union[
        Callable[[Solution], float],
        Callable[[Solution], TupleFitness],
    ],
    Union[
        Callable[[dict[str, float]], float],
        Callable[[dict[str, float]], TupleFitness],
    ],
    ParamsObjectiveFunction,
]:
    """Build evaluation function from the problem and the params of objective function.

    If params_objective_function is None then we compute inside this function the default ParamsObjectiveFunction.

    Args:
        problem: problem to build evaluation function from
        params_objective_function: params of the objective function.

    Returns: the function returns a 3-uple :
                    -first element is a function of Solution->Union[float,TupleFitness]
                    -second element is a function of (dict[str,float])->Union[float, TupleFitness]
                    -third element, return the params_objective_function (either the object passed in argument of the
                    function, or the one created inside the function.)

    """
    if params_objective_function is None:
        params_objective_function = get_default_objective_setup(problem)
    eval_sol, eval_dict = build_evaluate_function_aggregated(
        problem=problem, params_objective_function=params_objective_function
    )
    return eval_sol, eval_dict, params_objective_function


def build_evaluate_function_aggregated(
    problem: Problem,
    params_objective_function: Optional[ParamsObjectiveFunction] = None,
) -> Union[
    tuple[Callable[[Solution], float], Callable[[dict[str, float]], float]],
    tuple[
        Callable[[Solution], TupleFitness], Callable[[dict[str, float]], TupleFitness]
    ],
]:
    """Build 2 eval functions based from the problem and params of objective function.

    The 2 eval function are callable with a Solution for the first one, and a dict[str, float]
    (output of Problem.evaluate function) for the second one. Those two eval function will return either a scalar for
    monoobjective problem or a TupleFitness for multiobjective.
    those aggregated function will be the one actually called by an optimisation algorithm at the end.

    Args:
        problem (Problem): problem to build the evaluation function s
        params_objective_function (ParamsObjectiveFunction): params of the objective function.

    Returns: the function returns a 2-uple :
                    -first element is a function of Solution->Union[float,TupleFitness]
                    -second element is a function of (dict[str,float])->Union[float, TupleFitness]
    """
    if params_objective_function is None:
        params_objective_function = get_default_objective_setup(problem)
    sign = 1
    objectives = params_objective_function.objectives
    weights = params_objective_function.weights
    objective_handling = params_objective_function.objective_handling
    if objective_handling == ObjectiveHandling.AGGREGATE:
        length = len(objectives)

        def eval_sol(solution: Solution) -> float:
            dict_values = problem.evaluate(solution)
            val = sum([dict_values[objectives[i]] * weights[i] for i in range(length)])
            return sign * val

        def eval_from_dict_values(dict_values: dict[str, float]) -> float:
            val = sum([dict_values[objectives[i]] * weights[i] for i in range(length)])
            return sign * val

        return eval_sol, eval_from_dict_values

    elif objective_handling == ObjectiveHandling.SINGLE:
        length = len(objectives)

        def eval_sol(solution: Solution) -> float:
            dict_values = problem.evaluate(solution)
            return sign * dict_values[objectives[0]] * weights[0]

        def eval_from_dict_values(dict_values: dict[str, float]) -> float:
            return sign * dict_values[objectives[0]] * weights[0]

        return eval_sol, eval_from_dict_values

    else:  # objective_handling == ObjectiveHandling.MULTI_OBJ:
        length = len(objectives)

        def eval_sol2(solution: Solution) -> TupleFitness:
            d = problem.evaluate(solution)
            return (
                TupleFitness(
                    np.array([weights[i] * d[objectives[i]] for i in range(length)]),
                    length,
                )
                * sign
            )

        def eval_from_dict_values2(dict_values: dict[str, float]) -> TupleFitness:
            return (
                TupleFitness(
                    np.array(
                        [weights[i] * dict_values[objectives[i]] for i in range(length)]
                    ),
                    length,
                )
                * sign
            )

        return eval_sol2, eval_from_dict_values2
