"""Base module for the problem implementation in discrete-optimization library."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt

from discrete_optimization.generic_tools.result_storage.multiobj_utils import (
    TupleFitness,
)

logger = logging.getLogger(__name__)


class TypeAttribute(Enum):
    """Enum class to specify how are defined the attributes of a Solution.

    This specification will be particularly usefull if you want to give a try to local search algorithms, which will use
    the information to use the right local moves.
    """

    LIST_INTEGER = 0
    LIST_BOOLEAN = 1
    PERMUTATION = 2
    PERMUTATION_TSP = 3
    PERMUTATION_RCPSP = 4
    SET_INTEGER = 5
    LIST_BOOLEAN_KNAP = 6
    LIST_INTEGER_SPECIFIC_ARITY = 7
    SET_TUPLE_INTEGER = 8
    VRP_PATHS = 9
    LIST_FLOATS = 10


class ModeOptim(Enum):
    """Enum class to specify minimization or maximization problems."""

    MAXIMIZATION = 0
    MINIMIZATION = 1


class EncodingRegister:
    """Placeholder class where the Solution definition is defined.

    Attributes:
        dict_attribute_to_type (Dict[str, Any]): specifies the encoding of a solution object.
        User may refer to example in the different implemented problem definition.

    Examples:
        in ColoringModel, to specify the colors attribute of the Solution, you will do the following.
        dict_register = {
            "colors": {
                "name": "colors",
                "type": [TypeAttribute.LIST_INTEGER],
                "n": self.number_of_nodes,
                "arrity": self.number_of_nodes,
            }
        }
    """

    dict_attribute_to_type: Dict[str, Any]

    def __init__(self, dict_attribute_to_type: Dict[str, Any]):
        self.dict_attribute_to_type = dict_attribute_to_type

    def get_types(self) -> List[TypeAttribute]:
        """Returns all the TypeAttribute that are present in our encoding."""
        return [
            t
            for k in self.dict_attribute_to_type
            for t in self.dict_attribute_to_type[k]["type"]
        ]

    def __str__(self) -> str:
        return "Encoding : " + str(self.dict_attribute_to_type)

    def lower_bound_vector_encoding(self, encoding_name: str) -> List[int]:
        """Return for an encoding that is of type LIST_INTEGER or associated, the lower bound vector.
        Examples: if the vector should contains value higher or equal to 1, the function will return a list full of 1.
        """
        dict_encoding = self.dict_attribute_to_type[encoding_name]
        return lower_bound_vector_encoding_from_dict(dict_encoding)

    def upper_bound_vector_encoding(self, encoding_name: str) -> List[int]:
        """Return for an encoding that is of type LIST_INTEGER or associated, the upper bound vector.
        Examples: if the vector should contains value higher or equal to 1, the function will return a list full of 1.
        """
        dict_encoding = self.dict_attribute_to_type[encoding_name]
        return upper_bound_vector_encoding_from_dict(dict_encoding)


def lower_bound_vector_encoding_from_dict(dict_encoding: Dict[str, Any]) -> List[int]:
    length_encoding = dict_encoding["n"]
    if "low" in dict_encoding:
        low_value: Union[int, Iterable[int]] = dict_encoding["low"]
        if isinstance(low_value, int):
            return [low_value for i in range(length_encoding)]
        else:
            return list(low_value)
    else:
        return [0 for i in range(length_encoding)]  # By default we start at zero.


def upper_bound_vector_encoding_from_dict(dict_encoding: Dict[str, Any]) -> List[int]:
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
    dict_objective_to_doc: Dict[str, ObjectiveDoc]

    def __init__(
        self,
        objective_sense: ModeOptim,
        objective_handling: ObjectiveHandling,
        dict_objective_to_doc: Dict[str, Any],
    ):
        self.objective_sense = objective_sense
        self.objective_handling = objective_handling
        self.dict_objective_to_doc = dict_objective_to_doc

    def get_list_objective_and_default_weight(self) -> Tuple[List[str], List[float]]:
        """Flatten the list of kpi names and default weight.

        Returns: list of kpi names, list of default weight for the aggregated objective function.
        """
        d = [
            (k, self.dict_objective_to_doc[k].default_weight)
            for k in self.dict_objective_to_doc
        ]
        return [s[0] for s in d], [s[1] for s in d]

    def __str__(self) -> str:
        s = "Objective Register :\n"
        s += "Obj sense : " + str(self.objective_sense) + "\n"
        s += "Obj handling : " + str(self.objective_handling) + "\n"
        s += "detail : " + str(self.dict_objective_to_doc)
        return s


class Solution:
    """Base class for a solution to a Problem."""

    @abstractmethod
    def copy(self) -> "Solution":
        """Deep copy of the solution.

        The copy() function should return a new object containing the same input as the current object,
        that respects the following expected behaviour:
        -y = x.copy()
        -if do some inplace change of y, the changes are not done in x.

        Returns: a new object from which you can manipulate attributes without changing the original object.
        """
        ...

    def lazy_copy(self) -> "Solution":
        """This function should return a new object but possibly with mutable attributes from the original objects.

        A typical use of lazy copy is in evolutionary algorithms or genetic algorithm where the use of
        local move don't need to do a possibly costly deepcopy.

        Returns (Solution): copy (possibly shallow) of the Solution

        """
        return self.copy()

    def get_attribute_register(self, problem: "Problem") -> EncodingRegister:
        """Returns how the Solution is encoded for the Problem.

        By default it returns the encoding register of the problem itself. However it can make sense that for the same
        Problem, you have different Solution class with different encoding.

        Returns (EncodingRegister): content of the encoding of the Solution.
        """
        return problem.get_attribute_register()

    @abstractmethod
    def change_problem(self, new_problem: "Problem") -> None:
        """If relevant to the optimisation problem, change the underlying problem instance for the solution.

        This method can be used to evaluate a solution for different instance of problems.

        Args:
            new_problem (Problem): another problem instance from which the solution can be evaluated

        Returns: None

        """
        ...


class Problem:
    """Base class for a discrete optimization problem."""

    @abstractmethod
    def evaluate(self, variable: Solution) -> Dict[str, float]:
        """Evaluate a given solution object for the given problem.

        This method should return a dictionnary of KPI, that can be then used for mono or multiobjective optimization.

        Args:
            variable (Solution): the Solution object to evaluate.

        Returns: Dictionnary of float kpi for the solution.

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
        obj_register = self.get_objective_register()
        keys = sorted(obj_register.dict_objective_to_doc.keys())
        dict_values = self.evaluate(variable)
        return TupleFitness(np.array([dict_values[k] for k in keys]), len(keys))

    def evaluate_mobj_from_dict(self, dict_values: Dict[str, float]) -> TupleFitness:
        """Return an multiobjective fitness from a dictionnary of kpi (output of evaluate function).

        It consists in flattening the evaluate() function and put in an array.
        User should probably custom this to be more efficient.

        Args:
            dict_values: output of evaluate() function

        Returns (TupleFitness): a flattened tuple fitness object representing the multi-objective criteria.

        """
        # output of evaluate(solution) typically
        keys = sorted(self.get_objective_register().dict_objective_to_doc.keys())
        return TupleFitness(np.array([dict_values[k] for k in keys]), len(keys))

    @abstractmethod
    def satisfy(self, variable: Solution) -> bool:
        """Computes if a solution satisfies or not the constraints of the problem.

        Args:
            variable: the Solution object to check satisfability

        Returns (bool): boolean true if the constraints are fulfilled, false elsewhere.

        """
        ...

    @abstractmethod
    def get_attribute_register(self) -> EncodingRegister:
        """Returns how the Solution should be encoded.

        Returns (EncodingRegister): content of the encoding of the solution
        """
        ...

    @abstractmethod
    def get_solution_type(self) -> Type[Solution]:
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

    def evaluate(self, variable: Solution) -> Dict[str, float]:
        """Aggregated evaluate function.

        Args:
            variable (Solution): Solution to evaluate on the different scenarios.

        Returns (Dict[str,float]): aggregated kpi on different scenarios.

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

    def get_solution_type(self) -> Type[Solution]:
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
    objectives: List[str]
    weights: List[float]
    sense_function: ModeOptim

    def __init__(
        self,
        objective_handling: ObjectiveHandling,
        objectives: List[str],
        weights: List[float],
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
) -> Tuple[
    Union[
        Callable[[Solution], float],
        Callable[[Solution], TupleFitness],
    ],
    Union[
        Callable[[Dict[str, float]], float],
        Callable[[Dict[str, float]], TupleFitness],
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
                    -second element is a function of (Dict[str,float])->Union[float, TupleFitness]
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
    Tuple[Callable[[Solution], float], Callable[[Dict[str, float]], float]],
    Tuple[
        Callable[[Solution], TupleFitness], Callable[[Dict[str, float]], TupleFitness]
    ],
]:
    """Build 2 eval functions based from the problem and params of objective function.

    The 2 eval function are callable with a Solution for the first one, and a Dict[str, float]
    (output of Problem.evaluate function) for the second one. Those two eval function will return either a scalar for
    monoobjective problem or a TupleFitness for multiobjective.
    those aggregated function will be the one actually called by an optimisation algorithm at the end.

    Args:
        problem (Problem): problem to build the evaluation function s
        params_objective_function (ParamsObjectiveFunction): params of the objective function.

    Returns: the function returns a 2-uple :
                    -first element is a function of Solution->Union[float,TupleFitness]
                    -second element is a function of (Dict[str,float])->Union[float, TupleFitness]
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

        def eval_from_dict_values(dict_values: Dict[str, float]) -> float:
            val = sum([dict_values[objectives[i]] * weights[i] for i in range(length)])
            return sign * val

        return eval_sol, eval_from_dict_values

    elif objective_handling == ObjectiveHandling.SINGLE:
        length = len(objectives)

        def eval_sol(solution: Solution) -> float:
            dict_values = problem.evaluate(solution)
            return sign * dict_values[objectives[0]] * weights[0]

        def eval_from_dict_values(dict_values: Dict[str, float]) -> float:
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

        def eval_from_dict_values2(dict_values: Dict[str, float]) -> TupleFitness:
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
