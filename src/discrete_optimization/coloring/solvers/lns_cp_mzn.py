#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from typing import Any, Iterable, Optional

from minizinc import Instance

from discrete_optimization.coloring.problem import ColoringProblem, ColoringSolution
from discrete_optimization.coloring.solvers import ColoringSolver
from discrete_optimization.coloring.solvers.cp_mzn import (
    CpColoringModel,
    CpColoringSolver,
)
from discrete_optimization.coloring.solvers.lns_cp import (
    InitialColoring,
    InitialColoringMethod,
    PostProcessSolutionColoring,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
)
from discrete_optimization.generic_tools.lns_cp_mzn import (
    LnsCpMzn,
    MznConstraintHandler,
)
from discrete_optimization.generic_tools.lns_tools import (
    InitialSolution,
    PostProcessSolution,
    TrivialPostProcessSolution,
)
from discrete_optimization.generic_tools.mzn_tools import MinizincCpSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class FixColorsMznConstraintHandler(MznConstraintHandler):
    """Constraint builder for LNS coloring problem.

    This constraint handler is pretty basic, it fixes a fraction_to_fix proportion of nodes color.

    Attributes:
        problem (ColoringProblem): input coloring problem
        fraction_to_fix (float): float between 0 and 1, representing the proportion of nodes to constrain.
    """

    hyperparameters = [
        FloatHyperparameter("fraction_to_fix", low=0.0, high=1.0, default=0.9),
    ]

    def __init__(self, problem: ColoringProblem, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix

    def adding_constraint_from_results_store(
        self,
        solver: MinizincCpSolver,
        result_storage: ResultStorage,
        result_storage_last_iteration: ResultStorage,
        child_instance: Instance,
        **kwargs: Any,
    ) -> Iterable[Any]:
        """Include constraint that fix decision on a subset of nodes, according to current solutions found.

        Args:
            solver: a coloring CpSolver
            result_storage: all results so far
            result_storage_last_iteration: results from last LNS iteration only
            child_instance: minizinc instance where to include the constraints
            **kwargs:

        Returns: an empty list, unused.

        """
        range_node = range(1, self.problem.number_of_nodes + 1)
        current_solution = self.extract_best_solution_from_last_iteration(
            result_storage=result_storage,
            result_storage_last_iteration=result_storage_last_iteration,
        )
        if current_solution is None:
            raise ValueError("result_storage.get_best_solution() should not be None.")
        if not isinstance(current_solution, ColoringSolution):
            raise ValueError(
                "result_storage.get_best_solution() should be a ColoringSolution."
            )
        if current_solution.colors is None:
            raise ValueError(
                "result_storage.get_best_solution().colors should not be None."
            )
        subpart_color = set(
            random.sample(
                range_node, int(self.fraction_to_fix * self.problem.number_of_nodes)
            )
        )
        dict_color = {
            i + 1: current_solution.colors[i] + 1
            for i in range(self.problem.number_of_nodes)
        }
        current_nb_color = max(dict_color.values())
        list_strings = []
        for i in range_node:
            if i in subpart_color and dict_color[i] < current_nb_color:
                str1 = (
                    "constraint color_graph["
                    + str(i)
                    + "] == "
                    + str(dict_color[i])
                    + ";\n"
                )
                child_instance.add_string(str1)
                list_strings.append(str1)
            str1 = (
                "constraint color_graph["
                + str(i)
                + "] <= "
                + str(current_nb_color)
                + ";\n"
            )
            child_instance.add_string(str1)
            list_strings.append(str1)
        return list_strings


class LnsCpColoringSolver(LnsCpMzn, ColoringSolver):
    """
    Most easy way to use LNS-CP for coloring with some default parameters for constraint handler.
    """

    hyperparameters = LnsCpMzn.copy_and_update_hyperparameters(
        subsolver=dict(choices=[CpColoringSolver]),
        initial_solution_provider=dict(choices=[InitialColoring]),
        constraint_handler=dict(choices=[FixColorsMznConstraintHandler]),
        post_process_solution=dict(
            choices=[TrivialPostProcessSolution, PostProcessSolutionColoring]
        ),
    )

    def __init__(
        self,
        problem: ColoringProblem,
        subsolver: Optional[MinizincCpSolver] = None,
        initial_solution_provider: Optional[InitialSolution] = None,
        constraint_handler: Optional[MznConstraintHandler] = None,
        post_process_solution: Optional[PostProcessSolution] = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem,
            subsolver=subsolver,
            initial_solution_provider=initial_solution_provider,
            constraint_handler=constraint_handler,
            post_process_solution=post_process_solution,
            params_objective_function=params_objective_function,
            build_default_initial_solution_provider=build_default_initial_solution,
            build_default_contraint_handler=build_default_constraint_handler,
            build_default_post_process_solution=build_default_postprocess,
            build_default_subsolver=build_default_cp_model,
            **kwargs,
        )


def build_default_cp_model(coloring_problem: ColoringProblem, **kwargs):
    cp_model = CpColoringSolver(problem=coloring_problem, **kwargs)
    if coloring_problem.use_subset:
        cp_model.init_model(
            cp_model=CpColoringModel.DEFAULT_WITH_SUBSET, object_output=True
        )
    else:
        cp_model.init_model(cp_model=CpColoringModel.DEFAULT, object_output=True)
    return cp_model


def build_default_constraint_handler(coloring_problem: ColoringProblem, **kwargs):
    return FixColorsMznConstraintHandler(
        problem=coloring_problem, fraction_to_fix=kwargs.get("fraction_to_fix", 0.9)
    )


def build_default_postprocess(
    coloring_problem: ColoringProblem,
    params_objective_function: Optional[ParamsObjectiveFunction] = None,
    **kwargs,
):
    if coloring_problem.has_constraints_coloring:
        return TrivialPostProcessSolution()
    else:
        return PostProcessSolutionColoring(
            problem=coloring_problem,
            params_objective_function=params_objective_function,
        )


def build_default_initial_solution(
    coloring_problem: ColoringProblem,
    params_objective_function: Optional[ParamsObjectiveFunction] = None,
    **kwargs,
):
    return InitialColoring(
        problem=coloring_problem,
        initial_method=InitialColoringMethod.GREEDY,
        params_objective_function=params_objective_function,
    )
