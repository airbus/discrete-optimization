"""Easy Large neighborhood search solver for coloring. """
#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Optional

from discrete_optimization.coloring.coloring_model import ColoringProblem
from discrete_optimization.coloring.solvers.coloring_cp_lns_solvers import (
    ConstraintHandlerFixColorsCP,
    InitialColoring,
    InitialColoringMethod,
    PostProcessSolutionColoring,
)
from discrete_optimization.coloring.solvers.coloring_cp_solvers import (
    ColoringCP,
    ColoringCPModel,
)
from discrete_optimization.coloring.solvers.coloring_solver import SolverColoring
from discrete_optimization.generic_tools.cp_tools import MinizincCPSolver
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.lns_cp import LNS_CP, MznConstraintHandler
from discrete_optimization.generic_tools.lns_mip import (
    InitialSolution,
    PostProcessSolution,
)
from discrete_optimization.generic_tools.lns_tools import TrivialPostProcessSolution


def build_default_cp_model(coloring_model: ColoringProblem, **kwargs):
    cp_model = ColoringCP(problem=coloring_model, **kwargs)
    if coloring_model.use_subset:
        cp_model.init_model(
            cp_model=ColoringCPModel.DEFAULT_WITH_SUBSET, object_output=True
        )
    else:
        cp_model.init_model(cp_model=ColoringCPModel.DEFAULT, object_output=True)
    return cp_model


def build_default_constraint_handler(coloring_model: ColoringProblem, **kwargs):
    return ConstraintHandlerFixColorsCP(
        problem=coloring_model, fraction_to_fix=kwargs.get("fraction_to_fix", 0.9)
    )


def build_default_postprocess(
    coloring_model: ColoringProblem,
    params_objective_function: Optional[ParamsObjectiveFunction] = None,
    **kwargs,
):
    if coloring_model.has_constraints_coloring:
        return TrivialPostProcessSolution()
    else:
        return PostProcessSolutionColoring(
            problem=coloring_model, params_objective_function=params_objective_function
        )


def build_default_initial_solution(
    coloring_model: ColoringProblem,
    params_objective_function: Optional[ParamsObjectiveFunction] = None,
    **kwargs,
):
    return InitialColoring(
        problem=coloring_model,
        initial_method=InitialColoringMethod.GREEDY,
        params_objective_function=params_objective_function,
    )


class LnsCpColoring(LNS_CP, SolverColoring):
    """
    Most easy way to use LNS-CP for coloring with some default parameters for constraint handler.
    """

    hyperparameters = LNS_CP.copy_and_update_hyperparameters(
        subsolver=dict(choices=[ColoringCP]),
        initial_solution_provider=dict(choices=[InitialColoring]),
        constraint_handler=dict(choices=[ConstraintHandlerFixColorsCP]),
        post_process_solution=dict(
            choices=[TrivialPostProcessSolution, PostProcessSolutionColoring]
        ),
    )

    def __init__(
        self,
        problem: ColoringProblem,
        subsolver: Optional[MinizincCPSolver] = None,
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
