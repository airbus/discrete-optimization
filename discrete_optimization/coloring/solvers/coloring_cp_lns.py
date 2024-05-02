"""Easy Large neighborhood search solver for coloring. """
#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import List, Optional

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
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import MinizincCPSolver, ParametersCP
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    SubBrickHyperparameter,
    SubBrickKwargsHyperparameter,
)
from discrete_optimization.generic_tools.lns_cp import (
    LNS_CP,
    ConstraintHandler,
    TrivialPostProcessSolution,
)
from discrete_optimization.generic_tools.lns_mip import (
    InitialSolution,
    PostProcessSolution,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


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
        cp_solver_cls=dict(choices=[ColoringCP]),
        initial_solution_provider_cls=dict(choices=[InitialColoring]),
        constraint_handler_cls=dict(choices=[ConstraintHandlerFixColorsCP]),
        post_process_solution_cls=dict(
            choices=[TrivialPostProcessSolution, PostProcessSolutionColoring]
        ),
    )

    def __init__(
        self,
        problem: ColoringProblem,
        cp_solver: Optional[MinizincCPSolver] = None,
        initial_solution_provider: Optional[InitialSolution] = None,
        constraint_handler: Optional[ConstraintHandler] = None,
        post_process_solution: Optional[PostProcessSolution] = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        SolverColoring.__init__(
            self, problem=problem, params_objective_function=params_objective_function
        )
        kwargs = self.complete_with_default_hyperparameters(kwargs)

        if cp_solver is None:
            if kwargs["cp_solver_kwargs"] is None:
                cp_solver_kwargs = kwargs
            else:
                cp_solver_kwargs = kwargs["cp_solver_kwargs"]
            if kwargs["cp_solver_cls"] is None:
                cp_solver = build_default_cp_model(
                    coloring_model=self.problem, **cp_solver_kwargs
                )
            else:
                cp_solver_cls = kwargs["cp_solver_cls"]
                cp_solver = cp_solver_cls(problem=self.problem, **cp_solver_kwargs)
                cp_solver.init_model(**cp_solver_kwargs)
        self.cp_solver = cp_solver

        if constraint_handler is None:
            if kwargs["constraint_handler_kwargs"] is None:
                constraint_handler_kwargs = kwargs
            else:
                constraint_handler_kwargs = kwargs["constraint_handler_kwargs"]
            if kwargs["constraint_handler_cls"] is None:
                constraint_handler = build_default_constraint_handler(
                    coloring_model=self.problem, **constraint_handler_kwargs
                )
            else:
                constraint_handler_cls = kwargs["constraint_handler_cls"]
                constraint_handler = constraint_handler_cls(
                    problem=self.problem, **constraint_handler_kwargs
                )
        self.constraint_handler = constraint_handler

        if post_process_solution is None:
            if kwargs["post_process_solution_kwargs"] is None:
                post_process_solution_kwargs = kwargs
            else:
                post_process_solution_kwargs = kwargs["post_process_solution_kwargs"]
            if kwargs["post_process_solution_cls"] is None:
                post_process_solution = build_default_postprocess(
                    coloring_model=self.problem,
                    params_objective_function=self.params_objective_function,
                )
            else:
                post_process_solution_cls = kwargs["post_process_solution_cls"]
                post_process_solution = post_process_solution_cls(
                    problem=self.problem,
                    params_objective_function=self.params_objective_function,
                    **post_process_solution_kwargs
                )
        self.post_process_solution = post_process_solution

        if initial_solution_provider is None:
            if kwargs["initial_solution_provider_kwargs"] is None:
                initial_solution_provider_kwargs = kwargs
            else:
                initial_solution_provider_kwargs = kwargs[
                    "initial_solution_provider_kwargs"
                ]
            if kwargs["initial_solution_provider_cls"] is None:
                initial_solution_provider = build_default_initial_solution(
                    coloring_model=self.problem,
                    params_objective_function=self.params_objective_function,
                )
            else:
                initial_solution_provider_cls = kwargs["initial_solution_provider_cls"]
                initial_solution_provider = initial_solution_provider_cls(
                    problem=self.problem,
                    params_objective_function=self.params_objective_function,
                    **initial_solution_provider_kwargs
                )
        self.initial_solution_provider = initial_solution_provider
