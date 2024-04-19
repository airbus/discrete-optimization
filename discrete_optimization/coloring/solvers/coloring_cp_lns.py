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
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    SubBrickHyperparameter,
    SubBrickKwargsHyperparameter,
)
from discrete_optimization.generic_tools.lns_cp import (
    LNS_CP,
    TrivialPostProcessSolution,
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


class LnsCpColoring(SolverColoring):
    """
    Most easy way to use LNS-CP for coloring with some default parameters for constraint handler.
    """

    hyperparameters = [
        SubBrickHyperparameter("cp_solver_cls", choices=[ColoringCP], default=None),
        SubBrickKwargsHyperparameter(
            "cp_solver_kwargs", subbrick_hyperparameter="cp_solver_cls"
        ),
        SubBrickHyperparameter(
            "initial_solution_provider_cls", choices=[InitialColoring], default=None
        ),
        SubBrickKwargsHyperparameter(
            "initial_solution_provider_kwargs",
            subbrick_hyperparameter="initial_solution_provider_cls",
        ),
        SubBrickHyperparameter(
            "constraint_handler_cls",
            choices=[ConstraintHandlerFixColorsCP],
            default=None,
        ),
        SubBrickKwargsHyperparameter(
            "constraint_handler_kwargs",
            subbrick_hyperparameter="constraint_handler_cls",
        ),
        SubBrickHyperparameter(
            "post_process_solution_cls",
            choices=[TrivialPostProcessSolution, PostProcessSolutionColoring],
            default=None,
        ),
        SubBrickKwargsHyperparameter(
            "post_process_solution_kwargs",
            subbrick_hyperparameter="post_process_solution_cls",
        ),
    ]

    def __init__(
        self,
        problem: ColoringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )

        kwargs = self.complete_with_default_hyperparameters(kwargs)

        solver = kwargs.get("cp_solver", None)
        if solver is None:
            if kwargs["cp_solver_kwargs"] is None:
                cp_solver_kwargs = kwargs
            else:
                cp_solver_kwargs = kwargs["cp_solver_kwargs"]
            if kwargs["cp_solver_cls"] is None:
                solver = build_default_cp_model(
                    coloring_model=self.problem, **cp_solver_kwargs
                )
            else:
                cp_solver_cls = kwargs["cp_solver_cls"]
                solver = cp_solver_cls(problem=self.problem, **cp_solver_kwargs)
                solver.init_model(**cp_solver_kwargs)
        self.cp_solver = solver

        self.parameters_cp = kwargs.get("parameters_cp", ParametersCP.default())

        self.constraint_handler = kwargs.get("constraint_handler", None)
        if self.constraint_handler is None:
            if kwargs["constraint_handler_kwargs"] is None:
                constraint_handler_kwargs = kwargs
            else:
                constraint_handler_kwargs = kwargs["constraint_handler_kwargs"]
            if kwargs["constraint_handler_cls"] is None:
                self.constraint_handler = build_default_constraint_handler(
                    coloring_model=self.problem, **constraint_handler_kwargs
                )
            else:
                constraint_handler_cls = kwargs["constraint_handler_cls"]
                self.constraint_handler = constraint_handler_cls(
                    problem=self.problem, **constraint_handler_kwargs
                )

        self.post_pro = kwargs.get("post_process_solution", None)
        if self.post_pro is None:
            if kwargs["post_process_solution_kwargs"] is None:
                post_process_solution_kwargs = kwargs
            else:
                post_process_solution_kwargs = kwargs["post_process_solution_kwargs"]
            if kwargs["post_process_solution_cls"] is None:
                self.post_pro = build_default_postprocess(
                    coloring_model=self.problem,
                    params_objective_function=self.params_objective_function,
                )
            else:
                post_process_solution_cls = kwargs["post_process_solution_cls"]
                self.post_pro = post_process_solution_cls(
                    problem=self.problem,
                    params_objective_function=self.params_objective_function,
                    **post_process_solution_kwargs
                )

        self.initial_solution_provider = kwargs.get("initial_solution_provider", None)
        if self.initial_solution_provider is None:
            if kwargs["initial_solution_provider_kwargs"] is None:
                initial_solution_provider_kwargs = kwargs
            else:
                initial_solution_provider_kwargs = kwargs[
                    "initial_solution_provider_kwargs"
                ]
            if kwargs["initial_solution_provider_cls"] is None:
                self.initial_solution_provider = build_default_initial_solution(
                    coloring_model=self.problem,
                    params_objective_function=self.params_objective_function,
                )
            else:
                initial_solution_provider_cls = kwargs["initial_solution_provider_cls"]
                self.initial_solution_provider = initial_solution_provider_cls(
                    problem=self.problem,
                    params_objective_function=self.params_objective_function,
                    **initial_solution_provider_kwargs
                )

        self.lns_solver = LNS_CP(
            problem=self.problem,
            cp_solver=self.cp_solver,
            initial_solution_provider=self.initial_solution_provider,
            constraint_handler=self.constraint_handler,
            post_process_solution=self.post_pro,
            params_objective_function=self.params_objective_function,
        )

    def solve(
        self,
        nb_iteration_lns: int,
        parameters_cp: Optional[ParametersCP] = None,
        nb_iteration_no_improvement: Optional[int] = None,
        skip_first_iteration: bool = False,
        stop_first_iteration_if_optimal: bool = True,
        callbacks: Optional[List[Callback]] = None,
        **args
    ) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        return self.lns_solver.solve_lns(
            parameters_cp=parameters_cp,
            skip_first_iteration=skip_first_iteration,
            stop_first_iteration_if_optimal=stop_first_iteration_if_optimal,
            nb_iteration_no_improvement=nb_iteration_no_improvement,
            nb_iteration_lns=nb_iteration_lns,
            callbacks=callbacks,
        )
