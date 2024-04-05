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

    def __init__(
        self,
        problem: ColoringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )

        solver = kwargs.get("cp_solver", None)
        if solver is None:
            solver = build_default_cp_model(coloring_model=self.problem, **kwargs)
        self.cp_solver = solver
        self.parameters_cp = kwargs.get("parameters_cp", ParametersCP.default())
        self.constraint_handler = kwargs.get("constraint_handler", None)
        if self.constraint_handler is None:
            self.constraint_handler = build_default_constraint_handler(
                coloring_model=self.problem, **kwargs
            )
        self.post_pro = kwargs.get("post_process_solution", None)
        if self.post_pro is None:
            self.post_pro = build_default_postprocess(
                coloring_model=self.problem,
                params_objective_function=self.params_objective_function,
            )
        self.initial_solution_provider = kwargs.get("initial_solution_provider", None)
        if self.initial_solution_provider is None:
            self.initial_solution_provider = build_default_initial_solution(
                coloring_model=self.problem,
                params_objective_function=self.params_objective_function,
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
