#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    IntegerHyperparameter,
    SubBrick,
    SubBrickHyperparameter,
)
from discrete_optimization.multibatching.problem import MultibatchingProblem
from discrete_optimization.multibatching.solvers import MultibatchingSolver
from discrete_optimization.multibatching.solvers.cpsat import (
    CpsatMultibatchingSolver,
    ModelingMultiBatch,
)
from discrete_optimization.multibatching.solvers.milp_flow import (
    GurobiMultibatchingSolver,
)
from discrete_optimization.multibatching.solvers.netx import NetxMultibatchingSolver
from discrete_optimization.multibatching.solvers.packing_subproblem import (
    CpsatPackingSubproblem,
    GreedyPackingForMultibatching,
    PackingSubproblemSolver,
    PackingViaBinPacking,
)

logger = logging.getLogger(__name__)


class TwoStepMultibatchingSolver(SolverDO):
    hyperparameters = [
        SubBrickHyperparameter(
            name="flow_solver",
            choices=[
                CpsatMultibatchingSolver,
                GurobiMultibatchingSolver,
                NetxMultibatchingSolver,
            ],
            default=SubBrick(
                cls=CpsatMultibatchingSolver,
                kwargs={"modeling": ModelingMultiBatch.FLOW},
            ),
        ),
        SubBrickHyperparameter(
            name="packing_solver",
            choices=[
                CpsatPackingSubproblem,
                GreedyPackingForMultibatching,
                PackingViaBinPacking,
            ],
            default=SubBrick(cls=GreedyPackingForMultibatching, kwargs=None),
        ),
        IntegerHyperparameter(
            name="best_n_flow_solution", low=1, high=10, step=1, default=1
        ),
    ]
    problem: MultibatchingProblem

    def __init__(self, problem: MultibatchingProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)

    def solve(self, callbacks: list[Callback] = None, **args):
        cb = CallbackList(callbacks)
        cb.on_solve_start(self)
        args = self.complete_with_default_hyperparameters(args)
        sub_brick_flow: SubBrick = args["flow_solver"]
        sub_brick_pack: SubBrick = args["packing_solver"]

        solver_flow: MultibatchingSolver = sub_brick_flow.cls(
            problem=self.problem, **sub_brick_flow.kwargs
        )
        solver_flow.init_model(**sub_brick_flow.kwargs)
        res = solver_flow.solve(**sub_brick_flow.kwargs)
        best_n_flow_solution = args["best_n_flow_solution"]
        res_final = self.create_result_storage([])
        best_obj = float("inf")
        for i in range(best_n_flow_solution):
            if len(res) >= best_n_flow_solution:
                solver_pack: PackingSubproblemSolver = sub_brick_pack.cls(
                    problem=self.problem
                )
                solver_pack.init_from_solution(res[-1 - i][0])
                res_ = solver_pack.solve(**sub_brick_pack.kwargs)
                res_final.extend(res_.list_solution_fits)
                cb.on_step_end(i, res_final, self)
                best_obj_ = res_.get_best_solution_fit()[1]
                logger.info(
                    f"Current solution fit:{best_obj_}, current best: {best_obj}, "
                )
                if best_obj_ < best_obj:
                    best_obj = best_obj_
                    logger.info(f"Best solution found at iteration {i}, {best_obj}")
            else:
                break
        cb.on_solve_end(res_final, self)
        return res_final
