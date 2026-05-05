#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from copy import copy
from typing import Any, Optional

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    SubBrick,
    SubBrickHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.multibatching.problem import (
    MultibatchingProblem,
    Product,
    TransportLink,
)
from discrete_optimization.multibatching.solvers.cpsat import (
    CpsatMultibatchingSolver,
    ModelingMultiBatch,
)
from discrete_optimization.multibatching.solvers.packing_subproblem import (
    CpsatPackingSubproblem,
    GreedyPackingForMultibatching,
    PackingSubproblemSolver,
    PackingViaBinPacking,
)


class CpsatBendersMultibatchingSolver(SolverDO):
    hyperparameters = [
        SubBrickHyperparameter(
            name="packing_solver",
            choices=[
                CpsatPackingSubproblem,
                GreedyPackingForMultibatching,
                PackingViaBinPacking,
            ],
            default=SubBrick(cls=GreedyPackingForMultibatching, kwargs=None),
        )
    ]
    problem: MultibatchingProblem

    def __init__(self, problem: MultibatchingProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables = {}
        # self.modeling: ModelingMultiBatch = None
        self.base_solution = None
        self.solver_master: CpsatMultibatchingSolver = None
        self.solver_subproblem: PackingSubproblemSolver = None
        self.scaling_factor = kwargs.get("scaling_factor", 1)
        self.nb_cuts = 0

    def init_model(self, **kwargs: Any) -> None:
        self.solver_master = CpsatMultibatchingSolver(
            problem=self.problem, scaling_factor=self.scaling_factor
        )
        self.solver_master.init_model(modeling=ModelingMultiBatch.FLOW, **kwargs)

    def update_model_master(
        self,
        changed: list[tuple[TransportLink, dict[Product, int], int, int]],
        best_lb: int,
    ):
        for link, flows, new_nb_trip, prev_nb_trip in changed:
            index_link = self.problem.transport_links_to_index[link]
            indicators = []
            for all_index_link in [index_link]:
                for product in flows:
                    ind = self.solver_master.cp_model.NewBoolVar(
                        f"flow_{all_index_link}_{product.id}_equal_{flows[product]}"
                    )
                    var = self.solver_master.variables["flows"][all_index_link][
                        self.problem.product_to_index[product]
                    ]
                    self.solver_master.cp_model.Add(
                        var == flows[product]
                    ).OnlyEnforceIf(ind)
                    self.solver_master.cp_model.Add(
                        var != flows[product]
                    ).OnlyEnforceIf(ind.Not())
                    indicators.append(ind)
                all_equal = self.solver_master.cp_model.NewBoolVar(
                    f"indicator_cut_{all_index_link}_{self.nb_cuts}"
                )
                self.solver_master.cp_model.Add(all_equal == 1).OnlyEnforceIf(
                    *indicators
                )

                (
                    self.solver_master.cp_model.Add(
                        self.solver_master.variables["nb_trips"][all_index_link]
                        == new_nb_trip
                    ).OnlyEnforceIf(all_equal)
                )
                self.nb_cuts += 1
        # self.solver_master.cp_model.Add(self.solver_master.variables["total_obj"]
        #                                 >= int(math.floor(best_lb)))

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        time_limit_first_iter: int = 100,
        params_solver_master: dict = None,
        nb_iteration: int = 10,
        **kwargs: Any,
    ) -> ResultStorage:
        if self.solver_master is None:
            self.init_model(**kwargs)
        kwargs = self.complete_with_default_hyperparameters(kwargs)

        callback = CallbackList()
        callback.on_solve_start(solver=self)
        if params_solver_master is None:
            params_solver_master = dict(
                parameters_cp=ParametersCp.default_cpsat(),
                time_limit=5,
                ortools_cpsat_solver_kwargs={"log_search_progress": True},
            )
        subbrick_packing_solver: SubBrick = kwargs["packing_solver"]
        params_solver_sub = subbrick_packing_solver.kwargs
        if params_solver_sub is None:
            params_solver_sub = dict(
                parameters_cp=ParametersCp.default_cpsat(),
                time_limit=5,
                ortools_cpsat_solver_kwargs={"log_search_progress": True},
            )
        res_merged = self.create_result_storage([])
        current_lbs = []
        for i in range(nb_iteration):
            if i >= 1:
                bs, _ = res_merged.get_best_solution_fit(satisfying=self.problem)
                self.solver_master.set_warm_start(bs)
            if i == 0:
                kwargs = copy(params_solver_master)
                kwargs["time_limit"] = time_limit_first_iter
                res = self.solver_master.solve(callbacks=callbacks, **kwargs)
            else:
                res = self.solver_master.solve(
                    callbacks=callbacks, **params_solver_master
                )
            if len(res) > 0:
                current_lbs.append(
                    self.solver_master.get_current_best_internal_objective_bound()
                )
                base_solution = res[-1][0]
                print("Status master : ", self.solver_master.status_solver)
                print(
                    "Cur solution",
                    self.problem.evaluate(base_solution),
                    self.problem.satisfy(base_solution),
                )
                print("Obj ", self.aggreg_from_sol(base_solution))
                subsolver: PackingSubproblemSolver = subbrick_packing_solver.cls(
                    problem=self.problem
                )
                subsolver.init_from_solution(solution=base_solution)
                self.solver_subproblem = subsolver
                res_sub_problem = subsolver.solve(**params_solver_sub)
                res_merged.extend(res_sub_problem)
                print("Status subsolver : ", subsolver.status_solver)
                print(
                    "Solution post pro",
                    self.problem.evaluate(res_sub_problem[-1][0]),
                    self.problem.satisfy(res_sub_problem[-1][0]),
                )
                print("Obj ", self.aggreg_from_sol(res_sub_problem[-1][0]))
                callback.on_step_end(i, res=res_sub_problem, solver=self)
                changes = subsolver.analyse_solution(
                    new_solution=res_sub_problem[-1][0]
                )
                if len(changes) == 0:
                    print("stop")
                    break
                print("Nb changes :", len(changes))
                best_lb = max(current_lbs)
                self.update_model_master(changed=changes, best_lb=best_lb)
                bs, best_ub = res_merged.get_best_solution_fit(satisfying=self.problem)
                print("Best lb", best_lb)
                print("Best Ub", best_ub)
        return res_merged
