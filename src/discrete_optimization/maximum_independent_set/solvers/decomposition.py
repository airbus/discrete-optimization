#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
import random
from typing import Any, Optional

import numpy as np

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_solver import (
    TrivialSolverFromSolution,
    WarmstartMixin,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
    IntegerHyperparameter,
    SubBrick,
    SubBrickHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.maximum_independent_set.problem import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver
from discrete_optimization.maximum_independent_set.solvers.networkx import (
    NetworkxMisSolver,
)
from discrete_optimization.maximum_independent_set.solvers_map import solve, solvers_map

cur_folder = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)


subsolvers = [s for s in solvers_map]


class DecomposedMisSolver(MisSolver, WarmstartMixin):
    """
    This solver is based on the current observation. From a given mis model and one current solution,
    if we decide to freeze the decision variable for a subset of items, the remaining problem to solve is also a
    mis problem, with fewer nodes.
    DecomposedMisSolver is a basic iterative solver that starts from a given solution, then freeze random items,
    solve subproblem with a custom root solver, rebuild original solution and repeat the process.
    """

    hyperparameters = [
        FloatHyperparameter(
            name="proportion_to_remove", low=0.0, high=1.0, default=0.7
        ),
        IntegerHyperparameter(name="nb_iteration", low=0, high=int(10e6), default=100),
        SubBrickHyperparameter(
            name="initial_solver",
            choices=subsolvers,
            default=SubBrick(cls=NetworkxMisSolver, kwargs={}),
        ),
        SubBrickHyperparameter(
            name="root_solver",
            choices=subsolvers,
            default=SubBrick(cls=NetworkxMisSolver, kwargs={}),
        ),
    ]

    initial_solution: Optional[MisSolution] = None
    """Initial solution used for warm start."""

    def rebuild_sol(
        self,
        sol: MisSolution,
        original_mis_model: MisProblem,
        original_solution: MisSolution,
        indexes_to_remove: set[int] = None,
    ):
        """
        Rebuild a full Mus solution object from a partial solution.
        :param sol: solution to a sub-mis problem
        :param original_knapsack_problem: original knapsack model to solve
        :param original_solution: original base solution
        :param indexes_to_remove: indexes of item removed when building the sub-knapsack problem.
        :return: A new solution object for the original problem.
        """
        list_taken = [
            original_solution.chosen[i] for i in range(original_mis_model.number_nodes)
        ]
        for j in range(len(sol.chosen)):
            original_index = original_mis_model.nodes_to_index[
                sol.problem.index_to_nodes[j]
            ]
            list_taken[original_index] = sol.chosen[j]
        solution = MisSolution(problem=original_mis_model, chosen=list_taken)
        return solution

    def set_warm_start(self, solution: MisSolution) -> None:
        """Make the solver warm start from the given solution.

        Will be ignored if arg `initial_solver` is set and not None in call to `solve()`.

        """
        self.initial_solution = solution

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        # wrap all callbacks in a single one
        callbacks_list = CallbackList(callbacks=callbacks)
        # start of solve callback
        callbacks_list.on_solve_start(solver=self)

        # manage warm start if self.initial_solution is set
        if self.initial_solution is not None:
            kwargs["initial_solver"] = SubBrick(
                cls=TrivialSolverFromSolution,
                kwargs=dict(solution=self.initial_solution),
            )
        kwargs = self.complete_with_default_hyperparameters(kwargs)

        initial_solver: SubBrick = kwargs["initial_solver"]
        initial_solver_cls: type[MisSolver] = initial_solver.cls
        initial_solver_kwargs = initial_solver.kwargs

        root_solver: SubBrick = kwargs["root_solver"]
        root_solver_cls: type[MisSolver] = root_solver.cls
        root_solver_kwargs = root_solver.kwargs

        nb_iteration = kwargs["nb_iteration"]
        proportion_to_remove = kwargs["proportion_to_remove"]
        initial_results = solve(
            method_solver=initial_solver_cls,
            problem=self.problem,
            **initial_solver_kwargs,
        )
        results_storage = self.create_result_storage(
            initial_results.list_solution_fits,
        )
        logger.info(
            f"Initial solution fitness : {results_storage.get_best_solution_fit()[1]}"
        )
        all_nodes = set(self.problem.nodes_to_index.keys())

        for j in range(nb_iteration):
            sol, fit = results_storage.get_best_solution_fit()
            sol: MisSolution
            nb_chosen = sum(sol.chosen)
            idx_chosen = list(np.where(sol.chosen)[0])
            subpart_chosen = set(
                random.sample(
                    idx_chosen,
                    int(proportion_to_remove * nb_chosen),
                )
            )
            nodes_to_remove_from_subproblem = set()
            for index_node in subpart_chosen:
                n = self.problem.index_to_nodes[index_node]
                neighbors = self.problem.graph_nx.neighbors(n)
                nodes_to_remove_from_subproblem.add(n)
                for nn in neighbors:
                    nodes_to_remove_from_subproblem.add(nn)
            subgraph = self.problem.graph_nx.subgraph(
                all_nodes.difference(nodes_to_remove_from_subproblem)
            )
            new_problem = MisProblem(
                graph=subgraph, attribute_aggregate=self.problem.attribute_aggregate
            )
            res = solve(
                method_solver=root_solver_cls, problem=new_problem, **root_solver_kwargs
            )
            best_sol, fit = res.get_best_solution_fit()
            if best_sol is not None:
                reb_sol = self.rebuild_sol(
                    sol=best_sol,
                    original_solution=sol,
                    original_mis_model=self.problem,
                    indexes_to_remove=None,
                )
                fit = self.aggreg_from_sol(reb_sol)
                logger.info(f"Iteration {j}/{nb_iteration} : --- Current fitness {fit}")
                results_storage.append((reb_sol, fit))

            stopping = callbacks_list.on_step_end(
                step=j, res=results_storage, solver=self
            )
            if stopping:
                break

        # end of solve callback
        callbacks_list.on_solve_end(res=results_storage, solver=self)
        return results_storage
