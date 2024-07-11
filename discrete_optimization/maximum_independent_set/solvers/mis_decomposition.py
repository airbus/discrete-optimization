#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
import random
from typing import Any, Dict, List, Optional, Set, Type

import numpy as np

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
    IntegerHyperparameter,
    SubBrickHyperparameter,
    SubBrickKwargsHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.maximum_independent_set.mis_model import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.mis_solvers import solve, solvers_map
from discrete_optimization.maximum_independent_set.solvers.mis_networkx import (
    MisNetworkXSolver,
)
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver

cur_folder = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)


subsolvers = [s for s in solvers_map]


class MisDecomposedSolver(MisSolver):
    """
    This solver is based on the current observation. From a given mis model and one current solution,
    if we decide to freeze the decision variable for a subset of items, the remaining problem to solve is also a
    mis problem, with fewer nodes.
    MisDecomposedSolver is a basic iterative solver that starts from a given solution, then freeze random items,
    solve subproblem with a custom root solver, rebuild original solution and repeat the process.
    """

    hyperparameters = [
        FloatHyperparameter(
            name="proportion_to_remove", low=0.0, high=1.0, default=0.7
        ),
        IntegerHyperparameter(name="nb_iteration", low=0, high=int(10e6), default=100),
        SubBrickHyperparameter(
            name="initial_solver", choices=subsolvers, default=MisNetworkXSolver
        ),
        SubBrickKwargsHyperparameter(
            name="initial_solver_kwargs", subbrick_hyperparameter="initial_solver"
        ),
        SubBrickHyperparameter(
            name="root_solver", choices=subsolvers, default=MisNetworkXSolver
        ),
        SubBrickKwargsHyperparameter(
            name="root_solver_kwargs", subbrick_hyperparameter="root_solver"
        ),
    ]

    def rebuild_sol(
        self,
        sol: MisSolution,
        original_mis_model: MisProblem,
        original_solution: MisSolution,
        indexes_to_remove: Set[int] = None,
    ):
        """
        Rebuild a full Mus solution object from a partial solution.
        :param sol: solution to a sub-mis problem
        :param original_knapsack_model: original knapsack model to solve
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

    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        # wrap all callbacks in a single one
        callbacks_list = CallbackList(callbacks=callbacks)
        # start of solve callback
        callbacks_list.on_solve_start(solver=self)

        kwargs = self.complete_with_default_hyperparameters(kwargs)
        initial_solver: Type[MisSolver] = kwargs["initial_solver"]
        if kwargs["initial_solver_kwargs"] is None:
            initial_solver_kwargs: Dict[str, Any] = {}
        else:
            initial_solver_kwargs = kwargs["initial_solver_kwargs"]
        sub_solver: Type[MisSolver] = kwargs["root_solver"]
        if kwargs["root_solver_kwargs"] is None:
            sub_solver_kwargs: Dict[str, Any] = {}
        else:
            sub_solver_kwargs = kwargs["root_solver_kwargs"]
        nb_iteration = kwargs["nb_iteration"]
        proportion_to_remove = kwargs["proportion_to_remove"]
        initial_results = solve(
            method_solver=initial_solver, problem=self.problem, **initial_solver_kwargs
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
                method_solver=sub_solver, problem=new_problem, **sub_solver_kwargs
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
