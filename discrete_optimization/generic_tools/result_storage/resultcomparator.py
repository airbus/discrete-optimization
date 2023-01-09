#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Dict, List, Optional, Tuple, cast

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from discrete_optimization.generic_tools.do_problem import (
    Problem,
    Solution,
    TupleFitness,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ParetoFront,
    ResultStorage,
    fitness_class,
    plot_pareto_2d,
    result_storage_to_pareto_front,
)

logger = logging.getLogger(__name__)


class ResultComparator:
    # If test problem is None, then we use the fitnesses from the ResultStorage
    def __init__(
        self,
        list_result_storage: List[ResultStorage],
        result_storage_names: List[str],
        objectives_str: List[str],
        objective_weights: List[int],
        test_problems: Optional[List[Problem]] = None,
    ):
        self.list_result_storage = list_result_storage
        self.result_storage_names = result_storage_names
        self.objectives_str = objectives_str
        self.objective_weights = objective_weights
        self.test_problems = test_problems
        self.reevaluated_results: Dict[int, Dict[str, List[float]]] = {}

        if self.test_problems is not None:
            self.reevaluate_result_storages()

    def reevaluate_result_storages(self) -> None:
        if self.test_problems is None:
            raise RuntimeError(
                "self.test_problems cannot be None when calling reevaluate_result_storages()."
            )
        for res in self.list_result_storage:
            self.reevaluated_results[self.list_result_storage.index(res)] = {}
            for obj in self.objectives_str:
                self.reevaluated_results[self.list_result_storage.index(res)][obj] = []
                for scenario in self.test_problems:
                    best_sol = res.get_best_solution()
                    if best_sol is None:
                        raise RuntimeError(
                            "res.get_best_solution() cannot be None "
                            "for any res in self.list_result_storage"
                            "when calling reevaluate_result_storages()."
                        )
                    best_sol.change_problem(scenario)
                    val = scenario.evaluate(best_sol)[obj]
                    self.reevaluated_results[self.list_result_storage.index(res)][
                        obj
                    ].append(val)
        logger.debug(f"reevaluated_results: {self.reevaluated_results}")

    def plot_distribution_for_objective(self, objective_str: str) -> Figure:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        for i in range(len(self.result_storage_names)):
            sns.distplot(
                self.reevaluated_results[i][objective_str],
                rug=True,
                bins=max(1, len(self.reevaluated_results[i][objective_str]) // 10),
                label=self.result_storage_names[i],
                ax=ax,
            )
        ax.legend()
        ax.set_title(
            objective_str.upper()
            + " distribution over test instances, for different optimisation approaches"
        )
        return fig

    def print_test_distribution(self) -> None:
        ...

    def get_best_by_objective_by_result_storage(
        self, objectif_str: str
    ) -> Dict[str, Tuple[Solution, fitness_class]]:
        obj_index = self.objectives_str.index(objectif_str)
        val: Dict[str, Tuple[Solution, fitness_class]] = {}
        for i in range(len(self.list_result_storage)):
            fit_array = [
                cast(
                    TupleFitness, fitness
                ).vector_fitness[  # indicate to mypy that we are in multiobjective case
                    obj_index
                ]
                for solution, fitness in self.list_result_storage[i].list_solution_fits
            ]  # create fit array
            if self.list_result_storage[i].maximize:
                best_fit = max(fit_array)
            else:
                best_fit = min(fit_array)

            best_index = fit_array.index(best_fit)
            best_sol = self.list_result_storage[i].list_solution_fits[best_index]
            val[self.result_storage_names[i]] = best_sol
        return val

    def generate_super_pareto(self) -> ParetoFront:
        sols = []
        for rs in self.list_result_storage:
            for s in rs.list_solution_fits:
                sols.append(s)
        rs = ResultStorage(list_solution_fits=sols, best_solution=None)
        pareto_store = result_storage_to_pareto_front(result_storage=rs, problem=None)
        return pareto_store

    def plot_all_2d_paretos_single_plot(
        self, objectives_str: Optional[List[str]] = None
    ) -> Axes:

        if objectives_str is None:
            objecives_names = self.objectives_str[:2]
            objectives_index = [0, 1]
        else:
            objecives_names = objectives_str
            objectives_index = []
            for obj in objectives_str:
                obj_index = self.objectives_str.index(obj)
                objectives_index.append(obj_index)

        colors = cm.rainbow(np.linspace(0, 1, len(self.list_result_storage)))
        fig, ax = plt.subplots(1)
        ax.set_xlabel(objecives_names[0])
        ax.set_ylabel(objecives_names[1])

        for i in range(len(self.list_result_storage)):
            ax.scatter(
                x=[
                    p[1].vector_fitness[objectives_index[0]]  # type: ignore
                    for p in self.list_result_storage[i].list_solution_fits
                ],
                y=[
                    p[1].vector_fitness[objectives_index[1]]  # type: ignore
                    for p in self.list_result_storage[i].list_solution_fits
                ],
                color=colors[i],
            )
        ax.legend(self.result_storage_names)
        return ax

    def plot_all_2d_paretos_subplots(
        self, objectives_str: Optional[List[str]] = None
    ) -> Figure:

        if objectives_str is None:
            objecives_names = self.objectives_str[:2]
            objectives_index = [0, 1]
        else:
            objecives_names = objectives_str
            objectives_index = []
            for obj in objectives_str:
                obj_index = self.objectives_str.index(obj)
                objectives_index.append(obj_index)

        cols = 2
        rows = math.ceil(
            len(self.list_result_storage) / cols
        )  # I have to do this to ensure at least 2 rows or else it creates axs with only 1 diumension and it crashes
        fig, axs = plt.subplots(rows, cols)
        axis = axs.flatten()
        colors = cm.rainbow(np.linspace(0, 1, len(self.list_result_storage)))
        for i, ax in zip(
            range(len(self.list_result_storage)), axis[: len(self.list_result_storage)]
        ):
            x = [
                p[1].vector_fitness[objectives_index[0]]  # type: ignore
                for p in self.list_result_storage[i].list_solution_fits
            ]
            y = [
                p[1].vector_fitness[objectives_index[1]]  # type: ignore
                for p in self.list_result_storage[i].list_solution_fits
            ]
            ax.scatter(x=x, y=y, color=colors[i])
            ax.set_title(self.result_storage_names[i])
        fig.tight_layout(pad=3.0)

        return fig

    def plot_super_pareto(self) -> None:
        super_pareto = self.generate_super_pareto()
        plot_pareto_2d(pareto_front=super_pareto, name_axis=self.objectives_str)
        plt.title("Pareto front obtained by merging solutions from all result stores")

    def plot_all_best_by_objective(self, objectif_str: str) -> None:
        obj_index = self.objectives_str.index(objectif_str)
        data = self.get_best_by_objective_by_result_storage(objectif_str)
        x = list(data.keys())
        y = [data[key][1].vector_fitness[obj_index] for key in x]  # type: ignore
        y_pos = np.arange(len(x))

        plt.bar(y_pos, y)
        plt.xticks(y_pos, x, rotation=45)
        plt.title("Comparison on " + objectif_str)
