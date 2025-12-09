#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from collections.abc import Callable
from multiprocessing import Pool
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib import cm

from discrete_optimization.generic_rcpsp_tools.mutation import RcpspMutation
from discrete_optimization.generic_tools.do_problem import (
    BaseMethodAggregating,
    MethodAggregating,
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
    Problem,
    Solution,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.ls.hill_climber import HillClimberPareto
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandlerLimit,
)
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    create_mutations_portfolio_from_problem,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    TupleFitness,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.problem_robust import AggregRcpspProblem
from discrete_optimization.rcpsp.solution import RcpspSolution

logger = logging.getLogger(__name__)


class RobustnessTool:
    def __init__(
        self,
        base_instance: RcpspProblem,
        all_instances: list[RcpspProblem],
        train_instance: Optional[list[RcpspProblem]] = None,
        test_instance: Optional[list[RcpspProblem]] = None,
        proportion_train: float = 0.8,
    ):
        self.base_instance = base_instance
        self.all_instances = all_instances
        self.solutions: list[RcpspSolution] = []
        self.models: list[Union[RcpspProblem, AggregRcpspProblem]] = []
        self.tags: list[str] = []
        if train_instance is None or test_instance is None:
            random.shuffle(self.all_instances)
            len_train = int(proportion_train * len(self.all_instances))
            self.train_instance = self.all_instances[:len_train]
            self.test_instance = self.all_instances[len_train:]
        else:
            self.train_instance = train_instance
            self.test_instance = test_instance
        self.model_aggreg_mean = AggregRcpspProblem(
            list_problem=self.train_instance,
            method_aggregating=MethodAggregating(BaseMethodAggregating.MEAN),
        )
        self.model_aggreg_max = AggregRcpspProblem(
            list_problem=self.train_instance,
            method_aggregating=MethodAggregating(BaseMethodAggregating.MAX),
        )
        self.model_aggreg_min = AggregRcpspProblem(
            list_problem=self.train_instance,
            method_aggregating=MethodAggregating(BaseMethodAggregating.MIN),
        )
        self.model_aggreg_median = AggregRcpspProblem(
            list_problem=self.train_instance,
            method_aggregating=MethodAggregating(BaseMethodAggregating.MEDIAN),
        )

    def get_models(
        self, aggreg_apriori: bool = True, aggreg_aposteriori: bool = True
    ) -> list[RcpspProblem]:
        models: list[RcpspProblem] = []
        tags: list[str] = []
        if aggreg_apriori:
            models += [
                self.model_aggreg_mean,
                self.model_aggreg_max,
                self.model_aggreg_min,
                self.model_aggreg_median,
            ]
            tags += ["post_mean", "post_max", "post_min", "post_median"]
        if aggreg_aposteriori:
            model_apriori_mean = self.model_aggreg_mean.get_unique_rcpsp_problem()
            model_apriori_max = self.model_aggreg_max.get_unique_rcpsp_problem()
            model_apriori_min = self.model_aggreg_min.get_unique_rcpsp_problem()
            model_apriori_median = self.model_aggreg_median.get_unique_rcpsp_problem()
            models += [
                model_apriori_mean,
                model_apriori_max,
                model_apriori_min,
                model_apriori_median,
            ]
            tags += ["prio_mean", "prio_max", "prio_min", "prio_median"]
        models += [self.base_instance]
        tags += ["original"]
        self.models = models
        self.tags = tags
        return models

    def solve_and_retrieve(
        self,
        solve_models_function: Callable[[RcpspProblem], ResultStorage],
        apriori: bool = True,
        aposteriori: bool = True,
        nb_process: int = 8,
    ) -> npt.NDArray[np.float64]:
        models = self.get_models(apriori, aposteriori)
        p = Pool(min(nb_process, len(models)))
        l = p.map(solve_models_function, models)
        solutions: list[RcpspSolution] = [li.get_best_solution_fit()[0] for li in l]  # type: ignore
        self.solutions = [li.get_best_solution_fit()[0] for li in l]
        results = np.zeros((len(solutions), len(self.test_instance), 3))
        for index_instance in range(len(self.test_instance)):
            logger.debug(f"Evaluating in instance #{index_instance}")
            instance = self.test_instance[index_instance]
            for index_pareto in range(len(solutions)):
                sol_ = RcpspSolution(
                    problem=instance,
                    rcpsp_permutation=solutions[index_pareto].rcpsp_permutation,
                    rcpsp_modes=solutions[index_pareto].rcpsp_modes,
                )
                fit = instance.evaluate(sol_)
                results[index_pareto, index_instance, 0] = (
                    1 if sol_.rcpsp_schedule_feasible else 0
                )
                results[index_pareto, index_instance, 1] = fit["makespan"]
                results[index_pareto, index_instance, 2] = fit["mean_resource_reserve"]
        return results

    def get_statistics_df(self, results: npt.NDArray[np.float64]) -> pd.DataFrame:
        """
        Computes aggregate statistics for each method.
        Returns a Pandas DataFrame with Mean, Std, Min, Max, and Feasibility %.
        """
        stats = []
        for i, tag in enumerate(self.tags):
            makespans = results[i, :, 1]
            feasibility = results[i, :, 0]

            row = {
                "Method": tag,
                "Feasibility (%)": np.mean(feasibility) * 100,
                "Mean Makespan": np.mean(makespans),
                "Std Makespan": np.std(makespans),
                "Min Makespan": np.min(makespans),
                "Max Makespan": np.max(makespans),
            }
            stats.append(row)

        df = pd.DataFrame(stats)
        df.set_index("Method", inplace=True)
        # Sort by Mean Makespan for easier reading
        return df.sort_values("Mean Makespan")

    def plot(self, results: npt.NDArray[np.float64], image_tag: str = "") -> None:
        """Original plot method (Histograms)"""
        mean_makespan = np.mean(results[:, :, 1], axis=1)
        max_makespan = np.max(results[:, :, 1], axis=1)
        logger.debug(f"Mean makespan over test instances : {mean_makespan}")
        logger.debug(f"Max makespan over test instances : {max_makespan}")
        logger.debug(f"methods {self.tags}")

        fig, ax = plt.subplots(1, figsize=(10, 10))
        for tag, i in zip(self.tags, range(len(self.tags))):
            sns.histplot(
                results[i, :, 1],
                kde=True,
                bins=max(1, len(self.all_instances) // 10),
                label=tag,
                ax=ax,
                alpha=0.3,
            )
        plt.legend()
        plt.title("Makespan distribution over test instances")
        fig.savefig(str(image_tag) + "_comparaison_methods_robust_hist.png")

    def plot_boxplots(
        self, results: npt.NDArray[np.float64], image_tag: str = ""
    ) -> None:
        """Plots Boxplots for better comparison of distributions and outliers."""
        fig, ax = plt.subplots(1, figsize=(12, 6))

        # Prepare data for seaborn
        data_list = []
        for i, tag in enumerate(self.tags):
            for val in results[i, :, 1]:
                data_list.append({"Method": tag, "Makespan": val})
        df_melted = pd.DataFrame(data_list)

        sns.boxplot(x="Method", y="Makespan", data=df_melted, ax=ax, palette="Set3")
        sns.stripplot(
            x="Method",
            y="Makespan",
            data=df_melted,
            ax=ax,
            color=".25",
            alpha=0.5,
            jitter=True,
        )

        ax.set_title("Makespan Variability by Method (Test Instances)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(str(image_tag) + "_comparaison_methods_robust_boxplot.png")
        plt.show()

    def visualize_scenarios(self, method_tag: str, nb_scenarios: int = 5):
        """
        Visualizes the Gantt chart of the solution found by 'method_tag'
        simulated on 'nb_scenarios' random test instances.

        This uses the native RcpspSolution evaluation mechanism (no Executor).
        """
        if method_tag not in self.tags:
            logger.error(f"Method {method_tag} not found in {self.tags}")
            return

        method_idx = self.tags.index(method_tag)
        base_sol = self.solutions[method_idx]

        # Pick random test instances
        scenarios = random.sample(
            self.test_instance, min(nb_scenarios, len(self.test_instance))
        )

        fig, axs = plt.subplots(
            nb_scenarios, 1, figsize=(12, 3 * nb_scenarios), sharex=True
        )
        if nb_scenarios == 1:
            axs = [axs]

        colors = cm.get_cmap("tab20", self.base_instance.n_jobs)

        for i, instance in enumerate(scenarios):
            # Create the solution wrapper for this specific instance
            # This triggers the internal SGS (schedule generation) inside RcpspSolution
            test_sol = RcpspSolution(
                problem=instance,
                rcpsp_permutation=base_sol.rcpsp_permutation,
                rcpsp_modes=base_sol.rcpsp_modes,
            )

            # fit contains makespan, etc.
            fit = instance.evaluate(test_sol)
            makespan = fit["makespan"]

            # Access the computed schedule directly
            # Format: {job_id: {'start_time': X, 'end_time': Y}}
            schedule = test_sol.rcpsp_schedule

            ax = axs[i]
            for job in schedule:
                start = schedule[job]["start_time"]
                end = schedule[job]["end_time"]
                duration = end - start

                # Draw bar
                ax.barh(
                    y=f"Job {job}",
                    width=duration,
                    left=start,
                    color=colors(job % 20),
                    edgecolor="black",
                    alpha=0.8,
                )

                # Add job label
                ax.text(
                    start + duration / 2,
                    job,
                    str(job),
                    va="center",
                    ha="center",
                    fontsize=8,
                    color="white",
                )

            ax.set_title(
                f"Scenario {i + 1} - Makespan: {makespan} (Feasible: {test_sol.rcpsp_schedule_feasible})"
            )
            ax.grid(True, axis="x", linestyle="--", alpha=0.5)

            # If too many jobs, hide Y labels
            if self.base_instance.n_jobs > 20:
                ax.set_yticks([])

        plt.xlabel("Time")
        plt.suptitle(
            f"Robustness Check: Method '{method_tag}' across Random Test Scenarios"
        )
        plt.tight_layout()
        plt.show()


def solve_model(
    model: Problem, postpro: bool = True, nb_iteration: int = 500
) -> ResultStorage:
    dummy: Solution = model.get_dummy_solution()  # type: ignore
    mixed_mutation = create_mutations_portfolio_from_problem(
        problem=model, selected_mutations={RcpspMutation}
    )

    objectives = ["makespan"]
    objective_weights = [-1.0]
    if postpro:
        params_objective_function = ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=objectives,
            weights=objective_weights,
            sense_function=ModeOptim.MAXIMIZATION,
        )
        aggreg_from_sol: Callable[[Solution], float]
        aggreg_from_sol, _, _ = build_aggreg_function_and_params_objective(  # type: ignore
            model, params_objective_function
        )
        res = RestartHandlerLimit(200)
        sa = SimulatedAnnealing(
            problem=model,
            mutator=mixed_mutation,
            restart_handler=res,
            temperature_handler=TemperatureSchedulingFactor(
                temperature=0.5, restart_handler=res, coefficient=0.9999
            ),
            mode_mutation=ModeMutation.MUTATE,
            params_objective_function=params_objective_function,
            store_solution=True,
        )
        result_ls = sa.solve(initial_variable=dummy, nb_iteration_max=nb_iteration)
    else:
        params_objective_function = ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.MULTI_OBJ,
            objectives=objectives,
            weights=objective_weights,
            sense_function=ModeOptim.MAXIMIZATION,
        )
        aggreg_from_sol2: Callable[[Solution], TupleFitness]
        aggreg_from_sol2, _, _ = build_aggreg_function_and_params_objective(  # type: ignore
            model, params_objective_function
        )
        res = RestartHandlerLimit(200)
        sa_mo = HillClimberPareto(
            problem=model,
            mutator=mixed_mutation,
            restart_handler=res,
            params_objective_function=params_objective_function,
            mode_mutation=ModeMutation.MUTATE,
            store_solution=True,
        )
        result_ls = sa_mo.solve(
            initial_variable=dummy,
            nb_iteration_max=nb_iteration,
            update_iteration_pareto=10000,
        )
    return result_ls
