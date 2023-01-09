#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from multiprocessing import Pool
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

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
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    TupleFitness,
)
from discrete_optimization.rcpsp.mutations.mutation_rcpsp import (
    PermutationMutationRCPSP,
)
from discrete_optimization.rcpsp.rcpsp_model import (
    Aggreg_RCPSPModel,
    RCPSPModel,
    RCPSPSolution,
)

logger = logging.getLogger(__name__)


class RobustnessTool:
    def __init__(
        self,
        base_instance: RCPSPModel,
        all_instances: List[RCPSPModel],
        train_instance: Optional[List[RCPSPModel]] = None,
        test_instance: Optional[List[RCPSPModel]] = None,
        proportion_train: float = 0.8,
    ):
        self.base_instance = base_instance
        self.all_instances = all_instances
        if train_instance is None or test_instance is None:
            random.shuffle(self.all_instances)
            len_train = int(proportion_train * len(self.all_instances))
            self.train_instance = self.all_instances[:len_train]
            self.test_instance = self.all_instances[len_train:]
        else:
            self.train_instance = train_instance
            self.test_instance = test_instance
        self.model_aggreg_mean = Aggreg_RCPSPModel(
            list_problem=self.train_instance,
            method_aggregating=MethodAggregating(BaseMethodAggregating.MEAN),
        )
        self.model_aggreg_max = Aggreg_RCPSPModel(
            list_problem=self.train_instance,
            method_aggregating=MethodAggregating(BaseMethodAggregating.MAX),
        )
        self.model_aggreg_min = Aggreg_RCPSPModel(
            list_problem=self.train_instance,
            method_aggregating=MethodAggregating(BaseMethodAggregating.MIN),
        )
        self.model_aggreg_median = Aggreg_RCPSPModel(
            list_problem=self.train_instance,
            method_aggregating=MethodAggregating(BaseMethodAggregating.MEDIAN),
        )

    def get_models(
        self, apriori: bool = True, aposteriori: bool = True
    ) -> List[RCPSPModel]:
        models: List[RCPSPModel] = []
        tags: List[str] = []
        if aposteriori:
            models += [
                self.model_aggreg_mean,
                self.model_aggreg_max,
                self.model_aggreg_min,
                self.model_aggreg_median,
            ]
            tags += ["post_mean", "post_max", "post_min", "post_median"]
        if apriori:
            model_apriori_mean = self.model_aggreg_mean.get_unique_rcpsp_model()
            model_apriori_max = self.model_aggreg_max.get_unique_rcpsp_model()
            model_apriori_min = self.model_aggreg_min.get_unique_rcpsp_model()
            model_apriori_median = self.model_aggreg_median.get_unique_rcpsp_model()
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
        solve_models_function: Callable[[RCPSPModel], ResultStorage],
        apriori: bool = True,
        aposteriori: bool = True,
    ) -> npt.NDArray[np.float_]:
        models = self.get_models(apriori, aposteriori)
        p = Pool(min(8, len(models)))
        l = p.map(solve_models_function, models)
        solutions: List[RCPSPSolution] = [li.best_solution for li in l]  # type: ignore
        results = np.zeros((len(solutions), len(self.test_instance), 3))
        for index_instance in range(len(self.test_instance)):
            logger.debug(f"Evaluating in instance #{index_instance}")
            instance = self.test_instance[index_instance]
            for index_pareto in range(len(solutions)):
                sol_ = RCPSPSolution(
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

    def plot(self, results: npt.NDArray[np.float_], image_tag: str = "") -> None:
        mean_makespan = np.mean(results[:, :, 1], axis=1)
        max_makespan = np.max(results[:, :, 1], axis=1)
        logger.debug(f"Mean makespan over test instances : {mean_makespan}")
        logger.debug(f"Max makespan over test instances : {max_makespan}")
        logger.debug(f"methods {self.tags}")
        fig, ax = plt.subplots(1, figsize=(10, 10))
        for tag, i in zip(self.tags, range(len(self.tags))):
            sns.distplot(
                results[i, :, 1],
                rug=True,
                bins=max(1, len(self.all_instances) // 10),
                label=tag,
            )
        plt.legend()
        plt.figure(
            "Makespan distribution over test instances, for different optimisation approaches"
        )
        fig.savefig(str(image_tag) + "_comparaison_methods_robust.png")


def solve_model(
    model: Problem, postpro: bool = True, nb_iteration: int = 500
) -> ResultStorage:
    dummy: Solution = model.get_dummy_solution()  # type: ignore
    _, mutations = get_available_mutations(model, dummy)
    list_mutation = [
        mutate[0].build(model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
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
        aggreg_sol: Callable[[Solution], float]
        aggreg_sol, _, _ = build_aggreg_function_and_params_objective(  # type: ignore
            model, params_objective_function
        )
        res = RestartHandlerLimit(
            200, cur_solution=dummy, cur_objective=aggreg_sol(dummy)
        )
        sa = SimulatedAnnealing(
            evaluator=model,
            mutator=mixed_mutation,
            restart_handler=res,
            temperature_handler=TemperatureSchedulingFactor(
                temperature=0.5, restart_handler=res, coefficient=0.9999
            ),
            mode_mutation=ModeMutation.MUTATE,
            params_objective_function=params_objective_function,
            store_solution=True,
            nb_solutions=10000,
        )
        result_ls = sa.solve(dummy, nb_iteration_max=nb_iteration, pickle_result=False)
    else:
        params_objective_function = ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.MULTI_OBJ,
            objectives=objectives,
            weights=objective_weights,
            sense_function=ModeOptim.MAXIMIZATION,
        )
        aggreg_sol2: Callable[[Solution], TupleFitness]
        aggreg_sol2, _, _ = build_aggreg_function_and_params_objective(  # type: ignore
            model, params_objective_function
        )
        res = RestartHandlerLimit(
            200, cur_solution=dummy, cur_objective=aggreg_sol2(dummy)
        )
        sa_mo = HillClimberPareto(
            evaluator=model,
            mutator=mixed_mutation,
            restart_handler=res,
            params_objective_function=params_objective_function,
            mode_mutation=ModeMutation.MUTATE,
            store_solution=True,
            nb_solutions=10000,
        )
        result_ls = sa_mo.solve(
            dummy,
            nb_iteration_max=nb_iteration,
            pickle_result=False,
            update_iteration_pareto=10000,
        )
    return result_ls
