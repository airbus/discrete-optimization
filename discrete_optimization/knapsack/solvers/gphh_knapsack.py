import operator
from enum import Enum
from typing import List, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from deap import algorithms, creator, gp, tools
from deap.base import Fitness, Toolbox
from deap.gp import (
    Primitive,
    PrimitiveSet,
    PrimitiveSetTyped,
    PrimitiveTree,
    Terminal,
    genHalfAndHalf,
)

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackSolutionMultidimensional,
    MultidimensionalKnapsack,
)


def protected_div(left, right):
    if right != 0.0:
        return left / right
    else:
        return 1.0


def max_operator(left, right):
    return max(left, right)


def min_operator(left, right):
    return min(left, right)


def max_operator_list(list_):
    return max(list_)


def min_operator_list(list_):
    return min(list_)


class FeatureEnum(Enum):
    PROFIT = "profit"
    CAPACITIES = "capacities"
    RES_CONSUMPTION_ARRAY = "res_consumption"
    AVG_RES_CONSUMPTION_DELTA_CAPACITY = "avg_res_consumption_delta_capacity"


def get_profit(problem: MultidimensionalKnapsack, item_index, **kwargs):
    return problem.list_items[item_index].value


def get_capacities(problem: MultidimensionalKnapsack, **kwargs):
    return problem.max_capacities


def get_res_consumption(problem: MultidimensionalKnapsack, item_index, **kwargs):
    return problem.list_items[item_index].weights


def get_avg_res_consumption_delta_capacity(
    problem: MultidimensionalKnapsack, item_index, **kwargs
):
    return sum(
        [
            (problem.max_capacities[j] - problem.list_items[item_index].weights[j])
            / problem.max_capacities[j]
            for j in range(len(problem.max_capacities))
        ]
    ) / len(problem.max_capacities)


feature_function_map = {
    FeatureEnum.PROFIT: get_profit,
    FeatureEnum.CAPACITIES: get_capacities,
    FeatureEnum.RES_CONSUMPTION_ARRAY: get_res_consumption,
    FeatureEnum.AVG_RES_CONSUMPTION_DELTA_CAPACITY: get_avg_res_consumption_delta_capacity,
}


class ParametersGPHH:
    set_feature: Set[FeatureEnum] = None
    set_primitves: PrimitiveSet = None
    tournament_ratio: float = None
    pop_size: int = None
    n_gen: int = None
    min_tree_depth: int = None
    max_tree_depth: int = None
    crossover_rate: float = None
    mutation_rate: float = None
    deap_verbose: bool = None

    def __init__(
        self,
        set_feature,
        set_primitves,
        tournament_ratio,
        pop_size,
        n_gen,
        min_tree_depth,
        max_tree_depth,
        crossover_rate,
        mutation_rate,
        deap_verbose,
    ):
        self.set_feature = set_feature
        self.set_primitves = set_primitves
        self.tournament_ratio = tournament_ratio
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.min_tree_depth = min_tree_depth
        self.max_tree_depth = max_tree_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.deap_verbose = deap_verbose

    @staticmethod
    def default():
        set_feature = [
            FeatureEnum.PROFIT,
            FeatureEnum.CAPACITIES,
            FeatureEnum.AVG_RES_CONSUMPTION_DELTA_CAPACITY,
        ]
        pset = PrimitiveSetTyped("main", [float, list, float], float)
        # take profit, list of ressource consumption, avearage delta consumption
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.sub, [float, float], float)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(protected_div, [float, float], float)
        pset.addPrimitive(max_operator, [float, float], float)
        pset.addPrimitive(min_operator, [float, float], float)
        pset.addPrimitive(operator.neg, [float], float)
        pset.addPrimitive(max_operator_list, [list], float, name="max_operator_list")
        pset.addPrimitive(min_operator_list, [list], float, name="min_operator_list")
        pset.addPrimitive(lambda x: sum(x) / len(x), [list], float, name="mean_list")
        pset.addPrimitive(
            lambda x, y: [xx - yy for xx, yy in zip(x, y)],
            [list, list],
            list,
            name="sub_list",
        )
        pset.addPrimitive(
            lambda x, y: [xx + yy for xx, yy in zip(x, y)],
            [list, list],
            list,
            name="plus_list",
        )
        return ParametersGPHH(
            set_feature=set_feature,
            set_primitves=pset,
            tournament_ratio=0.1,
            pop_size=10,
            n_gen=2,
            min_tree_depth=1,
            max_tree_depth=4,
            crossover_rate=0.7,
            mutation_rate=0.3,
            deap_verbose=True,
        )


class GPHH(SolverDO):
    training_domains: List[Problem]
    verbose: bool
    weight: int
    pset: PrimitiveSet
    toolbox: Toolbox
    params_gphh: ParametersGPHH

    def __init__(
        self,
        training_domains: List[Problem],
        domain_model: Problem,
        weight: int = 1,
        params_gphh: ParametersGPHH = None,
        params_objective_function: ParamsObjectiveFunction = None,
        verbose: bool = False,
    ):
        self.training_domains = training_domains
        self.domain_model = domain_model
        self.params_gphh = params_gphh
        if self.params_gphh is None:
            self.params_gphh = ParametersGPHH.default()
        self.set_feature = self.params_gphh.set_feature
        print("self.set_feature: ", self.set_feature)
        self.list_feature = list(self.set_feature)
        self.list_feature_names = [value.value for value in list(self.list_feature)]
        self.verbose = verbose
        self.pset = self.init_primitives(self.params_gphh.set_primitves)
        self.weight = weight
        (
            self.aggreg_from_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.domain_model,
            params_objective_function=params_objective_function,
        )

    def init_model(self):
        tournament_ratio = self.params_gphh.tournament_ratio
        pop_size = self.params_gphh.pop_size
        min_tree_depth = self.params_gphh.min_tree_depth
        max_tree_depth = self.params_gphh.max_tree_depth

        creator.create("FitnessMin", Fitness, weights=(self.weight,))
        creator.create("Individual", PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox = Toolbox()
        self.toolbox.register(
            "expr",
            genHalfAndHalf,
            pset=self.pset,
            min_=min_tree_depth,
            max_=max_tree_depth,
        )
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register(
            "evaluate", self.evaluate_heuristic, domains=self.training_domains
        )
        self.toolbox.register(
            "select", tools.selTournament, tournsize=int(tournament_ratio * pop_size)
        )
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=max_tree_depth)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset
        )
        self.toolbox.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )
        self.toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

    def solve(self, **kwargs):
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        pop = self.toolbox.population(n=self.params_gphh.pop_size)
        hof = tools.HallOfFame(1000)
        self.hof = hof
        pop, log = algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=self.params_gphh.crossover_rate,
            mutpb=self.params_gphh.mutation_rate,
            ngen=self.params_gphh.n_gen,
            stats=mstats,
            halloffame=hof,
            verbose=True,
        )
        self.best_heuristic = hof[0]
        print("best_heuristic: ", self.best_heuristic)
        self.final_pop = pop
        self.func_heuristic = self.toolbox.compile(expr=self.best_heuristic)
        solution = self.build_solution(
            domain=self.domain_model, func_heuristic=self.func_heuristic
        )
        return ResultStorage(
            list_solution_fits=[(solution, self.aggreg_from_sol(solution))],
            mode_optim=self.params_objective_function.sense_function,
        )

    def build_result_storage_for_domain(self, domain):
        solution = self.build_solution(
            domain=domain, func_heuristic=self.func_heuristic
        )
        return ResultStorage(
            list_solution_fits=[
                (solution, self.aggreg_dict(domain.evaluate(solution)))
            ],
            mode_optim=self.params_objective_function.sense_function,
        )

    def init_primitives(self, pset) -> PrimitiveSet:
        for i in range(len(self.list_feature)):
            pset.renameArguments(**{"ARG" + str(i): self.list_feature[i].value})
        return pset

    def build_solution(self, domain, individual=None, func_heuristic=None):
        if func_heuristic is None:
            func_heuristic = self.toolbox.compile(expr=individual)
        d: MultidimensionalKnapsack = domain
        raw_values = []
        for j in range(len(d.list_items)):
            input_features = [
                feature_function_map[lf](problem=domain, item_index=j)
                for lf in self.list_feature
            ]
            output_value = func_heuristic(*input_features)
            raw_values.append(output_value)
        sorted_indexes = [
            x
            for x in sorted(
                range(len(raw_values)), key=lambda k: raw_values[k], reverse=True
            )
        ]
        current_weight = [0] * len(d.max_capacities)
        k = 0
        list_taken = [0] * len(d.list_items)
        value = 0
        while all(
            current_weight[j] <= d.max_capacities[j]
            for j in range(len(d.max_capacities))
        ) and k < len(sorted_indexes):
            if all(
                current_weight[j] + d.list_items[sorted_indexes[k]].weights[j]
                <= d.max_capacities[j]
                for j in range(len(d.max_capacities))
            ):
                list_taken[sorted_indexes[k]] = 1
                value += d.list_items[sorted_indexes[k]].value
                for j in range(len(d.max_capacities)):
                    current_weight[j] = (
                        current_weight[j] + d.list_items[sorted_indexes[k]].weights[j]
                    )
            k += 1
        solution = KnapsackSolutionMultidimensional(
            problem=d, list_taken=list_taken, value=value, weights=current_weight
        )
        return solution

    def evaluate_heuristic(
        self, individual, domains: List[MultidimensionalKnapsack]
    ) -> float:
        vals = []
        func_heuristic = self.toolbox.compile(expr=individual)
        for domain in domains:
            solution = self.build_solution(
                individual=individual, domain=domain, func_heuristic=func_heuristic
            )
            value = self.aggreg_dict(domain.evaluate(solution))
            vals.append(value)
        fitness = [np.mean(vals)]
        return [fitness[0] - 10 * self.evaluate_complexity(individual)]

    def evaluate_complexity(self, individual):
        all_primitives_list = []
        all_features_list = []
        for i in range(len(individual)):
            if isinstance(individual[i], Primitive):
                all_primitives_list.append(individual[i].name)
            elif isinstance(individual[i], Terminal):
                all_features_list.append(individual[i].value)
        n_operators = len(all_primitives_list)
        n_features = len(all_features_list)
        val = 1.0 * n_operators + 1.0 * n_features
        return val

    def plot_solution(self, show=True):
        nodes, edges, labels = gp.graph(self.best_heuristic)
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = nx.drawing.spring_layout(g)

        nx.draw_networkx_nodes(g, pos)
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels)
        if show:
            plt.show()
