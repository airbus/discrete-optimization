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

from discrete_optimization.facility.facility_model import FacilityProblem
from discrete_optimization.facility.solvers.greedy_solvers import (
    GreedySolverDistanceBased,
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


def distance(problem: FacilityProblem, customer_index, **kwargs):
    return [
        problem.evaluate_customer_facility(
            facility=f, customer=problem.customers[customer_index]
        )
        for f in problem.facilities
    ]


def demand_minus_capacity(problem: FacilityProblem, customer_index, **kwargs):
    return [
        problem.customers[customer_index].demand - f.capacity
        for f in problem.facilities
    ]


def capacity(problem: FacilityProblem, customer_index, **kwargs):
    return [f.capacity for f in problem.facilities]


def closest_facility(problem: FacilityProblem, customer_index, **kwargs):
    return min(
        range(len(problem.facilities)),
        key=lambda x: problem.evaluate_customer_facility(
            facility=problem.facilities[x], customer=problem.customers[customer_index]
        ),
    )


def index_min(list_or_array):
    return np.argmin(list_or_array)


def index_max(list_or_array):
    return np.argmax(list_or_array)


def argsort(list_or_array):
    return np.argsort(list_or_array)


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
    DISTANCE = "distance"
    CAPACITIES = "capacities"
    DEMAND_MINUS_CAPACITY = "demand_minus_capacity"


feature_function_map = {
    FeatureEnum.DISTANCE: distance,
    FeatureEnum.CAPACITIES: capacity,
    FeatureEnum.DEMAND_MINUS_CAPACITY: demand_minus_capacity,
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
            FeatureEnum.DISTANCE,
            FeatureEnum.DEMAND_MINUS_CAPACITY,
        ]
        pset = PrimitiveSetTyped("main", [list, list], list)
        # take profit, list of ressource consumption, avearage delta consumption
        pset.addPrimitive(
            lambda x, y: [max(xx, yy) for xx, yy in zip(x, y)],
            [list, list],
            list,
            name="max_element",
        )
        pset.addPrimitive(
            lambda x, y: [min(xx, yy) for xx, yy in zip(x, y)],
            [list, list],
            list,
            name="min_element",
        )
        pset.addPrimitive(
            lambda x, y: [protected_div(xx, yy) for xx, yy in zip(x, y)],
            [list, list],
            list,
            name="protected_div_list",
        )
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
        pset.addTerminal(1, int, name="dummy")
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
        training_domains: List[FacilityProblem],
        domain_model: FacilityProblem,
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
        self.greedy_solver = GreedySolverDistanceBased(
            facility_problem=self.domain_model
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
        result = self.build_solution(
            domain=self.domain_model, func_heuristic=self.func_heuristic
        )
        return result

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
        d: FacilityProblem = domain
        raw_values = []

        for j in range(d.customer_count):
            input_features = [
                feature_function_map[lf](problem=domain, customer_index=j)
                for lf in self.list_feature
            ]
            output_value = func_heuristic(*input_features)
            raw_values.append(argsort(output_value))
        result = self.greedy_solver.solve(
            **{"prio": {c: raw_values[c] for c in range(len(raw_values))}}
        )
        return result

    def evaluate_heuristic(self, individual, domains: List[FacilityProblem]) -> list:
        vals = []
        func_heuristic = self.toolbox.compile(expr=individual)
        for domain in domains:
            result = self.build_solution(
                individual=individual, domain=domain, func_heuristic=func_heuristic
            )
            value = result.get_best_solution_fit()[1]
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

    def plot_solution(self):
        nodes, edges, labels = gp.graph(self.best_heuristic)
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = nx.drawing.spring_layout(g)

        nx.draw_networkx_nodes(g, pos)
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels)
        plt.show()
