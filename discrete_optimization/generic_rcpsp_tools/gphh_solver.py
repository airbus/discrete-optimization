#  Copyright (c) 2022-2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import operator
import random
from enum import Enum
from typing import Dict, List, Set

import numpy as np
from deap import algorithms, creator, gp, tools
from deap.base import Fitness, Toolbox
from deap.gp import PrimitiveSet, PrimitiveTree, genHalfAndHalf

from discrete_optimization.generic_rcpsp_tools.generic_rcpsp_solver import (
    SolverGenericRCPSP,
)
from discrete_optimization.generic_rcpsp_tools.typing import ANY_RCPSP
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from discrete_optimization.rcpsp.solver.cpm import CPM
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPModel,
    MS_RCPSPSolution_Variant,
)

logger = logging.getLogger(__name__)


def if_then_else(input1, output1, output2):
    if input1:
        return output1
    else:
        return output2


def protected_div(left, right):
    if right != 0.0:
        return left / right
    else:
        return 1.0


def max_operator(left, right):
    return max(left, right)


def min_operator(left, right):
    return min(left, right)


def feature_task_duration(problem: ANY_RCPSP, cpm, cpm_esd, task_id: int, **kwargs):
    return problem.mode_details[task_id][1]["duration"]


def feature_total_n_res(problem: ANY_RCPSP, cpm, cpm_esd, task_id: int, **kwargs):
    val = 0
    mode_consumption = problem.mode_details[task_id][1]
    for res in mode_consumption:
        if res == "duration":
            continue
        val += mode_consumption[res]
    return val


def feature_n_successors(problem: ANY_RCPSP, cpm, cpm_esd, task_id: int, **kwargs):
    return len(problem.successors[task_id]) / problem.n_jobs


def feature_n_predecessors(problem: ANY_RCPSP, cpm, cpm_esd, task_id: int, **kwargs):
    return len(problem.graph.get_predecessors(task_id)) / problem.n_jobs


def get_resource_requirements_across_duration(
    problem: ANY_RCPSP, task_id: int, **kwargs
):
    values = []
    mode_consumption = problem.mode_details[task_id][1]
    duration = mode_consumption["duration"]
    if duration > 0:
        for res in problem.resources_list:
            need = mode_consumption.get(res, 0) / problem.get_max_resource_capacity(res)
            values.append(need / duration)
    else:
        values = [0.0]
    return values


def feature_average_resource_requirements(
    problem: ANY_RCPSP, cpm, cpm_esd, task_id: int, **kwargs
):
    values = get_resource_requirements_across_duration(problem=problem, task_id=task_id)
    if len(values) == 0:
        return 0
    val = np.mean(values)
    return val


def feature_minimum_resource_requirements(
    problem: ANY_RCPSP, cpm, cpm_esd, task_id: int, **kwargs
):
    values = get_resource_requirements_across_duration(problem=problem, task_id=task_id)
    if len(values) == 0:
        return 0
    val = np.min(values)
    return val


def feature_non_zero_minimum_resource_requirements(
    problem: ANY_RCPSP, cpm, cpm_esd, task_id: int, **kwargs
):
    values = get_resource_requirements_across_duration(problem=problem, task_id=task_id)
    if len(values) == 0:
        return 0
    if np.sum(values) > 0.0:
        val = np.min([x for x in values if x > 0.0])
    else:
        val = np.min(values)
    return val


def feature_maximum_resource_requirements(
    problem: ANY_RCPSP, cpm, cpm_esd, task_id: int, **kwargs
):
    values = get_resource_requirements_across_duration(problem=problem, task_id=task_id)
    if len(values) == 0:
        return 0
    val = np.max(values)
    return val


def feature_resource_requirements(
    problem: ANY_RCPSP, cpm, cpm_esd, task_id: int, **kwargs
):
    values = get_resource_requirements_across_duration(problem=problem, task_id=task_id)
    if len(values) > 0:
        val = len([x for x in values if x > 0.0]) / len(values)
        return val
    else:
        return 0


def feature_all_descendants(problem: ANY_RCPSP, cpm, cpm_esd, task_id: int, **kwargs):
    return len(problem.graph.full_successors[task_id]) / problem.n_jobs


def compute_cpm(problem: ANY_RCPSP):
    cpm_solver = CPM(problem)
    path = cpm_solver.run_classic_cpm()
    cpm = cpm_solver.map_node
    cpm_esd = cpm[path[-1]]._ESD  # to normalize...
    return cpm, cpm_esd


def feature_esd(problem: ANY_RCPSP, cpm, cpm_esd, task_id: int, **kwargs):
    """Will only work if you store cpm results into the object. dirty trick"""
    return cpm[task_id]._ESD / cpm_esd


def feature_lsd(problem: ANY_RCPSP, cpm, cpm_esd, task_id: int, **kwargs):
    """Will only work if you store cpm results into the object. dirty trick"""
    return cpm[task_id]._LSD / cpm_esd


def feature_efd(problem: ANY_RCPSP, cpm, cpm_esd, task_id: int, **kwargs):
    """Will only work if you store cpm results into the object. dirty trick"""
    return cpm[task_id]._EFD / cpm_esd


def feature_lfd(problem: ANY_RCPSP, cpm, cpm_esd, task_id: int, **kwargs):
    """Will only work if you store cpm results into the object. dirty trick"""
    return cpm[task_id]._LFD / cpm_esd


def get_dummy(problem: ANY_RCPSP, cpm, cpm_esd, task_id: int, increase: int, **kwargs):
    """Will only work if you store cpm results into the object. dirty trick"""
    return increase


class FeatureEnum(Enum):
    TASK_DURATION = "task_duration"
    RESSOURCE_TOTAL = "total_nres"
    N_SUCCESSORS = "n_successors"
    N_PREDECESSORS = "n_predecessors"
    RESSOURCE_REQUIRED = "res_requ"
    RESSOURCE_AVG = "avg_res_requ"
    RESSOURCE_MIN = "min_res_requ"
    RESSOURCE_NZ_MIN = "nz_min_res_requ"
    RESSOURCE_MAX = "max_res_requ"
    ALL_DESCENDANTS = "all_descendants"
    EARLIEST_START_DATE = "ESD"
    LATEST_START_DATE = "LSD"
    EARLIEST_FINISH_DATE = "EFD"
    LATEST_FINISH_DATE = "LFD"
    DUMMY = "DUMMY"


feature_function_map = {
    FeatureEnum.TASK_DURATION: feature_task_duration,
    FeatureEnum.RESSOURCE_TOTAL: feature_total_n_res,
    FeatureEnum.N_SUCCESSORS: feature_n_successors,
    FeatureEnum.N_PREDECESSORS: feature_n_predecessors,
    FeatureEnum.RESSOURCE_REQUIRED: feature_resource_requirements,
    FeatureEnum.RESSOURCE_AVG: feature_average_resource_requirements,
    FeatureEnum.RESSOURCE_MIN: feature_minimum_resource_requirements,
    FeatureEnum.RESSOURCE_NZ_MIN: feature_non_zero_minimum_resource_requirements,
    FeatureEnum.RESSOURCE_MAX: feature_maximum_resource_requirements,
    FeatureEnum.ALL_DESCENDANTS: feature_all_descendants,
    FeatureEnum.EARLIEST_START_DATE: feature_esd,
    FeatureEnum.EARLIEST_FINISH_DATE: feature_efd,
    FeatureEnum.LATEST_START_DATE: feature_lsd,
    FeatureEnum.LATEST_FINISH_DATE: feature_lfd,
    FeatureEnum.DUMMY: get_dummy,
}

feature_static_map = {
    FeatureEnum.TASK_DURATION: True,
    FeatureEnum.RESSOURCE_TOTAL: True,
    FeatureEnum.N_SUCCESSORS: True,
    FeatureEnum.N_PREDECESSORS: True,
    FeatureEnum.RESSOURCE_REQUIRED: True,
    FeatureEnum.RESSOURCE_AVG: True,
    FeatureEnum.RESSOURCE_MIN: True,
    FeatureEnum.RESSOURCE_NZ_MIN: True,
    FeatureEnum.RESSOURCE_MAX: True,
    FeatureEnum.ALL_DESCENDANTS: True,
    FeatureEnum.EARLIEST_START_DATE: True,
    FeatureEnum.EARLIEST_FINISH_DATE: True,
    FeatureEnum.LATEST_START_DATE: True,
    FeatureEnum.LATEST_FINISH_DATE: True,
    FeatureEnum.DUMMY: False,
}


class EvaluationGPHH(Enum):
    SGS = 0
    PERMUTATION_DISTANCE = 1


class PermutationDistance(Enum):
    KTD = 0
    HAMMING = 1
    KTD_HAMMING = 2


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
    evaluation: EvaluationGPHH = None
    permutation_distance = PermutationDistance.KTD

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
        evaluation,
        permutation_distance,
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
        self.evaluation = evaluation
        self.permutation_distance = permutation_distance

    @staticmethod
    def default():
        set_feature = {
            FeatureEnum.EARLIEST_FINISH_DATE,
            FeatureEnum.EARLIEST_START_DATE,
            FeatureEnum.LATEST_FINISH_DATE,
            FeatureEnum.LATEST_START_DATE,
            FeatureEnum.N_PREDECESSORS,
            FeatureEnum.N_SUCCESSORS,
            FeatureEnum.ALL_DESCENDANTS,
            FeatureEnum.RESSOURCE_REQUIRED,
            FeatureEnum.RESSOURCE_AVG,
            FeatureEnum.RESSOURCE_MAX,
            FeatureEnum.RESSOURCE_NZ_MIN,
        }

        pset = PrimitiveSet("main", len(set_feature))
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protected_div, 2)
        pset.addPrimitive(max_operator, 2)
        pset.addPrimitive(min_operator, 2)
        pset.addPrimitive(operator.neg, 1)

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
            evaluation=EvaluationGPHH.SGS,
            permutation_distance=PermutationDistance.KTD,
        )

    @staticmethod
    def fast_test():
        set_feature = {
            FeatureEnum.EARLIEST_FINISH_DATE,
            FeatureEnum.EARLIEST_START_DATE,
            FeatureEnum.LATEST_FINISH_DATE,
            FeatureEnum.LATEST_START_DATE,
            FeatureEnum.N_PREDECESSORS,
            FeatureEnum.N_SUCCESSORS,
            FeatureEnum.ALL_DESCENDANTS,
            FeatureEnum.RESSOURCE_REQUIRED,
            FeatureEnum.RESSOURCE_AVG,
            FeatureEnum.RESSOURCE_MAX,
            FeatureEnum.RESSOURCE_NZ_MIN,
        }

        pset = PrimitiveSet("main", len(set_feature))
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protected_div, 2)
        pset.addPrimitive(max_operator, 2)
        pset.addPrimitive(min_operator, 2)
        pset.addPrimitive(operator.neg, 1),
        pset.addEphemeralConstant(lambda: random.uniform(-1, 1), float)
        pset.addEphemeralConstant(lambda: random.uniform(-1, 1), float)
        pset.addTerminal(1.0, float)
        pset.addTerminal(1, bool)
        pset.addTerminal(0, bool)

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
            evaluation=EvaluationGPHH.SGS,
            permutation_distance=PermutationDistance.KTD,
        )

    @staticmethod
    def ms_fast():
        set_feature = {
            FeatureEnum.EARLIEST_FINISH_DATE,
            FeatureEnum.EARLIEST_START_DATE,
            FeatureEnum.LATEST_FINISH_DATE,
            FeatureEnum.LATEST_START_DATE,
            FeatureEnum.N_PREDECESSORS,
            FeatureEnum.N_SUCCESSORS,
            FeatureEnum.ALL_DESCENDANTS,
        }

        pset = PrimitiveSet("main", len(set_feature))
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protected_div, 2)
        pset.addPrimitive(max_operator, 2)
        pset.addPrimitive(min_operator, 2)
        pset.addPrimitive(operator.neg, 1)

        return ParametersGPHH(
            set_feature=set_feature,
            set_primitves=pset,
            tournament_ratio=0.1,
            pop_size=4,
            n_gen=2,
            min_tree_depth=1,
            max_tree_depth=4,
            crossover_rate=0.7,
            mutation_rate=0.3,
            deap_verbose=True,
            evaluation=EvaluationGPHH.SGS,
            permutation_distance=PermutationDistance.KTD,
        )

    @staticmethod
    def ms_default():
        set_feature = {
            FeatureEnum.EARLIEST_FINISH_DATE,
            FeatureEnum.EARLIEST_START_DATE,
            FeatureEnum.LATEST_FINISH_DATE,
            FeatureEnum.LATEST_START_DATE,
            FeatureEnum.N_PREDECESSORS,
            FeatureEnum.N_SUCCESSORS,
            FeatureEnum.ALL_DESCENDANTS,
        }

        pset = PrimitiveSet("main", len(set_feature))
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protected_div, 2)
        pset.addPrimitive(max_operator, 2)
        pset.addPrimitive(min_operator, 2)
        pset.addPrimitive(operator.neg, 1)

        return ParametersGPHH(
            set_feature=set_feature,
            set_primitves=pset,
            tournament_ratio=0.1,
            pop_size=40,
            n_gen=100,
            min_tree_depth=1,
            max_tree_depth=4,
            crossover_rate=0.7,
            mutation_rate=0.3,
            deap_verbose=True,
            evaluation=EvaluationGPHH.SGS,
            permutation_distance=PermutationDistance.KTD,
        )

    @staticmethod
    def default_for_set_features(set_feature: Set[FeatureEnum]):
        pset = PrimitiveSet("main", len(set_feature))
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(max_operator, 2)
        pset.addPrimitive(min_operator, 2)
        pset.addPrimitive(operator.neg, 1)

        return ParametersGPHH(
            set_feature=set_feature,
            set_primitves=pset,
            tournament_ratio=0.25,
            pop_size=20,
            n_gen=20,
            min_tree_depth=1,
            max_tree_depth=4,
            crossover_rate=0.7,
            mutation_rate=0.1,
            deap_verbose=True,
            evaluation=EvaluationGPHH.PERMUTATION_DISTANCE,
            permutation_distance=PermutationDistance.KTD,
        )


class GPHH(SolverGenericRCPSP):
    training_domains: List[Problem]
    weight: int
    pset: PrimitiveSet
    toolbox: Toolbox
    params_gphh: ParametersGPHH
    evaluation_method: EvaluationGPHH
    reference_permutations: Dict
    permutation_distance: PermutationDistance

    def __init__(
        self,
        training_domains: List[Problem],
        problem: Problem,
        weight: int = 1,
        params_gphh: ParametersGPHH = None,
        params_objective_function: ParamsObjectiveFunction = None,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.training_domains = training_domains
        self.params_gphh = params_gphh
        if self.params_gphh is None:
            self.params_gphh = ParametersGPHH.default()
        self.set_feature = self.params_gphh.set_feature
        self.list_feature = list(self.set_feature)
        self.list_feature_names = [value.value for value in list(self.list_feature)]
        self.pset = self.init_primitives(self.params_gphh.set_primitves)
        self.weight = weight
        self.evaluation_method = self.params_gphh.evaluation
        model = self.problem
        try:
            if model.graph.full_successors is None:
                model.graph.full_predecessors = model.graph.ancestors_map()
                model.graph.full_successors = model.graph.descendants_map()
        except:
            pass
        self.initialize_cpm_data_for_training()
        self.graphs = {}
        self.toolbox = None

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

        if self.evaluation_method == EvaluationGPHH.SGS:
            self.toolbox.register(
                "evaluate", self.evaluate_heuristic, domains=self.training_domains
            )
        elif self.evaluation_method == EvaluationGPHH.PERMUTATION_DISTANCE:
            self.toolbox.register(
                "evaluate",
                self.evaluate_heuristic_permutation,
                domains=self.training_domains,
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
        if self.toolbox is None:
            self.init_model()
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
        logger.debug(f"best_heuristic: {self.best_heuristic}")
        self.final_pop = pop
        self.func_heuristic = self.toolbox.compile(expr=self.best_heuristic)
        solution = self.build_solution(
            domain=self.problem, func_heuristic=self.func_heuristic
        )
        return ResultStorage(
            list_solution_fits=[(solution, self.aggreg_from_sol(solution))],
            mode_optim=self.params_objective_function.sense_function,
        )

    def init_primitives(self, pset) -> PrimitiveSet:
        for i in range(len(self.list_feature)):
            pset.renameArguments(**{"ARG" + str(i): self.list_feature[i].value})
        return pset

    def build_solution(self, domain, individual=None, func_heuristic=None):
        if func_heuristic is None:
            func_heuristic = self.toolbox.compile(expr=individual)
        d: ANY_RCPSP = domain
        modes = [1 for j in range(d.n_jobs_non_dummy)]
        cpm = self.cpm_data[domain]["cpm"]
        cpm_esd = self.cpm_data[domain]["cpm_esd"]
        raw_values = []
        for task_id in d.tasks_list:
            input_features = [
                feature_function_map[lf](
                    problem=domain,
                    cpm=cpm,
                    cpm_esd=cpm_esd,
                    task_id=task_id,
                    increase=1,
                )
                for lf in self.list_feature
            ]
            output_value = func_heuristic(*input_features)
            raw_values.append(output_value)

        normalized_values = sorted(
            range(len(raw_values)), key=lambda k: raw_values[k], reverse=False
        )
        normalized_values_for_do = [
            d.index_task_non_dummy[d.tasks_list[t]]
            for t in normalized_values
            if d.tasks_list[t] in d.index_task_non_dummy
        ]
        if isinstance(domain, MS_RCPSPModel):
            solution = MS_RCPSPSolution_Variant(
                problem=d,
                priority_list_task=normalized_values_for_do,
                priority_worker_per_task=[
                    [w for w in d.employees_list] for i in range(d.n_jobs_non_dummy)
                ],
                modes_vector=modes,
            )
        else:
            solution = RCPSPSolution(
                problem=d, rcpsp_permutation=normalized_values_for_do, rcpsp_modes=modes
            )
        return solution

    def evaluate_heuristic(self, individual, domains) -> float:
        vals = []
        func_heuristic = self.toolbox.compile(expr=individual)
        for domain in domains:
            solution = self.build_solution(
                individual=individual, domain=domain, func_heuristic=func_heuristic
            )
            do_makespan = -solution.get_end_time(domain.sink_task)
            vals.append(do_makespan)
        fitness = [np.mean(vals)]
        return fitness

    def initialize_cpm_data_for_training(self):
        self.cpm_data = {}
        for domain in self.training_domains:
            cpm, cpm_esd = compute_cpm(domain)
            self.cpm_data[domain] = {"cpm": cpm, "cpm_esd": cpm_esd}
