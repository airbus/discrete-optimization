#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    BaseConstraintExtractor,
    ConstraintExtractorList,
    ConstraintExtractorPortfolio,
    DummyConstraintExtractor,
    SchedulingConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    ObjectiveSubproblem,
    TasksConstraintHandler,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.neighbor_tools import (
    NeighborBuilderMix,
    NeighborBuilderSubPart,
    NeighborRandom,
)
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import BaseLnsCp, LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import InitialSolutionFromSolver
from discrete_optimization.singlemachine.parser import get_data_available, parse_file
from discrete_optimization.singlemachine.problem import WeightedTardinessProblem
from discrete_optimization.singlemachine.solvers.cpsat import CpsatWTSolver
from discrete_optimization.singlemachine.solvers.greedy import GreedySingleMachineWSPT

logging.basicConfig(level=logging.INFO)


def run_lns():
    problems = parse_file(get_data_available()[0])
    print(len(problems), " problems in the file")
    problem = parse_file(get_data_available()[0])[1]
    subsolver = CpsatWTSolver(problem)
    parameters_cp = ParametersCp.default_cpsat()
    parameters_cp.nb_process = 16
    initial_solution = InitialSolutionFromSolver(GreedySingleMachineWSPT(problem))
    extractors: list[BaseConstraintExtractor] = [
        SchedulingConstraintExtractor(
            minus_delta_primary=1000,
            plus_delta_primary=1000,
            minus_delta_secondary=200,
            plus_delta_secondary=200,
        ),
        DummyConstraintExtractor(),
    ]
    neighbors = [
        NeighborRandom(
            problem=problem,
            fraction_subproblem=0.7,
            delta_abs_time_from_makespan_to_not_fix=0,
            delta_rel_time_from_makespan_to_not_fix=0,
        ),
        NeighborBuilderSubPart(problem=problem, nb_cut_part=5),
    ]
    neighbor = NeighborBuilderMix(neighbors, weight_neighbor=[0.5, 0.5])
    constraints_extractor = ConstraintExtractorPortfolio(extractors=extractors)
    constraint_handler = TasksConstraintHandler(
        problem=problem,
        neighbor_builder=neighbor,
        constraints_extractor=constraints_extractor,
        objective_subproblem=ObjectiveSubproblem.INITIAL_OBJECTIVE,
    )
    solver = LnsOrtoolsCpSat(
        problem=problem,
        initial_solution_provider=initial_solution,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
    )
    res = solver.solve(
        callbacks=[WarmStartCallback()],
        nb_iteration_lns=100,
        time_limit_subsolver_iter0=3,
        time_limit_subsolver=5,
        parameters_cp=parameters_cp,
        skip_initial_solution_provider=False,
    )
    sol = res.get_best_solution()
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    run_lns()
