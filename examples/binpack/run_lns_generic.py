#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.solvers.cpsat import (
    CpSatBinPackSolver,
    ModelingBinPack,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    BaseConstraintExtractor,
    ConstraintExtractorList,
    ConstraintExtractorPortfolio,
    DummyConstraintExtractor,
    NbChangesAllocationConstraintExtractor,
    NbUsagesAllocationConstraintExtractor,
    SchedulingConstraintExtractor,
    SubresourcesAllocationConstraintExtractor,
    SubtasksAllocationConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    TasksConstraintHandler,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.neighbor_tools import (
    NeighborBuilderMix,
    NeighborBuilderSubPart,
    NeighborRandom,
)
from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat

logging.basicConfig(level=logging.INFO)


def run_lns():
    f = [ff for ff in get_data_available_bppc() if "BPPC_6_2_9.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)
    subsolver = CpSatBinPackSolver(problem=problem)
    subsolver.init_model(modeling=ModelingBinPack.BINARY, upper_bound=450)
    print("finished initializing")
    extractors: list[BaseConstraintExtractor] = [
        NbChangesAllocationConstraintExtractor(nb_changes_max=20),
        NbUsagesAllocationConstraintExtractor(plus_delta_nb_usages_total=10),
        SubresourcesAllocationConstraintExtractor(
            frac_random_fixed_unary_resources=0.2
        ),
        SubtasksAllocationConstraintExtractor(
            fix_secondary_tasks_allocation=False, frac_random_fixed_tasks=0.2
        ),
        DummyConstraintExtractor(),
    ]
    constraints_extractor = ConstraintExtractorPortfolio(
        extractors=extractors, weights=[1 / len(extractors)] * len(extractors)
    )
    # constraints_extractor = ConstraintExtractorList(extractors=extractors)
    constraint_handler = TasksConstraintHandler(
        problem=problem,
        constraints_extractor=constraints_extractor,
    )
    solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
    )
    parameters_cp = ParametersCp.default_cpsat()
    parameters_cp.nb_process = 15
    res = solver.solve(
        callbacks=[WarmStartCallback()],
        nb_iteration_lns=30,
        time_limit_subsolver=20,
        time_limit_subsolver_iter0=20,
        parameters_cp=parameters_cp,
        skip_initial_solution_provider=True,
        ortools_cpsat_solver_kwargs={
            "log_search_progress": False,
            "fix_variables_to_their_hinted_value": False,
        },
    )
    sol = res.get_best_solution()
    problem.satisfy(sol)


def run_lns_scheduling():
    f = [ff for ff in get_data_available_bppc() if "BPPC_4_1_1.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)
    subsolver = CpSatBinPackSolver(problem=problem)
    subsolver.init_model(modeling=ModelingBinPack.SCHEDULING, upper_bound=450)
    print("finished initializing")
    n = NeighborBuilderMix(
        list_neighbor=[
            NeighborBuilderSubPart(problem=problem, nb_cut_part=5),
            NeighborRandom(problem=problem, fraction_subproblem=0.5),
        ],
        weight_neighbor=[1 / 2] * 2,
        verbose=True,
    )
    extractors: list[BaseConstraintExtractor] = [
        SchedulingConstraintExtractor(
            plus_delta_primary=100,
            minus_delta_primary=100,
            plus_delta_secondary=2,
            minus_delta_secondary=2,
        ),
        DummyConstraintExtractor(),
    ]
    constraints_extractor = ConstraintExtractorPortfolio(
        extractors=extractors, weights=[0.8, 0.2]
    )
    constraint_handler = TasksConstraintHandler(
        problem=problem,
        neighbor_builder=n,
        constraints_extractor=constraints_extractor,
    )
    solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
    )
    parameters_cp = ParametersCp.default_cpsat()
    parameters_cp.nb_process = 15
    res = solver.solve(
        callbacks=[WarmStartCallback()],
        nb_iteration_lns=30,
        time_limit_subsolver=10,
        time_limit_subsolver_iter0=10,
        parameters_cp=parameters_cp,
        skip_initial_solution_provider=True,
        ortools_cpsat_solver_kwargs={
            "log_search_progress": False,
            "fix_variables_to_their_hinted_value": False,
        },
    )
    sol = res.get_best_solution()
    problem.satisfy(sol)


if __name__ == "__main__":
    run_lns_scheduling()
