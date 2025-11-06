#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


import logging

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.plot import plot_coloring_solution, plt
from discrete_optimization.coloring.problem import (
    ColoringConstraints,
    transform_coloring_problem,
)
from discrete_optimization.coloring.solvers.cpsat import (
    CpSatColoringSolver,
    ModelingCpSat,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    BaseConstraintExtractor,
    ConstraintExtractorPortfolio,
    DummyConstraintExtractor,
    NbChangesAllocationConstraintExtractor,
    NbUsagesAllocationConstraintExtractor,
    SubresourcesAllocationConstraintExtractor,
    SubtasksAllocationConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    TasksConstraintHandler,
)
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat

logging.basicConfig(level=logging.INFO)


def run_lns():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "gc_250_5" in f][0]
    color_problem = parse_file(file)
    subsolver = CpSatColoringSolver(color_problem, params_objective_function=None)
    subsolver.init_model(
        modeling=ModelingCpSat.BINARY,
        do_warmstart=True,
        value_sequence_chain=False,
        used_variable=True,
    )
    p = ParametersCp.default_cpsat()
    print("finished initializing")
    extractors: list[BaseConstraintExtractor] = [
        NbChangesAllocationConstraintExtractor(nb_changes_max=20),
        NbUsagesAllocationConstraintExtractor(plus_delta_nb_usages_total=10),
        # SubresourcesAllocationConstraintExtractor(frac_random_fixed_unary_resources=0.2),
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
        problem=color_problem,
        constraints_extractor=constraints_extractor,
    )
    solver = LnsOrtoolsCpSat(
        problem=color_problem,
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
    print(color_problem.satisfy(sol))


if __name__ == "__main__":
    run_lns()
