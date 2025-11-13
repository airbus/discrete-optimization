#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os.path

from discrete_optimization.facility.parser import get_data_available, parse_file
from discrete_optimization.facility.problem import FacilityProblem, FacilitySolution
from discrete_optimization.facility.solvers.cpsat import CpSatFacilitySolver
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    BaseConstraintExtractor,
    ConstraintExtractorList,
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
from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallback,
    WarmStartCallbackLastRun,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat

logging.basicConfig(level=logging.INFO)


def run_lns_cp_facility():
    file = [f for f in get_data_available() if "fl_200_6" in f][0]
    problem: FacilityProblem = parse_file(file)
    print("customer : ", problem.customer_count, "facility : ", problem.facility_count)
    subsolver = CpSatFacilitySolver(problem=problem)
    print("Initializing...")
    subsolver.init_model()
    print("finished initializing")
    extractors: list[BaseConstraintExtractor] = [
        # NbChangesAllocationConstraintExtractor(nb_changes_max=20),
        NbUsagesAllocationConstraintExtractor(plus_delta_nb_usages_total=5),
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
        time_limit_subsolver=10,
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


if __name__ == "__main__":
    run_lns_cp_facility()
