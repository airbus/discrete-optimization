#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from copy import deepcopy

from discrete_optimization.facility.parser import get_data_available, parse_file
from discrete_optimization.facility.problem import FacilityProblem
from discrete_optimization.facility.transformations.to_generic import (
    FacilityToGenericSchedulingTransformation,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.cpmpy.auto_impl import (
    GenericSchedulingAutoCpmpyImplSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.auto_impl import (
    GenericSchedulingAutoOptalImplSolver,
)
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hub_solver.optal.optalcp_tools import (
    optalcp_available,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver

logging.basicConfig(level=logging.INFO)


def generic_scheduling_solver_cls_kwargs_check(
    cp_framework, optal_preview_only: bool = False
):
    p = ParametersCp.default_cpsat()
    kwargs = {
        "time_limit": 100,
        "parameters_cp": p,
    }
    check_solution = True
    match cp_framework:
        case "cpsat":
            cls = GenericSchedulingAutoCpSatImplSolver
            kwargs["ortools_cpsat_solver_kwargs"] = {"log_search_progress": True}
        case "cpmpy":
            cls = GenericSchedulingAutoCpmpyImplSolver
            kwargs["verbose"] = True
        case "optal":
            if not optalcp_available:
                raise Exception("Install optal cp")
            cls = GenericSchedulingAutoOptalImplSolver
            if optal_preview_only:
                kwargs["do_not_retrieve_solutions"] = True
                check_solution = False
            import optalcp

            fds_worker = optalcp.WorkerParameters(
                searchType="fds", cumulPropagationLevel=3
            )
            kwargs["workers"] = [fds_worker, fds_worker]
        case _:
            raise NotImplementedError()

    return cls, kwargs, check_solution


def cp_facility_example(cp_framework: str = "cpsat"):
    file = [f for f in get_data_available() if "fl_16_1" in f][0]  # "fl_100_5"
    problem: FacilityProblem = parse_file(file)
    print("customer : ", problem.customer_count, "facility : ", problem.facility_count)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    cls, kwargs, _ = generic_scheduling_solver_cls_kwargs_check(cp_framework)
    solver = TransformationSolver(
        transformation=FacilityToGenericSchedulingTransformation(),
        solver_brick=SubBrick(
            cls,
            kwargs,
        ),
        source_problem=problem,
    )
    kwargs_init = deepcopy(kwargs)
    solver.init_model(**kwargs_init)
    res = solver.solve(
        callbacks=[  # ObjectiveGapStopper(0, 0),
            BasicStatsCallback()
        ],
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    cp_facility_example(cp_framework="optal")
