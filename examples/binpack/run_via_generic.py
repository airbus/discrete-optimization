#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.transformations.to_generic import (
    BinpackToGenericSchedulingTransformation,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.cpmpy.auto_impl import (
    GenericSchedulingAutoCpmpyImplSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    ObjectiveGapStopper,
)
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.sequential_metasolver import (
    SubBrick,
)

logging.basicConfig(level=logging.DEBUG)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.auto_impl import (
    GenericSchedulingAutoOptalImplSolver,
)
from discrete_optimization.generic_tools.hub_solver.optal.optalcp_tools import (
    optalcp_available,
)
from discrete_optimization.generic_tools.transformation import TransformationSolver


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
        case _:
            raise NotImplementedError()

    return cls, kwargs, check_solution


def run_via_generic(cp_framework: str = "cpsat"):
    f = [ff for ff in get_data_available_bppc() if "BPPC_4_1_1.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    cls, kwargs, _ = generic_scheduling_solver_cls_kwargs_check(cp_framework)
    solver = TransformationSolver(
        transformation=BinpackToGenericSchedulingTransformation(),
        solver_brick=SubBrick(
            cls,
            kwargs,
        ),
        source_problem=problem,
    )
    res = solver.solve(
        callbacks=[ObjectiveGapStopper(0, 0), BasicStatsCallback()],
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    run_via_generic("optal")
