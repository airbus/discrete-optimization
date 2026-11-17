#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import random

import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.rcpsp_multiskill.parser_mslib import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.solvers.optal import (
    OptalMSRcpspSolver,
    optalcp_available,
)


@pytest.mark.skipif(
    not optalcp_available, reason="You need optalcp to test this solver."
)
@pytest.mark.parametrize(
    "one_skill_used_per_worker, one_worker_per_task, feasible, time_limit, no_solution",
    [
        (False, False, True, 7, False),
        (True, False, True, 7, False),
        (False, True, False, 5, False),
        (False, False, True, 0.001, True),
    ],
)
@pytest.mark.parametrize("multimode", [False, True])
def test_optal(
    multimode,
    one_skill_used_per_worker,
    one_worker_per_task,
    feasible,
    time_limit,
    no_solution,
    optal_preview_only,
):
    files_dict = get_data_available()
    file = [f for f in files_dict["MSLIB4"] if "MSLIB_Set4_1003.msrcp" in f][0]
    problem = parse_file(file, skill_level_version=False)
    if multimode:
        # singlemode -> multimode
        for task, task_details in problem.mode_details.items():
            for original_mode, original_mode_task_details in task_details.items():
                ...
            new_mode = original_mode + 1
            new_mode_task_details = dict(original_mode_task_details)
            new_mode_task_details["duration"] = int(
                random.uniform(0.5, 2.0) * original_mode_task_details["duration"]
            )
            task_details[new_mode] = new_mode_task_details
        problem.update_problem()
        assert problem.is_multimode
    solver = OptalMSRcpspSolver(problem=problem)
    solver.init_model(
        one_skill_used_per_worker=one_skill_used_per_worker,
        one_worker_per_task=one_worker_per_task,
    )
    res = solver.solve(
        time_limit=time_limit,
        do_not_retrieve_solutions=optal_preview_only,
        callbacks=[NbIterationStopper(1)],
    )
    if feasible:
        if no_solution:
            assert solver.status_solver == StatusSolver.UNKNOWN
        else:
            assert solver.status_solver in (
                StatusSolver.OPTIMAL,
                StatusSolver.SATISFIED,
            )
            assert solver.current_obj >= solver.current_bound
    else:
        assert solver.status_solver == StatusSolver.UNSATISFIABLE

    if not optal_preview_only:
        assert problem.satisfy(res.get_best_solution())
