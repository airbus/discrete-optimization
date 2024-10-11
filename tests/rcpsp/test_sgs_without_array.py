#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from sortedcontainers import SortedDict

from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.sgs_without_array import SgsWithoutArray
from discrete_optimization.rcpsp.solution import RcpspSolution


def test_sgs_wo_array():
    # Problem initialisation
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)
    solution = rcpsp_problem.get_dummy_solution()
    rcpsp_problem.horizon = 100000

    solution.generate_schedule_from_permutation_serial_sgs(do_fast=False)
    assert rcpsp_problem.evaluate(solution) == {
        "makespan": 49,
        "mean_resource_reserve": 0,
        "constraint_penalty": 0.0,
    }
    sgs = SgsWithoutArray(rcpsp_problem=rcpsp_problem)
    (
        rcpsp_schedule,
        rcpsp_schedule_feasible,
        resources,
    ) = sgs.generate_schedule_from_permutation_serial_sgs(
        solution=solution, rcpsp_problem=sgs.rcpsp_problem
    )
    assert rcpsp_schedule_feasible
    assert resources == {
        "R1": SortedDict(
            {
                0: 8,
                6: 2,
                8: 2,
                12: 1,
                15: 1,
                17: 5,
                18: 9,
                24: 12,
                32: 10,
                33: 6,
                36: 10,
                39: 9,
                41: 12,
            }
        ),
        "R2": SortedDict(
            {
                0: 13,
                8: 8,
                12: 7,
                17: 12,
                21: 5,
                23: 4,
                24: 5,
                26: 3,
                33: 6,
                40: 13,
                41: 4,
                44: 5,
                47: 6,
                49: 13,
            }
        ),
        "R3": SortedDict({0: 4, 17: 0, 24: 4, 47: 2, 49: 4}),
        "R4": SortedDict(
            {0: 9, 6: 11, 8: 3, 13: 4, 16: 7, 18: 0, 23: 7, 26: 4, 32: 6, 34: 5, 42: 12}
        ),
    }
    sol = RcpspSolution(
        problem=rcpsp_problem,
        rcpsp_schedule=rcpsp_schedule,
        rcpsp_schedule_feasible=rcpsp_schedule_feasible,
    )
    assert rcpsp_problem.evaluate(sol) == {
        "makespan": 49,
        "mean_resource_reserve": 0,
        "constraint_penalty": 0.0,
    }
    assert rcpsp_problem.satisfy(sol)
    assert rcpsp_schedule[rcpsp_problem.sink_task] == {"start_time": 49, "end_time": 49}


if __name__ == "__main__":
    test_sgs_wo_array()
