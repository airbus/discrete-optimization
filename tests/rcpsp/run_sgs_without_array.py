import time

import numpy as np
from discrete_optimization.rcpsp.rcpsp_model import (
    RCPSPModel,
    RCPSPSolution,
    SGSWithoutArray,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from sortedcontainers import SortedDict, SortedList


def main():
    # Problem initialisation
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)
    solution = rcpsp_model.get_dummy_solution()
    rcpsp_model.horizon = 100000
    import time

    t = time.time()
    solution.generate_schedule_from_permutation_serial_sgs(do_fast=False)
    t2 = time.time()
    print(t2 - t, " second basic sgs ")
    print(rcpsp_model.evaluate(solution))
    sgs = SGSWithoutArray(rcpsp_model=rcpsp_model)

    t = time.time()
    (
        rcpsp_schedule,
        rcpsp_schedule_feasible,
        resources,
    ) = sgs.generate_schedule_from_permutation_serial_sgs(
        solution=solution, rcpsp_problem=sgs.rcpsp_model
    )
    print(rcpsp_schedule_feasible)
    print(resources)
    sol = RCPSPSolution(
        problem=rcpsp_model,
        rcpsp_schedule=rcpsp_schedule,
        rcpsp_schedule_feasible=rcpsp_schedule_feasible,
    )
    print(rcpsp_model.evaluate(sol))
    print(rcpsp_model.satisfy(sol))
    t2 = time.time()
    print(t2 - t, " second one")
    print(rcpsp_schedule[rcpsp_model.sink_task])


if __name__ == "__main__":
    main()
