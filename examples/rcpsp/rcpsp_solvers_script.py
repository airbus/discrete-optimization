#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import gc
import time

from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_solvers import (
    RCPSPModel,
    look_for_solver,
    solve,
    solvers,
    solvers_map,
)


def script():
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    methods = solvers.keys()
    results = {}
    for method in methods:
        results[method] = {}
        print("method : ", method)
        for submethod in solvers[method]:
            print(submethod[0])
            t = time.time()
            solution = solve(submethod[0], rcpsp_model, **submethod[1])
            print(time.time() - t, " seconds to solve")
            print("Solution Best fit: ", solution.get_best_solution_fit())
            results[method][submethod[0]] = solution.get_best_solution_fit()


def script_choose_solver():
    def release_token():
        gc.collect()

    release_token()
    solvers_choice = [
        (method, submethod[0], submethod[1])
        for method in solvers
        for submethod in solvers[method]
    ]
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    solvers_choice = look_for_solver(rcpsp_model)
    while True:
        # Ask user input to select solver
        choice = int(
            input(
                "\nChoose a solver:\n{solvers}\n".format(
                    solvers="\n".join(
                        ["0. Quit"]
                        + [
                            f"{i + 1}. {(solvers_map[k][0], k.__name__)}"
                            for i, k in enumerate(solvers_choice)
                        ]
                    )
                )
            )
        )
        if choice == 0:  # the user wants to quit
            break
        else:
            selected_solver = solvers_choice[choice - 1]
            solution = solve(
                selected_solver, rcpsp_model, **solvers_map[selected_solver][1]
            )
            print("Solver ran ! ")
            print(solution.get_best_solution_fit())
            print(solution.get_best_solution())


if __name__ == "__main__":
    script_choose_solver()
