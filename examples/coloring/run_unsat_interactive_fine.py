#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.solvers.cpmpy import CpmpyColoringSolver
from discrete_optimization.generic_tools.do_solver import StatusSolver

instance = "gc_20_1"
kwargs_init_model = dict(nb_colors=2)
kwargs_mus = {}

# problem to solve
filepath = [f for f in get_data_available() if instance in f][0]
color_problem = parse_file(filepath)


print("Initializing the solver...")
solver = CpmpyColoringSolver(color_problem)
solver.init_model(**kwargs_init_model)

done = False
removed_constraints = []
while not done:
    print("Solving...")
    result_store = solver.solve()

    if solver.status_solver == StatusSolver.UNSATISFIABLE:
        mus = solver.explain_unsat_fine(**kwargs_mus)
        print("The problem is unsatisfiable.")
        str_list_cstr = "\n".join(
            f"{i_cstr}: {cstr}" for i_cstr, cstr in enumerate(mus)
        )
        i_cstr = int(
            input(
                "Choose a constraint to remove among this minimal unsatisfiable subset:\n"
                + str_list_cstr
                + "\n> "
            )
        )
        # remove constraint from model
        cstr = mus[i_cstr]
        solver.model.constraints = [
            c for c in solver.model.constraints if c is not cstr
        ]
        removed_constraints.append(cstr)
    elif len(result_store) > 0:
        print(f"The problem was solved with status {solver.status_solver.value}.")
        if len(removed_constraints) > 0:
            print(
                f"We removed {len(removed_constraints)} meta-constraint{'s' if len(removed_constraints)>0 else ''}: "
                f"{ removed_constraints}"
            )
        break
    else:
        print(f"No solution found and status is {solver.status_solver}. Exiting.")
        break
