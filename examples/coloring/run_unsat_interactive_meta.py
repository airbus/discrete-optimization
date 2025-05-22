#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from copy import copy

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

# copy meta constraints as we will update them later
meta_constraints = [copy(m) for m in solver.get_soft_meta_constraints()]
done = False
removed_meta = []
while not done:
    print("Solving...")
    result_store = solver.solve()

    if solver.status_solver == StatusSolver.UNSATISFIABLE:
        meta_mus = solver.explain_unsat_meta(soft=meta_constraints, **kwargs_mus)
        print("The problem is unsatisfiable.")
        str_list_meta = "\n".join(
            f"{i_meta}: {meta.name}" for i_meta, meta in enumerate(meta_mus)
        )
        i_meta = int(
            input(
                "Choose a meta-constraint to remove among this minimal unsatisfiable subset:\n"
                + str_list_meta
                + "\n> "
            )
        )
        # remove meta-constraint:
        # - remove subconstraints from model
        # - remove subconstraints from other meta-constraints
        # - remove meta-constraint from list of meta-constraints
        meta = meta_mus[i_meta]
        subconstraints_ids = {id(c_) for c_ in meta.constraints}
        solver.model.constraints = [
            c for c in solver.model.constraints if id(c) not in subconstraints_ids
        ]
        meta_constraints = [m for m in meta_constraints if m is not meta]
        for other_meta in meta_constraints:
            other_meta.constraints = [
                c for c in other_meta.constraints if id(c) not in subconstraints_ids
            ]
        removed_meta.append(meta)
    elif len(result_store) > 0:
        print(f"The problem was solved with status {solver.status_solver.value}.")
        if len(removed_meta) > 0:
            print(
                f"We removed {len(removed_meta)} meta-constraint{'s' if len(removed_meta)>0 else ''}: "
                f"{[meta.name for meta in removed_meta]}"
            )
        break
    else:
        print(f"No solution found and status is {solver.status_solver}. Exiting.")
        break
