#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.lotsizing.uncapacitatedsingleitem.problem import (
    generate_random_instance,
)
from discrete_optimization.lotsizing.uncapacitatedsingleitem.solvers.cpsat import (
    CpSatUncapacitatedSingleItemSolver,
)
from discrete_optimization.lotsizing.uncapacitatedsingleitem.solvers.dp_wagner import (
    WagnerWhitinSolver,
)
from discrete_optimization.lotsizing.uncapacitatedsingleitem.solvers.lp import (
    GurobiUncapacitatedSingleItemSolver,
)
from discrete_optimization.lotsizing.uncapacitatedsingleitem.solvers.toulbar import (
    ToulbarUncapacitatedSingleItemSolver,
)
from discrete_optimization.lotsizing.utils import (
    plot_inventory_and_costs,
    plot_production_schedule,
    plot_solution_summary,
    plt,
)

logging.basicConfig(level=logging.INFO)


def solve_with(solver: SolverDO, **kwargs):
    solver.init_model(**kwargs)
    res = solver.solve(**kwargs)
    print("status :", solver.status_solver)
    return res


def script():
    problem = generate_random_instance(
        horizon=30,
        avg_demand=5,
        setup_cost=1,
        production_cost=5,
        inventory_cost=3,
        seed=42,
    )
    solver_tag = "cpsat"
    if solver_tag == "gurobi":
        solver = GurobiUncapacitatedSingleItemSolver(problem)
        p = ParametersMilp.default()
        res = solve_with(
            solver,
            **dict(
                gurobi_solver_kwargs={
                    "NoRelHeurTime": 3,
                    "Heuristics": 0.2,
                    "Threads": 10,
                },
                parameters_milp=p,
                time_limit=30,
            ),
        )
    if solver_tag == "wagner":
        solver = WagnerWhitinSolver(problem)
        res = solve_with(solver)
    if solver_tag == "cpsat":
        solver = CpSatUncapacitatedSingleItemSolver(problem)
        res = solve_with(
            solver, **dict(parameters_cp=ParametersCp.default_cpsat(), time_limit=30)
        )
    if solver_tag == "toulbar":
        solver = ToulbarUncapacitatedSingleItemSolver(problem)
        res = solve_with(solver, **dict(time_limit=30))
    sol = res[-1][0]
    plot_production_schedule(problem, sol)
    plot_inventory_and_costs(problem, sol)
    plot_solution_summary(problem, sol)
    plt.show()
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    script()
