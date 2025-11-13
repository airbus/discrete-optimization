#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.solvers.cpsat import (
    CpSatBinPackSolver,
    ModelingBinPack,
)
from discrete_optimization.binpack.solvers.greedy import GreedyBinPackSolver
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.sequential_metasolver import (
    SequentialMetasolver,
    SubBrick,
)


def run_cpsat():
    f = [ff for ff in get_data_available_bppc() if "BPPC_4_1_1.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)
    solver = CpSatBinPackSolver(problem=problem)
    solver.init_model(upper_bound=450, modeling=ModelingBinPack.SCHEDULING)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    res = solver.solve(
        parameters_cp=p,
        time_limit=100,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol = res[-1][0]
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


def run_cpsat_ws():
    f = [ff for ff in get_data_available_bppc() if "BPPC_4_3_2.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)
    greedy = GreedyBinPackSolver(problem=problem)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    sequential_solver = SequentialMetasolver(
        problem=problem,
        list_subbricks=[
            SubBrick(GreedyBinPackSolver, {}),
            SubBrick(
                CpSatBinPackSolver,
                kwargs=dict(
                    modeling=ModelingBinPack.SCHEDULING,
                    parameters_cp=p,
                    time_limit=100,
                    ortools_cpsat_solver_kwargs={"log_search_progress": True},
                ),
                kwargs_from_solution={"upper_bound": lambda sol: max(sol.allocation)},
            ),
        ],
    )
    res = sequential_solver.solve()
    sol = res[-1][0]
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


if __name__ == "__main__":
    run_cpsat()
