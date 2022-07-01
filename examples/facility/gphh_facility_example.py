import os

from discrete_optimization.facility.facility_model import (
    FacilityProblem,
    FacilitySolution,
)
from discrete_optimization.facility.facility_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.facility.facility_solvers import (
    CPSolverName,
    FacilityCP,
    GreedySolverDistanceBased,
    GreedySolverFacility,
    LP_Facility_Solver_PyMip,
    solve,
    solvers_map,
)
from discrete_optimization.facility.solvers.facility_cp_solvers import FacilityCPModel
from discrete_optimization.facility.solvers.gphh_facility import GPHH, ParametersGPHH


def run_gphh():
    model: FacilityProblem = parse_file(get_data_available()[3])
    params_gphh = ParametersGPHH.default()
    params_gphh.pop_size = 25
    params_gphh.crossover_rate = 0.7
    params_gphh.mutation_rate = 0.1
    params_gphh.n_gen = 50
    params_gphh.min_tree_depth = 1
    params_gphh.max_tree_depth = 6
    gphh_solver = GPHH(
        training_domains=[model], domain_model=model, params_gphh=params_gphh
    )
    gphh_solver.init_model()
    rs = gphh_solver.solve()
    sol, fit = rs.get_best_solution_fit()
    print(fit)


def run_greedy():
    model: FacilityProblem = parse_file(get_data_available()[5])
    for method in [
        GreedySolverFacility,
        GreedySolverDistanceBased,
        LP_Facility_Solver_PyMip,
    ]:
        result = solve(method, model, **solvers_map[method][1])
        print(method, result.get_best_solution_fit()[1])
    solver = FacilityCP(model)
    solution, fit = solver.solve_lns(
        fraction_to_fix=0.65,
        nb_iteration=2,
        limit_time_s=20,
        greedy_start=True,
        cp_model=FacilityCPModel.DEFAULT_INT_LNS,
        verbose=True,
    )
    print(fit)
    print(solver.aggreg_sol(solution))
    # 889538


if __name__ == "__main__":
    run_greedy()
