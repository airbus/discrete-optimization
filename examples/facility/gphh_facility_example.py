#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.facility.facility_model import FacilityProblem
from discrete_optimization.facility.facility_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.facility.solvers.gphh_facility import GPHH, ParametersGPHH


def run_gphh():
    model: FacilityProblem = parse_file(get_data_available()[3])
    params_gphh = ParametersGPHH.default()
    params_gphh.pop_size = 25
    params_gphh.crossover_rate = 0.7
    params_gphh.mutation_rate = 0.1
    params_gphh.n_gen = 10
    params_gphh.min_tree_depth = 1
    params_gphh.max_tree_depth = 6
    gphh_solver = GPHH(
        training_domains=[model], domain_model=model, params_gphh=params_gphh
    )
    gphh_solver.init_model()
    rs = gphh_solver.solve()
    sol, fit = rs.get_best_solution_fit()
    print(fit)


if __name__ == "__main__":
    run_gphh()
