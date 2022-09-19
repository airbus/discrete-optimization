#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.ea.ga import DeapMutation
from discrete_optimization.generic_tools.ea.nsga import Nsga
from discrete_optimization.generic_tools.result_storage.result_storage import (
    plot_storage_2d,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file


def test_single_mode_nsga_2obj():
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, +1]
    ga_solver = Nsga(
        rcpsp_model,
        encoding="rcpsp_permutation",
        objectives=objectives,
        objective_weights=objective_weights,
        mutation=mutation,
    )
    ga_solver._max_evals = 2000
    result_storage = ga_solver.solve()
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)


if __name__ == "__main__":
    test_single_mode_nsga_2obj()
