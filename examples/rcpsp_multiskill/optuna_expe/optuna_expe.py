#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.optuna.utils import (
    generic_optuna_experiment_monoproblem,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_mslib_parser import (
    get_data_available,
    parse_file_mslib,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_solvers import (
    GA_MSRCPSP_Solver,
    LargeNeighborhoodSearchScheduling,
    LS_RCPSP_Solver,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solver_mspsp_instlib import (
    CP_MSPSP_MZN,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import CP_MS_MRCPSP_MZN
from discrete_optimization.rcpsp_multiskill.solvers.cpsat_msrcpsp_solver import (
    CPSatMSRCPSPSolver,
)


def script_optuna():
    files_dict = get_data_available()
    file = [f for f in files_dict["MSLIB4"] if "MSLIB_Set4_1003.msrcp" in f][0]
    problem = parse_file_mslib(file, skill_level_version=False)
    problem = problem.to_variant_model()
    generic_optuna_experiment_monoproblem(
        problem=problem,
        solvers_to_test=[
            CPSatMSRCPSPSolver,
            CP_MS_MRCPSP_MZN,
            CP_MSPSP_MZN,
            LS_RCPSP_Solver,
            GA_MSRCPSP_Solver,
        ],
        kwargs_fixed_by_solver={
            CPSatMSRCPSPSolver: {"time_limit": 20},
            CP_MS_MRCPSP_MZN: {"time_limit": 20},
            CP_MSPSP_MZN: {"time_limit": 20},
        },
        n_trials=100,
    )


if __name__ == "__main__":
    script_optuna()
