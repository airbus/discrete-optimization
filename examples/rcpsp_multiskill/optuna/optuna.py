#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.optuna.utils import (
    generic_optuna_experiment_monoproblem,
)
from discrete_optimization.rcpsp_multiskill.parser_mslib import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_mspsp_instlib import (
    CpMspspMznMultiskillRcpspSolver,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_mzn import (
    CpMultiskillRcpspSolver,
)
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)
from discrete_optimization.rcpsp_multiskill.solvers_map import (
    GaMultiskillRcpspSolver,
    LsGenericRcpspSolver,
)


def script_optuna():
    files_dict = get_data_available()
    file = [f for f in files_dict["MSLIB4"] if "MSLIB_Set4_1003.msrcp" in f][0]
    problem = parse_file(file, skill_level_version=False)
    problem = problem.to_variant_model()
    generic_optuna_experiment_monoproblem(
        problem=problem,
        solvers_to_test=[
            CpSatMultiskillRcpspSolver,
            CpMultiskillRcpspSolver,
            CpMspspMznMultiskillRcpspSolver,
            LsGenericRcpspSolver,
            GaMultiskillRcpspSolver,
        ],
        kwargs_fixed_by_solver={
            CpSatMultiskillRcpspSolver: {"time_limit": 20},
            CpMultiskillRcpspSolver: {"time_limit": 20},
            CpMspspMznMultiskillRcpspSolver: {"time_limit": 20},
        },
        n_trials=100,
    )


if __name__ == "__main__":
    script_optuna()
