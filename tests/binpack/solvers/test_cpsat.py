#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.problem import ItemBinPack
from discrete_optimization.binpack.solvers.cpsat import (
    BinPackProblem,
    CpSatBinPackSolver,
    ModelingBinPack,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp


@pytest.mark.parametrize(
    "modeling",
    [ModelingBinPack.BINARY, ModelingBinPack.SCHEDULING],
)
def test_cpsat(modeling):
    # f = [ff for ff in get_data_available_bppc() if "BPPC_4_1_1.txt" in ff][0]
    # problem = parse_bin_packing_constraint_file(f)
    problem = BinPackProblem(
        list_items=[ItemBinPack(index=i, weight=10) for i in range(10)],
        capacity_bin=10,
        incompatible_items={(0, 1), (1, 9)},
    )
    solver = CpSatBinPackSolver(problem=problem)
    solver.init_model(upper_bound=20, modeling=modeling)
    p = ParametersCp.default_cpsat()
    res = solver.solve(
        parameters_cp=p,
        time_limit=3,
    )
    sol = res[-1][0]
    assert problem.satisfy(sol)
