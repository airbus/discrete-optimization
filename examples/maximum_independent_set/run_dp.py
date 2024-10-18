#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import re

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.maximum_independent_set.parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.problem import MisProblem
from discrete_optimization.maximum_independent_set.solvers.cpsat import CpSatMisSolver
from discrete_optimization.maximum_independent_set.solvers.dp import (
    DpMisSolver,
    DpModeling,
    dp,
)

logging.basicConfig(level=logging.DEBUG)

sol__ = """1   28   29   40   43   50   71   74   81  120  123  134  137  176  183
 186  207  214  217  228  229  256  260  261  288  303  310  313  334  340
 341  355  383  396  397  403  418  446  449  476  477  488  491  498  515
 543  558  564  565  588  589  595  610  638  648  651  658  673  700  701
 728  731  743  746  753  775  778  785  824  827  848  855  858  870  873
 911  918  921  932  933  960  963  991 1006 1012 1013 1026 1054 1068 1069
1075 1096 1099 1106 1121 1148 1149 1159 1162 1169 1208 1211 1232 1239 1242
1254 1257 1286 1289 1328 1335 1338 1359 1366 1369 1380 1381 1408 1422 1428
1429 1443 1471 1474 1502 1516 1517 1523 1540 1541 1568 1583 1590 1593 1614
1620 1621 1635 1663 1676 1677 1683 1698 1726 1729 1756 1757 1768 1771 1778
1800 1803 1810 1825 1852 1853 1880 1883 1895 1898 1905 1936 1943 1946 1958
1961 1988 1989 2016 2031 2038 2041"""


def extract_ints(word):
    return tuple(int(num) for num in re.findall(r"\d+", word))


def run_dip_solver():
    small_example = [f for f in get_data_available() if "1dc.1024" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    # sol_ = MisSolution(problem=mis_model,
    #                    chosen=[0 for i in range(mis_model.number_nodes)])
    # for n in extract_ints(sol__):
    #     sol_.chosen[mis_model.nodes_to_index[n]] = 1
    # print(mis_model.satisfy(sol_))
    # print(mis_model.evaluate(sol_))

    solver = DpMisSolver(problem=mis_model)
    solver.init_model(modeling=DpModeling.ORDER)
    res = solver.solve(solver=dp.CABS, time_limit=100)
    sol, fit = res.get_best_solution_fit()
    print(mis_model.evaluate(sol))
    print(mis_model.satisfy(sol))


def run_dip_solver_ws():
    small_example = [f for f in get_data_available() if "1tc.1024" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver_ws = CpSatMisSolver(problem=mis_model)
    sol_ws = solver_ws.solve(
        time_limit=5, callbacks=[ObjectiveLogger(step_verbosity_level=logging.INFO)]
    )[-1][0]
    print(solver_ws.is_optimal())
    solver = DpMisSolver(problem=mis_model)
    solver.init_model(modeling=DpModeling.ANY_ORDER)
    solver.set_warm_start(sol_ws)
    res = solver.solve(
        solver=dp.LNBS,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        retrieve_intermediate_solutions=True,
        time_limit=100,
    )
    sol = res[0][0]
    print(mis_model.evaluate(sol))
    print(mis_model.satisfy(sol))
    assert sol.chosen == sol_ws.chosen


if __name__ == "__main__":
    run_dip_solver_ws()
