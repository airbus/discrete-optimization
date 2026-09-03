#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Define the usecases on which we want non-regression.

(We will store previous results and compare newer versions of solvers to them)

"""

from pytest_cases import parametrize


@parametrize("i_row", range(30))
def case_rcpsp(i_row, nonregression_db):
    row = nonregression_db.reset_index().iloc[i_row, :]
    return row.test_id, row.status, row.objective, row.mode_optim
