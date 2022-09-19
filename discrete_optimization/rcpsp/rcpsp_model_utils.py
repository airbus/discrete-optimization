#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Union

import numpy as np

from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPModelCalendar


def create_fake_tasks(rcpsp_problem: Union[RCPSPModel, RCPSPModelCalendar]):
    if not rcpsp_problem.is_varying_resource():
        return []
    else:
        ressources_arrays = {
            r: np.array(rcpsp_problem.resources[r])
            for r in rcpsp_problem.resources_list
        }
        max_capacity = {r: np.max(ressources_arrays[r]) for r in ressources_arrays}
        fake_tasks = []
        for r in ressources_arrays:
            delta = ressources_arrays[r][:-1] - ressources_arrays[r][1:]
            index_non_zero = np.nonzero(delta)[0]
            if ressources_arrays[r][0] < max_capacity[r]:
                consume = {
                    r: int(max_capacity[r] - ressources_arrays[r][0]),
                    "duration": int(index_non_zero[0] + 1),
                    "start": 0,
                }
                fake_tasks += [consume]
            for j in range(len(index_non_zero) - 1):
                ind = index_non_zero[j]
                value = ressources_arrays[r][ind + 1]
                if value != max_capacity[r]:
                    consume = {
                        r: int(max_capacity[r] - value),
                        "duration": int(index_non_zero[j + 1] - ind),
                        "start": int(ind + 1),
                    }
                    fake_tasks += [consume]
        return fake_tasks
