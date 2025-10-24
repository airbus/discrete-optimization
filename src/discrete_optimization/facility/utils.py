#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import numpy as np

from discrete_optimization.facility.problem import FacilityProblem


def compute_matrix_distance_facility_problem(problem: FacilityProblem):
    matrix_distance = np.zeros((problem.customer_count, problem.facility_count))
    for k in range(problem.customer_count):
        for j in range(problem.facility_count):
            matrix_distance[k, j] = problem.evaluate_customer_facility(
                facility=problem.facilities[j], customer=problem.customers[k]
            )
    return matrix_distance
