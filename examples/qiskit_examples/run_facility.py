#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.facility.problem import (
    Customer,
    Facility,
    Facility2DProblem,
    Point,
)
from discrete_optimization.facility.solvers.quantum import (
    VqeFacilitySolver,
)


def facility_example():
    """
    We are using a quantum simulator here, these simulator can assume a very low number of variable,
    so we can use it only on very little example
    """

    f1 = Facility(0, 2, 5, Point(1, 1))
    f2 = Facility(1, 1, 2, Point(-1, -1))

    c1 = Customer(0, 2, Point(2, 2))
    c2 = Customer(1, 5, Point(0, -1))

    facilityProblem = Facility2DProblem(2, 2, [f1, f2], [c1, c2])
    facilitySolver = VqeFacilitySolver(facilityProblem)
    facilitySolver.init_model()
    kwargs = {"maxiter": 500}
    res = facilitySolver.solve(**kwargs)
    sol, _ = res.get_best_solution_fit()
    print(sol.facility_for_customers)
    print(facilityProblem.satisfy(sol))
