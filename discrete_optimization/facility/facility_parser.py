#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
from typing import Optional

from discrete_optimization.datasets import get_data_home
from discrete_optimization.facility.facility_model import (
    Customer,
    Facility,
    FacilityProblem,
    FacilityProblem2DPoints,
    Point,
)


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
):
    """Get datasets available for facility.

    Params:
        data_folder: folder where datasets for facility whould be find.
            If None, we look in "facility" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/facility"

    try:
        datasets = [
            os.path.abspath(os.path.join(data_folder, f))
            for f in os.listdir(data_folder)
        ]
    except FileNotFoundError:
        datasets = []
    return datasets


def parse(input_data):
    # parse the input
    lines = input_data.split("\n")
    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    facilities = []
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.append(
            Facility(
                i - 1,
                float(parts[0]),
                int(parts[1]),
                Point(float(parts[2]), float(parts[3])),
            )
        )
    customers = []
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.append(
            Customer(
                i - 1 - facility_count,
                int(parts[0]),
                Point(float(parts[1]), float(parts[2])),
            )
        )
    problem = FacilityProblem2DPoints(
        facility_count, customer_count, facilities, customers
    )
    return problem


def parse_file(file_path) -> FacilityProblem:
    with open(file_path, "r", encoding="utf-8") as input_data_file:
        input_data = input_data_file.read()
        facility_model = parse(input_data)
        return facility_model
