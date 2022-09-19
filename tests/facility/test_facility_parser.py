#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.facility.facility_parser import (
    get_data_available,
    parse_file,
)


def test_parsing():
    file_path = get_data_available()[0]
    facility_model = parse_file(file_path)
    facility_model.facility_count
    facility_model.customer_count
    facility_model.facilities
    facility_model.customers


if __name__ == "__main__":
    test_parsing()
