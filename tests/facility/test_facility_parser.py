from discrete_optimization.facility.facility_parser import (
    get_data_available,
    parse_file,
)


def parsing():
    file_path = get_data_available()[0]
    facility_model = parse_file(file_path)
    print(facility_model)
    print(
        facility_model.facility_count,
        facility_model.customer_count,
        facility_model.facilities,
        facility_model.customers,
    )


if __name__ == "__main__":
    parsing()
