from discrete_optimization.rcpsp.rcpsp_parser import (
    files_available,
    get_data_available,
    parse_file,
    parse_results_file,
    results_available,
)


def parsing():
    # file_path = files_available[0]

    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    # files = [f for f in files if 'j1010_5.mm' in f]  # Multi mode RCPSP

    file_path = files[0]

    rcpsp_model = parse_file(file_path)
    print("Model:")
    print(rcpsp_model)
    print("Successors details:")
    print(rcpsp_model.successors)
    print("Mode details:")
    print(rcpsp_model.mode_details)
    print("Resource details:")
    print(rcpsp_model.resources)
    print("List of non-renewable resources:")
    print(rcpsp_model.non_renewable_resources)
    print("Horizon:")
    print(rcpsp_model.horizon)


def parsing_results():
    # file_path = files_available[0]
    files = [f for f in results_available]  # Single mode RCPSP
    # files = [f for f in files if 'j1010_5.mm' in f]  # Multi mode RCPSP
    for f in files:
        parse_results_file(f)


if __name__ == "__main__":
    parsing_results()
