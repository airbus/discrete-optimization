from discrete_optimization.rcpsp.rcpsp_model import (
    MultiModeRCPSPModel,
    RCPSPModel,
    RCPSPSolution,
    SingleModeRCPSPModel,
)
from discrete_optimization.rcpsp.rcpsp_parser import (
    files_available,
    get_data_available,
    parse_file,
)


def test_single_mode_random_permutation_solution():
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)
    # Create solution (mode = 1 for each task, identity permutation)
    print(rcpsp_model.n_jobs)
    permutation = [i for i in range(rcpsp_model.n_jobs_non_dummy)]
    permutation = [
        25,
        0,
        19,
        29,
        21,
        27,
        18,
        15,
        28,
        14,
        26,
        3,
        17,
        9,
        24,
        16,
        13,
        8,
        1,
        6,
        10,
        20,
        7,
        11,
        4,
        2,
        5,
        22,
        12,
        23,
    ]
    print(permutation)
    mode_list = [1 for i in range(rcpsp_model.n_jobs_non_dummy)]
    print(mode_list)
    rcpsp_sol = RCPSPSolution(
        problem=rcpsp_model, rcpsp_permutation=permutation, rcpsp_modes=mode_list
    )
    print("schedule feasible: ", rcpsp_sol.rcpsp_schedule_feasible)
    print("schedule: ", rcpsp_sol.rcpsp_schedule)
    print("rcpsp_modes:", rcpsp_sol.rcpsp_modes)
    fitnesses = rcpsp_model.evaluate(rcpsp_sol)
    print("fitnesses: ", fitnesses)
    resource_consumption = rcpsp_model.compute_resource_consumption(rcpsp_sol)
    print("resource_consumption: ", resource_consumption)
    print("mean_resource_reserve:", rcpsp_sol.compute_mean_resource_reserve())


def test_non_existing_modes_solution():

    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    permutation = [i for i in range(rcpsp_model.n_jobs_non_dummy)]
    mode_list = [2 for i in range(rcpsp_model.n_jobs_non_dummy)]
    print(mode_list)
    rcpsp_sol = RCPSPSolution(
        problem=rcpsp_model, rcpsp_permutation=permutation, rcpsp_modes=mode_list
    )
    print("schedule feasible: ", rcpsp_sol.rcpsp_schedule_feasible)
    print("schedule: ", rcpsp_sol.rcpsp_schedule)
    print("rcpsp_modes:", rcpsp_sol.rcpsp_modes)


def test_unfeasible_modes_solution():
    files = get_data_available()
    files = [f for f in files if "j1010_5.mm" in f]  # Multi-mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    permutation = [i for i in range(rcpsp_model.n_jobs_non_dummy)]
    unfeasible_modes = [2, 1, 1, 1, 3, 1, 2, 1, 1, 3]
    rcpsp_sol = RCPSPSolution(
        problem=rcpsp_model, rcpsp_permutation=permutation, rcpsp_modes=unfeasible_modes
    )
    print("schedule feasible: ", rcpsp_sol.rcpsp_schedule_feasible)
    print("schedule: ", rcpsp_sol.rcpsp_schedule)
    print("rcpsp_modes:", rcpsp_sol.rcpsp_modes)
    print(rcpsp_model.satisfy(rcpsp_sol))


def test_feasible_modes_solution():
    files = get_data_available()
    files = [f for f in files if "j1010_1.mm" in f]  # Multi-mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    permutation = [i for i in range(rcpsp_model.n_jobs_non_dummy)]
    feasible_modes = [1, 1, 3, 3, 2, 2, 3, 3, 3, 2]

    rcpsp_sol = RCPSPSolution(
        problem=rcpsp_model, rcpsp_permutation=permutation, rcpsp_modes=feasible_modes
    )
    print("schedule feasible: ", rcpsp_sol.rcpsp_schedule_feasible)
    print("schedule: ", rcpsp_sol.rcpsp_schedule)
    print("rcpsp_modes:", rcpsp_sol.rcpsp_modes)

    print(rcpsp_model.satisfy(rcpsp_sol))

    print("mean_resource_reserve:", rcpsp_sol.compute_mean_resource_reserve())


if __name__ == "__main__":
    # test_single_mode_random_permutation_solution()
    # test_unfeasible_modes_solution()
    test_single_mode_random_permutation_solution()
