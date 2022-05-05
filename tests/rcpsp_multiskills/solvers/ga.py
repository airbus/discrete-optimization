from typing import Dict, List, Set

from discrete_optimization.generic_tools.do_problem import ObjectiveHandling
from discrete_optimization.generic_tools.ea.alternating_ga import AlternatingGa
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
    Ga,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    Employee,
    MS_RCPSPModel,
    MS_RCPSPSolution,
    MS_RCPSPSolution_Variant,
    SkillDetail,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.solvers.lp_model import (
    LP_Solver_MRSCPSP,
    MilpSolverName,
    ParametersMilp,
)

from tests.rcpsp_multiskills.rcpsp_multiskills_runs import (
    create_toy_msrcpsp,
    create_toy_msrcpsp_variant,
)


def alternating_ga_specific_mode_arrity():
    msrcpsp_model = create_toy_msrcpsp_variant()
    files = [f for f in get_data_available() if "100_5_64_9.def" in f]
    msrcpsp_model, new_tame_to_original_task_id = parse_file(files[0], max_horizon=2000)
    msrcpsp_model = msrcpsp_model.to_variant_model()

    total_evals = 50
    number_of_meta_iterations = 1
    evals_per_ga_runs_perm = 0.33 * (total_evals / number_of_meta_iterations)
    evals_per_ga_runs_modes = 0.33 * (
        total_evals / number_of_meta_iterations
    )  # total_evals/(2*number_of_meta_iterations)
    evals_per_ga_runs_resource_perm = 0.34 * (total_evals / number_of_meta_iterations)

    mode_mutation = DeapMutation.MUT_UNIFORM_INT
    task_permutation_mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    resource_permutation_mutation = (
        DeapMutation.MUT_SHUFFLE_INDEXES
    )  # TODO: Using adjacent swap would make more sense for this encoding

    # Initialise the task permutation that will be used to first search through the modes
    initial_task_permutation = [i for i in range(msrcpsp_model.n_jobs_non_dummy)]
    msrcpsp_model.set_fixed_task_permutation(initial_task_permutation)

    # Initialise the resource permutation that will be used to first search through the modes
    initial_resource_permutation = [
        i for i in range(len(msrcpsp_model.tasks) * len(msrcpsp_model.employees.keys()))
    ]
    msrcpsp_model.set_fixed_priority_worker_per_task_from_permutation(
        initial_resource_permutation
    )

    # initial_resource_priorities = [[w for w in msrcpsp_model.employees] for i in range(msrcpsp_model.n_jobs_non_dummy)]
    # msrcpsp_model.set_fixed_priority_worker_per_task(initial_resource_priorities)

    for it in range(number_of_meta_iterations):
        # Run a GA for evals_per_ga_runs evals on modes
        ga_solver = Ga(
            msrcpsp_model,
            encoding="modes_arrity_fix_from_0",
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["makespan"],
            objective_weights=[-1],
            mutation=mode_mutation,
            max_evals=evals_per_ga_runs_modes,
        )
        tmp_sol = ga_solver.solve().get_best_solution()
        print("best after modes search iteration: ", msrcpsp_model.evaluate(tmp_sol))
        # Fix the resulting modes
        print("tmp_sol.modes_vector:", tmp_sol.modes_vector)
        msrcpsp_model.set_fixed_modes(tmp_sol.modes_vector)

        # Run a GA for evals_per_ga_runs evals on permutation
        ga_solver = Ga(
            msrcpsp_model,
            encoding="priority_list_task",
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["makespan"],
            objective_weights=[-1],
            mutation=task_permutation_mutation,
            max_evals=evals_per_ga_runs_perm,
        )
        tmp_sol = ga_solver.solve().get_best_solution()
        print(
            "best after task permutation search iteration: ",
            msrcpsp_model.evaluate(tmp_sol),
        )

        # Fix the resulting permutation
        msrcpsp_model.set_fixed_task_permutation(tmp_sol.priority_list_task)

        # Run a GA for evals_per_ga_runs evals on permutation resource
        ga_solver = Ga(
            msrcpsp_model,
            encoding="priority_worker_per_task_perm",
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["makespan"],
            objective_weights=[-1],
            mutation=resource_permutation_mutation,
            max_evals=evals_per_ga_runs_resource_perm,
        )
        tmp_sol = ga_solver.solve().get_best_solution()
        print(
            "best after resource permutation search iteration: ",
            msrcpsp_model.evaluate(tmp_sol),
        )

        # Fix the resulting permutation
        msrcpsp_model.set_fixed_priority_worker_per_task(
            tmp_sol.priority_worker_per_task
        )

    sol = tmp_sol
    print(sol)
    fitnesses = msrcpsp_model.evaluate(sol)
    print("fitnesses: ", fitnesses)
    print("satisfy : ", msrcpsp_model.satisfy(sol))


def alternating_ga_specific_mode_arrity_single_solver():
    msrcpsp_model = create_toy_msrcpsp_variant()

    total_evals = 1000

    sub_evals = [50, 50, 50]

    ga_solver = AlternatingGa(
        msrcpsp_model,
        encodings=[
            "modes_arrity_fix_from_0",
            "priority_list_task",
            "priority_worker_per_task_perm",
        ],
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["makespan"],
        objective_weights=[-1],
        mutations=[
            DeapMutation.MUT_UNIFORM_INT,
            DeapMutation.MUT_SHUFFLE_INDEXES,
            DeapMutation.MUT_SHUFFLE_INDEXES,
        ],
        crossovers=[
            DeapCrossover.CX_ONE_POINT,
            DeapCrossover.CX_PARTIALY_MATCHED,
            DeapCrossover.CX_PARTIALY_MATCHED,
        ],
        max_evals=total_evals,
        sub_evals=sub_evals,
    )

    tmp_sol = ga_solver.solve().get_best_solution()
    print("best at end of alternating GA: ", msrcpsp_model.evaluate(tmp_sol))
    print("satisfy : ", msrcpsp_model.satisfy(tmp_sol))


if __name__ == "__main__":
    # alternating_ga_specific_mode_arrity()
    # test_alternating_ga_specific_mode_arrity()
    alternating_ga_specific_mode_arrity_single_solver()
