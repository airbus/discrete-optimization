import matplotlib.pyplot as plt
import numpy as np
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.ls.hill_climber import (
    HillClimberPareto,
    ObjectiveHandling,
)
from discrete_optimization.generic_tools.ls.local_search import RestartHandlerLimit
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    ModeMutation,
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    PermutationMutationRCPSP,
    get_available_mutations,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ParetoFront,
    plot_pareto_2d,
    plot_storage_2d,
    result_storage_to_pareto_front,
)
from discrete_optimization.rcpsp.rcpsp_model import MultiModeRCPSPModel, RCPSPModel
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_utils import (
    plot_resource_individual_gantt,
    plot_ressource_view,
)


def local_search():
    # file_path = files_available[0]
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]  # Single mode RCPSP
    # files = [f for f in files if 'j1010_5.mm' in f]  # Multi mode RCPSP
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    dummy = rcpsp_model.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_model, dummy)
    print(mutations)
    list_mutation = [
        mutate[0].build(rcpsp_model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )

    objectives = ["makespan"]
    # objective_weights = [-1]
    objectives = ["mean_resource_reserve"]
    objective_weights = [-1]
    res = RestartHandlerLimit(
        200, cur_solution=dummy, cur_objective=rcpsp_model.evaluate(dummy)
    )
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=objectives,
        weights=objective_weights,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    sa = SimulatedAnnealing(
        evaluator=rcpsp_model,
        mutator=mixed_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(1, res, 0.9999),
        mode_mutation=ModeMutation.MUTATE,
        params_objective_function=params_objective_function,
        store_solution=False,
        nb_solutions=10,
    )
    import time

    import matplotlib.pyplot as plt

    t = time.time()
    store = sa.solve(dummy, nb_iteration_max=5000, pickle_result=False)
    print("Optim done in ", time.time() - t, " seconds ")
    plot_ressource_view(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=store.get_best_solution(),
        title_figure="best makespan",
    )
    plot_resource_individual_gantt(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=store.get_best_solution(),
        title_figure="best makespan",
    )
    plt.show()


def local_search_multimode():
    # file_path = files_available[0]
    files = get_data_available()
    files = [f for f in files if "j1010_1.mm" in f]  # Single mode RCPSP
    # files = [f for f in files if 'j1010_5.mm' in f]  # Multi mode RCPSP
    file_path = files[0]
    rcpsp_model: MultiModeRCPSPModel = parse_file(file_path)
    rcpsp_model.set_fixed_modes([1 for i in range(rcpsp_model.n_jobs)])
    dummy = rcpsp_model.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_model, dummy)
    print(mutations)
    list_mutation = [
        mutate[0].build(rcpsp_model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    # and mutate[1]["other_mutation"] == TwoOptMutation]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    objectives = ["makespan"]
    # objective_weights = [-1]
    # objectives = ['mean_resource_reserve']
    objective_weights = [-1]
    res = RestartHandlerLimit(
        200, cur_solution=dummy, cur_objective=rcpsp_model.evaluate(dummy)
    )
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=objectives,
        weights=objective_weights,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    sa = SimulatedAnnealing(
        evaluator=rcpsp_model,
        mutator=mixed_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(1, res, 0.9999),
        mode_mutation=ModeMutation.MUTATE,
        params_objective_function=params_objective_function,
        store_solution=False,
        nb_solutions=10,
    )
    import matplotlib.pyplot as plt

    store = sa.solve(dummy, nb_iteration_max=300, pickle_result=False)
    plot_ressource_view(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=store.best_solution,
        title_figure="best makespan",
    )
    plot_resource_individual_gantt(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=store.best_solution,
        title_figure="best makespan",
    )
    plt.show()


def local_search_multiobj():
    # file_path = files_available[0]
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]  # Single mode RCPSP
    # files = [f for f in files if 'j1010_5.mm' in f]  # Multi mode RCPSP
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    dummy = rcpsp_model.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_model, dummy)
    list_mutation = [
        mutate[0].build(rcpsp_model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    # and mutate[1]["other_mutation"] == TwoOptMutation]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(
        200, cur_solution=dummy, cur_objective=rcpsp_model.evaluate(dummy)
    )
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, 1]
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.MULTI_OBJ,
        objectives=objectives,
        weights=objective_weights,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    sa = HillClimberPareto(
        evaluator=rcpsp_model,
        mutator=mixed_mutation,
        restart_handler=res,
        params_objective_function=params_objective_function,
        mode_mutation=ModeMutation.MUTATE,
        store_solution=True,
        nb_solutions=10000,
    )
    result_sa = sa.solve(
        dummy, nb_iteration_max=2000, pickle_result=False, update_iteration_pareto=10000
    )
    pareto_store: ParetoFront = result_sa.result_storage
    fig, ax = plt.subplots(1)
    print("Pareto length : ", pareto_store.len_pareto_front())
    plot_storage_2d(result_storage=pareto_store, name_axis=objectives, ax=ax)
    plot_pareto_2d(pareto_store, name_axis=["makespan", "mean_resource_reserve"], ax=ax)
    extreme_points = pareto_store.compute_extreme_points()
    plot_ressource_view(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[0][0],
        title_figure="best makespan",
    )
    plot_resource_individual_gantt(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[0][0],
        title_figure="best makespan",
    )
    plot_ressource_view(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[1][0],
        title_figure="best availability",
    )
    plot_resource_individual_gantt(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[1][0],
        title_figure="best availability",
    )

    plt.show()
    # plot_resource_individual_gantt(rcpsp_model, sol)
    # plot_ressource_view(rcpsp_model, sol)
    # plt.show()


def local_search_postpro_multiobj():
    # file_path = files_available[0]
    files = get_data_available()
    files = [f for f in files if "j601_1.sm" in f]  # Single mode RCPSP
    # files = [f for f in files if 'j1010_5.mm' in f]  # Multi mode RCPSP
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    dummy = rcpsp_model.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_model, dummy)
    print(mutations)
    list_mutation = [
        mutate[0].build(rcpsp_model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    #  and mutate[1]["other_mutation"] == TwoOptMutation]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(
        500, cur_solution=dummy, cur_objective=rcpsp_model.evaluate(dummy)
    )
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, 100]
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=objectives,
        weights=objective_weights,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    sa = SimulatedAnnealing(
        evaluator=rcpsp_model,
        mutator=mixed_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(2, res, 0.9999),
        mode_mutation=ModeMutation.MUTATE,
        params_objective_function=params_objective_function,
        store_solution=True,
        nb_solutions=10000,
    )
    result_sa = sa.solve(dummy, nb_iteration_max=2000, pickle_result=False)
    store = result_sa.result_storage
    pareto_store = result_storage_to_pareto_front(
        result_storage=store, problem=rcpsp_model
    )
    print(len(store.list_solution_fits))
    print(pareto_store.len_pareto_front())
    fig, ax = plt.subplots(1)
    plot_storage_2d(result_storage=pareto_store, name_axis=objectives, ax=ax)
    plot_pareto_2d(pareto_front=pareto_store, name_axis=objectives, ax=ax)
    extreme_points = pareto_store.compute_extreme_points()
    plot_ressource_view(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[0][0],
        title_figure="best makespan",
    )
    plot_resource_individual_gantt(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[0][0],
        title_figure="best makespan",
    )
    plot_ressource_view(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[1][0],
        title_figure="best availability",
    )
    plot_resource_individual_gantt(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[1][0],
        title_figure="best availability",
    )
    plt.show()


def local_search_multiobj_multimode():
    # file_path = files_available[0]
    files = get_data_available()
    files = [f for f in files if "j1010_1.mm" in f]  # Single mode RCPSP
    # files = [f for f in files if 'j1010_5.mm' in f]  # Multi mode RCPSP
    file_path = files[0]
    rcpsp_model: MultiModeRCPSPModel = parse_file(file_path)
    rcpsp_model.set_fixed_modes([1 for i in range(rcpsp_model.n_jobs)])
    dummy = rcpsp_model.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_model, dummy)
    list_mutation = [
        mutate[0].build(rcpsp_model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    # and mutate[1]["other_mutation"] == TwoOptMutation]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(
        200, cur_solution=dummy, cur_objective=rcpsp_model.evaluate(dummy)
    )
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, 100]
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.MULTI_OBJ,
        objectives=objectives,
        weights=objective_weights,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    sa = HillClimberPareto(
        evaluator=rcpsp_model,
        mutator=mixed_mutation,
        restart_handler=res,
        params_objective_function=params_objective_function,
        mode_mutation=ModeMutation.MUTATE,
        store_solution=True,
        nb_solutions=10000,
    )
    result_sa = sa.solve(
        dummy,
        nb_iteration_max=10000,
        pickle_result=False,
        update_iteration_pareto=10000,
    )
    pareto_store: ParetoFront = result_sa.result_storage
    pareto_store.list_solution_fits = [
        l for l in pareto_store.list_solution_fits if l[0].rcpsp_schedule_feasible
    ]
    pareto_store = result_storage_to_pareto_front(pareto_store, rcpsp_model)
    fig, ax = plt.subplots(1)
    print("Pareto length : ", pareto_store.len_pareto_front())
    plot_storage_2d(result_storage=pareto_store, name_axis=objectives, ax=ax)
    plot_pareto_2d(pareto_store, name_axis=["makespan", "mean_resource_reserve"], ax=ax)
    fig.savefig("multimode_pareto.png")
    extreme_points = pareto_store.compute_extreme_points()
    plot_ressource_view(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[0][0],
        title_figure="best makespan",
    )
    plot_resource_individual_gantt(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[0][0],
        title_figure="best makespan",
    )
    plot_ressource_view(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[1][0],
        title_figure="best availability",
    )
    plot_resource_individual_gantt(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[1][0],
        title_figure="best availability",
    )

    plt.show()
    # plot_resource_individual_gantt(rcpsp_model, sol)
    # plot_ressource_view(rcpsp_model, sol)
    # plt.show()


def local_search_postpro_multiobj_multimode():
    # file_path = files_available[0]
    files = get_data_available()
    files = [f for f in files if "j1010_1.mm" in f]  # Single mode RCPSP
    # files = [f for f in files if 'j1010_5.mm' in f]  # Multi mode RCPSP
    file_path = files[0]
    rcpsp_model: MultiModeRCPSPModel = parse_file(file_path)
    rcpsp_model.set_fixed_modes([1 for i in range(rcpsp_model.n_jobs)])
    dummy = rcpsp_model.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_model, dummy)
    print(mutations)
    list_mutation = [
        mutate[0].build(rcpsp_model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    #  and mutate[1]["other_mutation"] == TwoOptMutation]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(
        500, cur_solution=dummy, cur_objective=rcpsp_model.evaluate(dummy)
    )
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, 100]
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=objectives,
        weights=objective_weights,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    sa = SimulatedAnnealing(
        evaluator=rcpsp_model,
        mutator=mixed_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(10, res, 0.99999),
        mode_mutation=ModeMutation.MUTATE,
        params_objective_function=params_objective_function,
        store_solution=True,
        nb_solutions=10000,
    )
    result_sa = sa.solve(dummy, nb_iteration_max=10000, pickle_result=False)
    result_sa.list_solution_fits = [
        l for l in result_sa.list_solution_fits if l[0].rcpsp_schedule_feasible
    ]
    pareto_store = result_storage_to_pareto_front(
        result_storage=result_sa, problem=rcpsp_model
    )
    print("Nb Pareto : ", pareto_store.len_pareto_front())
    fig, ax = plt.subplots(1)
    plot_storage_2d(result_storage=pareto_store, name_axis=objectives, ax=ax)
    plot_pareto_2d(pareto_front=pareto_store, name_axis=objectives, ax=ax)
    extreme_points = pareto_store.compute_extreme_points()
    plot_ressource_view(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[0][0],
        title_figure="best makespan",
    )
    plot_resource_individual_gantt(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[0][0],
        title_figure="best makespan",
    )
    plot_ressource_view(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[1][0],
        title_figure="best availability",
    )
    plot_resource_individual_gantt(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=extreme_points[1][0],
        title_figure="best availability",
    )
    plt.show()


if __name__ == "__main__":
    local_search()
