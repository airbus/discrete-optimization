#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random

import numpy as np
import pytest

from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.ls.hill_climber import HillClimberPareto
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
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_utils import (
    plot_resource_individual_gantt,
    plot_ressource_view,
)


@pytest.fixture()
def random_seed():
    random.seed(42)
    np.random.seed(42)


def test_local_search_sm(random_seed):
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    dummy = rcpsp_model.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_model, dummy)
    list_mutation = [
        mutate[0].build(rcpsp_model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )

    objectives = ["makespan"]
    objectives = ["mean_resource_reserve"]
    objective_weights = [-1]
    res = RestartHandlerLimit(200)
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=objectives,
        weights=objective_weights,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    sa = SimulatedAnnealing(
        problem=rcpsp_model,
        mutator=mixed_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(1, res, 0.9999),
        mode_mutation=ModeMutation.MUTATE,
        params_objective_function=params_objective_function,
        store_solution=False,
    )

    sol = sa.solve(dummy, nb_iteration_max=500).get_best_solution()
    assert rcpsp_model.satisfy(sol)
    plot_ressource_view(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=sol,
        title_figure="best makespan",
    )
    plot_resource_individual_gantt(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=sol,
        title_figure="best makespan",
    )


def test_local_search_mm(random_seed):
    # file_path = files_available[0]
    files = get_data_available()
    files = [f for f in files if "j1010_1.mm" in f]  # Multi mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)
    rcpsp_model.set_fixed_modes([1 for i in range(rcpsp_model.n_jobs)])
    dummy = rcpsp_model.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_model, dummy)
    list_mutation = [
        mutate[0].build(rcpsp_model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    objectives = ["makespan"]
    objective_weights = [-1]
    res = RestartHandlerLimit(200)
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=objectives,
        weights=objective_weights,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    sa = SimulatedAnnealing(
        problem=rcpsp_model,
        mutator=mixed_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(1, res, 0.9999),
        mode_mutation=ModeMutation.MUTATE,
        params_objective_function=params_objective_function,
        store_solution=False,
    )

    sol = sa.solve(dummy, nb_iteration_max=300).get_best_solution()
    assert rcpsp_model.satisfy(sol)
    plot_ressource_view(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=sol,
        title_figure="best makespan",
    )
    plot_resource_individual_gantt(
        rcpsp_model=rcpsp_model,
        rcpsp_sol=sol,
        title_figure="best makespan",
    )


def test_local_search_sm_multiobj(random_seed):
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    dummy = rcpsp_model.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_model, dummy)
    list_mutation = [
        mutate[0].build(rcpsp_model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(200)
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, 1]
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.MULTI_OBJ,
        objectives=objectives,
        weights=objective_weights,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    sa = HillClimberPareto(
        problem=rcpsp_model,
        mutator=mixed_mutation,
        restart_handler=res,
        params_objective_function=params_objective_function,
        mode_mutation=ModeMutation.MUTATE,
        store_solution=True,
    )
    pareto_store = sa.solve(dummy, nb_iteration_max=100, update_iteration_pareto=100)
    assert isinstance(pareto_store, ParetoFront)


def test_local_search_sm_postpro_multiobj(random_seed):
    files = get_data_available()
    files = [f for f in files if "j601_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    dummy = rcpsp_model.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_model, dummy)
    list_mutation = [
        mutate[0].build(rcpsp_model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(500)
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, 100]
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=objectives,
        weights=objective_weights,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    sa = SimulatedAnnealing(
        problem=rcpsp_model,
        mutator=mixed_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(2, res, 0.9999),
        mode_mutation=ModeMutation.MUTATE,
        params_objective_function=params_objective_function,
        store_solution=True,
    )
    store = sa.solve(dummy, nb_iteration_max=100)
    pareto_store = result_storage_to_pareto_front(
        result_storage=store, problem=rcpsp_model
    )
    assert isinstance(pareto_store, ParetoFront)
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


def test_local_search_mm_multiobj(random_seed):
    files = get_data_available()
    files = [f for f in files if "j1010_1.mm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)
    rcpsp_model.set_fixed_modes([1 for i in range(rcpsp_model.n_jobs)])
    dummy = rcpsp_model.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_model, dummy)
    list_mutation = [
        mutate[0].build(rcpsp_model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(200)
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, 100]
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.MULTI_OBJ,
        objectives=objectives,
        weights=objective_weights,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    sa = HillClimberPareto(
        problem=rcpsp_model,
        mutator=mixed_mutation,
        restart_handler=res,
        params_objective_function=params_objective_function,
        mode_mutation=ModeMutation.MUTATE,
        store_solution=True,
    )
    pareto_store = sa.solve(
        dummy,
        nb_iteration_max=500,
        update_iteration_pareto=100,
    )
    assert isinstance(pareto_store, ParetoFront)
    pareto_store.list_solution_fits = [
        l for l in pareto_store.list_solution_fits if l[0].rcpsp_schedule_feasible
    ]
    pareto_store = result_storage_to_pareto_front(pareto_store, rcpsp_model)
    plot_storage_2d(result_storage=pareto_store, name_axis=objectives)
    plot_pareto_2d(pareto_store, name_axis=["makespan", "mean_resource_reserve"])
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


def test_local_search_postpro_multiobj_multimode(random_seed):
    files = get_data_available()
    files = [f for f in files if "j1010_1.mm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)
    rcpsp_model.set_fixed_modes([1 for i in range(rcpsp_model.n_jobs)])
    dummy = rcpsp_model.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_model, dummy)
    list_mutation = [
        mutate[0].build(rcpsp_model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(500)
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, 100]
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=objectives,
        weights=objective_weights,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    sa = SimulatedAnnealing(
        problem=rcpsp_model,
        mutator=mixed_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(10, res, 0.99999),
        mode_mutation=ModeMutation.MUTATE,
        params_objective_function=params_objective_function,
        store_solution=True,
    )
    result_sa = sa.solve(dummy, nb_iteration_max=100)
    result_sa.list_solution_fits = [
        l for l in result_sa.list_solution_fits if l[0].rcpsp_schedule_feasible
    ]
    pareto_store = result_storage_to_pareto_front(
        result_storage=result_sa, problem=rcpsp_model
    )
