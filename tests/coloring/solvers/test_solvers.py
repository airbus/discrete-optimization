#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import random
import sys

import numpy as np
import pytest
from minizinc.solver import Solver
from ortools.math_opt.python import mathopt

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.problem import (
    ColoringConstraints,
    ColoringProblem,
    ColoringSolution,
    transform_coloring_problem,
)
from discrete_optimization.coloring.solvers.asp import AspColoringSolver
from discrete_optimization.coloring.solvers.cp_mzn import (
    CpColoringModel,
    CpColoringSolver,
)
from discrete_optimization.coloring.solvers.cpmpy import CpmpyColoringSolver
from discrete_optimization.coloring.solvers.cpsat import (
    CpSatColoringSolver,
    ModelingCpSat,
)
from discrete_optimization.coloring.solvers.greedy import (
    GreedyColoringSolver,
    NxGreedyColoringMethod,
)
from discrete_optimization.coloring.solvers.lp import MathOptColoringSolver
from discrete_optimization.coloring.solvers_map import (
    GurobiColoringSolver,
    solve,
    solvers_map,
)
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.callbacks.loggers import (
    NbIterationTracker,
    ObjectiveLogger,
)
from discrete_optimization.generic_tools.cp_tools import (
    CpSolverName,
    ParametersCp,
    find_right_minizinc_solver_name,
)
from discrete_optimization.generic_tools.cpmpy_tools import MetaCpmpyConstraint
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
    TypeAttribute,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.ea.ga import DeapMutation, Ga
from discrete_optimization.generic_tools.ea.nsga import Nsga
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    plot_storage_2d,
)
from discrete_optimization.generic_tools.unsat_tools import MetaConstraint

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


try:
    Solver.lookup(find_right_minizinc_solver_name(CpSolverName.ORTOOLS))
except LookupError:
    mzn_ortools_available = False
else:
    mzn_ortools_available = True

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MyCallbackNok(Callback):
    def on_step_end(self, step: int, res, solver):
        raise RuntimeError("Explicit crash")


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


@pytest.mark.parametrize("coloring_problem_file", get_data_available())
def test_load_file(coloring_problem_file):
    coloring_problem: ColoringProblem = parse_file(coloring_problem_file)
    dummy_solution = coloring_problem.get_dummy_solution()
    assert coloring_problem.satisfy(dummy_solution)


@pytest.mark.parametrize("solver_class", solvers_map)
def test_solvers(solver_class):
    if solver_class == GurobiColoringSolver and not gurobi_available:
        pytest.skip("You need Gurobi to test this solver.")

    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    coloring_problem: ColoringProblem = parse_file(small_example)
    results = solve(
        method=solver_class, problem=coloring_problem, **solvers_map[solver_class][1]
    )
    sol, fit = results.get_best_solution_fit()


@pytest.mark.parametrize("solver_class", solvers_map)
def test_solvers_subset(solver_class):
    if solver_class == GurobiColoringSolver and not gurobi_available:
        pytest.skip("You need Gurobi to test this solver.")

    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    coloring_problem: ColoringProblem = parse_file(small_example)
    coloring_problem = transform_coloring_problem(
        coloring_problem,
        subset_nodes=set(range(10)),
        constraints_coloring=ColoringConstraints(color_constraint={0: 0, 1: 1, 2: 2}),
    )
    assert coloring_problem.graph is not None
    assert coloring_problem.number_of_nodes is not None
    assert coloring_problem.graph.nodes_name is not None

    results = solve(
        method=solver_class, problem=coloring_problem, **solvers_map[solver_class][1]
    )
    sol, fit = results.get_best_solution_fit()
    print(f"Solver {solver_class}, fitness = {fit}")
    print(f"Evaluation : {coloring_problem.evaluate(sol)}")


def test_mzn_solver_cb(caplog):
    small_example = [f for f in get_data_available() if "gc_50_9" in f][0]
    coloring_problem: ColoringProblem = parse_file(small_example)
    parameters_cp = ParametersCp.default()
    kwargs = {
        "cp_solver_name": CpSolverName.CHUFFED,
        "cp_model": CpColoringModel.DEFAULT,
        "parameters_cp": parameters_cp,
        "time_limit": 10,
        "greedy_start": False,
    }
    stopper = NbIterationStopper(nb_iteration_max=2)
    callbacks = [
        ObjectiveLogger(),
        stopper,
    ]
    solver = CpColoringSolver(problem=coloring_problem, **kwargs)
    solver.init_model(**kwargs)
    with caplog.at_level(logging.INFO):
        res = solver.solve(callbacks=callbacks, **kwargs)

    assert len(res) == min(stopper.nb_iteration_max, 3)
    assert f"Solve finished after {len(res)} iterations" in caplog.text


@pytest.mark.skipif(
    not mzn_ortools_available, reason="Needs ortools configured for minizinc."
)
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Hangs for ever on windows.")
def test_mzn_solver_ortools_parallel_cb(caplog):
    small_example = [f for f in get_data_available() if "gc_50_9" in f][0]
    coloring_problem: ColoringProblem = parse_file(small_example)
    parameters_cp = ParametersCp.default()
    parameters_cp.multiprocess = True
    parameters_cp.nb_process = 4
    kwargs = {
        "cp_solver_name": CpSolverName.ORTOOLS,
        "cp_model": CpColoringModel.DEFAULT,
        "parameters_cp": parameters_cp,
        "time_limit": 10,
        "greedy_start": False,
    }
    callbacks = [
        ObjectiveLogger(),
    ]
    solver = CpColoringSolver(problem=coloring_problem, **kwargs)
    solver.init_model(**kwargs)
    with caplog.at_level(logging.DEBUG):
        res = solver.solve(callbacks=callbacks, **kwargs)

    caploglines = caplog.text.splitlines()
    nb_callbacks_steps = len([line for line in caploglines if "Iteration #" in line])
    assert len(res) == nb_callbacks_steps
    assert f"Solve finished after {len(res)} iterations" in caplog.text


@pytest.mark.parametrize("modeling", [ModelingCpSat.BINARY, ModelingCpSat.INTEGER])
def test_cpsat_solver(modeling):
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(small_example)
    solver = CpSatColoringSolver(color_problem)
    solver.init_model(nb_colors=20, modeling=modeling)
    p = ParametersCp.default()
    result_store = solver.solve(parameters_cp=p)
    solution, fit = result_store.get_best_solution_fit()
    assert color_problem.satisfy(solution)

    # test warm start
    start_solution = solver.get_starting_solution()

    # first solution is not start_solution
    assert result_store[0][0].colors != start_solution.colors

    # warm start at first solution
    solver.set_warm_start(start_solution)
    # force first solution to be the hinted one
    result_store = solver.solve(
        parameters_cp=p,
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
    )
    assert result_store[0][0].colors == start_solution.colors


def test_cpsat_solver_internal_bound_and_objective():
    small_example = [f for f in get_data_available() if "gc_50_1" in f][0]
    color_problem = parse_file(small_example)
    solver = CpSatColoringSolver(color_problem)
    # timeout => bound and obj is None (not a single solution)
    result_store = solver.solve(time_limit=1e-5)
    assert solver.get_current_best_internal_objective_bound() is None
    assert solver.get_current_best_internal_objective_value() is None
    # not optimal => obj>bound
    result_store = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])
    assert solver.status_solver == StatusSolver.SATISFIED
    assert 0 < solver.get_current_best_internal_objective_bound()
    assert (
        solver.get_current_best_internal_objective_bound()
        < solver.get_current_best_internal_objective_value()
    )
    # optimal => obj == bound
    result_store = solver.solve()
    assert solver.status_solver == StatusSolver.OPTIMAL
    assert (
        solver.get_current_best_internal_objective_bound()
        == solver.get_current_best_internal_objective_value()
    )
    # infeasible => None
    solver.cp_model.add(solver.variables["nbc"] <= 1)
    res = solver.solve()
    assert solver.get_current_best_internal_objective_bound() is None
    assert solver.get_current_best_internal_objective_value() is None


def test_cpsat_solver_finetuned():
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(small_example)
    solver = CpSatColoringSolver(color_problem)
    solver.init_model(nb_colors=20)
    p = ParametersCp.default()

    # must use existing attribute name for ortools CpSolver
    with pytest.raises(AttributeError):
        result_store = solver.solve(
            parameters_cp=p, ortools_cpsat_solver_kwargs=dict(toto=4)
        )
    # must use correct value
    with pytest.raises(ValueError):
        result_store = solver.solve(
            parameters_cp=p, ortools_cpsat_solver_kwargs=dict(search_branching=-4)
        )
    # works
    from ortools.sat.sat_parameters_pb2 import SatParameters

    result_store = solver.solve(
        parameters_cp=p,
        ortools_cpsat_solver_kwargs=dict(
            search_branching=SatParameters.PSEUDO_COST_SEARCH
        ),
    )

    solution, fit = result_store.get_best_solution_fit()
    assert color_problem.satisfy(solution)


def test_asp_solver():
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(small_example)
    solver = AspColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(max_models=50, nb_colors=20)
    result_store = solver.solve(time_limit=5)
    solution, fit = result_store.get_best_solution_fit()
    assert color_problem.satisfy(solution)


def test_asp_solver_cb_log():
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(small_example)
    solver = AspColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(max_models=50, nb_colors=20)
    tracker = NbIterationTracker()
    callbacks = [tracker]
    result_store = solver.solve(time_limit=5, callbacks=callbacks)
    solution, fit = result_store.get_best_solution_fit()
    assert color_problem.satisfy(solution)
    assert tracker.nb_iteration > 1


def test_asp_solver_cb_stop():
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(small_example)
    solver = AspColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(max_models=50, nb_colors=20)
    stopper = NbIterationStopper(nb_iteration_max=1)
    callbacks = [stopper]
    result_store = solver.solve(time_limit=5, callbacks=callbacks)
    solution, fit = result_store.get_best_solution_fit()
    assert color_problem.satisfy(solution)
    assert stopper.nb_iteration == 1


def test_asp_solver_cb_exception():
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(small_example)
    solver = AspColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(max_models=50, nb_colors=20)
    with pytest.raises(RuntimeError, match="Explicit crash"):
        solver.solve(time_limit=5, callbacks=[MyCallbackNok()])


def test_model_satisfy():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)
    dummy_solution = color_problem.get_dummy_solution()
    assert color_problem.satisfy(dummy_solution)
    color_problem.evaluate(dummy_solution)
    bad_solution = ColoringSolution(color_problem, [1] * color_problem.number_of_nodes)
    color_problem.evaluate(bad_solution)
    assert not color_problem.satisfy(bad_solution)


def test_greedy_coloring():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = GreedyColoringSolver(color_problem, params_objective_function=None)
    result_store = solver.solve(strategy=NxGreedyColoringMethod.connected_sequential)
    solution = result_store.get_best_solution_fit()[0]
    assert color_problem.satisfy(solution)


def test_greedy_best_coloring():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = GreedyColoringSolver(color_problem, params_objective_function=None)
    result_store = solver.solve(strategy=NxGreedyColoringMethod.best)
    solution = result_store.get_best_solution_fit()[0]
    assert color_problem.satisfy(solution)


def test_ga_coloring_1(random_seed):
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)
    ga_solver = Ga(
        color_problem,
        encoding="colors_from0",
        mutation=DeapMutation.MUT_UNIFORM_INT,
        objectives=["nb_colors"],
        objective_weights=[-1],
        max_evals=5000,
    )
    color_sol = ga_solver.solve().get_best_solution()
    color_problem.evaluate(color_sol)
    assert color_problem.satisfy(color_sol)


def test_ga_coloring_2(random_seed):
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)
    kwargs = dict(
        problem=color_problem,
        encoding="colors",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["nb_colors", "nb_violations"],
        objective_weights=[-1, -2],
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=5000,
    )
    ga_solver = Ga(**kwargs)
    color_sol, fit = ga_solver.solve().get_best_solution_fit()
    color_problem.evaluate(color_sol)
    assert color_problem.satisfy(color_sol)

    # test warm start
    start_solution, start_fit = (
        GreedyColoringSolver(color_problem, params_objective_function=None)
        .solve()
        .get_best_solution_fit()
    )

    ga_solver = Ga(**kwargs)
    ga_solver.set_warm_start(start_solution)
    result_store = ga_solver.solve()
    color_sol, fit2 = result_store.get_best_solution_fit()
    color_problem.evaluate(color_sol)
    assert color_problem.satisfy(color_sol)

    # previous solution was worse than the start fit used now
    assert fit < start_fit

    # new solution can only improve the fitness
    assert fit2 >= start_fit


def test_ga_coloring_3(random_seed):
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)

    encoding = {
        "name": "colors",
        "type": [TypeAttribute.LIST_INTEGER],
        "n": 70,
        "arity": 10,
    }

    ga_solver = Ga(
        color_problem,
        encoding=encoding,
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["nb_colors", "nb_violations"],
        objective_weights=[-1, -2],
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=5000,
    )
    color_sol = ga_solver.solve().get_best_solution()
    color_problem.evaluate(color_sol)
    assert color_problem.satisfy(color_sol)


def test_coloring_nsga_1():

    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)

    objectives = ["nb_colors", "nb_violations"]
    objectives_weights = [-1, -1]
    params_objective_function = ParamsObjectiveFunction(
        objectives=objectives,
        objective_handling=ObjectiveHandling.AGGREGATE,
        weights=objectives_weights,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    kwargs = dict(
        problem=color_problem,
        encoding="colors",
        objectives=objectives,
        objective_weights=objectives_weights,
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=3000,
    )
    ga_solver = Nsga(**kwargs)

    result_storage = ga_solver.solve()
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)
    color_sol, fit = result_storage.get_best_solution_fit()

    # test warm start
    start_solution, start_fit = (
        GreedyColoringSolver(
            color_problem, params_objective_function=params_objective_function
        )
        .solve()
        .get_best_solution_fit()
    )
    start_fit = ga_solver.aggreg_from_sol(start_solution)

    ga_solver = Nsga(**kwargs)
    ga_solver.set_warm_start(start_solution)
    result_store = ga_solver.solve()
    color_sol, fit2 = result_store.get_best_solution_fit()

    print(start_fit.vector_fitness, fit.vector_fitness, fit2.vector_fitness)

    # previous solution was worse than the start fit used now
    assert fit < start_fit

    # new solution can only improve the fitness
    assert fit2 >= start_fit


def test_coloring_nsga_2():

    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)

    encoding = {
        "name": "colors",
        "type": [TypeAttribute.LIST_INTEGER],
        "n": 70,
        "arity": 10,
    }

    objectives = ["nb_colors", "nb_violations"]
    ga_solver = Nsga(
        color_problem,
        encoding=encoding,
        objectives=objectives,
        objective_weights=[-1, -1],
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=3000,
    )

    result_storage = ga_solver.solve()
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
@pytest.mark.parametrize("use_cliques", [False, True])
@pytest.mark.parametrize("greedy_start", [True, False])
def test_color_lp_gurobi(use_cliques, greedy_start):
    file = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(file)
    solver = GurobiColoringSolver(
        color_problem,
        params_objective_function=get_default_objective_setup(color_problem),
    )
    kwargs = dict(
        use_cliques=use_cliques,
        greedy_start=greedy_start,
        parameters_milp=ParametersMilp.default(),
    )
    result_store = solver.solve(**kwargs)
    solution = result_store.get_best_solution_fit()[0]
    assert color_problem.satisfy(solution)

    # Test warm-start only once
    if greedy_start and not use_cliques:

        # first solution is not start_solution
        assert result_store[0][0].colors != solver.start_solution.colors

        # warm start => first solution is start_solution
        solver.set_warm_start(solver.start_solution)
        result_store = solver.solve(**kwargs)
        assert result_store[0][0].colors == solver.start_solution.colors


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_color_lp_gurobi_explain_unsat():
    file = [f for f in get_data_available() if "gc_50_1" in f][0]
    color_problem = parse_file(file)
    solver = GurobiColoringSolver(
        color_problem,
    )
    solver.solve()
    assert solver.status_solver == StatusSolver.OPTIMAL

    # add impossible constraint "improve optimal bound"
    obj = solver.model.getObjective()
    extra_constraint = solver.add_linear_constraint(obj <= obj.getValue() / 2)
    extra_meta_constraint = MetaConstraint(name="extra", constraints=[extra_constraint])
    solver.model.update()

    # re-solve => unsatisfiable
    solver.solve()
    assert solver.status_solver == StatusSolver.UNSATISFIABLE

    # explain it
    constraints = solver.explain_unsat_fine()
    assert len(constraints) > 0

    # explain it via meta-constraints
    mus_meta_constraints = solver.explain_unsat_meta()
    assert len(mus_meta_constraints) > 0
    assert len(mus_meta_constraints) < len(solver.get_meta_constraints())
    assert isinstance(mus_meta_constraints[0], MetaConstraint)

    # custom meta-constraints
    mus_meta_constraints_extra = solver.explain_unsat_meta(
        meta_constraints=solver.get_meta_constraints() + [extra_meta_constraint]
    )
    assert mus_meta_constraints == [
        meta for meta in mus_meta_constraints_extra if meta != extra_meta_constraint
    ]


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_color_lp_gurobi_cb_log():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = GurobiColoringSolver(
        color_problem,
        params_objective_function=get_default_objective_setup(color_problem),
    )
    tracker = NbIterationTracker()
    callbacks = [tracker]
    result_store = solver.solve(
        parameters_milp=ParametersMilp.default(), callbacks=callbacks
    )
    solution = result_store.get_best_solution_fit()[0]
    assert len(result_store) > 1
    # check tracker called at each solution found
    assert tracker.nb_iteration == len(result_store)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_color_lp_gurobi_cb_stop():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = GurobiColoringSolver(
        color_problem,
        params_objective_function=get_default_objective_setup(color_problem),
    )
    stopper = NbIterationStopper(nb_iteration_max=1)
    callbacks = [stopper]
    result_store = solver.solve(
        parameters_milp=ParametersMilp.default(), callbacks=callbacks
    )
    # check stop after 1st iteration
    assert len(result_store) == 1


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_color_lp_gurobi_current_obj_not_inf():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = GurobiColoringSolver(
        color_problem,
    )
    stopper = NbIterationStopper(nb_iteration_max=1)
    callbacks = [stopper]
    result_store = solver.solve(
        parameters_milp=ParametersMilp.default(), callbacks=callbacks
    )
    # check stop after 1st iteration
    assert len(result_store) == 1
    assert solver.get_current_best_internal_objective_value() < gurobipy.GRB.INFINITY


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="On windows guthub runner, it finds a solution despite the time limit.",
)
def test_color_lp_gurobi_current_obj_timeout():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = GurobiColoringSolver(
        color_problem,
    )
    result_store = solver.solve(time_limit=1e-5)
    assert solver.get_current_best_internal_objective_value() == float("inf")
    assert solver.get_current_best_internal_objective_bound() == -float("inf")


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_color_lp_gurobi_current_obj_infeasible():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = GurobiColoringSolver(
        color_problem,
    )
    solver.init_model()
    obj = solver.model.getObjective()
    solver.add_linear_constraint(obj <= 1)
    solver.model.update()
    solver.solve()
    assert solver.status_solver == StatusSolver.UNSATISFIABLE
    assert solver.get_current_best_internal_objective_value() is None
    assert solver.get_current_best_internal_objective_bound() == float("inf")


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_color_lp_gurobi_current_bnd_optimal():
    file = [f for f in get_data_available() if "gc_50_1" in f][0]
    color_problem = parse_file(file)
    solver = GurobiColoringSolver(
        color_problem,
    )
    result_store = solver.solve()
    # check stop after 1st iteration
    assert solver.status_solver == StatusSolver.OPTIMAL
    assert (
        solver.get_current_best_internal_objective_value()
        == solver.get_current_best_internal_objective_bound()
    )


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_color_lp_gurobi_cb_exception():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = GurobiColoringSolver(
        color_problem,
        params_objective_function=get_default_objective_setup(color_problem),
    )
    with pytest.raises(RuntimeError, match="Explicit crash"):
        solver.solve(
            parameters_milp=ParametersMilp.default(), callbacks=[MyCallbackNok()]
        )


@pytest.mark.parametrize("use_cliques", [False, True])
@pytest.mark.parametrize("greedy_start", [True, False])
@pytest.mark.parametrize(
    "solver_type", [mathopt.SolverType.CP_SAT, mathopt.SolverType.HIGHS]
)
def test_color_lp_ortools_mathopt(use_cliques, greedy_start, solver_type):
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = MathOptColoringSolver(
        color_problem,
        params_objective_function=get_default_objective_setup(color_problem),
    )
    kwargs = dict(
        use_cliques=use_cliques,
        greedy_start=greedy_start,
        parameters_milp=ParametersMilp.default(),
        mathopt_solver_type=solver_type,
    )
    result_store = solver.solve(**kwargs)
    solution = result_store.get_best_solution_fit()[0]
    assert color_problem.satisfy(solution)

    # Test warm-start only once (and not for HiGHS as only 1 solution found)
    if greedy_start and not use_cliques and solver_type != mathopt.SolverType.HIGHS:
        # first solution is not start_solution
        assert result_store[0][0].colors != solver.start_solution.colors

        # warm start => first solution is start_solution
        solver.set_warm_start(solver.start_solution)
        result_store = solver.solve(**kwargs)
        assert result_store[0][0].colors == solver.start_solution.colors


def test_color_lp_ortools_mathopt_cb_log():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = MathOptColoringSolver(
        color_problem,
        params_objective_function=get_default_objective_setup(color_problem),
    )
    tracker = NbIterationTracker()
    callbacks = [tracker]
    result_store = solver.solve(
        parameters_milp=ParametersMilp.default(), callbacks=callbacks
    )
    solution = result_store.get_best_solution_fit()[0]
    assert len(result_store) > 1
    # check tracker called at each solution found
    assert tracker.nb_iteration == len(result_store)


def test_color_lp_ortools_mathopt_stop():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = MathOptColoringSolver(
        color_problem,
        params_objective_function=get_default_objective_setup(color_problem),
    )
    stopper = NbIterationStopper(nb_iteration_max=1)
    callbacks = [stopper]
    result_store = solver.solve(
        parameters_milp=ParametersMilp.default(), callbacks=callbacks
    )
    # check stop after 1st iteration
    assert len(result_store) == 1


def test_color_lp_ortools_mathopt_optimal_bnd():
    file = [f for f in get_data_available() if "gc_50_1" in f][0]
    color_problem = parse_file(file)
    solver = MathOptColoringSolver(
        color_problem,
    )
    result_store = solver.solve()
    # check obj and bound
    assert solver.status_solver == StatusSolver.OPTIMAL
    assert solver.get_current_best_internal_objective_value() is not None
    assert solver.get_current_best_internal_objective_bound() is not None
    assert (
        solver.get_current_best_internal_objective_value()
        == solver.get_current_best_internal_objective_bound()
    )


def test_color_lp_ortools_mathopt_current_obj_timeout():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = MathOptColoringSolver(
        color_problem,
    )
    solver.solve(time_limit=1e-5)
    assert solver.get_current_best_internal_objective_value() == float("inf")
    assert solver.get_current_best_internal_objective_bound() == -float("inf")


def test_color_lp_ortools_mathopt_current_obj_infeasible():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = MathOptColoringSolver(
        color_problem,
    )
    solver.init_model()
    obj = solver.model.objective.as_linear_expression()
    solver.add_linear_constraint(obj <= 1)
    solver.solve()
    assert solver.status_solver == StatusSolver.UNSATISFIABLE
    assert solver.get_current_best_internal_objective_value() == float("inf")
    assert solver.get_current_best_internal_objective_bound() == float("inf")


def test_color_lp_ortools_mathopt_cb_exception():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = MathOptColoringSolver(
        color_problem,
        params_objective_function=get_default_objective_setup(color_problem),
    )
    with pytest.raises(RuntimeError, match="Explicit crash"):
        solver.solve(
            parameters_milp=ParametersMilp.default(), callbacks=[MyCallbackNok()]
        )


def test_cpmpy_solver():
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(small_example)
    solver = CpmpyColoringSolver(color_problem)
    solver.init_model(nb_colors=20)
    result_store = solver.solve()
    solution, fit = result_store.get_best_solution_fit()
    assert color_problem.satisfy(solution)
    assert solver.status_solver == StatusSolver.OPTIMAL
    nb_colors_optimal = solution.nb_color

    # add impossible constraint (nb_colors < optimal)
    solver = CpmpyColoringSolver(color_problem)
    solver.init_model(nb_colors=nb_colors_optimal - 1)
    solver.solve()
    assert solver.status_solver == StatusSolver.UNSATISFIABLE

    # explain (fine and meta)
    assert len(solver.explain_unsat_fine()) < len(solver.get_soft_constraints())
    metaconstraints_mus = solver.explain_unsat_meta()
    assert len(metaconstraints_mus) < len(solver.get_soft_meta_constraints())
    for meta in metaconstraints_mus:
        assert isinstance(meta, MetaCpmpyConstraint)

    # correct meta
    meta_mcs = solver.correct_unsat_meta()
    assert 0 < len(meta_mcs) < len(solver.get_soft_meta_constraints())
    subconstraints_mcs_ids = set()
    for meta in meta_mcs:
        subconstraints_mcs_ids.update({id(c) for c in meta.constraints})
    solver.model.constraints = [
        c for c in solver.model.constraints if id(c) not in subconstraints_mcs_ids
    ]  # NB: solver.model.constraints.remove(cstr) not working as expected
    solver.solve()
    assert solver.status_solver == StatusSolver.OPTIMAL

    # tranform meta-constraints into nodes
    meta_mcs_nodes = solver.convert_metaconstraints2nodes(meta_mcs)
    assert len(meta_mcs_nodes) == len(meta_mcs)
    for node in meta_mcs_nodes:
        assert node in color_problem.graph.nodes_name

    # tranform meta-constraints into edges
    constraints = meta_mcs[0]
    edges = solver.convert_constraints2edges(constraints)
    assert len(constraints) == len(edges)
    for edge in edges:
        assert edge[0] in color_problem.graph.nodes_name
        assert edge[1] in color_problem.graph.nodes_name


if __name__ == "__main__":
    test_solvers()
