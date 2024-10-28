import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.maximum_independent_set.parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.solvers.asp import AspMisSolver
from discrete_optimization.maximum_independent_set.solvers.decomposition import (
    DecomposedMisSolver,
    MisProblem,
)
from discrete_optimization.maximum_independent_set.solvers.dp import (
    DpMisSolver,
    DpModeling,
)
from discrete_optimization.maximum_independent_set.solvers.gurobi import (
    GurobiQuadraticMisSolver,
)
from discrete_optimization.maximum_independent_set.solvers.toulbar import (
    ToulbarMisSolver,
)
from discrete_optimization.maximum_independent_set.solvers_map import (
    CpSatMisSolver,
    KamisMisSolver,
)

logging.basicConfig(level=logging.INFO)


def run_decomposition():
    small_example = [f for f in get_data_available() if "1zc.4096" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = DecomposedMisSolver(problem=mis_model)
    p = ParametersCp.default_cpsat()
    res = solver.solve(
        initial_solver=SubBrick(
            cls=CpSatMisSolver, kwargs={"parameters_cp": p, "time_limit": 10}
        ),
        root_solver=SubBrick(
            cls=KamisMisSolver, kwargs={"method": "redumis", "time_limit": 3}
        ),
        proportion_to_remove=0.5,
        nb_iteration=10000,
    )


def run_decomposition_ortools():
    small_example = [f for f in get_data_available() if "1zc.4096" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = DecomposedMisSolver(problem=mis_model)
    p = ParametersCp.default_cpsat()
    res = solver.solve(
        initial_solver=SubBrick(
            cls=CpSatMisSolver, kwargs={"parameters_cp": p, "time_limit": 5}
        ),
        root_solver=SubBrick(
            cls=CpSatMisSolver, kwargs={"parameters_cp": p, "time_limit": 5}
        ),
        proportion_to_remove=0.6,
        nb_iteration=10000,
    )


def run_decomposition_asp():
    small_example = [f for f in get_data_available() if "1zc.4096" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = DecomposedMisSolver(problem=mis_model)
    p = ParametersCp.default_cpsat()
    res = solver.solve(
        initial_solver=SubBrick(
            cls=CpSatMisSolver, kwargs={"parameters_cp": p, "time_limit": 5}
        ),
        root_solver=SubBrick(cls=AspMisSolver, kwargs={"time_limit": 10}),
        proportion_to_remove=0.5,
        nb_iteration=10000,
    )


def run_decomposition_toulbar():
    small_example = [f for f in get_data_available() if "1dc.2048" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = DecomposedMisSolver(problem=mis_model)
    p = ParametersCp.default_cpsat()
    res = solver.solve(
        initial_solver=SubBrick(
            cls=CpSatMisSolver, kwargs={"parameters_cp": p, "time_limit": 5}
        ),
        root_solver=SubBrick(
            cls=ToulbarMisSolver,
            kwargs={"time_limit": 10},
        ),
        proportion_to_remove=0.5,
        nb_iteration=10000,
    )


def run_decomposition_gurobi():
    small_example = [f for f in get_data_available() if "1dc.2048" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = DecomposedMisSolver(problem=mis_model)
    p = ParametersCp.default_cpsat()
    p_milp = ParametersMilp.default()
    res = solver.solve(
        initial_solver=SubBrick(
            cls=CpSatMisSolver, kwargs={"parameters_cp": p, "time_limit": 5}
        ),
        root_solver=SubBrick(
            cls=GurobiQuadraticMisSolver,
            kwargs={"parameters_milp": p_milp, "time_limit": 5},
        ),
        proportion_to_remove=0.2,
        nb_iteration=10000,
    )


def run_decomposition_dp():
    small_example = [f for f in get_data_available() if "1dc.1024" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = DecomposedMisSolver(problem=mis_model)
    res = solver.solve(
        initial_solver=SubBrick(
            cls=DpMisSolver,
            kwargs={"modeling": DpModeling.ORDER, "time_limit": 10, "quiet": True},
        ),
        root_solver=SubBrick(
            cls=DpMisSolver,
            kwargs={"modeling": DpModeling.ORDER, "time_limit": 3, "quiet": True},
        ),
        proportion_to_remove=0.4,
        nb_iteration=10000,
    )


if __name__ == "__main__":
    run_decomposition_dp()
