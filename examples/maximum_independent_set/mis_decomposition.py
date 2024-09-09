import logging

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.maximum_independent_set.mis_parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.mis_solvers import (
    MisKamisSolver,
    MisOrtoolsSolver,
    ParametersCP,
)
from discrete_optimization.maximum_independent_set.solvers.mis_asp import MisASPSolver
from discrete_optimization.maximum_independent_set.solvers.mis_decomposition import (
    MisDecomposedSolver,
    MisNetworkXSolver,
    MisProblem,
)
from discrete_optimization.maximum_independent_set.solvers.mis_gurobi import (
    MisMilpSolver,
    MisQuadraticSolver,
)
from discrete_optimization.maximum_independent_set.solvers.mis_toulbar import (
    MisToulbarSolver,
)

logging.basicConfig(level=logging.INFO)


def run_decomposition():
    small_example = [f for f in get_data_available() if "1zc.4096" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = MisDecomposedSolver(problem=mis_model)
    p = ParametersCP.default_cpsat()
    res = solver.solve(
        initial_solver=SubBrick(
            cls=MisOrtoolsSolver, kwargs={"parameters_cp": p, "time_limit": 10}
        ),
        root_solver=SubBrick(
            cls=MisKamisSolver, kwargs={"method": "redumis", "time_limit": 3}
        ),
        proportion_to_remove=0.5,
        nb_iteration=10000,
    )


def run_decomposition_ortools():
    small_example = [f for f in get_data_available() if "1zc.4096" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = MisDecomposedSolver(problem=mis_model)
    p = ParametersCP.default_cpsat()
    res = solver.solve(
        initial_solver=SubBrick(
            cls=MisOrtoolsSolver, kwargs={"parameters_cp": p, "time_limit": 5}
        ),
        root_solver=SubBrick(
            cls=MisOrtoolsSolver, kwargs={"parameters_cp": p, "time_limit": 5}
        ),
        proportion_to_remove=0.6,
        nb_iteration=10000,
    )


def run_decomposition_asp():
    small_example = [f for f in get_data_available() if "1zc.4096" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = MisDecomposedSolver(problem=mis_model)
    p = ParametersCP.default_cpsat()
    res = solver.solve(
        initial_solver=SubBrick(
            cls=MisOrtoolsSolver, kwargs={"parameters_cp": p, "time_limit": 5}
        ),
        root_solver=SubBrick(cls=MisASPSolver, kwargs={"time_limit": 10}),
        proportion_to_remove=0.5,
        nb_iteration=10000,
    )


def run_decomposition_toulbar():
    small_example = [f for f in get_data_available() if "1dc.1024" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = MisDecomposedSolver(problem=mis_model)
    p = ParametersCP.default_cpsat()
    res = solver.solve(
        initial_solver=SubBrick(
            cls=MisOrtoolsSolver, kwargs={"parameters_cp": p, "time_limit": 5}
        ),
        root_solver=SubBrick(
            cls=MisToulbarSolver,
            kwargs={"time_limit": 10},
        ),
        proportion_to_remove=0.5,
        nb_iteration=10000,
    )


def run_decomposition_gurobi():
    small_example = [f for f in get_data_available() if "1dc.2048" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = MisDecomposedSolver(problem=mis_model)
    p = ParametersCP.default_cpsat()
    p_milp = ParametersMilp.default()
    res = solver.solve(
        initial_solver=SubBrick(
            cls=MisOrtoolsSolver, kwargs={"parameters_cp": p, "time_limit": 5}
        ),
        root_solver=SubBrick(
            cls=MisQuadraticSolver,
            kwargs={"parameters_milp": p_milp, "time_limit": 5},
        ),
        proportion_to_remove=0.2,
        nb_iteration=10000,
    )


if __name__ == "__main__":
    run_decomposition_gurobi()
