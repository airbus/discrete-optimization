import logging

from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    ConstraintExtractorList,
    NbChangesAllocationConstraintExtractor,
    ParamsConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    TasksConstraintHandler,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.neighbor_tools import (
    NeighborBuilderMix,
    NeighborBuilderTaskThresholdTime,
    NeighborRandom,
)
from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import (
    InitialSolutionFromSolver,
    ReinitModelCallback,
)
from discrete_optimization.rcalbp_l.parser import get_data_available, parse_rcalbpl_json
from discrete_optimization.rcalbp_l.problem import plot_rcalbpl_dashboard
from discrete_optimization.rcalbp_l.solvers.cpsat import CpSatRCALBPLSolver
from discrete_optimization.rcalbp_l.solvers.meta_solvers import (
    BackwardSequentialRCALBPLSolver,
)

logging.basicConfig(level=logging.INFO)


def main():
    file = [f for f in get_data_available() if "187_2_26_2880.json" in f][0]
    problem = parse_rcalbpl_json(file)
    # problem.nb_periods = 5
    # problem.periods = range(problem.nb_periods)
    solver = CpSatRCALBPLSolver(problem)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    res = solver.solve(
        time_limit=300,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
        parameters_cp=p,
    )
    sol = res[-1][0]


def main_lns():
    file = [f for f in get_data_available() if "187_2_26_2880.json" in f][0]
    problem = parse_rcalbpl_json(file)
    p = ParametersCp.default_cpsat()
    p.nb_process = 8
    initial_solver = BackwardSequentialRCALBPLSolver(
        problem=problem,
        future_chunk_size=1,
        phase2_chunk_size=5,
        time_limit_phase1=20,
        time_limit_phase2=20,
        use_sgs_warm_start=True,
    )
    initial_solution_provider = InitialSolutionFromSolver(
        solver=initial_solver,
        params_objective_function=p,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
    )

    constraints_extractor = ConstraintExtractorList(
        extractors=[
            NbChangesAllocationConstraintExtractor(
                nb_changes_max=int(0.1 * len(problem.tasks_list))
            ),
        ]
    )
    constraint_handler = TasksConstraintHandler(
        problem=problem,
        neighbor_builder=NeighborBuilderMix(
            [
                NeighborRandom(problem=problem, fraction_subproblem=0.3),
                NeighborBuilderTaskThresholdTime(
                    problem=problem, threshold=problem.c_target
                ),
            ],
            [0.0, 1],
        ),
        params_constraint_extractor=ParamsConstraintExtractor(
            constraint_to_current_solution_makespan=False,
            margin_rel_to_current_solution_makespan=0.05,
            fix_primary_tasks_modes=False,
            fix_secondary_tasks_modes=False,
        ),
        constraints_extractor=constraints_extractor,
    )
    subsolver = CpSatRCALBPLSolver(problem=problem)
    subsolver.init_model(add_heuristic_constraint=False)
    solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
        initial_solution_provider=initial_solution_provider,
    )
    res = solver.solve(
        callbacks=[
            ReinitModelCallback(),
            WarmStartCallback(
                warm_start_best_solution=True, warm_start_last_solution=False
            ),
        ],
        nb_iteration_lns=10,
        time_limit_subsolver_iter0=20,
        time_limit_subsolver=100,
        parameters_cp=p,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
    )
    sol = res.get_best_solution()

    print("Satisfy : ", problem.satisfy(sol))
    print("Evaluate : ", problem.evaluate(sol))
    fig, slider = plot_rcalbpl_dashboard(problem, sol)


if __name__ == "__main__":
    main_lns()
