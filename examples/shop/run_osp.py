import logging

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.shop.jsp.parser import get_data_available, parse_file
from discrete_optimization.shop.osp.problem import OpenShopProblem
from discrete_optimization.shop.osp.solvers.cpsat import CpSatOspSolver
from discrete_optimization.shop.osp.solvers.dp import DpOspSolver, dp

logging.basicConfig(level=logging.DEBUG)


def run_osp():
    problem = parse_file(get_data_available()[10])
    osp = OpenShopProblem(
        list_jobs=problem.list_jobs,
        n_jobs=problem.n_jobs,
        n_machines=problem.n_machines,
        horizon=problem.horizon,
    )
    solver_cpsat = CpSatOspSolver(osp)
    solver_cpsat.init_model(use_cpm_for_task_bounds=False, use_energy_constraints=False)
    res = solver_cpsat.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=20,
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
    )
    print(osp.evaluate(res[-1][0]), osp.satisfy(res[-1][0]))

    if False:
        solver = DpOspSolver(osp)
        solver.init_model()
        res = solver.solve(
            time_limit=20,
            callbacks=[
                # NbIterationStopper(nb_iteration_max=2),
                ProblemEvaluateLogger(
                    step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
                )
            ],
            solver=dp.LNBS,
            threads=6,
            retrieve_intermediate_solutions=True,
        )
    print(osp.evaluate(res[-1][0]), osp.satisfy(res[-1][0]))


if __name__ == "__main__":
    run_osp()
