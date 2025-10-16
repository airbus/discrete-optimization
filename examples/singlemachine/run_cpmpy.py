#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cpmpy_tools import (
    CpmpyCorrectUnsatMethod,
    CpmpyExplainUnsatMethod,
)
from discrete_optimization.singlemachine.parser import get_data_available, parse_file
from discrete_optimization.singlemachine.solvers.cpmpy_solver import (
    CpmpySingleMachineSolver,
    SingleMachineModel,
)

logging.basicConfig(level=logging.INFO)


def run_cpmpy():
    problem = parse_file(get_data_available()[0])[0]
    # 2. Initialize the solver (using the CP model by default)
    solver_cp = CpmpySingleMachineSolver(problem, solver_name="ortools")
    # 3. Solve the problem
    solver_cp.init_model(model_type=SingleMachineModel.CP)
    from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger

    result_storage_cp = solver_cp.solve(
        time_limit=30,
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
    )
    best_solution_cp, best_fit_cp = result_storage_cp.get_best_solution_fit()
    print(f"CP Model - Best solution fit: {best_fit_cp}, {solver_cp.status_solver}")
    if best_solution_cp.schedule is not None:
        print(f"CP Model - Schedule: {best_solution_cp.schedule}")
    # --- You can also try the LP model ---
    solver_lp = CpmpySingleMachineSolver(problem, solver_name="gurobi")
    solver_lp.init_model(
        model_type=SingleMachineModel.LP,
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
    )
    result_storage_lp = solver_lp.solve(time_limit=30)
    best_solution_lp, best_fit_lp = result_storage_lp.get_best_solution_fit()
    print(f"LP Model - Best solution fit: {best_fit_lp}, {solver_lp.status_solver}")
    # --- Example of using explanation tools if a model is UNSAT ---
    # if solver_cp.status_solver == StatusSolver.UNSATISFIABLE:
    #     print("Model is unsatisfiable. Computing explanation...")
    #     # This will return the minimal set of soft constraints causing the issue
    #     explanation = solver_cp.explain_unsat_meta()
    #     print(f"Explanation (Minimal Unsatisfiable Set): {[e.name for e in explanation]}")


def run_explanation():
    problem = parse_file(get_data_available()[0])[0]
    solver_lp = CpmpySingleMachineSolver(problem, solver_name="ortools")
    solver_lp.init_model(
        model_type=SingleMachineModel.LP, add_impossible_constraints=True
    )
    result_storage_lp = solver_lp.solve(
        time_limit=1,
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
    )
    # best_solution_lp, best_fit_lp = result_storage_lp.get_best_solution_fit()
    # print(f"LP Model - Best solution fit: {best_fit_lp}, {solver_lp.status_solver}")
    soft_constraints = []
    soft_constraints.append(solver_lp.meta_constraints["impossible_deadline"])
    for i in range(problem.num_jobs):
        soft_constraints.append(solver_lp.meta_constraints[f"completion_time_{i}"])
    explanation = solver_lp.explain_unsat_meta(
        soft=soft_constraints,
        hard=solver_lp.get_hard_meta_constraints(),
        cpmpy_method=CpmpyExplainUnsatMethod.mus,
    )
    for meta in explanation:
        print("Conflict : ", meta.name)
        print(meta.constraints)
    print(explanation)
    # --- Example of using explanation tools if a model is UNSAT ---
    # if solver_cp.status_solver == StatusSolver.UNSATISFIABLE:
    #     print("Model is unsatisfiable. Computing explanation...")
    #     # This will return the minimal set of soft constraints causing the issue
    #     explanation = solver_cp.explain_unsat_meta()
    #     print(f"Explanation (Minimal Unsatisfiable Set): {[e.name for e in explanation]}")


def run_explanation_brut():
    problem = parse_file(get_data_available()[0])[0]
    solver_lp = CpmpySingleMachineSolver(problem, solver_name="exact")
    solver_lp.init_model(
        model_type=SingleMachineModel.LP, add_impossible_constraints=True
    )
    result_storage_lp = solver_lp.solve(
        time_limit=1,
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
    )
    best_solution_lp, best_fit_lp = result_storage_lp.get_best_solution_fit()
    print(f"LP Model - Best solution fit: {best_fit_lp}, {solver_lp.status_solver}")
    explanation = solver_lp.explain_unsat_fine(cpmpy_method=CpmpyExplainUnsatMethod.mus)
    for cstr in explanation:
        print("Conflict : ", cstr)
    # --- Example of using explanation tools if a model is UNSAT ---
    # if solver_cp.status_solver == StatusSolver.UNSATISFIABLE:
    #     print("Model is unsatisfiable. Computing explanation...")
    #     # This will return the minimal set of soft constraints causing the issue
    #     explanation = solver_cp.explain_unsat_meta()
    #     print(f"Explanation (Minimal Unsatisfiable Set): {[e.name for e in explanation]}")


if __name__ == "__main__":
    run_explanation()
