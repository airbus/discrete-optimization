import logging
import re
import time
from collections import defaultdict
from os.path import basename
from typing import Any, Dict, Type

import numpy as np
import optuna
from optuna.storages import JournalFileStorage, JournalStorage
from optuna.trial import Trial, TrialState

from discrete_optimization.generic_rcpsp_tools.large_neighborhood_search_scheduling import (
    LargeNeighborhoodSearchScheduling,
)
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.lp_tools import (
    MilpSolverName,
    ParametersMilp,
    gurobi_available,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_solvers import look_for_solver
from discrete_optimization.rcpsp.solver.rcpsp_lp_solver import LP_MRCPSP, LP_RCPSP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

seed = 42  # set this to an integer to get reproducible results, else to None
optuna_nb_trials = 100  # number of trials to launch
gurobi_full_license_available = False  # is the installed gurobi having a full license? (contrary to the license installed by `pip install gurobipy`)
create_another_study = True  # True: generate a study name with timestamp to avoid overwriting previous study, False: keep same study name
overwrite = False  # True: delete previous studies with same name (in particular, if create_another_study=False), False: keep the study and add trials to the existing ones
max_time_per_solver = 20  # max duration per solver (seconds)

problem_pattern = "j301_.*\.sm"
nb_problems = 10
study_basename = f"rcpsp_multiple_instances-{problem_pattern}-{nb_problems}"
problems_files = [
    f for f in get_data_available() if re.match(problem_pattern, basename(f))
]
if nb_problems > 0:
    problems_files = problems_files[:nb_problems]
problems = [parse_file(f) for f in problems_files]

solvers_to_test = look_for_solver(problems[0])

suffix = f"-{time.time()}" if create_another_study else ""
study_name = f"{study_basename}{suffix}"
storage_path = "./optuna-journal.log"  # NFS path for distributed optimization

parameters_cp = ParametersCP.default_cpsat()
parameters_cp.nb_process = 6
parameters_cp.time_limit = max_time_per_solver
parameters_milp = ParametersMilp.default()
parameters_milp.time_limit = max_time_per_solver

kwargs_fixed_by_solver: Dict[Type[SolverDO], Dict[str, Any]] = defaultdict(
    dict,  # default kwargs for unspecified solvers
    {
        LargeNeighborhoodSearchScheduling: dict(
            nb_iteration_lns=10, parameters_cp=parameters_cp
        ),
        LP_RCPSP: dict(parameters_milp=parameters_milp),
        LP_MRCPSP: dict(parameters_milp=parameters_milp),
    },
)

# restrict some hyperparameters choices, for some solvers (making use of `kwargs_by_name` of `suggest_with_optuna`)
suggest_optuna_kwargs_by_name_by_solver: Dict[
    Type[SolverDO], Dict[str, Dict[str, Any]]
] = defaultdict(
    dict,  # default kwargs_by_name for unspecified solvers
    {},
)
if not gurobi_available or not gurobi_full_license_available:
    # Remove possibility for gurobi if not available
    suggest_optuna_kwargs_by_name_by_solver[LP_RCPSP].update(
        {"lp_solver": dict(choices=[MilpSolverName.CBC])}
    )
    suggest_optuna_kwargs_by_name_by_solver[LP_MRCPSP].update(
        {"lp_solver": dict(choices=[MilpSolverName.CBC])}
    )

# we need to map the classes to a unique string, to be seen as a categorical hyperparameter by optuna
# by default, we use the class name, but if there are identical names, f"{cls.__module__}.{cls.__name__}" could be used.
solvers_by_name: Dict[str, Type[SolverDO]] = {
    cls.__name__: cls for cls in solvers_to_test
}

# sense of optimization
direction = problems[0].get_optuna_study_direction()

# objective definition
def objective(trial: Trial):
    # hyperparameters to test

    # first parameter: solver choice
    solver_name: str = trial.suggest_categorical("solver", choices=solvers_by_name)
    solver_class = solvers_by_name[solver_name]

    # hyperparameters for the chosen solver
    suggested_hyperparameters_kwargs = solver_class.suggest_hyperparameters_with_optuna(
        trial=trial,
        prefix=solver_name + ".",
        kwargs_by_name=suggest_optuna_kwargs_by_name_by_solver[
            solver_class
        ],  # options to restrict the choices of some hyperparameter
        fixed_hyperparameters=kwargs_fixed_by_solver[solver_class],
    )

    # use existing value if corresponding to a previous complete trial (it may happen that the sampler repropose same params)
    states_to_consider = (TrialState.COMPLETE,)
    trials_to_consider = trial.study.get_trials(
        deepcopy=False, states=states_to_consider
    )
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            msg = "Trial with same hyperparameters as a previous complete trial: returning previous fit."
            logger.warning(msg)
            trial.set_user_attr("Error", msg)
            trial.set_user_attr("pruned", True)
            return t.value

    # prune if corresponding to a previous failed trial
    states_to_consider = (TrialState.FAIL,)
    trials_to_consider = trial.study.get_trials(
        deepcopy=False, states=states_to_consider
    )
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            msg = "Pruning trial identical to a previous failed trial."
            trial.set_user_attr("Error", msg)
            trial.set_user_attr("pruned", True)
            raise optuna.TrialPruned(msg)

    logger.info(f"Launching trial {trial.number} with parameters: {trial.params}")

    # construct kwargs for __init__, init_model, and solve
    kwargs = dict(kwargs_fixed_by_solver[solver_class])  # copy the frozen kwargs dict
    kwargs.update(suggested_hyperparameters_kwargs)

    # loop on problem instances
    fitnesses = []
    # For best results, shuffle the evaluation order in each trial.
    instance_ids = np.random.permutation(len(problems))
    for instance_id in instance_ids:
        instance_id = int(instance_id)  # convert np.int64 into python int
        problem = problems[instance_id]

        try:
            # solver init
            if solver_class.__name__ == "GPHH":
                solver = solver_class(
                    problem=problem, training_domains=[problem], **kwargs
                )
            else:
                solver = solver_class(problem=problem, **kwargs)
            solver.init_model(**kwargs)

            # solve
            res = solver.solve(
                callbacks=[
                    ObjectiveLogger(
                        step_verbosity_level=logging.INFO,
                        end_verbosity_level=logging.INFO,
                    ),
                ],
                **kwargs,
            )
        except Exception as e:
            # Store exception message as trial user attribute
            msg = f"{e.__class__}: {e}"
            trial.set_user_attr("Error", msg)
            trial.set_user_attr("pruned", True)
            raise optuna.TrialPruned(msg)  # show failed

        # store result for this instance and report it as an intermediate value (=> dashboard + pruning)
        if len(res.list_solution_fits) != 0:
            _, fit = res.get_best_solution_fit()
            fitnesses.append(fit)
            trial.report(fit, instance_id)
            current_average = sum(fitnesses) / len(fitnesses)
            trial.set_user_attr("current_fitness_average", current_average)
            if trial.should_prune():
                # return current average instead of raising TrialPruned,
                # else optuna dashboard thinks that last intermediate fitness is the value for the trial
                trial.set_user_attr("pruned", True)
                trial.set_user_attr("Error", "Pruned by pruner.")
                return current_average

        else:
            trial.set_user_attr("pruned", True)
            msg = f"No solution found for problem #{instance_id}."
            trial.set_user_attr("Error", msg)
            raise optuna.TrialPruned(msg)  # show failed

    trial.set_user_attr("pruned", False)
    return sum(fitnesses) / len(fitnesses)


# create study + database to store it
storage = JournalStorage(JournalFileStorage(storage_path))
if overwrite:
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
    except:
        pass
study = optuna.create_study(
    study_name=study_name,
    direction=direction,
    sampler=optuna.samplers.TPESampler(seed=seed),
    pruner=optuna.pruners.WilcoxonPruner(p_threshold=0.1),
    storage=storage,
    load_if_exists=not overwrite,
)
study.optimize(objective, n_trials=optuna_nb_trials)
