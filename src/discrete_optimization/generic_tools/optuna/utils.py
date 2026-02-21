#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Utilities for optuna."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.callbacks.optuna import OptunaCallback
from discrete_optimization.generic_tools.do_problem import Problem
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    Hyperparameter,
    TrialDropped,
)
from discrete_optimization.generic_tools.optuna.timed_percentile_pruner import (
    TimedPercentilePruner,
)

logger = logging.getLogger(__name__)


try:
    import optuna
except ImportError:
    logger.warning("You should install optuna to use this module.")
else:
    from optuna.pruners import BasePruner, MedianPruner
    from optuna.samplers import BaseSampler
    from optuna.storages import JournalFileStorage, JournalStorage
    from optuna.trial import Trial, TrialState


def drop_already_tried_hyperparameters(trial: Trial) -> None:
    """Fail the trial if using same hyperparameters as a previous one."""
    states_to_consider = (
        TrialState.FAIL,
        TrialState.COMPLETE,
        TrialState.WAITING,
        TrialState.PRUNED,
    )
    trials_to_consider = trial.study.get_trials(
        deepcopy=False,
        states=states_to_consider,
    )
    for t in reversed(trials_to_consider):
        if trial.params == t.params and trial:
            msg = "Trial with same hyperparameters as a previous trial: dropping it."
            trial.set_user_attr("Error", msg)
            raise TrialDropped(msg)


def generic_optuna_experiment_monoproblem(
    problem: Problem,
    solvers_to_test: list[type[SolverDO]],
    kwargs_fixed_by_solver: Optional[dict[type[SolverDO], dict[str, Any]]] = None,
    suggest_optuna_kwargs_by_name_by_solver: Optional[
        dict[type[SolverDO], dict[str, dict[str, Any]]]
    ] = None,
    additional_hyperparameters_by_solver: Optional[
        dict[type[SolverDO], list[Hyperparameter]]
    ] = None,
    n_trials: int = 150,
    check_satisfy: bool = True,
    computation_time_in_study: bool = True,
    study_basename: str = "study",
    create_another_study: bool = True,
    overwrite_study=False,
    storage_path: str = "./optuna-journal.log",
    sampler: Optional[BaseSampler] = None,
    pruner: Optional[BasePruner] = None,
    seed: Optional[int] = None,
    min_time_per_solver: int = 5,
    callbacks: Optional[list[Callback]] = None,
) -> optuna.Study:
    """Create and run an optuna study to tune solvers hyperparameters on a given problem.

    The optuna study will choose a solver and its hyperparameters in order to optimize the fitness
    on the given problem.

    Pruning is potentially done at each optimization step thanks to dedicated callback.
    This can be done
    - either according to the optimization step number (but this is meaningful only when considering
      a single solver or at least solvers of a same family so that comparing step number can be done),
    - or according to the elapsed time (which should be more meaningful when comparing several types of solvers).

    The optuna study can be monitored with optuna-dashboard with

        optuna-dashboard optuna-journal.log

    (or the relevant path set by `storage_path`)

    Args:
        problem: problem to consider
        solvers_to_test: list of solvers to consider
        kwargs_fixed_by_solver: fixed hyperparameters by solver.
            Can also be other parameters needed by solvers' __init__(), init_model(), and solve() methods
        suggest_optuna_kwargs_by_name_by_solver: kwargs_by_name passed to solvers' suggest_with_optuna().
            Useful to restrict or specify choices, step, high, ...
        additional_hyperparameters_by_solver: additional user-defined hyperparameters by solver, to be suggested by optuna
        n_trials: Number of trials to be run in the optuna study
        check_satisfy: Decide whether checking if solution found satisfies the problem. If not satisfying,
            we consider the trial as failed and prune it without reporting the value.
        computation_time_in_study: if `True` the intermediate reporting and pruning will be labelled according to elapsed time
            instead of solver internal iteration number.
        study_basename: Base name of the study generated.
            If `create_another_study` is True, a timestamp will be added to this base name.
        create_another_study: if `True` a timestamp prefix will be added to the study base name in order to avoid
            overwriting or continuing a previously created study.
            Should be False, if one wants to add trials to an existing study.
        overwrite_study: if True, any study with the same name as the one generated here will be deleted before starting the optuna study.
            Should be False, if one wants to add trials to an existing study.
        storage_path: path to the journal used by optuna used to log the study. Can be a NFS path to allow parallelized optuna studies.
        sampler: sampler used by the optuna study. If None, a TPESampler is used with the provided `seed`.
        pruner: pruner used by the optuna study.
            If None,
              - if computation_time_in_study is True: TimedPercentilePruner(percentile=50, n_warmup_steps=min_time_per_solver)
              - else: MedianPruner()
            is used
        seed: used to create the sampler if `sampler` is None. Should be set to an integer if one wants to ensure
            reproducible results.
        min_time_per_solver: if no pruner is defined, and computation_time_in_study is True,
            we wait for these many seconds before allowing pruning.
        callbacks: list of callbacks to plug in solvers' solve(). By default, use
            `ObjectiveLogger(step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO)`
            Moreover a `OptunaCallback` will be added to report intermediate values and prune accordingly.

    Returns:

    """
    # default parameters
    if kwargs_fixed_by_solver is None:
        kwargs_fixed_by_solver = defaultdict(dict)
    if suggest_optuna_kwargs_by_name_by_solver is None:
        suggest_optuna_kwargs_by_name_by_solver = defaultdict(dict)
    if additional_hyperparameters_by_solver is None:
        additional_hyperparameters_by_solver = defaultdict(list)
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=seed)
    if pruner is None:
        if computation_time_in_study:
            pruner = TimedPercentilePruner(  # intermediate values interpolated at same "step"
                percentile=50,  # median pruner
                n_warmup_steps=min_time_per_solver,  # no pruning during first seconds
            )
        else:
            pruner = MedianPruner()
    if callbacks is None:
        callbacks = [
            ObjectiveLogger(
                step_verbosity_level=logging.INFO,
                end_verbosity_level=logging.INFO,
            )
        ]

    elapsed_time_attr = "elapsed_time"

    # study name
    suffix = f"-{time.time()}" if create_another_study else ""
    study_name = f"{study_basename}{suffix}"

    # we need to map the classes to a unique string, to be seen as a categorical hyperparameter by optuna
    # by default, we use the class name, but if there are identical names, f"{cls.__module__}.{cls.__name__}" could be used.
    solvers_by_name: dict[str, type[SolverDO]] = {
        cls.__name__: cls for cls in solvers_to_test
    }

    # sense of optimization
    direction = problem.get_optuna_study_direction()

    # add new user-defined hyperparameters to the solvers
    for (
        solver_cls,
        additional_hyperparameters,
    ) in additional_hyperparameters_by_solver.items():
        solver_cls.hyperparameters = (
            list(solver_cls.hyperparameters) + additional_hyperparameters
        )

    # objective definition
    def objective(trial: Trial):
        # hyperparameters to test

        # first parameter: solver choice
        solver_name: str = trial.suggest_categorical("solver", choices=solvers_by_name)
        solver_class = solvers_by_name[solver_name]

        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            solver_class.suggest_hyperparameters_with_optuna(
                trial=trial,
                prefix=solver_name + ".",
                kwargs_by_name=suggest_optuna_kwargs_by_name_by_solver.get(
                    solver_class, None
                ),
                fixed_hyperparameters=kwargs_fixed_by_solver.get(solver_class, None),
            )
        )

        # drop trial if corresponding to a previous trial (it may happen that the sampler repropose same params)
        drop_already_tried_hyperparameters(trial)

        logger.info(f"Launching trial {trial.number} with parameters: {trial.params}")

        # construct kwargs for __init__, init_model, and solve
        kwargs = dict(suggested_hyperparameters_kwargs)  # copy
        if solver_class in kwargs_fixed_by_solver:
            kwargs.update(kwargs_fixed_by_solver[solver_class])

        try:
            # solver init
            solver = solver_class(problem=problem, **kwargs)
            solver.init_model(**kwargs)

            # init timer
            starting_time = time.perf_counter()

            # add optuna callbacks
            optuna_callback = OptunaCallback(
                trial=trial,
                starting_time=starting_time,
                elapsed_time_attr=elapsed_time_attr,
                report_time=computation_time_in_study,
                # report intermediate values according to elapsed time instead of iteration number?
            )
            callbacks_for_optuna = (
                callbacks + [optuna_callback] + kwargs.pop("callbacks", [])
            )

            # solve
            res = solver.solve(
                callbacks=callbacks_for_optuna,
                **kwargs,
            )

        except Exception as e:
            if isinstance(e, optuna.TrialPruned):
                raise e  # pruning error managed directly by optuna
            else:
                # Store exception message as trial user attribute
                msg = f"{e.__class__}: {e}"
                trial.set_user_attr("Error", msg)
                raise optuna.TrialPruned(msg)  # show failed

        # store elapsed time
        elapsed_time = time.perf_counter() - starting_time
        trial.set_user_attr(elapsed_time_attr, elapsed_time)

        # store result for this instance and report it as an intermediate value (=> dashboard + pruning)
        if check_satisfy:
            # accept only satisfying solutions
            try:
                sol, fit = res.get_best_solution_fit(satisfying=problem)
            except Exception as e:
                # Store exception message as trial user attribute
                msg = f"{e.__class__}: {e}"
                trial.set_user_attr("Error", msg)
                raise optuna.TrialPruned(msg)  # show failed
        else:
            # take the solution with best fit regardless of satifaction
            sol, fit = res.get_best_solution_fit()

        if fit is None:
            # no solution found
            if check_satisfy:
                msg = f"No solution found satisfying the problem."
            else:
                msg = f"No solution found."
            trial.set_user_attr("Error", msg)
            raise optuna.TrialPruned(msg)  # show failed

        return fit

    # create study + database to store it
    storage = JournalStorage(JournalFileStorage(storage_path))
    if overwrite_study:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except:
            pass
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=not overwrite_study,
    )
    study.optimize(objective, n_trials=n_trials, catch=TrialDropped)
    return study


def generic_optuna_experiment_multiproblem(
    problems: list[Problem],
    solvers_to_test: list[type[SolverDO]],
    kwargs_fixed_by_solver: Optional[dict[type[SolverDO], dict[str, Any]]] = None,
    suggest_optuna_kwargs_by_name_by_solver: Optional[
        dict[type[SolverDO], dict[str, dict[str, Any]]]
    ] = None,
    additional_hyperparameters_by_solver: Optional[
        dict[type[SolverDO], list[Hyperparameter]]
    ] = None,
    n_trials: int = 150,
    check_satisfy: bool = True,
    study_basename: str = "study",
    create_another_study: bool = True,
    overwrite_study=False,
    storage_path: str = "./optuna-journal.log",
    sampler: Optional[BaseSampler] = None,
    pruner: Optional[BasePruner] = None,
    seed: Optional[int] = None,
    prop_startup_instances: float = 0.2,
    randomize_instances: bool = True,
    report_cumulated_fitness: bool = False,
    callbacks: Optional[list[Callback]] = None,
) -> optuna.Study:
    """Create and run an optuna study to tune solvers hyperparameters on several instances of a problem.

    The optuna study will choose a solver and its hyperparameters in order to optimize the average fitness
    on given problem instances.

    Pruning is potentially made after each instance is solved based on how previous solvers performed
    on this same instance.

    The optuna study can be monitored with optuna-dashboard with

        optuna-dashboard optuna-journal.log

    (or the relevant path set by `storage_path`)

    Args:
        problems: list of problem instances to consider
        solvers_to_test: list of solvers to consider
        kwargs_fixed_by_solver: fixed hyperparameters by solver.
            Can also be other parameters needed by solvers' __init__(), init_model(), and solve() methods
        suggest_optuna_kwargs_by_name_by_solver: kwargs_by_name passed to solvers' suggest_with_optuna().
            Useful to restrict or specify choices, step, high, ...
        additional_hyperparameters_by_solver: additional user-defined hyperparameters by solver, to be suggested by optuna
        n_trials: Number of trials to be run in the optuna study
        check_satisfy: Decide whether checking if solution found satisfies the problem. If not satisfying,
            we consider the trial as failed and prune it without reporting the value.
        study_basename: Base name of the study generated.
            If `create_another_study` is True, a timestamp will be added to this base name.
        create_another_study: if `True` a timestamp prefix will be added to the study base name in order to avoid
            overwriting or continuing a previously created study.
            Should be False, if one wants to add trials to an existing study.
        overwrite_study: if True, any study with the same name as the one generated here will be deleted before starting the optuna study.
            Should be False, if one wants to add trials to an existing study.
        storage_path: path to the journal used by optuna used to log the study. Can be a NFS path to allow parallelized optuna studies.
        sampler: sampler used by the optuna study. If None, a TPESampler is used with the provided `seed`.
        pruner: pruner used by the optuna study. If None, a WilcoxonPruner is used.
        seed: used to create the sampler if `sampler` is None. Should be set to an integer if one wants to ensure
            reproducible results.
        prop_startup_instances: used if Pruner is None. Proportion of instances used to startup before allowing pruning.
        randomize_instances: whether randomizing instances order when running a trial.
            Should probably set to False if report_cumulated_fitness is set to True.
        report_cumulated_fitness: whether reporting cumulated fitness instead of individual fitness for each problem instance.
            Should be set to False when using WilcoxonPruner.
        callbacks: list of callbacks to plug in solvers' solve(). By default, use
            `ObjectiveLogger(step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO)`

    Returns:

    """
    # default parameters
    if kwargs_fixed_by_solver is None:
        kwargs_fixed_by_solver = defaultdict(dict)
    if suggest_optuna_kwargs_by_name_by_solver is None:
        suggest_optuna_kwargs_by_name_by_solver = defaultdict(dict)
    if additional_hyperparameters_by_solver is None:
        additional_hyperparameters_by_solver = defaultdict(list)
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=seed)
    if pruner is None:
        n_startup_steps = math.ceil(prop_startup_instances * len(problems))
        pruner = optuna.pruners.WilcoxonPruner(
            p_threshold=0.1, n_startup_steps=n_startup_steps
        )
    if callbacks is None:
        callbacks = [
            ObjectiveLogger(
                step_verbosity_level=logging.INFO,
                end_verbosity_level=logging.INFO,
            )
        ]

    # cumulative fitness + Wilcoxon = incompatible
    if isinstance(pruner, optuna.pruners.WilcoxonPruner) and report_cumulated_fitness:
        raise ValueError(
            "`report_cumulated_fitness` cannot be true when using a WilcoxonPruner."
        )

    # study name
    suffix = f"-{time.time()}" if create_another_study else ""
    study_name = f"{study_basename}{suffix}"

    # we need to map the classes to a unique string, to be seen as a categorical hyperparameter by optuna
    # by default, we use the class name, but if there are identical names, f"{cls.__module__}.{cls.__name__}" could be used.
    solvers_by_name: dict[str, type[SolverDO]] = {
        cls.__name__: cls for cls in solvers_to_test
    }

    # sense of optimization
    direction = problems[0].get_optuna_study_direction()

    # add new user-defined hyperparameters to the solvers
    for (
        solver_cls,
        additional_hyperparameters,
    ) in additional_hyperparameters_by_solver.items():
        solver_cls.hyperparameters = (
            list(solver_cls.hyperparameters) + additional_hyperparameters
        )

    # objective definition
    def objective(trial: Trial):
        # hyperparameters to test

        # first parameter: solver choice
        solver_name: str = trial.suggest_categorical("solver", choices=solvers_by_name)
        solver_class = solvers_by_name[solver_name]

        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            solver_class.suggest_hyperparameters_with_optuna(
                trial=trial,
                prefix=solver_name + ".",
                kwargs_by_name=suggest_optuna_kwargs_by_name_by_solver.get(
                    solver_class, None
                ),  # options to restrict the choices of some hyperparameter
                fixed_hyperparameters=kwargs_fixed_by_solver.get(solver_class, None),
            )
        )

        # drop trial if corresponding to a previous trial (it may happen that the sampler repropose same params)
        drop_already_tried_hyperparameters(trial)

        logger.info(f"Launching trial {trial.number} with parameters: {trial.params}")

        # construct kwargs for __init__, init_model, and solve
        kwargs = dict(suggested_hyperparameters_kwargs)  # copy
        if solver_class in kwargs_fixed_by_solver:
            kwargs.update(kwargs_fixed_by_solver[solver_class])

        # loop on problem instances
        fitnesses = []
        cumulated_fitness = 0.0
        i_cumulated_fitness = 0
        # For best results, shuffle the evaluation order in each trial.
        if randomize_instances:
            instance_ids = np.random.permutation(len(problems))
        else:
            instance_ids = list(range(len(problems)))
        for instance_id in instance_ids:
            instance_id = int(instance_id)  # convert np.int64 into python int
            problem = problems[instance_id]

            try:
                # solver init
                solver = solver_class(problem=problem, **kwargs)
                solver.init_model(**kwargs)

                callbacks_for_trial = callbacks + kwargs.pop("callbacks", [])

                # solve
                res = solver.solve(
                    callbacks=callbacks_for_trial,
                    **kwargs,
                )
            except Exception as e:
                # Store exception message as trial user attribute
                msg = f"{e.__class__}: {e}"
                trial.set_user_attr("Error", msg)
                trial.set_user_attr("pruned", True)
                raise optuna.TrialPruned(msg)  # show failed

            # store result for this instance and report it as an intermediate value (=> dashboard + pruning)
            if check_satisfy:
                # accept only satisfying solutions
                try:
                    sol, fit = res.get_best_solution_fit(satisfying=problem)
                except Exception as e:
                    # Store exception message as trial user attribute
                    msg = f"{e.__class__}: {e}"
                    trial.set_user_attr("pruned", True)
                    trial.set_user_attr("Error", msg)
                    raise optuna.TrialPruned(msg)  # show failed
            else:
                # take the solution with best fit regardless of satifaction
                sol, fit = res.get_best_solution_fit()

            if fit is None:
                # no solution found
                if check_satisfy:
                    msg = f"No solution found satisfying problem #{instance_id}."
                else:
                    msg = f"No solution found for problem #{instance_id}."
                trial.set_user_attr("pruned", True)
                trial.set_user_attr("Error", msg)
                raise optuna.TrialPruned(msg)  # show failed
            else:
                fitnesses.append(fit)
                if report_cumulated_fitness:
                    cumulated_fitness += fit
                    trial.report(cumulated_fitness, i_cumulated_fitness)
                    i_cumulated_fitness += 1
                else:
                    trial.report(fit, instance_id)
                current_average = sum(fitnesses) / len(fitnesses)
                trial.set_user_attr("current_fitness_average", current_average)
                if trial.should_prune():
                    # return current average instead of raising TrialPruned,
                    # else optuna dashboard thinks that last intermediate fitness is the value for the trial
                    trial.set_user_attr("pruned", True)
                    trial.set_user_attr(
                        "Error", f"Pruned by pruner at problem instance #{instance_id}."
                    )
                    return current_average

        trial.set_user_attr("pruned", False)
        return sum(fitnesses) / len(fitnesses)

    # create study + database to store it
    storage = JournalStorage(JournalFileStorage(storage_path))
    if overwrite_study:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except:
            pass
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=not overwrite_study,
    )
    study.optimize(objective, n_trials=n_trials, catch=TrialDropped)

    return study
