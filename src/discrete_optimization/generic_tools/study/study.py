#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any, Optional

import pandas as pd

from discrete_optimization.generic_tools.do_problem import Problem
from discrete_optimization.generic_tools.do_solver import SolverDO, StatusSolver
from discrete_optimization.generic_tools.study import Hdf5Database
from discrete_optimization.generic_tools.study.config import ConfigStore
from discrete_optimization.generic_tools.study.database import (
    IS_EMPTY,
    is_empty_metrics,
)
from discrete_optimization.generic_tools.study.experiment import (
    CONFIG,
    INSTANCE,
    STATUS,
    Experiment,
    SolverConfig,
    SolverJsonableConfig,
)

logger = logging.getLogger(__name__)


class Study(Iterable[tuple[Problem, SolverDO, dict[str, Any]]]):
    """Small wrapper to manage d-o studies

    - loop over instance x config
    - manage mapping instance_name <-> instance, config_name <-> config
    - allow warmstart from previous run, with skip of successful experiments
    - automatic retry of unsuccessful experiments

    Manage database to store experiment.
    For now, use Hdf5Database with default path beigin <study_name>.h5

    To loop over a study, do:

    ```python
    for problem, solver, solver_kwargs in study:
        ...
        # solve
        solver.solve(..., **solver_kwargs)

        # retrieve metrics
        metrics = ...

        study.store_current_xp(metrics, ...)
    ```

    """

    def __init__(
        self,
        name: str,
        instances: list[str],
        solver_configs: dict[str, SolverConfig],
        problem_factory: Callable[[str], Problem],
        overwrite: bool = False,
        max_retry: int = 0,
        database_filepath: Optional[str] = None,
        solver_factory: Optional[Callable[[Problem, SolverConfig], SolverDO]] = None,
    ):
        """

        Args:
            name: study name
            instances: list of instance names (as they are to appear in database/dashboard)
            solver_configs: mapping a config name to an actual solver config
            problem_factory: mapping an instance name to a d-o problem instance
            overwrite: if True, the previous database is erased. Else we warmstart with previous database to avoid relaunching successful experiments
            max_retry: when a experiment is unsuccessful (potentially from a previous stufy run) we retry it until reaching max_retry.
                Default to 0, which means no retry.
            database_filepath: hdf5 database file path, default to <name>.h5
            solver_factory: mapping a problem and solver config to a solver.
                Default to calling solver_config.cls __init__ + init_model with solver_config.kwargs.

        """
        self.max_retry = max_retry
        self.overwrite = overwrite
        self.name = name
        self.instances = instances
        self.solver_configs = solver_configs
        self.problem_factory = problem_factory
        if database_filepath is None:
            self.database_filepath = f"{self.name}.h5"
        else:
            self.database_filepath = database_filepath
        if solver_factory is None:

            def solver_factory(
                problem: Problem, solver_config: SolverConfig
            ) -> SolverDO:
                solver = solver_config.cls(problem, **solver_config.kwargs)
                solver.init_model(**solver_config.kwargs)
                return solver

        self.solver_factory = solver_factory

        self.config_store = ConfigStore()
        self.config_instance_success: dict[tuple[str, str], bool] = defaultdict(
            lambda: False
        )
        self.config_instance_n_attempts: dict[tuple[str, str], int] = defaultdict(
            lambda: 0
        )

        if self.overwrite:
            # erase previous database
            try:
                os.remove(self.database_filepath)
            except FileNotFoundError:
                pass
        else:
            # load success and n_attempts from previous database
            self.load_metadatas()

        self._in_the_loop = False

    def __iter__(self):
        self._in_the_loop = True
        # loop over instances x configs
        for i_instance, instance in enumerate(self.instances):
            self._instance = instance
            for i_config, (config_name, solver_config) in enumerate(
                self.solver_configs.items()
            ):
                self._solver_config = solver_config
                logging.info(
                    f"###### Instance {i_instance + 1}/{len(self.instances)}: {instance}, config {i_config + 1}/{len(self.solver_configs)}: {config_name} ######"
                )
                if not self.overwrite:
                    # check if config x instance already done
                    config = SolverJsonableConfig.from_solver_config(
                        solver_config, name=config_name
                    ).as_nested_dict()
                    self.config_store.add(config)
                    config_name_normalized = self.config_store.get_name(config)
                    key = (instance, config_name_normalized)
                    if self.config_instance_success[key]:
                        logging.info(
                            "config x instance already run successfully in a previous attempt. Skipping it."
                        )
                        continue
                    if self.config_instance_n_attempts[key] > self.max_retry:
                        logging.info(
                            f"config x instance already run (unsuccessfully) {self.config_instance_n_attempts[key]} times. Skipping it."
                        )
                        continue
                    n_attempts = self.config_instance_n_attempts[key]
                    self._success = self.config_instance_success[key]
                    self._config_name = config_name_normalized
                else:
                    n_attempts = 0
                    self._success = False
                    self._config_name = config_name

                # retry it if failing
                while not self._success and n_attempts <= self.max_retry:
                    n_attempts += 1
                    if self.max_retry > 0:
                        logging.info(
                            f"###### Attempt {n_attempts} / {self.max_retry + 1}"
                        )
                    try:
                        problem = self.problem_factory(instance)
                        solver = self.solver_factory(problem, solver_config)
                        yield problem, solver, solver_config.kwargs

                    except Exception as e:
                        # failed experiment from problem/solver instantiation
                        metrics = pd.DataFrame([])
                        status = StatusSolver.ERROR
                        logging.error(e)
                        reason = f"{type(e)}: {str(e)}"
                        success = False
                        self.store_current_xp(
                            metrics=metrics,
                            status=status,
                            reason=reason,
                            success=success,
                        )

    def store_current_xp(
        self,
        metrics: pd.DataFrame,
        status: str | StatusSolver,
        reason: str = "",
        success: Optional[bool] = None,
    ):
        """

        Args:
            metrics:
            status:
            reason:
            success:

        Returns:

        """
        if not self._in_the_loop:
            raise RuntimeError(
                "This method should be called inside a loop over the study:\n"
                "`for problem, solver, solver_kwargs in study: ...; study.store_current_xp(...)`"
            )

        if success is None:
            success = is_empty_metrics(metrics)
        with (
            Hdf5Database(self.database_filepath) as database
        ):  # ensure closing the database at the end of computation (even if error)
            xp_id = database.get_new_experiment_id()
            xp = Experiment.from_solver_config(
                xp_id=xp_id,
                instance=self._instance,
                config_name=self._config_name,
                solver_config=self._solver_config,
                metrics=metrics,
                status=status,
                reason=reason,
            )
            database.store(xp)
        self._success = success

    def get_current_instance(self) -> str:
        return self._instance

    def get_current_config_name(self) -> str:
        return self._config_name

    def load_metadatas(self):
        """Load and normalize metadatas"""
        self.config_instance_success: dict[tuple[str, str], bool] = defaultdict(
            lambda: False
        )
        self.config_instance_n_attempts: dict[tuple[str, str], int] = defaultdict(
            lambda: 0
        )
        try:
            with Hdf5Database(self.database_filepath) as database:
                metadatas = database.load_metadata()
        except KeyError:
            # database w/o metadata
            metadatas = []
        else:
            normalize_metadatas(metadatas, config_store=self.config_store)
            for metadata in metadatas:
                key = metadata[INSTANCE], metadata[CONFIG]
                self.config_instance_success[key] = (
                    self.config_instance_success[key] or not metadata[IS_EMPTY]
                )
                self.config_instance_n_attempts[key] += 1

        return metadatas


I_RUN_LABEL = "attempt"


def normalize_metadatas(
    metadatas: list[dict[str, Any]], config_store: ConfigStore
) -> list[dict[str, Any]]:
    n_runs_by_config_instance = defaultdict(lambda: 0)  # to compute attempt number
    for metadata in metadatas:
        config_store.add(metadata[CONFIG])
        normalize_metadata(
            metadata=metadata,
            config_store=config_store,
            n_runs_by_config_instance=n_runs_by_config_instance,
        )


def normalize_metadata(
    metadata: dict[str, Any],
    config_store: ConfigStore,
    n_runs_by_config_instance: dict[tuple[str, str], int],
) -> None:
    # status
    if STATUS not in metadata:
        metadata[STATUS] = StatusSolver.UNKNOWN
    else:
        if not isinstance(metadata[STATUS], StatusSolver):
            metadata[STATUS] = StatusSolver(metadata[STATUS].upper())
    # config -> config name
    metadata[CONFIG] = normalize_config(metadata[CONFIG], config_store=config_store)
    # compute an attempt number
    key = metadata[CONFIG], metadata[INSTANCE]
    metadata[I_RUN_LABEL] = n_runs_by_config_instance[key]
    n_runs_by_config_instance[key] += 1


def normalize_config(config: Any, config_store: ConfigStore) -> str:
    # config -> config name
    if isinstance(config, dict):
        config_name = config_store.get_name(config)
    elif isinstance(config, str):
        config_name = config
    else:
        raise ValueError(
            "For each result df, df.attrs['config'] must be either a dictionary "
            "or a string representing its name."
        )
    return config_name
