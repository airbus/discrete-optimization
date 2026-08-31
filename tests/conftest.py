#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os
import warnings

import pandas as pd
from pytest import fixture

from discrete_optimization.datasets import DO_DEFAULT_DATAHOME_ENVVARNAME
from discrete_optimization.generic_tools.do_problem import ModeOptim
from discrete_optimization.generic_tools.do_solver import StatusSolver


@fixture
def fake_data_home(monkeypatch):
    data_home = "~/discrete_optimization_data_not_existing"
    monkeypatch.setenv(DO_DEFAULT_DATAHOME_ENVVARNAME, data_home)


# common fixtures for non-regression tests
NONREGRESSION_DB_SUFFIX = "_db.csv"


@fixture(scope="module")
def populate_nonregression_database(request) -> bool:
    """Whether to populate the non-regression database or compare test results to previous ones.

    To change the behaviour for a given test module, set a variable "POPULATE_NONREGRESSION_DATABASE" to False or True
    inside the module.

    """
    return getattr(request.module, "NONREGRESSION_POPULATE_DATABASE", False)


@fixture(scope="module")
def nonregression_database_filepath(request) -> str:
    return os.path.splitext(request.module.__file__)[0] + NONREGRESSION_DB_SUFFIX


@fixture(scope="module")
def nonregression_db(populate_nonregression_database, nonregression_database_filepath):
    if populate_nonregression_database:
        # create database from scratch
        with open(nonregression_database_filepath, "wt") as f:
            f.write("test_id,status,mode_optim,objective\n")
        return
    else:
        # read the database for comparison
        df = pd.read_csv(nonregression_database_filepath, index_col="test_id")
        if isinstance(df.objective.iloc[0], str):
            df.objective = df.objective.apply(eval)
        df.status = df.status.apply(lambda x: StatusSolver(x))
        df.mode_optim = df.mode_optim.apply(lambda x: ModeOptim(x))
        return df


@fixture(scope="module")
def nonregression_margin(request) -> float:
    """Margin allowed on objectives.

    Default to 5%, set variable "NONREGRESSION_MARGIN_OBJECTIVE" in test module to change it.

    """
    return getattr(request.module, "NONREGRESSION_MARGIN_OBJECTIVE", 0.05)


@fixture(scope="module")
def check_nonregression_fn(
    request,
    nonregression_db,
    nonregression_database_filepath,
    populate_nonregression_database,
    nonregression_margin,
):
    def _check_nonregression(
        test_id: str,
        status: StatusSolver,
        objective: float | int | tuple[float | int, ...],
        mode_optim: ModeOptim,
    ):
        # comparison with previous runs
        previous_mode_optim = nonregression_db.mode_optim[test_id]
        previous_objective = nonregression_db.objective[test_id]
        previous_status = nonregression_db.status[test_id]
        info = (
            f"current_run(status={status}, objective={objective})"
            f" vs stored_run(status={previous_status}, objective={previous_objective})"
        )

        if mode_optim != previous_mode_optim:
            raise RuntimeError(
                "mode_optim has changed between current and stored test runs!"
            )
        if previous_status == StatusSolver.OPTIMAL:
            # previous runs were optimal => new versions should be optimal with same objective value
            assert status == StatusSolver.OPTIMAL, info
            assert objective == previous_objective, info
        else:
            # previous runs not optimal => newer versions should be better than the previous objective + a margin
            assert status in [StatusSolver.OPTIMAL, StatusSolver.SATISFIED], info
            if mode_optim == ModeOptim.MAXIMIZATION:
                # take - objective to be as if minimizing the objective
                if isinstance(objective, tuple):
                    objective = tuple(-obj for obj in objective)
                    previous_objective = tuple(-obj for obj in previous_objective)
                else:
                    objective = -objective
                    previous_objective = -previous_objective
            if isinstance(objective, tuple):
                # multiobjectives: first objective components equal and then less than previous one + margin
                i = 0
                while objective[i] == previous_objective[i] and i < len(objective) - 1:
                    i += 1
                assert objective[i] <= previous_objective[
                    i
                ] + nonregression_margin * abs(previous_objective[i]), info
                # warning if improvement beyond margin
                if objective[i] < previous_objective[i] - nonregression_margin * abs(
                    previous_objective[i]
                ):
                    warnings.warn(
                        f"Test module: {request.module.__name__}, usecase: {test_id}, objective: {objective}. Objective improved beyond set margin!"
                    )
            else:
                # monoobjective: less than previous one + margin
                assert objective <= previous_objective + nonregression_margin * abs(
                    previous_objective
                ), info
                # warning if improvement beyond margin
                if objective < previous_objective - nonregression_margin * abs(
                    previous_objective
                ):
                    warnings.warn(
                        f"Test module: {request.module.__name__}, usecase: {test_id}, objective: {objective}. Objective improved beyond set margin!"
                    )

    def check_nonregression(
        test_id: str,
        status: StatusSolver,
        objective: float | int | tuple[float | int, ...],
        mode_optim: ModeOptim,
    ):
        csv_line = f'{test_id},{status.value},{mode_optim.value},"{objective}"'
        if populate_nonregression_database:
            # populate the non-regression database
            if status in [StatusSolver.OPTIMAL, StatusSolver.SATISFIED]:
                with open(nonregression_database_filepath, "at") as f:
                    f.write(csv_line + "\n")
            else:
                raise RuntimeError(f"Usecase: {test_id}, solver status: {status}")
        else:
            # comparison with previous runs
            try:
                _check_nonregression(
                    test_id=test_id,
                    status=status,
                    objective=objective,
                    mode_optim=mode_optim,
                )
            except Exception as e:
                error_msg = f"{e}\nCSV_LINE:{csv_line}"
                raise type(e)(error_msg)

    return check_nonregression
