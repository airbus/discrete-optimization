#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
from collections import defaultdict
from typing import Any, Optional

import pandas as pd

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import SolverDO, StatusSolver
from discrete_optimization.generic_tools.encoding_register import EncodingRegister
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.generic_tools.study import SolverConfig, Study
from discrete_optimization.generic_tools.study.database import IS_EMPTY
from discrete_optimization.generic_tools.study.experiment import CONFIG, INSTANCE
from discrete_optimization.generic_tools.study.study import I_RUN_LABEL


class FakeProblem(Problem):
    def __init__(self, instance):
        self.instance = instance

    def evaluate(self, variable: Solution) -> dict[str, float]:
        pass

    def set_fixed_attributes(self, attribute_name: str, solution: Solution) -> None:
        pass

    def satisfy(self, variable: Solution) -> bool:
        pass

    def get_attribute_register(self) -> EncodingRegister:
        pass

    def get_solution_type(self) -> type[Solution]:
        pass

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={},
        )

    def get_dummy_solution(self) -> Solution:
        pass


def test_study_overwrite():
    class FakeSolver(SolverDO):
        n_tries = 0
        n_failing_tries = 3

        def solve(
            self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
        ) -> ResultStorage:
            FakeSolver.n_tries += 1
            if FakeSolver.n_tries <= self.n_failing_tries:
                raise RuntimeError("Try again")
            else:
                self.stats = pd.DataFrame([[1, 2], [3, 4]])
                return self.create_result_storage()

    class FakeSolver2(SolverDO):
        n_tries_per_instance = defaultdict(lambda: 0)
        n_failing_tries_per_instance = 1
        problem: FakeProblem

        def solve(
            self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
        ) -> ResultStorage:
            FakeSolver2.n_tries_per_instance[self.problem.instance] += 1
            if (
                FakeSolver2.n_tries_per_instance[self.problem.instance]
                <= self.n_failing_tries_per_instance
            ):
                raise RuntimeError("Try again")
            else:
                self.stats = pd.DataFrame([[1, 2], [3, 4]])
                return self.create_result_storage()

    overwrite = True
    max_retry = 1
    instances = ["i1", "i2", "i3"]
    solver_configs = {
        "fake": SolverConfig(cls=FakeSolver, kwargs={}),
        "fake2": SolverConfig(
            cls=FakeSolver2,
            kwargs=dict(),
        ),
    }

    def problem_factory(instance):
        return FakeProblem(instance)

    study = Study(
        name="fake_study_overwrite",
        instances=instances,
        solver_configs=solver_configs,
        overwrite=overwrite,
        max_retry=max_retry,
        problem_factory=problem_factory,
    )

    # loop over study
    i_xps = 0
    for problem, solver, solver_kwargs in study:
        try:
            # solve
            res = solver.solve(
                **solver_kwargs,
            )
        except Exception as e:
            # failed experiment
            metrics = pd.DataFrame([])
            status = StatusSolver.ERROR
            logging.error(e)
            reason = f"{type(e)}: {str(e)}"
            success = False
        else:
            # get metrics and solver status
            metrics = solver.stats
            status = StatusSolver.UNKNOWN
            success = len(metrics) > 0
            reason = ""
            logging.info("experiment successful")
        i_xps += 1
        # store corresponding experiment
        study.store_current_xp(metrics, status, reason, success)

    print(i_xps)

    metadatas = study.load_metadatas()
    assert len(metadatas) == 11
    for i, metadata in enumerate(metadatas):
        if metadata[CONFIG] == "fake":
            if metadata[INSTANCE] == "i1" or (
                metadata[INSTANCE] == "i2" and metadata[I_RUN_LABEL] < 1
            ):
                assert metadata[IS_EMPTY]
            else:
                assert not metadata[IS_EMPTY]
        else:
            if metadata[I_RUN_LABEL] < 1:
                assert metadata[IS_EMPTY]
            else:
                assert not metadata[IS_EMPTY]


def test_study_no_overwrite(caplog):
    class FakeSolver(SolverDO):
        n_tries = 0
        n_failing_tries = 3

        def solve(
            self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
        ) -> ResultStorage:
            FakeSolver.n_tries += 1
            if FakeSolver.n_tries <= self.n_failing_tries:
                raise RuntimeError("Try again")
            else:
                self.stats = pd.DataFrame([[1, 2], [3, 4]])
                return self.create_result_storage()

    class FakeSolver2(SolverDO):
        n_tries_per_instance = defaultdict(lambda: 0)
        n_failing_tries_per_instance = 1
        problem: FakeProblem

        def solve(
            self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
        ) -> ResultStorage:
            FakeSolver2.n_tries_per_instance[self.problem.instance] += 1
            if (
                FakeSolver2.n_tries_per_instance[self.problem.instance]
                <= self.n_failing_tries_per_instance
            ):
                raise RuntimeError("Try again")
            else:
                self.stats = pd.DataFrame([[1, 2], [3, 4]])
                return self.create_result_storage()

    overwrite = False
    max_retry = 1
    instances = ["i1", "i2", "i3"]
    name = "fake_study_no_overwrite"
    solver_configs = {
        "fake": SolverConfig(cls=FakeSolver, kwargs={}),
        "fake2": SolverConfig(
            cls=FakeSolver2,
            kwargs=dict(),
        ),
    }
    try:
        os.remove(f"{name}.h5")
    except FileNotFoundError:
        pass

    def problem_factory(instance):
        return FakeProblem(instance)

    def run_study(
        name, instances, solver_configs, overwrite, max_retry, problem_factory
    ) -> tuple[Study, int]:
        # Stop after 3 xps (simulate an issue on laptop
        study = Study(
            name=name,
            instances=instances,
            solver_configs=solver_configs,
            overwrite=overwrite,
            max_retry=max_retry,
            problem_factory=problem_factory,
        )
        i_xps = 0
        for problem, solver, solver_kwargs in study:
            print(problem.instance, type(solver))
            i_xps += 1
            if i_xps > 3:
                i_xps -= 1
                break
            try:
                # solve
                res = solver.solve(
                    **solver_kwargs,
                )
            except Exception as e:
                # failed experiment
                metrics = pd.DataFrame([])
                status = StatusSolver.ERROR
                logging.error(e)
                reason = f"{type(e)}: {str(e)}"
                success = False
            else:
                # get metrics and solver status
                metrics = solver.stats
                status = StatusSolver.UNKNOWN
                success = len(metrics) > 0
                reason = ""
                logging.info("experiment successful")
            # store corresponding experiment
            study.store_current_xp(metrics, status, reason, success)
        return study, i_xps

    # 1st run
    study, i_xps = run_study(
        name=name,
        instances=instances,
        solver_configs=solver_configs,
        overwrite=overwrite,
        max_retry=max_retry,
        problem_factory=problem_factory,
    )
    metadatas = study.load_metadatas()
    assert len(metadatas) == 3
    assert i_xps == 3
    print(metadatas)
    assert len(set([metadata[INSTANCE] for metadata in metadatas])) == 1

    # 2nd run: skip successful and max_retry xps
    with caplog.at_level(logging.INFO):
        study, i_xps = run_study(
            name=name,
            instances=instances,
            solver_configs=solver_configs,
            overwrite=overwrite,
            max_retry=max_retry,
            problem_factory=problem_factory,
        )
    assert "already run (unsuccessfully)" in caplog.text
    assert "already run successfully" not in caplog.text
    metadatas = study.load_metadatas()
    assert i_xps == 3
    assert len(metadatas) == 6

    # 3rd run
    with caplog.at_level(logging.INFO):
        study, i_xps = run_study(
            name=name,
            instances=instances,
            solver_configs=solver_configs,
            overwrite=overwrite,
            max_retry=max_retry,
            problem_factory=problem_factory,
        )
    assert "already run (unsuccessfully)" in caplog.text
    assert "already run successfully" in caplog.text
    metadatas = study.load_metadatas()
    assert i_xps == 3
    assert len(metadatas) == 9

    # 4th run
    with caplog.at_level(logging.INFO):
        study, i_xps = run_study(
            name=name,
            instances=instances,
            solver_configs=solver_configs,
            overwrite=overwrite,
            max_retry=max_retry,
            problem_factory=problem_factory,
        )
    assert i_xps == 2
    metadatas = study.load_metadatas()
    assert len(metadatas) == 11
