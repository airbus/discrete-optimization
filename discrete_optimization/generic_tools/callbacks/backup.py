#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import pickle
from datetime import datetime
from typing import Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class PickleBestSolutionBackup(Callback):
    def __init__(self, save_nb_steps: int, backup_path: str = "debug.pkl"):
        self.backup_path = backup_path
        self.save_nb_steps = save_nb_steps

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        if step % self.save_nb_steps == 0:
            logger.debug(f"Pickling best solution")
            pickle.dump(res.best_solution, open(self.backup_path, "wb"))
