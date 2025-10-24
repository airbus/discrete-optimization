#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
# You should compile the executable tempo_scheduler,
# from the tempo library developed here : https://gepgitlab.laas.fr/roc/emmanuel-hebrard/tempo
import datetime
import json
import logging
import os.path
import subprocess
from typing import Any, Optional

import numpy as np
import pandas as pd

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.hub_solver.tempo.tempo_tools import (
    FormatEnum,
    TempoSchedulingSolver,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

# Tempo wrapper for scheduling
from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
    export_scheduling_problem_json,
)
from discrete_optimization.workforce.scheduling.solvers import SolverAllocScheduling

logger = logging.getLogger(__name__)


class TempoScheduler(TempoSchedulingSolver, SolverAllocScheduling):
    def init_model(self, **kwargs: Any) -> None:
        self._input_format = FormatEnum.WORKFORCE
        this_folder = os.path.abspath(os.path.dirname(__file__))
        tmp_folder = os.path.join(this_folder, "data_temp/")
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
        d = export_scheduling_problem_json(problem=self.problem)
        json.dump(d, open(os.path.join(tmp_folder, "tmp.json"), "w"))
        self._file_input = os.path.join(tmp_folder, "tmp.json")

    def retrieve_solution(self, path_to_output: str, process_stdout: str) -> Solution:
        if os.path.exists(path_to_output):
            res = json.load(open(path_to_output, "r"))
            schedule = np.zeros((self.problem.number_tasks, 2))
            allocation = np.zeros(self.problem.number_tasks)
            used_resource = np.zeros((self.problem.number_teams, self.problem.horizon))
            for i in range(len(res["tasks"])):
                min_time = res["tasks"][i]["start"][0]
                duration = res["tasks"][i]["duration"]
                resource = int(res["tasks"][i]["resource"])

                schedule[i, 0] = min_time
                schedule[i, 1] = min_time + duration
                allocation[i] = int(res["tasks"][i]["resource"])
                used_resource[resource, min_time : min_time + duration] = 1
            solution = AllocSchedulingSolution(
                problem=self.problem, schedule=schedule, allocation=allocation
            )
            return solution
        else:
            logger.info(f"{path_to_output} file doesn't exist")
