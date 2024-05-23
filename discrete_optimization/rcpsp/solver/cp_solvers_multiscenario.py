#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, List, Optional, Union

from minizinc import Instance, Model, Solver

from discrete_optimization.generic_tools.cp_tools import (
    CPSolverName,
    MinizincCPSolver,
    find_right_minizinc_solver_name,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_model_preemptive import RCPSPModelPreemptive
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_utils import create_fake_tasks
from discrete_optimization.rcpsp.robust_rcpsp import AggregRCPSPModel

logger = logging.getLogger(__name__)
this_path = os.path.dirname(os.path.abspath(__file__))

files_mzn = {
    "multiscenario": os.path.join(this_path, "../minizinc/rcpsp_multiscenario.mzn")
}


def add_fake_task_cp_data(
    rcpsp_model: Union[RCPSPModel, RCPSPModelPreemptive],
    ignore_fake_task: bool = True,
    max_time_to_consider: int = None,
):
    if rcpsp_model.is_varying_resource() and not ignore_fake_task:
        fake_tasks = create_fake_tasks(rcpsp_problem=rcpsp_model)
        max_time_to_consider = (
            rcpsp_model.horizon
            if max_time_to_consider is None
            else max_time_to_consider
        )
        fake_tasks = [f for f in fake_tasks if f["start"] <= max_time_to_consider]
        n_fake_tasks = len(fake_tasks)
        fakestart = [fake_tasks[i]["start"] for i in range(len(fake_tasks))]
        fake_dur = [fake_tasks[i]["duration"] for i in range(len(fake_tasks))]
        max_duration_fake_task = max(fake_dur)
        fake_req = [
            [fake_tasks[i].get(res, 0) for i in range(len(fake_tasks))]
            for res in rcpsp_model.resources_list
        ]
        dict_to_add_in_instance = {
            "max_duration_fake_task": max_duration_fake_task,
            "n_fake_tasks": n_fake_tasks,
            "fakestart": fakestart,
            "fakedur": fake_dur,
            "fakereq": fake_req,
            "include_fake_tasks": True,
        }
        return dict_to_add_in_instance
    else:
        dict_to_add_in_instance = {
            "max_duration_fake_task": 0,
            "n_fake_tasks": 0,
            "fakestart": [],
            "fakedur": [],
            "fakereq": [[] for r in rcpsp_model.resources_list],
            "include_fake_tasks": False,
        }
        return dict_to_add_in_instance


class CP_MULTISCENARIO(MinizincCPSolver):
    hyperparameters = MinizincCPSolver.hyperparameters + [
        CategoricalHyperparameter(
            name="relax_ordering", default=False, choices=[True, False]
        ),
        IntegerHyperparameter(name="nb_incoherence_limit", low=0, high=10, default=3),
    ]
    problem: AggregRCPSPModel

    def __init__(
        self,
        problem: AggregRCPSPModel,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        params_objective_function: ParamsObjectiveFunction = None,
        silent_solve_error: bool = False,
        **kwargs,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.silent_solve_error = silent_solve_error
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = [
            "s"
        ]  # For now, I've put the var name of the CP model (not the rcpsp_model)

    @property
    def base_problem(self):
        return self.problem

    def init_model(self, **args):
        args = self.complete_with_default_hyperparameters(args)
        model_type = args.get("model_type", "multiscenario")
        max_time = args.get("max_time", self.problem.horizon)
        fake_tasks = args.get(
            "fake_tasks", True
        )  # to modelize varying quantity of resource.
        add_objective_makespan = args.get("add_objective_makespan", True)
        ignore_sec_objective = args.get("ignore_sec_objective", True)

        model = Model(files_mzn[model_type])
        solver = Solver.lookup(find_right_minizinc_solver_name(self.cp_solver_name))
        instance = Instance(solver, model)
        instance["add_objective_makespan"] = add_objective_makespan
        instance["ignore_sec_objective"] = ignore_sec_objective
        n_res = len(self.problem.resources_list)
        instance["n_res"] = n_res
        instance["n_scenario"] = self.problem.nb_problem
        dict_to_add = add_fake_task_cp_data(
            rcpsp_model=self.problem,
            ignore_fake_task=not fake_tasks,
            max_time_to_consider=max_time,
        )
        instance["max_time"] = max_time
        for key in dict_to_add:
            instance[key] = dict_to_add[key]

        sorted_resources = self.problem.resources_list
        self.resources_index = sorted_resources
        rcap = [
            int(self.problem.get_max_resource_capacity(x)) for x in sorted_resources
        ]
        instance["rc"] = rcap
        n_tasks = self.problem.n_jobs
        instance["n_tasks"] = n_tasks
        sorted_tasks = self.problem.tasks_list
        d = [
            [
                int(self.problem.list_problem[j].mode_details[key][1]["duration"])
                for j in range(self.problem.nb_problem)
            ]
            for key in sorted_tasks
        ]
        instance["d"] = d
        all_modes = [
            [
                (act, 1, self.problem.list_problem[j].mode_details[act][1])
                for act in sorted_tasks
            ]
            for j in range(self.problem.nb_problem)
        ]
        rr = [
            [
                [all_modes[j][i][2].get(res, 0) for j in range(self.problem.nb_problem)]
                for i in range(len(all_modes[0]))
            ]
            for res in sorted_resources
        ]
        instance["rr"] = rr
        suc = [
            set(
                [
                    self.problem.return_index_task(x, offset=1)
                    for x in self.problem.successors[task]
                ]
            )
            for task in sorted_tasks
        ]
        instance["suc"] = suc
        instance["relax_ordering"] = args["relax_ordering"]
        instance["nb_incoherence_limit"] = args["nb_incoherence_limit"]
        self.instance = instance
        self.index_in_minizinc = {
            task: self.problem.return_index_task(task, offset=1)
            for task in self.problem.tasks_list
        }
        self.instance["sink_task"] = self.index_in_minizinc[self.problem.sink_task]

    def retrieve_solution(
        self, _output_item: Optional[str] = None, **kwargs: Any
    ) -> RCPSPSolution:
        """Return a d-o solution from the variables computed by minizinc.

        Args:
            _output_item: string representing the minizinc solver output passed by minizinc to the solution constructor
            **kwargs: keyword arguments passed by minzinc to the solution contructor
                containing the objective value (key "objective"),
                and the computed variables as defined in minizinc model.

        Returns:

        """
        start = kwargs["s"]
        order = kwargs["ordering"]
        obj = kwargs["objective"]

        oo = [
            self.problem.index_task_non_dummy[self.problem.tasks_list[j - 1]]
            for j in order
            if self.problem.tasks_list[j - 1] in self.problem.index_task_non_dummy
        ]

        logger.debug(f"starts found : {start[-1]}")
        logger.debug(f"minizinc obj : {obj}")

        sol = RCPSPSolution(problem=self.problem, rcpsp_permutation=oo)
        sol.minizinc_obj = obj

        return sol
