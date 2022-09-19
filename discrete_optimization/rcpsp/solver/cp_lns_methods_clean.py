#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Iterable, List, Optional, Union

from deprecation import deprecated
from minizinc import Instance

from discrete_optimization.generic_rcpsp_tools.graph_tools_rcpsp import (
    GraphRCPSP,
    GraphRCPSPSpecialConstraints,
)
from discrete_optimization.generic_rcpsp_tools.neighbor_tools_rcpsp import (
    BasicConstraintBuilder,
    ParamsConstraintBuilder,
)
from discrete_optimization.generic_tools.cp_tools import SignEnum
from discrete_optimization.generic_tools.lns_cp import ConstraintHandler
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_model_preemptive import (
    RCPSPModelPreemptive,
    RCPSPSolutionPreemptive,
)
from discrete_optimization.rcpsp.solver import CP_MRCPSP_MZN, CP_RCPSP_MZN
from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_MRCPSP_MZN_PREEMMPTIVE,
    CP_RCPSP_MZN_PREEMMPTIVE,
)
from discrete_optimization.rcpsp.specialized_rcpsp.rcpsp_specialized_constraints import (
    RCPSPModelSpecialConstraints,
    RCPSPModelSpecialConstraintsPreemptive,
)

logger = logging.getLogger(__name__)


@deprecated(
    deprecated_in="0.1", details="Use rather generic_tools_rcpsp/neighbor_tools_rcpsp"
)
class NeighborSubproblem(ConstraintHandler):
    def __init__(
        self,
        problem: Union[
            RCPSPModel,
            RCPSPModelPreemptive,
            RCPSPSolutionPreemptive,
            RCPSPModelSpecialConstraints,
            RCPSPModelSpecialConstraintsPreemptive,
        ],
        basic_constraint_builder: BasicConstraintBuilder,
        params_list: List[ParamsConstraintBuilder] = None,
    ):
        self.problem = problem
        self.basic_constraint_builder = basic_constraint_builder
        if isinstance(
            self.problem,
            (RCPSPModelSpecialConstraintsPreemptive, RCPSPModelSpecialConstraints),
        ):
            self.graph_rcpsp = GraphRCPSPSpecialConstraints(problem=self.problem)
            self.special_constraints = True
        else:
            self.graph_rcpsp = GraphRCPSP(problem=self.problem)
            self.special_constraints = False
        if params_list is None:
            self.params_list = [
                ParamsConstraintBuilder(
                    minus_delta_primary=6000,
                    plus_delta_primary=6000,
                    minus_delta_secondary=1,
                    plus_delta_secondary=1,
                ),
                ParamsConstraintBuilder(
                    minus_delta_primary=6000,
                    plus_delta_primary=6000,
                    minus_delta_secondary=300,
                    plus_delta_secondary=300,
                ),
            ]
        else:
            self.params_list = params_list

    def adding_constraint_from_results_store(
        self,
        cp_solver: Union[
            CP_RCPSP_MZN_PREEMMPTIVE,
            CP_RCPSP_MZN,
            CP_MRCPSP_MZN,
            CP_MRCPSP_MZN_PREEMMPTIVE,
        ],
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        if last_result_store is not None:
            current_solution, fit = next(
                (
                    last_result_store.list_solution_fits[j]
                    for j in range(len(last_result_store.list_solution_fits))
                    if "opti_from_cp"
                    in last_result_store.list_solution_fits[j][0].__dict__.keys()
                ),
                (None, None),
            )
        else:
            current_solution, fit = next(
                (
                    result_storage.list_solution_fits[j]
                    for j in range(len(result_storage.list_solution_fits))
                    if "opti_from_cp"
                    in result_storage.list_solution_fits[j][0].__dict__.keys()
                ),
                (None, None),
            )
        if current_solution is None or fit != result_storage.get_best_solution_fit()[1]:
            current_solution, fit = result_storage.get_last_best_solution()
        current_solution: RCPSPSolutionPreemptive = current_solution
        evaluation = self.problem.evaluate(current_solution)
        logger.debug(f"Current Eval : {evaluation}")
        if evaluation.get("constraint_penalty", 0) == 0:
            p = self.params_list[min(1, len(self.params_list) - 1)]
        else:
            p = self.params_list[min(1, len(self.params_list) - 1)]
        (
            list_strings,
            subtasks_1,
            subtasks_2,
        ) = self.basic_constraint_builder.return_constraints(
            current_solution=current_solution,
            cp_solver=cp_solver,
            params_constraint_builder=p,
        )
        for s in list_strings:
            child_instance.add_string(s)
        if evaluation.get("constraint_penalty", 0) > 0:
            child_instance.add_string(
                "constraint objective=" + str(evaluation["makespan"]) + ";\n"
            )
        else:
            string = cp_solver.constraint_start_time_string(
                task=self.problem.sink_task,
                start_time=current_solution.get_start_time(self.problem.sink_task) + 20,
                sign=SignEnum.LEQ,
            )
            child_instance.add_string(string)
        child_instance.add_string(
            "constraint sec_objective<="
            + str(int(1.01 * 100 * evaluation.get("constraint_penalty", 0)) + 1000)
            + ";\n"
        )
        if evaluation.get("constraint_penalty", 0) > 0:
            strings = []
        else:
            strings = cp_solver.constraint_objective_max_time_set_of_jobs(subtasks_1)
        for s in strings:
            child_instance.add_string(s)
            list_strings += [s]
        return list_strings

    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: CP_RCPSP_MZN,
        child_instance,
        previous_constraints: Iterable[Any],
    ):
        pass
