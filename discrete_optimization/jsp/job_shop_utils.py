#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.jsp.job_shop_problem import JobShopProblem, Subjob
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel


def transform_jsp_to_rcpsp(jsp_problem: JobShopProblem) -> RCPSPModel:
    mode_details = {}
    successors = {}
    tasks_list = ["source"]
    successors["source"] = [(i, 0) for i in range(jsp_problem.n_jobs)]
    successors["sink"] = []
    mode_details["source"] = {1: {"duration": 0}}
    mode_details["sink"] = {1: {"duration": 0}}
    for i in range(jsp_problem.n_jobs):
        for j in range(len(jsp_problem.list_jobs[i])):
            tasks_list.append((i, j))
            mode_details[(i, j)] = {
                1: {
                    "duration": jsp_problem.list_jobs[i][j].processing_time,
                    f"machine_{jsp_problem.list_jobs[i][j].machine_id}": 1,
                }
            }
            if j < len(jsp_problem.list_jobs[i]) - 1:
                successors[(i, j)] = [(i, j + 1)]
        successors[(i, len(jsp_problem.list_jobs[i]) - 1)] = ["sink"]
    tasks_list.append("sink")

    rcpsp_problem = RCPSPModel(
        resources={f"machine_{i}": 1 for i in range(jsp_problem.n_machines)},
        non_renewable_resources=[],
        successors=successors,
        mode_details=mode_details,
        tasks_list=tasks_list,
        source_task="source",
        sink_task="sink",
        horizon=5000,
    )
    return rcpsp_problem
