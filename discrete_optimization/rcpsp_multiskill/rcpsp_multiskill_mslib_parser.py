#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
from typing import Dict, Optional

from discrete_optimization.datasets import fetch_data_from_mslib, get_data_home
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    Employee,
    MS_RCPSPModel,
    SkillDetail,
)

logger = logging.getLogger(__name__)
this_folder = os.path.dirname(os.path.abspath(__file__))


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
):
    """Get datasets available for knapsack.
    Params:
        data_folder: folder where datasets for knapsack whould be find.
            If None, we look in "knapsack" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "
    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/rcpsp_multiskill_mslib"
    if not os.path.exists(data_folder):
        logger.info(f"Fetching data from MSLIB webpage.")
        fetch_data_from_mslib(data_home)
    mslib_folder = os.path.join(data_folder, "MSLIB/")
    subfolders = [
        os.path.join(mslib_folder, "MSLIB1/Instances1"),
        os.path.join(mslib_folder, "MSLIB2/Instances2"),
        os.path.join(mslib_folder, "MSLIB3/Instances3"),
        os.path.join(mslib_folder, "MSLIB4/Instances4"),
    ]
    tags = ["MSLIB1", "MSLIB2", "MSLIB3", "MSLIB4"]
    return {
        tags[i]: sorted(
            [
                os.path.abspath(os.path.join(subfolders[i], f))
                for f in os.listdir(subfolders[i])
                if f[-5:] == "msrcp"
            ]
        )
        for i in range(len(tags))
    }


def parse_file_mslib(file_path, skill_level_version: bool = True):
    logger.info(f"Parsing file {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        f = file.readlines()
        summary_line = f[1]
        number_activities, number_units, number_skills, number_skill_level = tuple(
            map(int, summary_line.split("\t"))
        )
        logger.info(
            f"Number of activities {number_activities}, number units {number_units}, "
            f"number skills {number_skills}, {number_skill_level}"
        )
        horizon_1 = int(f[3])
        horizon_2 = int(f[5])
        range_activities = range(7, 7 + number_activities)
        successors_dict = {}
        tasks_list = list(range(number_activities))
        source = 0
        sink = number_activities - 1
        mode_details = {t: {1: {}} for t in tasks_list}
        for j, index_task in zip(range_activities, tasks_list):
            string_ = f[j].replace("\t", " ")
            # string_ = string_.replace("\t", " ")
            # string_ = string_.replace("\n", "")
            logger.debug(f"{string_.split(' ')[:-1]}")
            datas = list(map(int, string_.split(" ")[:-1]))
            duration = datas[0]
            mode_details[index_task][1]["duration"] = duration
            nb_successors = datas[1]
            if nb_successors >= 1:
                successors_dict[index_task] = list(map(lambda x: x - 1, datas[2:]))
            else:
                successors_dict[index_task] = []
        logger.debug(f"successors dict : {successors_dict}")
        logger.debug(f"mode_details : {mode_details}")

        index_workforce_module = next(
            j for j in range(len(f)) if "* Workforce Module *" in f[j]
        )
        index_workforce_module = next(
            j for j in range(len(f)) if "Workforce Module with Skill Levels" in f[j]
        )
        lines_workforce = range(
            index_workforce_module + 1, index_workforce_module + 1 + number_units
        )
        workers_list = [f"w-{i}" for i in range(number_units)]
        skills_list = [f"sk-{i}" for i in range(number_skills)]
        workers: Dict[str, Employee] = {
            w: Employee(
                dict_skill={}, calendar_employee=[True] * (2 * horizon_1 + 1), salary=0
            )
            for w in workers_list
        }
        for j, worker in zip(lines_workforce, workers_list):
            string_ = f[j].replace("\t", " ")
            datas_workforce = list(map(int, string_.split(" ")[:-1]))
            for k in range(len(datas_workforce)):
                if datas_workforce[k] > 0:
                    if skill_level_version:
                        workers[worker].dict_skill[
                            (skills_list[k], datas_workforce[k])
                        ] = SkillDetail(skill_value=1, efficiency_ratio=0, experience=0)
                    else:
                        workers[worker].dict_skill[skills_list[k]] = SkillDetail(
                            skill_value=1, efficiency_ratio=0, experience=0
                        )
        index_skill_req = next(
            j for j in range(len(f)) if "Skill Requirements Module" in f[j]
        )
        temp_dict = {t: [] for t in tasks_list}
        lines_skill_req = range(
            index_skill_req + 1, index_skill_req + 1 + number_activities
        )
        skills_set = set()
        for j, t in zip(lines_skill_req, tasks_list):
            string_ = f[j].replace("\t", " ")
            datas_skill_req = list(map(int, string_.split(" ")[:-1]))
            for k, skill in zip(range(len(datas_skill_req)), skills_list):
                if datas_skill_req[k] > 0:
                    temp_dict[t] += [(skill, datas_skill_req[k])]
                    if not skill_level_version:
                        mode_details[t][1][skill] = datas_skill_req[k]
                        skills_set.add(skill)
        if skill_level_version:
            index_skill_level_req = next(
                j for j in range(len(f)) if "Skill Level Requirements Module" in f[j]
            )
            lines_skill_req = range(
                index_skill_level_req + 1, index_skill_level_req + 1 + number_activities
            )
            for j, t in zip(lines_skill_req, tasks_list):
                string_ = f[j].replace("\t", " ")
                datas_skill_req = list(map(int, string_.split(" ")[:-1]))
                if len(temp_dict[t]) == 0:
                    continue
                cur_index = 0
                for k in range(len(temp_dict[t])):
                    skill, number = temp_dict[t][k]
                    for jj in range(cur_index, cur_index + number):
                        level_req = datas_skill_req[jj]
                        subkey_name = (skill, level_req)
                        if subkey_name not in mode_details[t][1]:
                            mode_details[t][1][subkey_name] = 0
                            skills_set.add(subkey_name)
                        mode_details[t][1][subkey_name] += 1
                    cur_index = cur_index + number
        logger.info(f"skills set {skills_set}")
        skills_list = sorted(list(skills_set))
        return MS_RCPSPModel(
            skills_set=set(skills_list),
            resources_set=set(),
            non_renewable_resources=set(),
            resources_availability={},
            employees=workers,
            mode_details=mode_details,
            successors=successors_dict,
            horizon=2 * horizon_1,
            tasks_list=tasks_list,
            employees_list=workers_list,
            sink_task=sink,
            source_task=source,
            one_unit_per_task_max=False,
            preemptive=False,
        )


if __name__ == "__main__":
    files_dict = get_data_available()
    file = files_dict["MSLIB1"][0]
    logger.info("file = ", file)
    model = parse_file_mslib(file).to_variant_model()
    solution = model.get_dummy_solution()
    from discrete_optimization.rcpsp.solver.ls_solver import LS_RCPSP_Solver
    from discrete_optimization.rcpsp_multiskill.plots.plot_solution import (
        plot_resource_individual_gantt,
        plt,
    )

    logging.basicConfig(level=logging.DEBUG)
    solver = LS_RCPSP_Solver(model=model)
    result = solver.solve(nb_iteration_max=5000)
    sol, fit = result.get_best_solution_fit()
    plot_resource_individual_gantt(rcpsp_model=model, rcpsp_sol=sol)
    plt.show()
