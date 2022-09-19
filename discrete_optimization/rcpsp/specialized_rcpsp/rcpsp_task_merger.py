#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import itertools
import logging

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_evaluate_function_aggregated,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.lns_mip import InitialSolution
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.specialized_rcpsp.rcpsp_specialized_constraints import (
    RCPSPModelSpecialConstraintsPreemptive,
    RCPSPSolutionPreemptive,
    SpecialConstraintsDescription,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPModel
from discrete_optimization.rcpsp_multiskill.solvers.cp_lns_solver import (
    InitialMethodRCPSP,
)

logger = logging.getLogger(__name__)


class InitialSolutionMS_RCPSP_TaskMerger(InitialSolution):
    def __init__(
        self,
        problem: MS_RCPSPModel,
        special_constraints: SpecialConstraintsDescription,
        params_objective_function: ParamsObjectiveFunction = None,
        initial_method: InitialMethodRCPSP = InitialMethodRCPSP.DUMMY,
        type_of_merges=None,
        generate_only_feasible_solutions=True,
    ):
        self.problem = problem
        self.params_objective_function = params_objective_function
        if self.params_objective_function is None:
            self.params_objective_function = get_default_objective_setup(
                problem=self.problem
            )
        if type_of_merges is None:
            type_of_merges = {
                "start_at_end": True,
                "start_together": True,
                "start_at_end_plus_offset": False,
                "start_after_nunit": False,
            }
        self.initial_method = initial_method
        self.special_constraints = special_constraints
        self.type_of_merges = type_of_merges
        self.generate_only_feasible_solutions = generate_only_feasible_solutions
        (
            self.simplified_rcpsp_model,
            self.original_to_simplified_tasks_dict,
            self.simplified_to_original_tasks_dict,
        ) = merge_tasks(self.problem, self.special_constraints, self.type_of_merges)
        logger.debug(
            f"type_simplified_rcpsp_model: {type(self.simplified_rcpsp_model)}"
        )

        self.aggreg, _ = build_evaluate_function_aggregated(
            problem=self.problem,
            params_objective_function=self.params_objective_function,
        )

    def get_starting_solution(self) -> ResultStorage:

        n_attempts_per_relaxation_config = 1
        gg = self.simplified_rcpsp_model.compute_graph()
        has_loop = gg.check_loop()

        list_solution_fits = []
        if has_loop is None:
            has_loop = []
        for l in range(0, len(has_loop) + 1):
            for subset in itertools.combinations(has_loop, l):
                simplified_rcpsp_model_relaxed = self.simplified_rcpsp_model.copy()
                for pc in subset:
                    simplified_rcpsp_model_relaxed.successors[pc[0]].remove(pc[1])
                    simplified_rcpsp_model_relaxed.predecessors[pc[1]].remove(pc[0])

                for attempt in range(n_attempts_per_relaxation_config):
                    try:
                        simplified_solution: RCPSPSolutionPreemptive = (
                            simplified_rcpsp_model_relaxed.get_dummy_solution(
                                random_perm=True
                            )
                        )
                        sgs_success = True
                    except:
                        sgs_success = False

                    if sgs_success:
                        decoded_solution = decode_merged_task_solution(
                            self.problem,
                            simplified_solution,
                            self.simplified_rcpsp_model,
                            self.original_to_simplified_tasks_dict,
                            self.simplified_to_original_tasks_dict,
                        )

                        if self.problem.satisfy(decoded_solution):
                            list_solution_fits += [
                                (decoded_solution, self.aggreg(decoded_solution))
                            ]
                        elif not self.generate_only_feasible_solutions:
                            list_solution_fits += [
                                (decoded_solution, self.aggreg(decoded_solution))
                            ]

        return ResultStorage(
            list_solution_fits=list_solution_fits,
            mode_optim=self.params_objective_function.sense_function,
        )


def merge_tasks(
    original_rcpsp_model: RCPSPModelSpecialConstraintsPreemptive,
    special_constraints: SpecialConstraintsDescription,
    type_of_merges=None,
):
    if type_of_merges is None:
        type_of_merges = {
            "start_at_end": True,
            "start_together": True,
            "start_at_end_plus_offset": False,
            "start_after_nunit": False,
        }
    simplified_rcpsp_model: RCPSPModelSpecialConstraintsPreemptive = (
        original_rcpsp_model.copy()
    )

    original_to_simplified_tasks_dict = {}
    simplified_to_original_tasks_dict = {}

    next_id = max(simplified_rcpsp_model.tasks_list) + 1

    if type_of_merges["start_at_end"]:
        #  Handle START_AT_END constraints
        for st in special_constraints.start_at_end:

            existing_simplified_id = None
            if st[0] in original_to_simplified_tasks_dict.keys():
                existing_simplified_id = original_to_simplified_tasks_dict[st[0]][
                    "new_id"
                ]
                unseen_task = st[1]
            elif st[1] in original_to_simplified_tasks_dict.keys():
                existing_simplified_id = original_to_simplified_tasks_dict[st[1]][
                    "new_id"
                ]
                unseen_task = st[0]

            if existing_simplified_id is None:
                new_duration = sum(
                    [
                        simplified_rcpsp_model.mode_details[st[0]][1]["duration"],
                        simplified_rcpsp_model.mode_details[st[1]][1]["duration"],
                    ]
                )
                new_mode = {"duration": new_duration}
                for r in [
                    k
                    for k in simplified_rcpsp_model.mode_details[st[0]][1].keys()
                    if k != "duration"
                ]:
                    new_mode[r] = max(
                        simplified_rcpsp_model.mode_details[st[0]][1][r],
                        simplified_rcpsp_model.mode_details[st[1]][1][r],
                    )

                simplified_rcpsp_model.mode_details[next_id] = {1: new_mode}

                original_to_simplified_tasks_dict[st[0]] = {
                    "new_id": next_id,
                    "simplification": {"start_at_end"},
                }
                original_to_simplified_tasks_dict[st[1]] = {
                    "new_id": next_id,
                    "simplification": {"start_at_end"},
                }
                simplified_to_original_tasks_dict[next_id] = {
                    "original_id": {st[0], st[1]},
                    "simplification": {"start_at_end"},
                }

                next_id += 1
            else:

                new_duration = sum(
                    [
                        simplified_rcpsp_model.mode_details[unseen_task][1]["duration"],
                        simplified_rcpsp_model.mode_details[existing_simplified_id][1][
                            "duration"
                        ],
                    ]
                )
                new_mode = {"duration": new_duration}

                all_keys = simplified_to_original_tasks_dict[existing_simplified_id][
                    "original_id"
                ].copy()
                all_keys.update([st[0], st[1]])
                for r in [
                    k
                    for k in simplified_rcpsp_model.mode_details[st[0]][1].keys()
                    if k != "duration"
                ]:
                    new_mode[r] = max(
                        [
                            simplified_rcpsp_model.mode_details[the_id][1][r]
                            for the_id in all_keys
                        ]
                    )

                simplified_rcpsp_model.mode_details[existing_simplified_id] = {
                    1: new_mode
                }

                if st[0] == unseen_task:
                    original_to_simplified_tasks_dict[st[0]] = {
                        "new_id": existing_simplified_id,
                        "simplification": {"start_at_end"},
                    }
                else:
                    original_to_simplified_tasks_dict[st[0]]["simplification"].add(
                        "start_at_end"
                    )
                if st[1] == unseen_task:
                    original_to_simplified_tasks_dict[st[1]] = {
                        "new_id": existing_simplified_id,
                        "simplification": {"start_at_end"},
                    }
                else:
                    original_to_simplified_tasks_dict[st[1]]["simplification"].add(
                        "start_at_end"
                    )
                simplified_to_original_tasks_dict[existing_simplified_id][
                    "original_id"
                ].update((st[0], st[1]))
                simplified_to_original_tasks_dict[existing_simplified_id][
                    "simplification"
                ].add("start_at_end")

    if type_of_merges["start_together"]:
        #  Handle START_TOGETHER constraints
        for st in special_constraints.start_together:
            existing_simplified_id = None
            unseen_task = [st[0], st[1]]
            if st[0] in original_to_simplified_tasks_dict.keys():
                existing_simplified_id = original_to_simplified_tasks_dict[st[0]][
                    "new_id"
                ]
                unseen_task.remove(st[0])
            elif st[1] in original_to_simplified_tasks_dict.keys():
                existing_simplified_id = original_to_simplified_tasks_dict[st[1]][
                    "new_id"
                ]
                unseen_task.remove(st[1])

            if len(unseen_task) == 0:
                logger.warning("Length of unseen_task is null")
            unseen_task = unseen_task[0]

            if existing_simplified_id is None:
                new_duration = max(
                    simplified_rcpsp_model.mode_details[st[0]][1]["duration"],
                    simplified_rcpsp_model.mode_details[st[1]][1]["duration"],
                )
                new_mode = {"duration": new_duration}
                for r in [
                    k
                    for k in simplified_rcpsp_model.mode_details[st[0]][1].keys()
                    if k != "duration"
                ]:
                    new_mode[r] = (
                        simplified_rcpsp_model.mode_details[st[0]][1][r]
                        + simplified_rcpsp_model.mode_details[st[1]][1][r]
                    )

                simplified_rcpsp_model.mode_details[next_id] = {1: new_mode}

                original_to_simplified_tasks_dict[st[0]] = {
                    "new_id": next_id,
                    "simplification": {"start_together"},
                }
                original_to_simplified_tasks_dict[st[1]] = {
                    "new_id": next_id,
                    "simplification": {"start_together"},
                }
                simplified_to_original_tasks_dict[next_id] = {
                    "original_id": {st[0], st[1]},
                    "simplification": {"start_together"},
                }

                next_id += 1
            else:
                if unseen_task == st[0]:
                    delta = (
                        simplified_rcpsp_model.mode_details[unseen_task][1]["duration"]
                        - simplified_rcpsp_model.mode_details[st[1]][1]["duration"]
                    )
                else:
                    delta = (
                        simplified_rcpsp_model.mode_details[unseen_task][1]["duration"]
                        - simplified_rcpsp_model.mode_details[st[0]][1]["duration"]
                    )

                if delta > 0:
                    # add delta to duration
                    new_duration = (
                        simplified_rcpsp_model.mode_details[existing_simplified_id][1][
                            "duration"
                        ]
                        + delta
                    )
                else:
                    new_duration = simplified_rcpsp_model.mode_details[
                        existing_simplified_id
                    ][1]["duration"]
                new_mode = {"duration": new_duration}
                all_keys = simplified_to_original_tasks_dict[existing_simplified_id][
                    "original_id"
                ].copy()
                all_keys.update([st[0], st[1]])
                for r in [
                    k
                    for k in simplified_rcpsp_model.mode_details[st[0]][1].keys()
                    if k != "duration"
                ]:
                    new_mode[r] = sum(
                        [
                            simplified_rcpsp_model.mode_details[the_id][1][r]
                            for the_id in all_keys
                        ]
                    )

                simplified_rcpsp_model.mode_details[existing_simplified_id] = {
                    1: new_mode
                }

                if st[0] == unseen_task:
                    original_to_simplified_tasks_dict[st[0]] = {
                        "new_id": existing_simplified_id,
                        "simplification": {"start_together"},
                    }
                else:
                    original_to_simplified_tasks_dict[st[0]]["simplification"].add(
                        "start_together"
                    )
                if st[1] == unseen_task:
                    original_to_simplified_tasks_dict[st[1]] = {
                        "new_id": existing_simplified_id,
                        "simplification": {"start_together"},
                    }
                else:
                    original_to_simplified_tasks_dict[st[1]]["simplification"].add(
                        "start_together"
                    )

                simplified_to_original_tasks_dict[existing_simplified_id][
                    "original_id"
                ].update((st[0], st[1]))
                simplified_to_original_tasks_dict[existing_simplified_id][
                    "simplification"
                ].add("start_together")

    #  Fix problem data (add task_ids, precedences ... remove original tasks ...,
    #  fix resource need to upper bound capacity)

    # Fix precedence links
    new_successors = {}
    for key in simplified_rcpsp_model.successors:
        if key in original_to_simplified_tasks_dict.keys():
            new_key = original_to_simplified_tasks_dict[key]["new_id"]
        else:
            new_key = key
        new_vals = []
        vals = simplified_rcpsp_model.successors[key]

        for v in vals:
            if v in original_to_simplified_tasks_dict.keys():
                new_vals.append(original_to_simplified_tasks_dict[v]["new_id"])
            else:
                new_vals.append(v)

        if new_key in new_successors.keys():
            new_successors[new_key] = list(set(new_successors[new_key] + new_vals))
        else:
            new_successors[new_key] = new_vals

    simplified_rcpsp_model.successors = new_successors

    # Fix loops in precedence graph
    gg = simplified_rcpsp_model.compute_graph()
    has_loop = gg.check_loop()
    if has_loop is None:
        has_loop = []
    for c in has_loop:
        if (
            c[0] not in simplified_to_original_tasks_dict.keys()
            and c[1] in simplified_to_original_tasks_dict.keys()
        ):
            simplified_to_original_tasks_dict[c[1]]["original_id"].add(c[0])
            simplified_to_original_tasks_dict[c[1]]["simplification"].add("loop_fix")
            original_to_simplified_tasks_dict[c[0]] = {
                "new_id": c[1],
                "simplification": {"loop_fix"},
            }

        elif (
            c[1] not in simplified_to_original_tasks_dict.keys()
            and c[0] in simplified_to_original_tasks_dict.keys()
        ):
            simplified_to_original_tasks_dict[c[0]]["original_id"].add(c[1])
            simplified_to_original_tasks_dict[c[0]]["simplification"].add("loop_fix")
            original_to_simplified_tasks_dict[c[1]] = {
                "new_id": c[0],
                "simplification": {"loop_fix"},
            }

    # Fix precedence links again !
    for key in [
        x
        for x in original_to_simplified_tasks_dict.keys()
        if "loop_fix" in original_to_simplified_tasks_dict[x]["simplification"]
    ]:
        new_key = original_to_simplified_tasks_dict[key]["new_id"]
        old_succ = simplified_rcpsp_model.successors[key]
        simplified_rcpsp_model.successors[new_key] = list(
            set(old_succ + simplified_rcpsp_model.successors[new_key])
        )
        simplified_rcpsp_model.successors.pop(key)

        for key2 in simplified_rcpsp_model.successors.keys():
            if key in simplified_rcpsp_model.successors[key2]:
                simplified_rcpsp_model.successors[key2].remove(key)
                if new_key not in simplified_rcpsp_model.successors[key2]:
                    simplified_rcpsp_model.successors[key2].append(new_key)
            if key2 in simplified_rcpsp_model.successors[key2]:
                simplified_rcpsp_model.successors[key2].remove(key2)

    # upper bound need
    for new_task in simplified_to_original_tasks_dict.keys():
        for r in [
            k
            for k in simplified_rcpsp_model.mode_details[new_task][1].keys()
            if k != "duration"
        ]:
            if simplified_rcpsp_model.mode_details[new_task][1][r] > max(
                simplified_rcpsp_model.resources[r]
            ):
                simplified_rcpsp_model.mode_details[new_task][1][r] = max(
                    simplified_rcpsp_model.resources[r]
                )

    # add task ids
    for new_task in simplified_to_original_tasks_dict.keys():
        simplified_rcpsp_model.tasks_list.append(new_task)
        simplified_rcpsp_model.tasks_list_non_dummy.append(new_task)
        simplified_rcpsp_model.name_task[new_task] = "merge_" + str(new_task)

        for old_task in list(
            simplified_to_original_tasks_dict[new_task]["original_id"]
        ):
            if old_task in simplified_rcpsp_model.tasks_list:
                simplified_rcpsp_model.tasks_list.remove(old_task)
                simplified_rcpsp_model.tasks_list_non_dummy.remove(old_task)
                simplified_rcpsp_model.mode_details.pop(old_task)
                simplified_rcpsp_model.name_task.pop(old_task)

        simplified_rcpsp_model.n_jobs = len(simplified_rcpsp_model.tasks_list)
        simplified_rcpsp_model.n_jobs_non_dummy = len(
            simplified_rcpsp_model.tasks_list_non_dummy
        )

    # fix duration subtask data
    simplified_rcpsp_model.duration_subtask = {
        t: (False, 0) for t in simplified_rcpsp_model.tasks_list
    }

    # Re-instantiate new problem
    new_special_constraints_start_together = []
    new_special_constraints_start_at_end = []
    new_special_constraints_start_at_end_plus_offset = []
    new_special_constraints_start_after_nunit = []

    if not type_of_merges["start_together"]:
        for c in simplified_rcpsp_model.special_constraints.start_together:
            vals = []
            for i in range(len(c)):
                if c[i] in original_to_simplified_tasks_dict.keys():
                    vals.append(original_to_simplified_tasks_dict[c[i]]["new_id"])
                else:
                    vals.append(c[i])
            new_special_constraints_start_together.append(tuple(vals))

    if not type_of_merges["start_at_end"]:
        for c in simplified_rcpsp_model.special_constraints.start_at_end:
            vals = []
            for i in range(len(c)):
                if c[i] in original_to_simplified_tasks_dict.keys():
                    vals.append(original_to_simplified_tasks_dict[c[i]]["new_id"])
                else:
                    vals.append(c[i])
            new_special_constraints_start_at_end.append(tuple(vals))

    if not type_of_merges["start_at_end_plus_offset"]:
        for c in simplified_rcpsp_model.special_constraints.start_at_end_plus_offset:
            vals = []
            for i in range(len(c) - 1):
                if c[i] in original_to_simplified_tasks_dict.keys():
                    vals.append(original_to_simplified_tasks_dict[c[i]]["new_id"])
                else:
                    vals.append(c[i])
            vals.append(c[-1])
            new_special_constraints_start_at_end_plus_offset.append(tuple(vals))

    if not type_of_merges["start_after_nunit"]:
        for c in simplified_rcpsp_model.special_constraints.start_after_nunit:
            vals = []
            transform_constraint = False
            for i in range(len(c) - 1):
                if c[i] in original_to_simplified_tasks_dict.keys():
                    vals.append(original_to_simplified_tasks_dict[c[i]]["new_id"])
                    transform_constraint = True
                else:
                    vals.append(c[i])
            vals.append(c[-1])
            if not transform_constraint:
                new_special_constraints_start_after_nunit.append(tuple(vals))
            else:
                new_special_constraints_start_at_end_plus_offset.append(tuple(vals))

    new_special_constraints = SpecialConstraintsDescription(
        start_together=new_special_constraints_start_together,
        start_at_end=new_special_constraints_start_at_end,
        start_at_end_plus_offset=new_special_constraints_start_at_end_plus_offset,
        start_after_nunit=new_special_constraints_start_after_nunit,
    )

    new_model = RCPSPModelSpecialConstraintsPreemptive(
        resources=simplified_rcpsp_model.resources,
        non_renewable_resources=simplified_rcpsp_model.non_renewable_resources,
        mode_details=simplified_rcpsp_model.mode_details,
        successors=simplified_rcpsp_model.successors,
        horizon=simplified_rcpsp_model.horizon,
        special_constraints=new_special_constraints,
        preemptive_indicator=simplified_rcpsp_model.preemptive_indicator,
        relax_the_start_at_end=simplified_rcpsp_model.relax_the_start_at_end,
        tasks_list=simplified_rcpsp_model.tasks_list,
        source_task=simplified_rcpsp_model.source_task,
        sink_task=simplified_rcpsp_model.sink_task,
        name_task=simplified_rcpsp_model.name_task,
    )

    return (
        new_model,
        original_to_simplified_tasks_dict,
        simplified_to_original_tasks_dict,
    )


def decode_merged_task_solution(
    original_rcpsp_model: RCPSPModelSpecialConstraintsPreemptive,
    simplified_solution: RCPSPSolutionPreemptive,
    simplified_rcpsp_model: RCPSPModelSpecialConstraintsPreemptive,
    original_to_simplified_tasks_dict,
    simplified_to_original_tasks_dict,
):
    all_start_at_end_second_tasks = [
        x[1] for x in original_rcpsp_model.special_constraints.start_at_end
    ]
    all_start_together_tasks = list(
        set(
            [
                a
                for a in [
                    x for x in original_rcpsp_model.special_constraints.start_together
                ]
            ]
        )
    )

    decoded_schedule = {}
    for sim_task in simplified_to_original_tasks_dict.keys():
        done = []
        to_explore = []
        original_tasks = simplified_to_original_tasks_dict[sim_task]["original_id"]
        for t in original_tasks:
            if t not in all_start_at_end_second_tasks:
                cur_task = t
                cur_t_start = simplified_solution.rcpsp_schedule[sim_task]["starts"][0]
                cur_t_end = (
                    cur_t_start
                    + original_rcpsp_model.mode_details[cur_task][1]["duration"]
                )
                decoded_schedule[cur_task] = {
                    "starts": [cur_t_start],
                    "ends": [cur_t_end],
                }
                done.append(cur_task)

                next_start_at_end = [
                    x[1]
                    for x in original_rcpsp_model.special_constraints.start_at_end
                    if x[0] == cur_task
                ]
                next_start_together = [
                    x[1]
                    for x in original_rcpsp_model.special_constraints.start_together
                    if x[0] == cur_task
                ] + [
                    x[0]
                    for x in original_rcpsp_model.special_constraints.start_together
                    if x[1] == cur_task
                ]
                to_explore = set(next_start_at_end + next_start_together)
        stop = False
        while not stop:
            cur_task = list(to_explore)[0]
            if cur_task in all_start_at_end_second_tasks:
                previous_task = [
                    x[0]
                    for x in original_rcpsp_model.special_constraints.start_at_end
                    if x[1] == cur_task
                ][
                    0
                ]  # Retrieve previous task from all_start_at_end_second_tasks (at index 1)
                cur_t_start = decoded_schedule[previous_task]["ends"][-1]
                cur_t_end = (
                    cur_t_start
                    + original_rcpsp_model.mode_details[cur_task][1]["duration"]
                )

            if cur_task in all_start_together_tasks:
                tmp = [
                    x[0]
                    for x in original_rcpsp_model.special_constraints.start_at_end
                    if x[1] == cur_task
                ] + [
                    x[1]
                    for x in original_rcpsp_model.special_constraints.start_at_end
                    if x[0] == cur_task
                ]
                previous_task = tmp[0]
                cur_t_start = decoded_schedule[previous_task]["starts"][-1]
                cur_t_end = (
                    cur_t_start
                    + original_rcpsp_model.mode_details[cur_task][1]["duration"]
                )

            decoded_schedule[cur_task] = {"starts": [cur_t_start], "ends": [cur_t_end]}
            done.append(cur_task)
            next_start_at_end = [
                x[1]
                for x in original_rcpsp_model.special_constraints.start_at_end
                if x[0] == cur_task and x[1] not in done
            ]
            next_start_together = [
                x[1]
                for x in original_rcpsp_model.special_constraints.start_together
                if x[0] == cur_task and x[1] not in done
            ] + [
                x[0]
                for x in original_rcpsp_model.special_constraints.start_together
                if x[1] == cur_task and x[0] not in done
            ]
            to_explore.update(next_start_at_end + next_start_together)
            to_explore.remove(cur_task)
            if len(to_explore) == 0:
                stop = True

    for t in original_rcpsp_model.tasks_list:
        if t not in original_to_simplified_tasks_dict.keys():
            cur_t_start = simplified_solution.rcpsp_schedule[t]["starts"]
            cur_t_end = simplified_solution.rcpsp_schedule[t]["ends"]
            decoded_schedule[t] = {"starts": cur_t_start, "ends": cur_t_end}

    decoded_solution = RCPSPSolutionPreemptive(
        problem=original_rcpsp_model,
        rcpsp_permutation=None,
        rcpsp_schedule=decoded_schedule,
        rcpsp_modes=None,
        rcpsp_schedule_feasible=None,
        standardised_permutation=None,
    )

    return decoded_solution
