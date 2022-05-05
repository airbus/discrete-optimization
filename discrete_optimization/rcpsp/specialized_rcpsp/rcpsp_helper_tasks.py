from typing import Dict

from discrete_optimization.rcpsp.rcpsp_model import SingleModeRCPSPModel, MultiModeRCPSPModel, RCPSPModel
from discrete_optimization.rcpsp.rcpsp_model_wip import RCPSPSolution
import numpy as np

class RCPSP_H_Model(SingleModeRCPSPModel):
    def __init__(self,
                 base_rcpsp_model: SingleModeRCPSPModel,
                 pre_helper_activities: Dict,
                 post_helper_activities: Dict):
        RCPSPModel.__init__(self, resources=base_rcpsp_model.resources,
                            non_renewable_resources=base_rcpsp_model.non_renewable_resources,
                            mode_details=base_rcpsp_model.mode_details,
                            successors=base_rcpsp_model.successors,
                            horizon=base_rcpsp_model.horizon,
                            horizon_multiplier=base_rcpsp_model.horizon_multiplier,
                            )
        self.pre_helper_activities = pre_helper_activities
        self.post_helper_activities = post_helper_activities
        # self.base_rcpsp_model = base_rcpsp_model

    def evaluate(self, rcpsp_sol: RCPSPSolution) -> Dict[str, float]:
        if rcpsp_sol._schedule_to_recompute:
            rcpsp_sol.generate_schedule_from_permutation_serial_sgs()
        rcpsp_sol.rcpsp_schedule = self.rcpsp_pre_helper_correction(rcpsp_sol)
        obj_makespan, obj_mean_resource_reserve = self.evaluate_function(rcpsp_sol)
        cumulated_helper_gap = 0
        for main_act_id in self.pre_helper_activities.keys():
            pre_gap = rcpsp_sol.rcpsp_schedule[main_act_id]['start_time'] - \
                      rcpsp_sol.rcpsp_schedule[self.pre_helper_activities[main_act_id][0]]['end_time']
            post_gap = rcpsp_sol.rcpsp_schedule[self.post_helper_activities[main_act_id][0]]['start_time'] - \
                       rcpsp_sol.rcpsp_schedule[main_act_id]['end_time']
            cumulated_helper_gap += pre_gap + post_gap

        return {'makespan': obj_makespan,
                'mean_resource_reserve': obj_mean_resource_reserve,
                'cumulated_helper_gap': cumulated_helper_gap}

    def evaluate_from_encoding(self, int_vector, encoding_name):
        if encoding_name == 'rcpsp_permutation':
            single_mode_list = [1 for i in range(self.n_jobs)]
            rcpsp_sol = RCPSPSolution(problem=self,
                                      rcpsp_permutation=int_vector,
                                      rcpsp_modes=single_mode_list)
        objectives = self.evaluate(rcpsp_sol)
        return objectives

    def rcpsp_pre_helper_correction(self, rcpsp_sol: RCPSPSolution):
        corrected_sol = rcpsp_sol.copy()

        # sort pre_helper activities by start time decreasing
        pre_helper_ids = []
        pre_helper_starts = []
        for main_id in self.pre_helper_activities:
            pre_helper_ids.append(self.pre_helper_activities[main_id][0])
            pre_helper_starts.append(rcpsp_sol.rcpsp_schedule[self.pre_helper_activities[main_id][0]]['start_time'])
        # print('pre_helper_ids: ', pre_helper_ids)
        # print('pre_helper_starts: ', pre_helper_starts)
        sorted_pre_helper_ids = [x for _, x in sorted(zip(pre_helper_starts, pre_helper_ids), reverse=True)]
        # print('sorted_pre_helper_ids: ', sorted_pre_helper_ids)

        # for each pre_helper, try to start as late as possible
        for id in sorted_pre_helper_ids:
            # print('id: ',id)
            # print('original_start: ', corrected_sol.rcpsp_schedule[id]['start_time'])
            # print('self.successors[id]: ', self.successors[id])
            # Latest possible cannot be later than the earliest start of its successors
            all_successor_starts = [corrected_sol.rcpsp_schedule[s_id]['start_time'] for s_id in self.successors[id]]
            # print('all_successor_starts: ', all_successor_starts)
            latest_end = min(all_successor_starts)
            # print('initial latest_end: ',latest_end)
            duration = (corrected_sol.rcpsp_schedule[id]['end_time'] - corrected_sol.rcpsp_schedule[id]['start_time'])
            latest_start = latest_end - duration
            # print('initial latest_start:', latest_start)

            # print('self.compute_resource_consumption(): ', self.compute_resource_consumption(corrected_sol))
            # Then iteratively check if the latest time is suitable resource-wise
            # if not try earlier

            # first copy the resource consumption array and remove consumption of the pre_helper activity
            consumption = np.copy(self.compute_resource_consumption(corrected_sol))
            # print('self.resources: ', self.resources)
            for i in range(len(list(self.resources.keys()))):
                res_str = list(self.resources.keys())[i]
                # print('res_str: ', res_str)
                for t in range(corrected_sol.rcpsp_schedule[id]['start_time'], corrected_sol.rcpsp_schedule[id]['end_time']):
                    # print('t: ', t)
                    consumption[i,t+1] -= self.mode_details[id][1][res_str]

            # print('consumption -2: ', consumption)

            # then start trying iteratively to fit the pre_helper activity as late as possible
            stop = False
            while not stop:
                all_good = True
                for t in range(latest_start, latest_start+duration):
                    # print('t: ',t)
                    for i in range(len(list(self.resources.keys()))):
                        res_str = list(self.resources.keys())[i]
                        if consumption[i, t+1] + self.mode_details[id][1][res_str] > self.resources[res_str]:
                            all_good = False
                            break
                if all_good:
                    corrected_sol.rcpsp_schedule[id]['start_time'] = latest_start
                    corrected_sol.rcpsp_schedule[id]['end_time'] = latest_start+duration
                    # print('Corrected start: ',corrected_sol.rcpsp_schedule[id]['start_time'])
                    # print('Corrected end: ', corrected_sol.rcpsp_schedule[id]['end_time'])
                    stop = True
                else:
                    latest_start -= 1

        # print(' ---------- ')
        return corrected_sol.rcpsp_schedule


class MRCPSP_H_Model(MultiModeRCPSPModel):
    def __init__(self,
                 base_rcpsp_model: MultiModeRCPSPModel,
                 pre_helper_activities: Dict,
                 post_helper_activities: Dict
                 ):
        RCPSPModel.__init__(self, resources=base_rcpsp_model.resources,
                            non_renewable_resources=base_rcpsp_model.non_renewable_resources,
                            mode_details=base_rcpsp_model.mode_details,
                            successors=base_rcpsp_model.successors,
                            horizon=base_rcpsp_model.horizon,
                            horizon_multiplier=base_rcpsp_model.horizon_multiplier,
                            )
        self.pre_helper_activities = pre_helper_activities
        self.post_helper_activities = post_helper_activities

    def evaluate_from_encoding(self, int_vector, encoding_name):
        if encoding_name == 'rcpsp_permutation':
            # change the permutation in the solution with int_vector and set the modes with self.fixed_modes
            rcpsp_sol = RCPSPSolution(problem=self,
                                      rcpsp_permutation=int_vector,
                                      rcpsp_modes=self.fixed_modes)
        elif encoding_name == 'rcpsp_modes':
            # change the modes in the solution with int_vector and set the permutation with self.fixed_permutation
            modes_corrected = [x+1 for x in int_vector]
            rcpsp_sol = RCPSPSolution(problem=self,
                                      rcpsp_permutation=self.fixed_permutation,
                                      rcpsp_modes=modes_corrected)
        objectives = self.evaluate(rcpsp_sol)
        return objectives
