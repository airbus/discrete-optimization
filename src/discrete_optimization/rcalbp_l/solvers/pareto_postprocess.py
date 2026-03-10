#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from copy import deepcopy
from typing import Any, Iterable

from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.generic_tools.do_solver import (
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcalbp_l.problem import RCALBPLProblem, RCALBPLSolution


class RampUpParetoSolverPostpro(OrtoolsCpSatSolver):
    problem: RCALBPLProblem

    def __init__(
        self,
        problem: RCALBPLProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.solution: RCALBPLSolution = None

    def init_model(self, from_solution: RCALBPLSolution, **kwargs: Any) -> None:
        super().init_model(**kwargs)
        self.solution = from_solution
        sol: RCALBPLSolution = from_solution
        cycle_time_used_per_period = self.problem.compute_actual_cycle_time_per_period(
            sol
        )
        cycle_time_chosen = deepcopy(sol.cyc)
        initial_cycle_time_setup = max(
            [cycle_time_used_per_period[p] for p in range(self.problem.nb_stations)]
        )
        for p in range(self.problem.nb_stations):
            cycle_time_chosen[p] = initial_cycle_time_setup

        variables = {
            p: self.cp_model.NewIntVar(
                lb=max(cycle_time_used_per_period[p], self.problem.c_target),
                ub=self.problem.c_max,
                name=f"cycle_time_chosen_{p}",
            )
            for p in self.problem.periods
        }
        for p in self.problem.periods:
            if p < self.problem.nb_stations:
                self.cp_model.add(variables[p] == initial_cycle_time_setup)

        ramp_up_cost_per_period = {
            p: self.cp_model.NewIntVar(
                lb=0, ub=self.problem.c_max, name=f"ramp_up_cost_per_period_{p}"
            )
            for p in self.problem.periods
        }
        change_cost = {
            p: self.cp_model.NewBoolVar(name=f"change_cost_{p}")
            for p in self.problem.periods[1:]
        }
        for i in range(1, len(self.problem.periods)):
            p_0 = self.problem.periods[i - 1]
            p_1 = self.problem.periods[i]
            # decreasing monotony
            self.cp_model.add(variables[p_1] <= variables[p_0])
            # change_cost[p_1] <-> cycletime[p_1] < cycletime[p_0]
            self.cp_model.add(variables[p_1] < variables[p_0]).only_enforce_if(
                change_cost[p_1]
            )
            self.cp_model.add(variables[p_1] == variables[p_0]).only_enforce_if(
                ~change_cost[p_1]
            )

            # If there is a change, we jump to the max("cycle_time_used[p], self.problem.c_target)
            self.cp_model.add(
                variables[p_1]
                == max(cycle_time_used_per_period[p_1], self.problem.c_target)
            ).only_enforce_if(change_cost[p_1])
        for p in ramp_up_cost_per_period:
            is_upper_than_ctarget = self.cp_model.NewBoolVar(
                name=f"is_upper_than_ctarget_{p}"
            )
            self.cp_model.add(variables[p] > self.problem.c_target).only_enforce_if(
                is_upper_than_ctarget
            )
            self.cp_model.add(variables[p] == self.problem.c_target).only_enforce_if(
                ~is_upper_than_ctarget
            )
            self.cp_model.add(
                ramp_up_cost_per_period[p] == variables[p]
            ).only_enforce_if(is_upper_than_ctarget)
            self.cp_model.add(ramp_up_cost_per_period[p] == 0).only_enforce_if(
                ~is_upper_than_ctarget
            )
        self.variables["objectives"] = {
            "change_cost": sum([change_cost[p] for p in change_cost]),
            "ramp_up_cost": sum(
                [ramp_up_cost_per_period[p] for p in ramp_up_cost_per_period]
            ),
        }
        self.variables["cycle_time_chosen"] = variables

    @staticmethod
    def implements_lexico_api() -> bool:
        return True

    def get_lexico_objectives_available(self) -> list[str]:
        return list(self.variables["objectives"].keys())

    def set_lexico_objective(self, obj: str) -> None:
        assert obj in self.get_lexico_objectives_available()
        self.cp_model.minimize(self.variables["objectives"][obj])

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        if res:
            sol = res[-1][0]
            kpis = self.problem.evaluate(sol)
            if obj == "change_cost":
                return kpis["nb_adjustments"]
            if obj == "ramp_up_cost":
                return kpis["ramp_up_duration"]
        return 0

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        return [self.cp_model.add(self.variables["objectives"][obj]) <= value]

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> RCALBPLSolution:
        cycle_time_chosen = {}
        for p in self.variables["cycle_time_chosen"]:
            cycle_time_chosen[p] = cpsolvercb.Value(
                self.variables["cycle_time_chosen"][p]
            )
        sol = self.solution.copy()
        sol.cyc = cycle_time_chosen
        return sol


import didppy as dp

from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver


class DpRCALBPLPostProSolver(DpSolver):
    problem: RCALBPLProblem
    solution: RCALBPLSolution

    def init_model(
        self,
        from_solution: RCALBPLSolution,
        max_nb_adjustments: int = None,
        **kwargs: Any,
    ) -> None:
        self.solution = from_solution
        sol: RCALBPLSolution = from_solution
        cycle_time_used_per_period = self.problem.compute_actual_cycle_time_per_period(
            sol
        )
        cycle_time_chosen = deepcopy(sol.cyc)
        initial_cycle_time_setup = max(
            [cycle_time_used_per_period[p] for p in range(self.problem.nb_stations)]
        )
        for p in range(self.problem.nb_stations):
            cycle_time_chosen[p] = initial_cycle_time_setup
        self.model = dp.Model(maximize=False, float_cost=False)
        decision_step = [
            p
            for p in self.problem.periods
            if p >= self.problem.nb_stations
            and cycle_time_chosen[p] > self.problem.c_target
        ]
        if decision_step[-1] + 1 in self.problem.periods:
            decision_step.append(decision_step[-1] + 1)
        print(decision_step)
        if max_nb_adjustments is None:
            max_nb_adjustments = len(decision_step)
        self.decision_step = decision_step
        decision_step_object = self.model.add_object_type(number=len(decision_step) + 1)
        index_decision = self.model.add_element_var(
            object_type=decision_step_object, target=0
        )
        nb_adjustment = self.model.add_int_var(target=0)
        self.model.add_state_constr(nb_adjustment <= max_nb_adjustments)
        current_cycle_time_chosen = self.model.add_int_var(
            target=initial_cycle_time_setup
        )
        cycle_time_array = [
            max(cycle_time_used_per_period[p], self.problem.c_target)
            for p in decision_step
        ] + [self.problem.c_target]
        self.cycle_time_array = cycle_time_array
        cycle_time_table = self.model.add_int_table(cycle_time_array)
        cost_if_accepting = self.model.add_int_state_fun(
            (cycle_time_table[index_decision] <= self.problem.c_target).if_then_else(
                0, cycle_time_table[index_decision]
            )
        )

        cost_if_skipping = self.model.add_int_state_fun(
            (current_cycle_time_chosen <= self.problem.c_target).if_then_else(
                0, current_cycle_time_chosen
            )
        )

        change_cycle_time = dp.Transition(
            name="change_cycle_time",
            cost=cost_if_accepting + dp.IntExpr.state_cost(),
            effects=[
                (nb_adjustment, nb_adjustment + 1),
                (index_decision, index_decision + 1),
                (current_cycle_time_chosen, cycle_time_table[index_decision]),
            ],
            preconditions=[index_decision < len(decision_step)],
        )
        change_cycle_time_forced = dp.Transition(
            name="change_cycle_time_forced",
            cost=cost_if_accepting + dp.IntExpr.state_cost(),
            effects=[
                (nb_adjustment, nb_adjustment + 1),
                (index_decision, index_decision + 1),
                (current_cycle_time_chosen, cycle_time_table[index_decision]),
            ],
            preconditions=[
                index_decision < len(decision_step),
                cycle_time_table[index_decision] == self.problem.c_target,
                current_cycle_time_chosen > self.problem.c_target,
            ],
        )
        self.model.add_base_case(
            [
                (current_cycle_time_chosen == self.problem.c_target)
                | (index_decision == len(decision_step))
            ]
        )
        self.model.add_transition(change_cycle_time)
        self.model.add_transition(change_cycle_time_forced, forced=True)
        skip = dp.Transition(
            name="skip",
            cost=cost_if_skipping + dp.IntExpr.state_cost(),
            effects=[(index_decision, index_decision + 1)],
            preconditions=[index_decision < len(decision_step)],
        )
        self.model.add_transition(skip)

    def retrieve_solution(self, sol: dp.Solution) -> RCALBPLSolution:
        cycle_time_chosen = {}
        for p in range(self.problem.nb_stations):
            cycle_time_chosen[p] = self.solution.cyc[p]
        periods_of_decision = self.decision_step
        index = 0
        prev_value = self.solution.cyc[0]
        print([t.name for t in sol.transitions])
        for transition in sol.transitions:
            if "change_cycle_time" in transition.name:
                cycle_time_chosen[periods_of_decision[index]] = self.cycle_time_array[
                    index
                ]
                prev_value = self.cycle_time_array[index]
            else:
                cycle_time_chosen[periods_of_decision[index]] = prev_value
            index += 1
        for p in self.solution.cyc:
            if p not in cycle_time_chosen:
                cycle_time_chosen[p] = self.solution.cyc[p]
        solution = self.solution.copy()
        solution.cyc = cycle_time_chosen
        return solution
