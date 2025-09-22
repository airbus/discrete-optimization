import logging
import re
from typing import Any

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver, dp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.workforce.allocation.problem import (
    TeamAllocationProblem,
    TeamAllocationProblemMultiobj,
    TeamAllocationSolution,
)
from discrete_optimization.workforce.allocation.solvers import TeamAllocationSolver
from discrete_optimization.workforce.allocation.utils import compute_equivalent_teams

logger = logging.getLogger(__name__)


class DpAllocationSolver(DpSolver, TeamAllocationSolver, WarmstartMixin):
    hyperparameters = DpSolver.hyperparameters + [
        CategoricalHyperparameter(
            name="symmbreak_on_used", choices=[True, False], default=True
        ),
        CategoricalHyperparameter(
            name="force_allocation_when_possible",
            choices=[True, False],
            depends_on=[("symmbreak_on_used", False)],
            default=False,
        ),
    ]
    problem: TeamAllocationProblem
    transitions: dict[str, list]

    def init_model_mono_objective(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        symmbreak_on_used = kwargs["symmbreak_on_used"]
        model = dp.Model()
        compatibility = self.problem.compute_allowed_team_index_all_task()
        task = model.add_object_type(number=self.problem.number_of_activity)
        teams = model.add_object_type(number=self.problem.number_of_teams)
        compatibility = model.add_set_table(compatibility, object_type=teams)
        tasks_allocated_to_team = [
            model.add_set_var(object_type=task, target=set())
            for i in range(self.problem.number_of_teams)
        ]
        # unallocated = model.add_set_var(object_type=task, target=range(self.problem.number_of_activity))
        neighbors = [set() for i in range(self.problem.number_of_activity)]
        for edge in self.problem.graph_activity.edges:
            ind1 = self.problem.index_activities_name[edge[0]]
            ind2 = self.problem.index_activities_name[edge[1]]
            neighbors[ind1].add(ind2)
            neighbors[ind2].add(ind1)
        neighbors = model.add_set_table(
            [model.create_set_const(object_type=task, value=n) for n in neighbors]
        )
        current_task = model.add_element_var(object_type=task, target=0)
        used_team = [
            model.add_int_var(target=0) for _ in range(self.problem.number_of_teams)
        ]
        finish = model.add_int_var(target=0)
        model.add_base_case(
            [current_task == self.problem.number_of_activity, finish == 1]
        )
        if symmbreak_on_used:
            groups = compute_equivalent_teams(team_allocation_problem=self.problem)
            for group in groups:
                for team_0, team_1 in zip(group[:-1], group[1:]):
                    model.add_state_constr(used_team[team_0] >= used_team[team_1])
        transitions_dict = {"new_team": [], "not_new_team": []}
        for team in range(self.problem.number_of_teams):
            # new_team = (used_team[team] == 0).if_then_else(1, 0)
            transition = dp.Transition(
                name=f"allocate_to_{team}",
                cost=1 + dp.IntExpr.state_cost(),
                effects=[
                    (
                        tasks_allocated_to_team[team],
                        tasks_allocated_to_team[team].add(current_task),
                    ),
                    # (unallocated, unallocated.remove(current_task)),
                    (current_task, current_task + 1),
                    (used_team[team], 1),
                ],
                preconditions=[
                    current_task < self.problem.number_of_activity,
                    used_team[team] == 0,
                    tasks_allocated_to_team[team]
                    .intersection(neighbors[current_task])
                    .is_empty(),
                    compatibility[current_task].contains(team),
                ],
            )
            model.add_transition(transition)
            transition_no_new_team = dp.Transition(
                name=f"allocate_to_{team}_",
                cost=dp.IntExpr.state_cost(),
                effects=[
                    (
                        tasks_allocated_to_team[team],
                        tasks_allocated_to_team[team].add(current_task),
                    ),
                    (current_task, current_task + 1),
                ],
                preconditions=[
                    current_task < self.problem.number_of_activity,
                    used_team[team] == 1,
                    tasks_allocated_to_team[team]
                    .intersection(neighbors[current_task])
                    .is_empty(),
                    compatibility[current_task].contains(team),
                ],
            )
            model.add_transition(
                transition_no_new_team, forced=kwargs["force_allocation_when_possible"]
            )
            transitions_dict["new_team"].append(transition)
            transitions_dict["not_new_team"].append(transition_no_new_team)

        finish_transition = dp.Transition(
            name=f"finish",
            cost=dp.IntExpr.state_cost(),
            effects=[(finish, 1)],
            preconditions=[current_task == self.problem.number_of_activity],
        )
        transitions_dict["finish"] = finish_transition
        from discrete_optimization.workforce.allocation.utils import (
            compute_all_overlapping,
        )

        overlaps = compute_all_overlapping(self.problem)
        model.add_dual_bound(max([len(x) for x in overlaps]) - sum(used_team))
        model.add_transition(finish_transition)
        model.add_dual_bound(0)
        self.model = model
        self.transitions = transitions_dict

    def init_model_multi_objective(self, **kwargs: Any) -> None:
        self.problem: TeamAllocationProblemMultiobj
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        symmbreak_on_used = kwargs["symmbreak_on_used"]
        model = dp.Model(float_cost=True)
        compatibility = self.problem.compute_allowed_team_index_all_task()
        task = model.add_object_type(number=self.problem.number_of_activity)
        teams = model.add_object_type(number=self.problem.number_of_teams)
        compatibility = model.add_set_table(compatibility, object_type=teams)
        tasks_allocated_to_team = [
            model.add_set_var(object_type=task, target=set())
            for i in range(self.problem.number_of_teams)
        ]
        # unallocated = model.add_set_var(object_type=task, target=range(self.problem.number_of_activity))
        neighbors = [set() for i in range(self.problem.number_of_activity)]
        for edge in self.problem.graph_activity.edges:
            ind1 = self.problem.index_activities_name[edge[0]]
            ind2 = self.problem.index_activities_name[edge[1]]
            neighbors[ind1].add(ind2)
            neighbors[ind2].add(ind1)
        neighbors = model.add_set_table(
            [model.create_set_const(object_type=task, value=n) for n in neighbors]
        )
        current_task = model.add_element_var(object_type=task, target=0)
        # used_team_set = model.add_set_var(object_type=teams, target=set())
        used_team = [
            model.add_int_var(target=0) for _ in range(self.problem.number_of_teams)
        ]
        cumuls = {}
        dict_values = {}
        max_cumul = {}
        min_cumul = {}
        for obj in self.problem.attributes_cumul_activities:
            d_value = self.problem.attributes_of_activities[obj]
            dict_values[obj] = model.add_float_table(
                [
                    d_value[self.problem.activities_name[i]]
                    for i in range(self.problem.number_of_activity)
                ]
            )
            cumuls[obj] = [
                model.add_float_var(target=0)
                for _ in range(self.problem.number_of_teams)
            ]
            max_cumul[obj] = model.add_float_var(target=0)
            min_cumul[obj] = model.add_float_var(target=0)
        finish = model.add_int_var(target=0)
        model.add_base_case(
            [current_task == self.problem.number_of_activity, finish == 1]
        )
        if symmbreak_on_used:
            groups = compute_equivalent_teams(team_allocation_problem=self.problem)
            for group in groups:
                for team_0, team_1 in zip(group[:-1], group[1:]):
                    model.add_state_constr(used_team[team_0] >= used_team[team_1])
        for obj in []:  # ["duration"]:
            for team in range(self.problem.number_of_teams):
                for team2 in range(self.problem.number_of_teams):
                    if team2 == team:
                        continue
                    model.add_state_constr(
                        ((used_team[team] == 1) & (used_team[team2] == 1)).if_then_else(
                            cumuls[obj][team] - cumuls[obj][team2], 0
                        )
                        <= 1000
                    )
        transitions_dict = {"new_team": [], "not_new_team": []}
        for team in range(self.problem.number_of_teams):
            # new_team = (used_team[team] == 0).if_then_else(1, 0)
            additional = 0
            for obj in max_cumul:
                additional += (
                    dp.max(
                        max_cumul[obj],
                        cumuls[obj][team] + dict_values[obj][current_task],
                    )
                    - max_cumul[obj]
                )
            transition = dp.Transition(
                name=f"allocate_to_{team}",
                cost=10000 + dp.FloatExpr.state_cost(),
                effects=[
                    (
                        tasks_allocated_to_team[team],
                        tasks_allocated_to_team[team].add(current_task),
                    ),
                    # (unallocated, unallocated.remove(current_task)),
                    (current_task, current_task + 1),
                    (used_team[team], 1),
                ]
                + [
                    (
                        cumuls[obj][team],
                        cumuls[obj][team] + dict_values[obj][current_task],
                    )
                    for obj in cumuls
                ]
                + [
                    (
                        max_cumul[obj],
                        dp.max(
                            max_cumul[obj],
                            cumuls[obj][team] + dict_values[obj][current_task],
                        ),
                    )
                    for obj in cumuls
                ],
                preconditions=[
                    current_task < self.problem.number_of_activity,
                    used_team[team] == 0,
                    tasks_allocated_to_team[team]
                    .intersection(neighbors[current_task])
                    .is_empty(),
                    compatibility[current_task].contains(team),
                ],
            )
            model.add_transition(transition)
            transition_no_new_team = dp.Transition(
                name=f"allocate_to_{team}_",
                cost=dp.FloatExpr.state_cost(),
                effects=[
                    (
                        tasks_allocated_to_team[team],
                        tasks_allocated_to_team[team].add(current_task),
                    ),
                    (current_task, current_task + 1),
                ]
                + [
                    (
                        cumuls[obj][team],
                        cumuls[obj][team] + dict_values[obj][current_task],
                    )
                    for obj in cumuls
                ]
                + [
                    (
                        max_cumul[obj],
                        dp.max(
                            max_cumul[obj],
                            cumuls[obj][team] + dict_values[obj][current_task],
                        ),
                    )
                    for obj in cumuls
                ],
                preconditions=[
                    current_task < self.problem.number_of_activity,
                    used_team[team] == 1,
                    tasks_allocated_to_team[team]
                    .intersection(neighbors[current_task])
                    .is_empty(),
                    compatibility[current_task].contains(team),
                ],
            )
            model.add_transition(transition_no_new_team)
            transitions_dict["new_team"].append(transition)
            transitions_dict["not_new_team"].append(transition_no_new_team)
        # expr = 0
        # for obj in max_cumul:
        #     expr += max_cumul[obj]
        expr = 0
        for obj in max_cumul:
            for i in range(self.problem.number_of_teams):
                for j in range(i + 1, self.problem.number_of_teams):
                    expr += abs(cumuls[obj][i] - cumuls[obj][j])
            expr /= self.problem.number_of_teams
        finish_transition = dp.Transition(
            name=f"finish",
            cost=expr + dp.FloatExpr.state_cost(),
            effects=[(finish, 1)],
            preconditions=[current_task == self.problem.number_of_activity],
        )
        transitions_dict["finish"] = finish_transition
        # from allocation.allocation_problem_utils import compute_all_overlapping
        # overlaps = compute_all_overlapping(self.problem)
        # model.add_dual_bound(max([len(x) for x in overlaps])-sum(used_team))
        model.add_transition(finish_transition)
        self.model = model
        self.transitions = transitions_dict

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        if isinstance(self.problem, TeamAllocationProblemMultiobj):
            self.init_model_multi_objective(**kwargs)
        else:
            self.init_model_mono_objective(**kwargs)

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        def extract_ints(word):
            return tuple(int(num) for num in re.findall(r"\d+", word))

        allocation = []
        for t in sol.transitions:
            if "finish" in t.name:
                continue
            team = extract_ints(t.name)[0]
            allocation.append(team)
        solution = TeamAllocationSolution(problem=self.problem, allocation=allocation)
        logger.info(f"{self.problem.evaluate(solution)}")
        return solution

    def set_warm_start(self, solution: TeamAllocationSolution) -> None:
        self.initial_solution = []
        teams_used = set()
        for i in range(len(solution.allocation)):
            t = solution.allocation[i]
            if t not in teams_used:
                self.initial_solution.append(self.transitions["new_team"][t])
            else:
                self.initial_solution.append(self.transitions["not_new_team"][t])
            teams_used.add(t)
        self.initial_solution.append(self.transitions["finish"])
