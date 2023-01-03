#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from typing import Any, Dict, Iterable, Optional, Tuple

from discrete_optimization.generic_tools.do_mutation import (
    LocalMove,
    LocalMoveDefault,
    Mutation,
)

# Relocate operator
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.vrp.vrp_model import BasicCustomer, VrpProblem, VrpSolution

# https://dial.uclouvain.be/memoire/ucl/fr/object/thesis%3A4615/datastream/PDF_01/
# view#:~:text=One%20of%20the%20approaches%20that,solution%20and%20iteratively%20improving%20it.


class RelocateMove(LocalMove):
    def __init__(
        self,
        index_vehicle_from: int,
        index_vehicle_to: int,
        index_from: int,
        index_to: int,
    ):
        self.index_vehicle_from = index_vehicle_from
        self.index_vehicle_to = index_vehicle_to
        self.index_from = index_from
        self.index_to = index_to

    def apply_local_move(self, solution: VrpSolution) -> VrpSolution:  # type: ignore # avoid isinstance checks for efficiency
        if (
            solution.length is None
            or solution.lengths is None
            or solution.capacities is None
        ):
            raise RuntimeError(
                "length, lengths, and capacities solution's attributes should not be None at this point."
            )
        self.index_from = min(
            [self.index_from, len(solution.list_paths[self.index_vehicle_from]) - 1]
        )
        self.index_to = min(
            [self.index_to, len(solution.list_paths[self.index_vehicle_to])]
        )
        city = solution.list_paths[self.index_vehicle_from][self.index_from]
        solution.capacities[self.index_vehicle_from] -= solution.problem.customers[
            city
        ].demand
        solution.capacities[self.index_vehicle_to] += solution.problem.customers[
            city
        ].demand
        solution.list_paths[self.index_vehicle_to].insert(self.index_to, city)
        previous_from_length = (
            solution.lengths[self.index_vehicle_from][self.index_from],
            solution.lengths[self.index_vehicle_from][self.index_from + 1],
        )

        def get_index_previous(index_vehicle: int, index_from: int) -> int:
            if index_from == 0:
                return solution.problem.start_indexes[index_vehicle]
            else:
                return solution.list_paths[index_vehicle][index_from - 1]

        def get_index_next(index_vehicle: int, index_from: int) -> int:
            if index_from >= len(solution.list_paths[index_vehicle]) - 1:
                return solution.problem.end_indexes[index_vehicle]
            else:
                return solution.list_paths[index_vehicle][index_from + 1]

        new_length_vehicle_from = solution.problem.evaluate_function_indexes(
            index_1=get_index_previous(self.index_vehicle_from, self.index_from),
            index_2=get_index_next(self.index_vehicle_from, self.index_from),
        )
        city = solution.list_paths[self.index_vehicle_from].pop(self.index_from)
        solution.lengths[self.index_vehicle_from].pop(self.index_from)
        solution.lengths[self.index_vehicle_from].pop(self.index_from)
        # remove the 2 length where the city was concerned
        solution.lengths[self.index_vehicle_from].insert(
            self.index_from, new_length_vehicle_from
        )
        # include new length

        previous_length_vehicle_to = solution.lengths[self.index_vehicle_to][
            self.index_to
        ]
        solution.lengths[self.index_vehicle_to].pop(self.index_to)

        new_length_vehicle_to_1 = solution.problem.evaluate_function_indexes(
            index_1=get_index_previous(self.index_vehicle_to, self.index_to),
            index_2=city,
        )
        solution.lengths[self.index_vehicle_to].insert(
            self.index_to, new_length_vehicle_to_1
        )
        new_length_vehicle_to_2 = solution.problem.evaluate_function_indexes(
            index_1=city, index_2=get_index_next(self.index_vehicle_to, self.index_to)
        )
        solution.lengths[self.index_vehicle_to].insert(
            self.index_to + 1, new_length_vehicle_to_2
        )
        delta = (
            new_length_vehicle_from
            + new_length_vehicle_to_1
            + new_length_vehicle_to_2
            - previous_length_vehicle_to
            - sum(previous_from_length)
        )
        solution.length = solution.length + delta

        return solution

    def backtrack_local_move(self, solution: VrpSolution) -> VrpSolution:  # type: ignore # avoid isinstance checks for efficiency
        move = RelocateMove(
            index_vehicle_from=self.index_vehicle_to,
            index_vehicle_to=self.index_vehicle_from,
            index_from=self.index_to,
            index_to=self.index_from,
        )
        return move.apply_local_move(solution)


class MutationRelocate(Mutation):
    @staticmethod
    def build(  # type: ignore # avoid isinstance checks for efficiency
        problem: VrpProblem, solution: VrpSolution, **kwargs
    ) -> "MutationRelocate":
        return MutationRelocate(problem)

    def __init__(self, vrp_model: VrpProblem):
        self.vrp_model = vrp_model
        self.customer_count = vrp_model.customer_count
        self.vehicle_count = vrp_model.vehicle_count

    def mutate(self, solution: VrpSolution) -> Tuple[VrpSolution, LocalMove]:  # type: ignore # avoid isinstance checks for efficiency
        vehicles_used = [
            v
            for v in range(self.vrp_model.vehicle_count)
            if len(solution.list_paths[v]) > 0
        ]
        some_vehicle = random.choice(vehicles_used)
        some_other_vehicle = random.choice(
            [v for v in range(self.vrp_model.vehicle_count) if v != some_vehicle]
        )
        index_from = random.choice(range(len(solution.list_paths[some_vehicle])))
        index_to = random.choice(
            range(max(1, len(solution.list_paths[some_other_vehicle])))
        )
        move = RelocateMove(
            index_vehicle_from=some_vehicle,
            index_vehicle_to=some_other_vehicle,
            index_from=index_from,
            index_to=index_to,
        )
        sol = move.apply_local_move(solution)
        return sol, move

    def mutate_and_compute_obj(  # type: ignore # avoid isinstance checks for efficiency
        self, solution: VrpSolution
    ) -> Tuple[VrpSolution, LocalMove, Dict[str, float]]:
        sol, move = self.mutate(solution)
        f = self.vrp_model.evaluate(sol)
        return sol, move, f


class SwapMove(LocalMove):
    def __init__(
        self,
        index_vehicle_from: int,
        index_vehicle_to: int,
        index_from: int,
        index_to: int,
    ):
        self.index_vehicle_from = index_vehicle_from
        self.index_vehicle_to = index_vehicle_to
        self.index_from = index_from
        self.index_to = index_to

    def apply_local_move(self, solution: VrpSolution) -> VrpSolution:  # type: ignore # avoid isinstance checks for efficiency
        if (
            solution.length is None
            or solution.lengths is None
            or solution.capacities is None
        ):
            raise RuntimeError(
                "length, lengths, and capacities solution's attributes should not be None at this point."
            )
        self.index_from = min(
            [self.index_from, len(solution.list_paths[self.index_vehicle_from]) - 1]
        )
        self.index_to = min(
            [self.index_to, len(solution.list_paths[self.index_vehicle_to]) - 1]
        )
        city_from = solution.list_paths[self.index_vehicle_from][self.index_from]
        city_to = solution.list_paths[self.index_vehicle_to][self.index_to]
        previous_from_length = (
            solution.lengths[self.index_vehicle_from][self.index_from],
            solution.lengths[self.index_vehicle_from][self.index_from + 1],
        )
        previous_to_length = (
            solution.lengths[self.index_vehicle_to][self.index_to],
            solution.lengths[self.index_vehicle_to][self.index_to + 1],
        )

        def get_index_previous(index_vehicle: int, index_from: int) -> int:
            if index_from == 0:
                return solution.problem.start_indexes[index_vehicle]
            else:
                return solution.list_paths[index_vehicle][index_from - 1]

        def get_index_next(index_vehicle: int, index_from: int) -> int:
            if index_from >= len(solution.list_paths[index_vehicle]) - 1:
                return solution.problem.end_indexes[index_vehicle]
            else:
                return solution.list_paths[index_vehicle][index_from + 1]

        new_length_vehicle_from_1 = solution.problem.evaluate_function_indexes(
            index_1=get_index_previous(self.index_vehicle_from, self.index_from),
            index_2=city_to,
        )
        new_length_vehicle_from_2 = solution.problem.evaluate_function_indexes(
            index_1=city_to,
            index_2=get_index_next(self.index_vehicle_from, self.index_from),
        )
        new_length_vehicle_to_1 = solution.problem.evaluate_function_indexes(
            index_1=get_index_previous(self.index_vehicle_to, self.index_to),
            index_2=city_from,
        )
        new_length_vehicle_to_2 = solution.problem.evaluate_function_indexes(
            index_1=city_from,
            index_2=get_index_next(self.index_vehicle_to, self.index_to),
        )
        solution.list_paths[self.index_vehicle_from][self.index_from] = city_to
        solution.list_paths[self.index_vehicle_to][self.index_to] = city_from
        solution.lengths[self.index_vehicle_from][
            self.index_from
        ] = new_length_vehicle_from_1
        solution.lengths[self.index_vehicle_from][
            self.index_from + 1
        ] = new_length_vehicle_from_2
        solution.lengths[self.index_vehicle_to][self.index_to] = new_length_vehicle_to_1
        solution.lengths[self.index_vehicle_to][
            self.index_to + 1
        ] = new_length_vehicle_to_2
        solution.capacities[self.index_vehicle_from] += (
            solution.problem.customers[city_to].demand
            - solution.problem.customers[city_from].demand
        )

        solution.capacities[self.index_vehicle_to] += (
            solution.problem.customers[city_from].demand
            - solution.problem.customers[city_to].demand
        )
        delta = (
            new_length_vehicle_to_1
            + new_length_vehicle_to_2
            + new_length_vehicle_from_1
            + new_length_vehicle_from_2
            - sum(previous_to_length)
            - sum(previous_from_length)
        )
        solution.length = solution.length + delta
        return solution

    def backtrack_local_move(self, solution: VrpSolution) -> VrpSolution:  # type: ignore # avoid isinstance checks for efficiency
        move = SwapMove(
            index_vehicle_from=self.index_vehicle_from,
            index_vehicle_to=self.index_vehicle_to,
            index_from=self.index_from,
            index_to=self.index_to,
        )
        return move.apply_local_move(solution)


class MutationSwap(Mutation):
    @staticmethod
    def build(problem: VrpProblem, solution: VrpSolution, **kwargs) -> "MutationSwap":  # type: ignore # avoid isinstance checks for efficiency
        return MutationSwap(problem)

    def __init__(self, vrp_model: VrpProblem):
        self.vrp_model = vrp_model
        self.customer_count = vrp_model.customer_count
        self.vehicle_count = vrp_model.vehicle_count

    def mutate(self, solution: VrpSolution) -> Tuple[VrpSolution, LocalMove]:  # type: ignore # avoid isinstance checks for efficiency
        vehicles_used = [
            v
            for v in range(self.vrp_model.vehicle_count)
            if len(solution.list_paths[v]) > 0
        ]
        some_vehicle = random.choice(vehicles_used)
        swap = True
        if len(vehicles_used) > 1:
            swap = True
            some_other_vehicle = random.choice(
                [v for v in vehicles_used if v != some_vehicle]
            )
        else:
            some_other_vehicle = random.choice(
                [v for v in range(self.vrp_model.vehicle_count) if v != some_vehicle]
            )
            swap = False
        index_from = random.choice(range(len(solution.list_paths[some_vehicle])))
        index_to = random.choice(
            range(max(1, len(solution.list_paths[some_other_vehicle])))
        )
        move: LocalMove
        if swap:
            move = SwapMove(
                index_vehicle_from=some_vehicle,
                index_vehicle_to=some_other_vehicle,
                index_from=index_from,
                index_to=index_to,
            )
            sol = move.apply_local_move(solution)
        else:
            move = RelocateMove(
                index_vehicle_from=some_vehicle,
                index_vehicle_to=some_other_vehicle,
                index_from=index_from,
                index_to=index_to,
            )
            sol = move.apply_local_move(solution)

        return sol, move

    def mutate_and_compute_obj(  # type: ignore # avoid isinstance checks for efficiency
        self, solution: VrpSolution
    ) -> Tuple[Solution, LocalMove, Dict[str, float]]:
        sol, move = self.mutate(solution)
        f = self.vrp_model.evaluate(sol)
        return sol, move, f


class MutationTwoOptVRP(Mutation):
    node_count: int

    @staticmethod
    def build(  # type: ignore # avoid isinstance checks for efficiency
        problem: VrpProblem, solution: VrpSolution, **kwargs
    ) -> "MutationTwoOptVRP":
        return MutationTwoOptVRP(problem, **kwargs)

    def __init__(
        self,
        vrp_model: VrpProblem,
        test_all: bool = False,
        nb_test: Optional[int] = None,
        return_only_improvement: bool = False,
        **kwargs: Any
    ):
        self.node_count = vrp_model.customer_count
        self.points = vrp_model.customers
        self.test_all = test_all
        self.evaluate_function_indexes = vrp_model.evaluate_function_indexes
        self.return_only_improvement = return_only_improvement
        self.vrp_model = vrp_model
        if nb_test is None:
            self.nb_test = max(1, self.node_count // 10)
        else:
            self.nb_test = min(nb_test, self.node_count - 1)

    def get_points(
        self, vehicle: int, it: int, jt: int, variable: VrpSolution
    ) -> Tuple[BasicCustomer, BasicCustomer, BasicCustomer, BasicCustomer]:
        perm = variable.list_paths[vehicle]
        if it == 0:
            point_before_i = self.points[variable.list_start_index[vehicle]]
        else:
            point_before_i = self.points[perm[it - 1]]
        point_i = self.points[perm[it]]
        point_j = self.points[perm[jt]]
        if jt >= len(perm) - 1:
            point_after_j = self.points[variable.list_end_index[vehicle]]
        else:
            point_after_j = self.points[perm[jt + 1]]
        return point_before_i, point_i, point_j, point_after_j

    def get_points_index(
        self, vehicle: int, it: int, jt: int, variable: VrpSolution
    ) -> Tuple[int, int, int, int]:
        i_before = None
        j_after = None
        perm = variable.list_paths[vehicle]
        i = perm[it]
        j = perm[jt]
        if it == 0:
            i_before = variable.list_start_index[vehicle]
        else:
            i_before = perm[it - 1]
        if jt >= len(perm) - 1:
            j_after = variable.list_end_index[vehicle]
        else:
            j_after = perm[jt + 1]
        return i_before, i, j, j_after

    def mutate_and_compute_obj(self, variable: VrpSolution) -> Tuple[VrpSolution, LocalMove, Dict[str, float]]:  # type: ignore # avoid isinstance checks for efficiency
        if (
            variable.length is None
            or variable.lengths is None
            or variable.capacities is None
        ):
            raise RuntimeError(
                "length, lengths, and capacities variable's attributes should not be None at this point."
            )
        vehicles_used = [
            v
            for v in range(self.vrp_model.vehicle_count)
            if len(variable.list_paths[v]) > 2
        ]
        some_vehicle = random.choice(vehicles_used)
        it = random.randint(0, len(variable.list_paths[some_vehicle]) - 2)
        jt = random.randint(it + 1, len(variable.list_paths[some_vehicle]) - 1)
        min_change = float("inf")
        length_permut = len(variable.list_paths[some_vehicle])
        range_its: Iterable[int] = (
            range(length_permut)
            if self.test_all
            else random.sample(range(length_permut), min(self.nb_test, length_permut))
        )
        for i in range_its:
            if i == length_permut - 1:
                range_jts: Iterable[int] = []
            else:
                range_jts = (
                    range(i + 1, length_permut)
                    if self.test_all
                    else random.sample(
                        range(i + 1, length_permut),
                        min(1, self.nb_test, length_permut - i - 1),
                    )
                )
            for j in range_jts:
                i_before, i_, j_, j_after = self.get_points_index(
                    some_vehicle, i, j, variable
                )
                change = (
                    self.evaluate_function_indexes(i_before, j_)
                    - self.evaluate_function_indexes(i_before, i_)
                    - self.evaluate_function_indexes(j_, j_after)
                    + self.evaluate_function_indexes(i_, j_after)
                )
                if change < min_change:
                    it = i
                    jt = j
                    min_change = change
        fitness = variable.length + min_change
        i_before, i_, j_, j_after = self.get_points_index(
            some_vehicle, it, jt, variable
        )
        permut = (
            variable.list_paths[some_vehicle][:it]
            + variable.list_paths[some_vehicle][it : jt + 1][::-1]
            + variable.list_paths[some_vehicle][jt + 1 :]
        )
        lengths = []
        if it > 0:
            lengths += variable.lengths[some_vehicle][:it]
        lengths += [self.evaluate_function_indexes(i_before, j_)]
        lengths += variable.lengths[some_vehicle][it + 1 : jt + 1][::-1]
        lengths += [self.evaluate_function_indexes(i_, j_after)]
        if jt < length_permut - 1:
            lengths += variable.lengths[some_vehicle][jt + 2 :]

        if min_change < 0 or not self.return_only_improvement:
            v = VrpSolution(
                list_start_index=variable.list_start_index,
                list_end_index=variable.list_end_index,
                list_paths=[
                    permut if j == some_vehicle else variable.list_paths[j]
                    for j in range(len(variable.list_paths))
                ],
                lengths=[
                    lengths if j == some_vehicle else variable.lengths[j]
                    for j in range(len(variable.lengths))
                ],
                length=fitness,
                capacities=variable.capacities,
                problem=self.vrp_model,
            )
            return v, LocalMoveDefault(variable, v), self.vrp_model.evaluate(v)
        else:
            return (
                variable,
                LocalMoveDefault(variable, variable),
                self.vrp_model.evaluate(variable),
            )

    def mutate(self, variable: VrpSolution) -> Tuple[VrpSolution, LocalMove]:  # type: ignore # avoid isinstance checks for efficiency
        v, move, f = self.mutate_and_compute_obj(variable)
        return v, move
