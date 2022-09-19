#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from typing import Dict, List, Tuple

from discrete_optimization.generic_tools.do_mutation import (
    LocalMove,
    LocalMoveDefault,
    Mutation,
)
from discrete_optimization.tsp.tsp_model import (
    Point,
    Point2D,
    SolutionTSP,
    TSPModel,
    TSPModel2D,
)


def ccw(A: Point2D, B: Point2D, C: Point2D):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


# Return true if line segments AB and CD intersect
def intersect(A: Point2D, B: Point2D, C: Point2D, D: Point2D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def find_intersection(
    variable: SolutionTSP, points: List[Point2D], test_all=False, nb_tests=10
):
    perm = variable.permutation
    intersects = []
    its = (
        range(len(perm))
        if test_all
        else random.sample(range(len(perm)), min(nb_tests, len(perm)))
    )
    jts = (
        range(len(perm) - 1)
        if test_all
        else random.sample(range(len(perm) - 1), min(nb_tests, len(perm) - 1))
    )
    for i in its:
        for j in jts:
            ii = i
            jj = j
            if jj <= ii + 1:
                continue
            A, B = points[perm[ii]], points[perm[ii + 1]]
            C, D = points[perm[jj]], points[perm[jj + 1]]
            if intersect(A, B, C, D):
                intersects += [(ii + 1, jj)]
                if len(intersects) > 5:
                    break
        if len(intersects) > 5:
            break
    return intersects


def get_points_index(it, jt, variable: SolutionTSP, length_permutation: int):
    perm = variable.permutation
    i = perm[it]
    j = perm[jt]
    if it == 0:
        i_before = variable.start_index
    else:
        i_before = perm[it - 1]
    if jt == length_permutation - 1:
        j_after = variable.end_index
    else:
        j_after = perm[jt + 1]
    return i_before, i, j, j_after


class Mutation2Opt(Mutation):
    node_count: int
    points: List[Point]

    @staticmethod
    def build(problem: TSPModel2D, solution: SolutionTSP, **kwargs):
        return Mutation2Opt(problem, **kwargs)

    def __init__(
        self,
        tsp_model: TSPModel2D,
        test_all=False,
        nb_test=None,
        return_only_improvement=False,
        **kwargs
    ):
        self.node_count = tsp_model.node_count
        self.length_permutation = tsp_model.length_permutation
        self.points = tsp_model.list_points
        self.test_all = test_all
        self.nb_test = min(nb_test, self.node_count - 1)
        self.evaluate_function_indexes = tsp_model.evaluate_function_indexes
        self.return_only_improvement = return_only_improvement
        self.tsp_model = tsp_model
        if self.nb_test is None:
            self.nb_test = max(1, self.node_count // 10)

    def get_points(self, it, jt, variable: SolutionTSP):
        perm = variable.permutation
        if it == 0:
            point_before_i = self.points[variable.start_index]
        else:
            point_before_i = self.points[perm[it - 1]]
        point_i = self.points[perm[it]]
        point_j = self.points[perm[jt]]
        if jt == self.length_permutation - 1:
            point_after_j = self.points[variable.end_index]
        else:
            point_after_j = self.points[perm[jt + 1]]
        return point_before_i, point_i, point_j, point_after_j

    def get_points_index(self, it, jt, variable: SolutionTSP):
        perm = variable.permutation
        i = perm[it]
        j = perm[jt]
        if it == 0:
            i_before = variable.start_index
        else:
            i_before = perm[it - 1]
        if jt == self.length_permutation - 1:
            j_after = variable.end_index
        else:
            j_after = perm[jt + 1]
        return i_before, i, j, j_after

    def mutate_and_compute_obj(self, variable: SolutionTSP):
        it = random.randint(0, self.length_permutation - 2)
        jt = random.randint(it + 1, self.length_permutation - 1)
        min_change = float("inf")
        range_its = (
            range(self.length_permutation)
            if self.test_all
            else random.sample(
                range(self.length_permutation),
                min(self.nb_test, self.length_permutation),
            )
        )
        for i in range_its:
            if i == self.length_permutation - 1:
                range_jts = []
            else:
                range_jts = (
                    range(i + 1, self.length_permutation)
                    if self.test_all
                    else random.sample(
                        range(i + 1, self.length_permutation),
                        min(1, self.nb_test, self.length_permutation - i - 1),
                    )
                )
            for j in range_jts:
                i_before, i_, j_, j_after = self.get_points_index(i, j, variable)
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
        i_before, i_, j_, j_after = self.get_points_index(it, jt, variable)
        permut = (
            variable.permutation[:it]
            + variable.permutation[it : jt + 1][::-1]
            + variable.permutation[jt + 1 :]
        )
        lengths = []
        if it > 0:
            lengths += variable.lengths[:it]
        lengths += [self.evaluate_function_indexes(i_before, j_)]
        lengths += variable.lengths[it + 1 : jt + 1][::-1]
        lengths += [self.evaluate_function_indexes(i_, j_after)]
        if jt < self.length_permutation - 1:
            lengths += variable.lengths[jt + 2 :]
        if min_change < 0 or not self.return_only_improvement:
            v = SolutionTSP(
                start_index=variable.start_index,
                end_index=variable.end_index,
                permutation=permut,
                lengths=lengths,
                length=fitness,
                problem=self.tsp_model,
            )
            return v, LocalMoveDefault(variable, v), {"length": fitness}
        else:
            return (
                variable,
                LocalMoveDefault(variable, variable),
                {"length": variable.length},
            )

    def mutate(self, variable: SolutionTSP):
        v, move, f = self.mutate_and_compute_obj(variable)
        return v, move


class Mutation2OptIntersection(Mutation2Opt):
    nodeCount: int
    points: List[Point]

    @staticmethod
    def build(problem: TSPModel2D, solution: SolutionTSP, **kwargs):
        return Mutation2OptIntersection(problem, **kwargs)

    def __init__(
        self,
        tsp_model: TSPModel2D,
        test_all=True,
        nb_test=None,
        return_only_improvement=False,
        i_j_pairs=None,
        **kwargs
    ):
        Mutation2Opt.__init__(
            self, tsp_model, test_all, nb_test, return_only_improvement
        )
        self.tsp_model = tsp_model
        self.i_j_pairs = i_j_pairs
        if self.i_j_pairs is None:
            self.i_j_pairs = None

    def mutate_and_compute_obj(self, variable: SolutionTSP):
        reset_end = True
        ints = find_intersection(
            variable, self.points, nb_tests=min(3000, self.node_count - 2)
        )
        self.i_j_pairs = ints
        if len(self.i_j_pairs) == 0:
            return (
                variable,
                LocalMoveDefault(variable, variable),
                {"length": variable.length},
            )
        min_change = float("inf")
        for i, j in self.i_j_pairs:
            i_before, i_, j_, j_after = self.get_points_index(i, j, variable)
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
        i_before, i_, j_, j_after = self.get_points_index(it, jt, variable)
        permut = (
            variable.permutation[:it]
            + variable.permutation[it : jt + 1][::-1]
            + variable.permutation[jt + 1 :]
        )
        lengths = []
        if it > 0:
            lengths += variable.lengths[:it]
        lengths += [self.evaluate_function_indexes(i_before, j_)]
        lengths += variable.lengths[it + 1 : jt + 1][::-1]
        lengths += [self.evaluate_function_indexes(i_, j_after)]
        if jt < self.length_permutation - 1:
            lengths += variable.lengths[jt + 2 :]
        if reset_end:
            self.i_j_pairs = None
        if min_change < 0 or not self.return_only_improvement:
            v = SolutionTSP(
                start_index=variable.start_index,
                end_index=variable.end_index,
                permutation=permut,
                lengths=lengths,
                length=fitness,
                problem=self.tsp_model,
            )
            return v, LocalMoveDefault(variable, v), {"length": fitness}
        else:
            return (
                variable,
                LocalMoveDefault(variable, variable),
                {"length": variable.length},
            )


class SwapTSPMove(LocalMove):
    def __init__(self, attribute, tsp_model: TSPModel, swap: Tuple[int, int]):
        self.attribute = attribute
        self.tsp_model = tsp_model
        self.swap = swap

    def apply_local_move(self, solution: SolutionTSP) -> SolutionTSP:
        current = getattr(solution, self.attribute)
        i1, i2 = self.swap
        v1, v2 = current[i1], current[i2]
        i_before, i, j, j_after = get_points_index(
            i1, i2, solution, self.tsp_model.length_permutation
        )
        current[i1], current[i2] = v2, v1
        previous = (
            solution.lengths[i1],
            solution.lengths[i1 + 1],
            solution.lengths[i2],
            solution.lengths[i2 + 1],
        )
        solution.lengths[i1] = self.tsp_model.evaluate_function_indexes(
            i_before, current[i1]
        )
        solution.lengths[i1 + 1] = self.tsp_model.evaluate_function_indexes(
            current[i1], current[i1 + 1]
        )
        solution.lengths[i2] = self.tsp_model.evaluate_function_indexes(
            current[i2 - 1], current[i2]
        )
        solution.lengths[i2 + 1] = self.tsp_model.evaluate_function_indexes(
            current[i2], j_after
        )
        solution.length = (
            solution.length
            + solution.lengths[i1]
            + solution.lengths[i1 + 1]
            + solution.lengths[i2]
            + solution.lengths[i2 + 1]
            - sum(previous)
        )
        return solution

    def backtrack_local_move(self, solution: SolutionTSP) -> SolutionTSP:
        return self.apply_local_move(solution)


class MutationSwapTSP(Mutation):
    @staticmethod
    def build(problem: TSPModel, solution: SolutionTSP, **kwargs):
        return MutationSwapTSP(problem)

    def __init__(self, tsp_model: TSPModel):
        self.tsp_model = tsp_model
        self.length_permutation = tsp_model.length_permutation

    def mutate(self, solution: SolutionTSP) -> Tuple[SolutionTSP, LocalMove]:
        i = random.randint(0, self.length_permutation - 3)
        j = random.randint(i + 2, min(self.length_permutation - 1, i + 1 + 4))
        two_opt_move = SwapTSPMove("permutation", self.tsp_model, (i, j))
        new_sol = two_opt_move.apply_local_move(solution)
        return new_sol, two_opt_move

    def mutate_and_compute_obj(
        self, solution: SolutionTSP
    ) -> Tuple[SolutionTSP, LocalMove, Dict[str, float]]:
        sol, move = self.mutate(solution)
        return sol, move, {"length": sol.length}
