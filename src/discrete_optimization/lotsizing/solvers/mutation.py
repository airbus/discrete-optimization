#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Lot sizing specific mutations.

Implements GPI (Generalized Pairwise Interchange) moves from:
Ceschia, Di Gaspero, Schaerf (2017) - "Solving discrete lot-sizing and
scheduling by simulated annealing and mixed integer programming"
"""

import random
from typing import Any, Optional

from discrete_optimization.generic_tools.do_mutation import (
    LocalMove,
    LocalMoveDefault,
    SingleAttributeMutation,
)
from discrete_optimization.generic_tools.do_problem import Problem, Solution
from discrete_optimization.generic_tools.encoding_register import ListInteger


class GPIInsertMutation(SingleAttributeMutation):
    """GPI Insert move: move element from position i to position j.

    This move removes the element at position i and inserts it at position j,
    shifting intermediate elements. This is one of the two main moves from the
    Ceschia et al. (2017) paper.

    Example:
        [A, B, C, D, E] with i=1, j=3
        -> [A, C, D, B, E]  (remove B, shift C and D left, insert B at position 3)
    """

    attribute_type_cls = ListInteger
    attribute_type: ListInteger

    def __init__(
        self,
        problem: Problem,
        attribute: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        self.attribute_type: ListInteger

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        s2 = solution.copy()
        vector = list(getattr(s2, self.attribute))  # Make a copy

        size = len(vector)
        if size <= 1:
            # Nothing to move
            return s2, LocalMoveDefault(solution, s2)

        # Select two random positions
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)

        if i != j:
            # Remove element at position i and insert at position j
            element = vector.pop(i)
            vector.insert(j, element)

        setattr(s2, self.attribute, vector)
        return s2, LocalMoveDefault(solution, s2)


class GPISwapMutation(SingleAttributeMutation):
    """GPI Swap move: swap elements at positions i and j.

    This is the second main move from the Ceschia et al. (2017) paper.

    Example:
        [A, B, C, D, E] with i=1, j=3
        -> [A, D, C, B, E]  (swap B and D)
    """

    attribute_type_cls = ListInteger
    attribute_type: ListInteger

    def __init__(
        self,
        problem: Problem,
        attribute: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        self.attribute_type: ListInteger

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        s2 = solution.copy()
        vector = list(getattr(s2, self.attribute))  # Make a copy

        size = len(vector)
        if size <= 1:
            # Nothing to swap
            return s2, LocalMoveDefault(solution, s2)

        # Select two random positions
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)

        if i != j:
            # Swap elements
            vector[i], vector[j] = vector[j], vector[i]

        setattr(s2, self.attribute, vector)
        return s2, LocalMoveDefault(solution, s2)


class GPIMixedMutation(SingleAttributeMutation):
    """Mixed GPI mutation combining Insert and Swap moves.

    Randomly chooses between Insert (with probability beta) and Swap
    (with probability 1-beta). Default beta=0.7 matches the paper.
    """

    attribute_type_cls = ListInteger
    attribute_type: ListInteger

    def __init__(
        self,
        problem: Problem,
        attribute: Optional[str] = None,
        beta: float = 0.7,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        self.beta = beta  # Probability of INSERT vs SWAP
        self.insert_mutation = GPIInsertMutation(problem, attribute, **kwargs)
        self.swap_mutation = GPISwapMutation(problem, attribute, **kwargs)

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        # Choose move type based on beta
        if random.random() < self.beta:
            return self.insert_mutation.mutate(solution)
        else:
            return self.swap_mutation.mutate(solution)


class GPIInsertStockAware(SingleAttributeMutation):
    """Stock-aware INSERT: move production to reduce stock holding costs.

    Identifies items that are produced too early (leading to high stock)
    and moves them closer to their deadlines.

    Strategy:
    - Find items with high stock accumulation
    - Move them later (closer to demand deadline)
    - Reduces stock holding time
    """

    attribute_type_cls = ListInteger
    attribute_type: ListInteger

    def __init__(
        self,
        problem: Problem,
        attribute: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        from discrete_optimization.lotsizing.problem import LotSizingProblem

        self.lotsizing_problem: LotSizingProblem = problem

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        s2 = solution.copy()
        vector = list(getattr(s2, self.attribute))

        size = len(vector)
        if size <= 1:
            return s2, LocalMoveDefault(solution, s2)

        # Find production positions (not idle)
        production_positions = [
            i
            for i in range(size)
            if vector[i] >= 0 and vector[i] < self.lotsizing_problem.nb_items_type
        ]

        if len(production_positions) < 2:
            # Not enough to move, fall back to random
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
        else:
            # Calculate which items have high stock cost potential
            # Items produced early have more stock time
            item_early_production = {}
            for pos in production_positions:
                item = vector[pos]
                if item not in item_early_production:
                    item_early_production[item] = []
                item_early_production[item].append(pos)

            # Find demands for each item
            item_demands = {}
            for item in range(self.lotsizing_problem.nb_items_type):
                demands = self.lotsizing_problem.demands[item]
                demand_times = [t for t, d in enumerate(demands) if d > 0]
                if demand_times:
                    item_demands[item] = demand_times

            # Score each production: stock_cost × (deadline - production_time)
            scored_productions = []
            for item, positions in item_early_production.items():
                if item in item_demands:
                    stock_cost = (
                        self.lotsizing_problem.stock_cost_per_type_per_time_per_unit[
                            item
                        ]
                    )
                    for pos in positions:
                        # Find closest demand after this production
                        future_demands = [d for d in item_demands[item] if d >= pos]
                        if future_demands:
                            deadline = future_demands[0]
                            stock_time = deadline - pos
                            score = stock_cost * stock_time
                            scored_productions.append((score, pos, deadline))

            if scored_productions:
                # Pick production with high stock cost (top 20%)
                scored_productions.sort(reverse=True)
                top_n = max(1, len(scored_productions) // 5)
                score, from_pos, deadline = random.choice(scored_productions[:top_n])

                # Move it closer to deadline (but before deadline)
                possible_targets = list(range(from_pos + 1, min(deadline + 1, size)))
                if possible_targets:
                    to_pos = random.choice(possible_targets)
                    element = vector.pop(from_pos)
                    vector.insert(to_pos, element)
                else:
                    # Can't move closer, random fallback
                    i = random.randint(0, size - 1)
                    j = random.randint(0, size - 1)
                    if i != j:
                        element = vector.pop(i)
                        vector.insert(j, element)
            else:
                # No scored productions, random fallback
                i = random.randint(0, size - 1)
                j = random.randint(0, size - 1)
                if i != j:
                    element = vector.pop(i)
                    vector.insert(j, element)

        setattr(s2, self.attribute, vector)
        return s2, LocalMoveDefault(solution, s2)


class GPISwapChangeoverAware(SingleAttributeMutation):
    """Changeover-aware SWAP: swap items to reduce changeover costs.

    Identifies expensive changeovers in the sequence and tries to batch
    same items together to reduce setup costs.

    Strategy:
    - Find consecutive productions with high changeover cost
    - Swap one with a same-item production elsewhere
    - Creates batches of same items
    """

    attribute_type_cls = ListInteger
    attribute_type: ListInteger

    def __init__(
        self,
        problem: Problem,
        attribute: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        from discrete_optimization.lotsizing.problem import LotSizingProblem

        self.lotsizing_problem: LotSizingProblem = problem

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        s2 = solution.copy()
        vector = list(getattr(s2, self.attribute))

        size = len(vector)
        if size <= 1:
            return s2, LocalMoveDefault(solution, s2)

        # Find production positions
        production_positions = [
            i
            for i in range(size)
            if vector[i] >= 0 and vector[i] < self.lotsizing_problem.nb_items_type
        ]

        if len(production_positions) < 2:
            # Random swap fallback
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            if i != j:
                vector[i], vector[j] = vector[j], vector[i]
        else:
            # Find expensive changeovers (consecutive different items)
            changeovers = []
            for idx in range(len(production_positions) - 1):
                pos1 = production_positions[idx]
                pos2 = production_positions[idx + 1]
                item1 = vector[pos1]
                item2 = vector[pos2]

                if item1 != item2:
                    cost = self.lotsizing_problem.changeover_costs[item1][item2]
                    changeovers.append((cost, pos1, pos2, item1, item2))

            if changeovers:
                # Pick expensive changeover (top 20%)
                changeovers.sort(reverse=True)
                top_n = max(1, len(changeovers) // 5)
                cost, pos1, pos2, item1, item2 = random.choice(changeovers[:top_n])

                # Try to batch: find another production of item1 or item2
                # and swap to create consecutive same items
                item_to_batch = random.choice([item1, item2])
                other_positions = [
                    p
                    for p in production_positions
                    if p not in [pos1, pos2] and vector[p] == item_to_batch
                ]

                if other_positions:
                    # Swap with position that would create batching
                    target_pos = random.choice(other_positions)
                    # Decide which position to swap (pos1 or pos2)
                    swap_pos = pos1 if item_to_batch == item2 else pos2
                    vector[swap_pos], vector[target_pos] = (
                        vector[target_pos],
                        vector[swap_pos],
                    )
                else:
                    # No batching opportunity, random swap
                    i = random.randint(0, size - 1)
                    j = random.randint(0, size - 1)
                    if i != j:
                        vector[i], vector[j] = vector[j], vector[i]
            else:
                # No changeovers, random swap
                i = random.randint(0, size - 1)
                j = random.randint(0, size - 1)
                if i != j:
                    vector[i], vector[j] = vector[j], vector[i]

        setattr(s2, self.attribute, vector)
        return s2, LocalMoveDefault(solution, s2)


class GPIInsertDeadlineAware(SingleAttributeMutation):
    """Deadline-aware INSERT: move production closer to demand deadlines.

    More aggressive than stock-aware: directly targets items that are
    far from their deadlines.

    Strategy:
    - Find items produced way before their first demand
    - Move them just before the demand
    - Minimizes stock holding time aggressively
    """

    attribute_type_cls = ListInteger
    attribute_type: ListInteger

    def __init__(
        self,
        problem: Problem,
        attribute: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        from discrete_optimization.lotsizing.problem import LotSizingProblem

        self.lotsizing_problem: LotSizingProblem = problem

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        s2 = solution.copy()
        vector = list(getattr(s2, self.attribute))

        size = len(vector)
        if size <= 1:
            return s2, LocalMoveDefault(solution, s2)

        # Find production positions
        production_positions = [
            i
            for i in range(size)
            if vector[i] >= 0 and vector[i] < self.lotsizing_problem.nb_items_type
        ]

        if len(production_positions) < 1:
            return s2, LocalMoveDefault(solution, s2)

        # Calculate deadline gaps for each production
        deadline_gaps = []
        for pos in production_positions:
            item = vector[pos]
            demands = self.lotsizing_problem.demands[item]

            # Find next demand after this production
            future_demands = [t for t in range(pos, size) if demands[t] > 0]
            if future_demands:
                deadline = future_demands[0]
                gap = deadline - pos
                if gap > 1:  # Only if there's room to move
                    deadline_gaps.append((gap, pos, deadline, item))

        if deadline_gaps:
            # Pick production with large gap (top 30%)
            deadline_gaps.sort(reverse=True)
            top_n = max(1, len(deadline_gaps) * 3 // 10)
            gap, from_pos, deadline, item = random.choice(deadline_gaps[:top_n])

            # Move closer to deadline (randomly between halfway and just before deadline)
            min_target = from_pos + gap // 2
            max_target = min(deadline, size - 1)

            if min_target < max_target:
                to_pos = random.randint(min_target, max_target)
                element = vector.pop(from_pos)
                vector.insert(to_pos, element)
            else:
                # Small gap, random move
                i = random.randint(0, size - 1)
                j = random.randint(0, size - 1)
                if i != j:
                    element = vector.pop(i)
                    vector.insert(j, element)
        else:
            # No gaps, random move
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            if i != j:
                element = vector.pop(i)
                vector.insert(j, element)

        setattr(s2, self.attribute, vector)
        return s2, LocalMoveDefault(solution, s2)
