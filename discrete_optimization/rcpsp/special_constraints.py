#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Dict, Hashable, List, Optional, Set, Tuple, Union

import networkx as nx


class PairModeConstraint:
    """
    This data structure exists to express the following constraints :
    if allowed_mode_assignment is not None :
    for each key of the dictionary (activity_1, activity_2),
    the allowed modes assignment to those 2 activity are listed in allowed_mode_assignment[(activity_1, activity_2)]
    This can be used to express that a given resource has to do both activities.

    Another way of expressing the constraint is :
    -each interesting tuple (task,mode) is given an int score. (score_mode)
    -We get a set of pairs of activities (same_score_mode) : indicating that the score_mode of the 2 activity should be
    equal.
    """

    def __init__(
        self,
        allowed_mode_assignment: Optional[
            Dict[Tuple[Hashable, Hashable], Set[Tuple[int, int]]]
        ] = None,
        same_score_mode: Optional[Set[Tuple[Hashable, Hashable]]] = None,
        score_mode: Dict[Tuple[Hashable, int], int] = None,
    ):
        self.allowed_mode_assignment = allowed_mode_assignment
        self.same_score_mode = same_score_mode
        self.score_mode = score_mode
        # We don't allow both ways of expressing the constraint.
        assert (
            allowed_mode_assignment is None
            and (same_score_mode is not None and score_mode is not None)
        ) or (allowed_mode_assignment is not None)


class SpecialConstraintsDescription:
    def __init__(
        self,
        task_mode: Optional[Dict[Hashable, int]] = None,
        start_times: Optional[Dict[Hashable, int]] = None,
        end_times: Optional[Dict[Hashable, int]] = None,
        start_times_window: Optional[
            Dict[Hashable, Tuple[Union[int, None], Union[int, None]]]
        ] = None,
        end_times_window: Optional[
            Dict[Hashable, Tuple[Union[int, None], Union[int, None]]]
        ] = None,
        partial_permutation: Optional[List[Hashable]] = None,
        list_partial_order: Optional[List[List[Hashable]]] = None,
        start_together: Optional[List[Tuple[Hashable, Hashable]]] = None,
        start_at_end: Optional[List[Tuple[Hashable, Hashable]]] = None,
        start_at_end_plus_offset: Optional[List[Tuple[Hashable, Hashable, int]]] = None,
        start_after_nunit: Optional[List[Tuple[Hashable, Hashable, int]]] = None,
        disjunctive_tasks: Optional[List[Tuple[Hashable, Hashable]]] = None,
        pair_mode_constraint: Optional[PairModeConstraint] = None,
    ):
        self.task_mode = task_mode
        self.start_times = start_times
        self.end_times = end_times
        self.partial_permutation = partial_permutation
        self.list_partial_order = list_partial_order
        if start_times_window is None:
            self.start_times_window = {}
        else:
            self.start_times_window = start_times_window
        if end_times_window is None:
            self.end_times_window = {}
        else:
            self.end_times_window = end_times_window
        if start_together is None:
            self.start_together = []
        else:
            self.start_together = start_together
        if start_at_end is None:
            self.start_at_end = []
        else:
            self.start_at_end = start_at_end
        if start_at_end_plus_offset is None:
            self.start_at_end_plus_offset = []
        else:
            self.start_at_end_plus_offset = start_at_end_plus_offset
        if start_after_nunit is None:
            self.start_after_nunit = []
        else:
            self.start_after_nunit = start_after_nunit
        if disjunctive_tasks is None:
            self.disjunctive_tasks = []
        else:
            self.disjunctive_tasks = disjunctive_tasks
        self.dict_start_together: Dict[Hashable, Set[Hashable]] = {}
        self.graph_start_together = nx.Graph()
        for i, j in self.start_together:
            if i not in self.graph_start_together:
                self.graph_start_together.add_node(i)
            if j not in self.graph_start_together:
                self.graph_start_together.add_node(j)
            self.graph_start_together.add_edge(i, j)
        self.components = [
            c for c in nx.connected_components(self.graph_start_together)
        ]
        for c in self.components:
            for j in c:
                self.dict_start_together[j] = set([k for k in c if k != j])
        self.dict_start_at_end: Dict[Hashable, Set[Hashable]] = {}
        self.dict_start_at_end_reverse: Dict[Hashable, Set[Hashable]] = {}
        for i, j in self.start_at_end:
            if i not in self.dict_start_at_end:
                self.dict_start_at_end[i] = set()
            if j not in self.dict_start_at_end_reverse:
                self.dict_start_at_end_reverse[j] = set()
            self.dict_start_at_end[i].add(j)
            self.dict_start_at_end_reverse[j].add(i)

        self.dict_start_at_end_offset: Dict[Hashable, Dict[Hashable, int]] = {}
        self.dict_start_at_end_offset_reverse: Dict[Hashable, Dict[Hashable, int]] = {}
        for i, j, off in self.start_at_end_plus_offset:
            if i not in self.dict_start_at_end_offset:
                self.dict_start_at_end_offset[i] = {}
            if j not in self.dict_start_at_end_offset_reverse:
                self.dict_start_at_end_offset_reverse[j] = {}
            self.dict_start_at_end_offset_reverse[j][i] = off
            self.dict_start_at_end_offset[i][j] = off

        self.dict_start_after_nunit: Dict[Hashable, Dict[Hashable, int]] = {}
        self.dict_start_after_nunit_reverse: Dict[Hashable, Dict[Hashable, int]] = {}
        for i, j, off in self.start_after_nunit:
            if i not in self.dict_start_after_nunit:
                self.dict_start_after_nunit[i] = {}
            if j not in self.dict_start_after_nunit_reverse:
                self.dict_start_after_nunit_reverse[j] = {}
            self.dict_start_after_nunit[i][j] = off
            self.dict_start_after_nunit_reverse[j][i] = off

        self.dict_disjunctive: Dict[Hashable, Set[Hashable]] = {}
        for i, j in self.disjunctive_tasks:
            if i not in self.dict_disjunctive:
                self.dict_disjunctive[i] = set()
            if j not in self.dict_disjunctive:
                self.dict_disjunctive[j] = set()
            self.dict_disjunctive[i].add(j)
            self.dict_disjunctive[j].add(i)
        self.pair_mode_constraint = pair_mode_constraint
