#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""CPM to compute lower/uppe bound on start/end of tasks"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import TYPE_CHECKING, Any, Generic, Optional

import networkx as nx

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    CumulativeResource,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.skill import (
    NonSkillCumulativeResource,
    Skill,
)

if TYPE_CHECKING:  # avoid circular imports due to annotations
    from discrete_optimization.generic_tasks_tools.generic_scheduling import (
        GenericSchedulingProblem,
    )

logger = logging.getLogger(__name__)


@dataclass
class Taskbounds:
    start_lower_bound: Optional[int] = None
    end_lower_bound: Optional[int] = None
    start_upper_bound: Optional[int] = None
    end_upper_bound: Optional[int] = None


class Cpm(Generic[Task, UnaryResource, CumulativeResource, NonRenewableResource]):
    """Propagator of bounds trough precedence graph according to minimum duration of each task.

    For that we use:
    - multimode-scheduling: we know all possible durations
    - precedence: we use the precedence constraints

    We first do a forward pass on all tasks in precedence graph for lower bounds and then a backward pass for upper bounds.

    We start from lower bound 0 and upper bound horizon (problem horizon can be overriden).

    This differs from classic CPM as we know that other non-precedence constraints can occur so that we cannot assume
    end lower_bounds = end upper bound for sink tasks. So sink tasks end upper bound will be set to horizon instead.

    For each task, we take best of propagated bound and existing problem bound on start/end
    (via `problem.get_task_start_or_end_tighter_lower_bound()` and `problem.get_task_start_or_end_tighter_upper_bound()`).

    Attributes:
        horizon: if set, override `problem.get_makespan_upper_bound()`.

    """

    def __init__(
        self,
        problem: GenericSchedulingProblem[
            Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
        ],
        horizon: Optional[int] = None,
    ):
        self.problem = problem
        if horizon is None:
            self.horizon = self.problem.get_makespan_upper_bound()
        else:
            self.horizon = horizon
        self.graph_nx = self.problem.get_precedence_graph().to_networkx()
        self.map_node: dict[Any, Taskbounds] = {
            n: Taskbounds(None, None, None, None) for n in self.graph_nx.nodes()
        }
        self.node_to_index = {
            node: i_node for i_node, node in enumerate(self.graph_nx.nodes())
        }
        self.index_to_node = {
            i_node: node for node, i_node in self.node_to_index.items()
        }
        self.immediate_successors = {
            node: set(nx.neighbors(self.graph_nx, node))
            for node in self.graph_nx.nodes()
        }
        self.immediate_predecessors = {
            node: set(self.graph_nx.predecessors(node))
            for node in self.graph_nx.nodes()
        }
        self.predecessors = {
            node: nx.algorithms.ancestors(self.graph_nx, node)
            for node in self.graph_nx.nodes()
        }
        self.successors = {
            node: nx.algorithms.descendants(self.graph_nx, node)
            for node in self.graph_nx.nodes()
        }
        self._computed = False

    def compute_task_bounds(self) -> None:
        """Compute task bounds by forxard/backward propagation through precedence graph

        Returns:

        """
        # Forward pass for lower bounds
        # starting nodes: without ancestors, start lower bound = 0
        queue = [
            (0, self.node_to_index[node])
            for node, preds in self.predecessors.items()
            if len(preds) == 0
        ]
        self._propagate_forward_bounds_through_graph(queue)

        # Check if horizon is sufficient for earliest finish date found
        last_tasks = [
            node for node, succs in self.successors.items() if len(succs) == 0
        ]
        # optimal makespan if there were no constraints apart from precedence
        self.makespan_cpm = max(
            self.map_node[task].end_lower_bound for task in last_tasks
        )
        if self.makespan_cpm > self.horizon:
            raise RuntimeError(
                f"The schedule cannot be done with given horizon {self.horizon}, as it is below the computed earliest finish date {self.makespan_cpm}"
            )

        # Backward pass for upper bounds
        # starting nodes: without descendants, end upper bound = horizon
        queue = [(-self.horizon, self.node_to_index[node]) for node in last_tasks]
        self._propagate_backward_bounds_through_graph(queue)

        self._computed = True

    def get_task_bounds(self) -> dict[Task, tuple[int, int, int, int]]:
        """Return computed bounds on task start and end.

        Returns:
            start_lower_bound, end_lower_bound, start_upper_bound, end_upper_bound

        """
        if not self._computed:
            self.compute_task_bounds()
        return {
            task: (
                bounds.start_lower_bound,
                bounds.end_lower_bound,
                bounds.start_upper_bound,
                bounds.end_upper_bound,
            )
            for task, bounds in self.map_node.items()
        }

    def get_a_critical_path(self) -> list[Task]:
        """Compute a critical path.

        It takes into account no other constraints that precedence constraints.
        It starts from a task without predecessor to a task without successor.

        Returns:

        """
        if not self._computed:
            self.compute_task_bounds()
        last_tasks = [
            node for node, succs in self.successors.items() if len(succs) == 0
        ]
        critical_path = []
        preds = []
        for node in last_tasks:
            if self.map_node[node].end_lower_bound == self.makespan_cpm:
                # critical last task
                critical_path.append(node)
                preds = self.immediate_predecessors[node]
                cur_node = node
                break
        while len(preds) > 0:
            new_preds = []
            for node in preds:
                if (
                    self.map_node[node].start_lower_bound
                    == self.map_node[node].start_upper_bound
                    - self.horizon
                    + self.makespan_cpm
                    and self.map_node[node].end_lower_bound
                    == self.map_node[cur_node].start_lower_bound
                ):
                    critical_path.append(node)
                    new_preds = self.immediate_predecessors[node]
                    cur_node = node
                    break
            preds = new_preds
        return critical_path[::-1]

    def get_critical_subgraph(self) -> nx.DiGraph:
        """Compute critical subgraph.

        It includes all critical nodes.

        Returns:
            a subgraph *view* of precedence graph `self.graph_nx`
            (so beware that attributes are shared with original graph)

        """
        if not self._computed:
            self.compute_task_bounds()
        last_tasks = [
            node for node, succs in self.successors.items() if len(succs) == 0
        ]
        critical_nodes = set()
        critical_nodes_to_review = []
        # Init: critical nodes among last tasks
        for node in last_tasks:
            if self.map_node[node].end_lower_bound == self.makespan_cpm:
                # critical last task
                critical_nodes_to_review.append(node)
        # Go backward from there
        while len(critical_nodes_to_review) > 0:
            critical_node = critical_nodes_to_review.pop()
            critical_nodes.add(critical_node)
            for node in self.immediate_predecessors[critical_node]:
                if (
                    self.map_node[node].start_lower_bound
                    == self.map_node[node].start_upper_bound
                    - self.horizon
                    + self.makespan_cpm
                    and self.map_node[node].end_lower_bound
                    == self.map_node[critical_node].start_lower_bound
                ):
                    critical_nodes_to_review.append(node)

        return self.graph_nx.subgraph(critical_nodes)

    def _propagate_forward_bounds_through_graph(self, queue):
        done = set()
        heapify(queue)
        while queue:
            time, i_node = heappop(queue)
            node = self.index_to_node[i_node]
            if node in done:
                # node already seen
                continue
            # update time to take into account problem bound
            time = max(
                time,
                self.problem.get_task_start_or_end_tighter_lower_bound(
                    task=node, start_or_end=StartOrEnd.START, use_cpm=False
                ),
            )
            self.map_node[node].start_lower_bound = time
            min_duration = min(
                self.problem.get_task_mode_duration(task=node, mode=mode)
                for mode in self.problem.get_task_modes(task=node)
            )
            # end lower bound : best of start lower bound + min duration and problem bound
            self.map_node[node].end_lower_bound = max(
                time + min_duration,
                self.problem.get_task_start_or_end_tighter_lower_bound(
                    task=node, start_or_end=StartOrEnd.END, use_cpm=False
                ),
            )
            done.add(node)
            next_nodes = self.immediate_successors[node]
            for next_node in next_nodes:
                if all(node in done for node in self.immediate_predecessors[next_node]):
                    next_node_start_lower_bound = max(
                        self.map_node[node].end_lower_bound
                        for node in self.immediate_predecessors[next_node]
                    )
                    heappush(
                        queue,
                        (next_node_start_lower_bound, self.node_to_index[next_node]),
                    )

    def _propagate_backward_bounds_through_graph(self, queue):
        done = set()
        heapify(queue)
        while queue:
            time, i_node = heappop(queue)
            node = self.index_to_node[i_node]
            if node in done:
                # node already seen
                continue
            # update time to take into account problem bound
            eub = min(
                -time,
                self.problem.get_task_start_or_end_tighter_upper_bound(
                    task=node, start_or_end=StartOrEnd.END, use_cpm=False
                ),
            )
            self.map_node[node].end_upper_bound = eub
            min_duration = min(
                self.problem.get_task_mode_duration(task=node, mode=mode)
                for mode in self.problem.get_task_modes(task=node)
            )
            # start upper bound: best of end upper bound - min_duration and problem bound
            self.map_node[node].start_upper_bound = min(
                eub - min_duration,
                self.problem.get_task_start_or_end_tighter_upper_bound(
                    task=node, start_or_end=StartOrEnd.START, use_cpm=False
                ),
            )
            done.add(node)

            next_nodes = self.immediate_predecessors[node]
            for next_node in next_nodes:
                if all(node in done for node in self.immediate_successors[next_node]):
                    next_node_end_upper_bound = min(
                        self.map_node[n].start_upper_bound
                        for n in self.immediate_successors[next_node]
                    )
                    heappush(
                        queue,
                        (-next_node_end_upper_bound, self.node_to_index[next_node]),
                    )
