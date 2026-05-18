#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""CPM toc ompute lower/uppe bound on start/end of tasks"""

import logging
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import Any, Generic, Optional

import networkx as nx

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    CumulativeResource,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
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
    (via `problem.get_task_start_or_end_lower_bound()`).

    Attributes:
        horizon: if set, override `problem.get_makespan_upper_bound()`.

    """

    def __init__(
        self,
        problem: GenericSchedulingProblem[
            Task, UnaryResource, CumulativeResource, NonRenewableResource
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

    def compute_task_bounds(self) -> None:
        # Forward pass for lower bounds
        # starting nodes: without ancestors, start lower bound = 0
        queue = [
            (0, self.node_to_index[node])
            for node in self.graph_nx.nodes()
            if len(nx.algorithms.ancestors(self.graph_nx, node)) == 0
        ]
        self._propagate_forward_bounds_through_graph(queue)

        # Check if horizon is sufficient for earliest finish date found
        last_tasks = [
            node
            for node in self.graph_nx.nodes()
            if len(nx.algorithms.descendants(self.graph_nx, node)) == 0
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

    def get_task_bounds(self) -> dict[Task, tuple[int, int, int, int]]:
        """Return computed bounds on task start and end.

        Returns:
            start_lower_bound, end_lower_bound, start_upper_bound, end_upper_bound

        """
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
        last_tasks = [
            node
            for node in self.graph_nx.nodes()
            if len(nx.algorithms.descendants(self.graph_nx, node)) == 0
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
                self.problem.get_task_start_or_end_lower_bound(
                    task=node, start_or_end=StartOrEnd.START
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
                self.problem.get_task_start_or_end_lower_bound(
                    task=node, start_or_end=StartOrEnd.END
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
                self.problem.get_task_start_or_end_upper_bound(
                    task=node, start_or_end=StartOrEnd.END
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
                self.problem.get_task_start_or_end_upper_bound(
                    task=node, start_or_end=StartOrEnd.START
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
