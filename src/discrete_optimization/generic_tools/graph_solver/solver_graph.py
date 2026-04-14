#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""DAG-based solver workflow (SolverGraph)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Optional

from discrete_optimization.generic_tools.do_problem import (
    Problem,
    Solution,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import SolverDO, WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)


@dataclass
class NodeData:
    """Data flowing through graph nodes.

    Attributes:
        problem: Problem instance (required for most nodes)
        result: ResultStorage from solver nodes (optional)
        solution: Single best solution for warmstart/kwargs extraction (optional)

    """

    problem: Optional[Problem] = None
    result: Optional[ResultStorage] = None
    solution: Optional[Solution] = None

    def __post_init__(self):
        """Auto-extract solution from result if available."""
        if self.solution is None and self.result is not None and len(self.result) > 0:
            self.solution = self.result.get_best_solution()


class GraphNode(ABC):
    """Base class for nodes in the solver graph.

    A node takes inputs, executes an operation, and produces outputs.
    Data flows through the graph as NodeData objects.
    """

    node_id: str
    inputs: dict[str, NodeData]  # upstream_node_id -> NodeData
    output: Optional[NodeData]  # Output data

    def __init__(self, node_id: str):
        """Initialize graph node.

        Args:
            node_id: Unique identifier for this node

        """
        self.node_id = node_id
        self.inputs = {}
        self.output = None

    @abstractmethod
    def execute(self, **kwargs: Any) -> NodeData:
        """Execute the node's operation.

        Args:
            **kwargs: Execution parameters (e.g., time_limit)

        Returns:
            NodeData with problem, result, and/or solution

        """
        ...

    def can_execute(self) -> bool:
        """Check if all required inputs are available.

        Default: can execute if we have at least one input.
        Override in subclasses for specific requirements.

        Returns:
            True if node can execute

        """
        return len(self.inputs) > 0

    def get_input_problem(self) -> Problem:
        """Extract problem from inputs.

        Returns:
            Problem instance from first input that has one

        """
        for input_data in self.inputs.values():
            if input_data.problem is not None:
                return input_data.problem
        raise ValueError(f"Node {self.node_id} has no problem in inputs")

    def get_input_solution(self) -> Optional[Solution]:
        """Extract best solution from inputs for warmstart.

        Returns:
            Solution if available, None otherwise

        """
        for input_data in self.inputs.values():
            if input_data.solution is not None:
                return input_data.solution
        return None

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(id={self.node_id})"


class RootNode(GraphNode):
    """Virtual root node that provides the source problem."""

    problem: Problem

    def __init__(self, node_id: str, problem: Problem):
        """Initialize root node.

        Args:
            node_id: Node identifier
            problem: Source problem

        """
        super().__init__(node_id)
        self.problem = problem

    def execute(self, **kwargs: Any) -> NodeData:
        """Return the source problem.

        Returns:
            NodeData with problem

        """
        return NodeData(problem=self.problem)

    def can_execute(self) -> bool:
        """Root can always execute."""
        return True


class TransformationNode(GraphNode):
    """Node that applies a problem transformation."""

    transformation: ProblemTransformation

    def __init__(self, node_id: str, transformation: ProblemTransformation):
        """Initialize transformation node.

        Args:
            node_id: Node identifier
            transformation: Transformation to apply

        """
        super().__init__(node_id)
        self.transformation = transformation

    def execute(self, **kwargs: Any) -> NodeData:
        """Transform the input problem.

        Returns:
            NodeData with transformed problem

        """
        input_problem = self.get_input_problem()
        output_problem = self.transformation.transform_problem(input_problem)
        return NodeData(problem=output_problem)


class SolverNode(GraphNode):
    """Node that runs a solver."""

    solver_brick: SubBrick
    solver: Optional[SolverDO]
    problem: Optional[Problem]

    def __init__(self, node_id: str, solver_brick: SubBrick):
        """Initialize solver node.

        Args:
            node_id: Node identifier
            solver_brick: Solver specification

        """
        super().__init__(node_id)
        self.solver_brick = solver_brick
        self.solver = None
        self.problem = None

    def execute(self, **kwargs: Any) -> NodeData:
        """Solve the input problem.

        Returns:
            NodeData with result, solution, and problem

        """
        # Get problem from inputs
        problem = self.get_input_problem()

        # Get solution for warmstart (if available)
        warmstart_solution = self.get_input_solution()

        # Update kwargs using kwargs_from_solution (like SequentialMetasolver)
        kwargs_updated = dict(self.solver_brick.kwargs)
        if (
            self.solver_brick.kwargs_from_solution is not None
            and warmstart_solution is not None
        ):
            kwargs_updated.update(
                {
                    k: fun(warmstart_solution)
                    for k, fun in self.solver_brick.kwargs_from_solution.items()
                }
            )

        # Instantiate solver if needed or problem changed
        if self.solver is None or self.problem != problem:
            self.problem = problem
            self.solver = self.solver_brick.cls(problem=problem, **kwargs_updated)
            self.solver.init_model(**kwargs_updated)

        # Warmstart if solution available and solver supports it
        if warmstart_solution is not None and isinstance(self.solver, WarmstartMixin):
            self.solver.set_warm_start(warmstart_solution)

        # Solve with updated kwargs
        result = self.solver.solve(**kwargs)

        # Return NodeData (solution auto-extracted in __post_init__)
        return NodeData(problem=problem, result=result)


class MergeNode(GraphNode):
    """Node that merges multiple result storages."""

    strategy: str  # "best", "all"

    def __init__(self, node_id: str, strategy: str = "best"):
        """Initialize merge node.

        Args:
            node_id: Node identifier
            strategy: Merge strategy ("best" or "all")

        """
        super().__init__(node_id)
        self.strategy = strategy

    def execute(self, **kwargs: Any) -> NodeData:
        """Merge input result storages.

        Returns:
            NodeData with merged result and solution

        """
        # Extract results from all inputs
        results = []
        for input_data in self.inputs.values():
            if input_data.result is not None:
                results.append(input_data.result)

        if len(results) == 0:
            raise ValueError("MergeNode requires at least one result in inputs")

        # Merge based on strategy
        if self.strategy == "best":
            # Keep only best solution from each result
            merged = ResultStorage(
                list_solution_fits=[], mode_optim=results[0].mode_optim
            )
            for res in results:
                if len(res) > 0:
                    best_sol, best_fit = res.get_best_solution_fit()
                    if best_sol is not None:
                        merged.append((best_sol, best_fit))

        elif self.strategy == "all":
            # Combine all solutions
            merged = ResultStorage(
                list_solution_fits=[], mode_optim=results[0].mode_optim
            )
            for res in results:
                merged.extend(res)

        else:
            raise ValueError(f"Unknown merge strategy: {self.strategy}")

        # Return NodeData (solution auto-extracted in __post_init__)
        return NodeData(result=merged)

    def can_execute(self) -> bool:
        """Can execute if we have at least one input."""
        return len(self.inputs) >= 1


class BackTransformNode(GraphNode):
    """Node that back-transforms solutions through a transformation."""

    transformation: ProblemTransformation
    source_problem: Problem

    def __init__(
        self,
        node_id: str,
        transformation: ProblemTransformation,
        source_problem: Problem,
    ):
        """Initialize back-transformation node.

        Args:
            node_id: Node identifier
            transformation: Transformation to reverse
            source_problem: Original source problem

        """
        super().__init__(node_id)
        self.transformation = transformation
        self.source_problem = source_problem

        # Build aggregation function for evaluating back-transformed solutions
        (
            self.aggreg_from_sol,
            self.aggreg_from_dict,
            _,
        ) = build_aggreg_function_and_params_objective(problem=source_problem)

    def execute(self, **kwargs: Any) -> NodeData:
        """Back-transform all solutions.

        Returns:
            NodeData with back-transformed result, solution, and source problem

        """
        # Extract result from first input that has one
        input_result = None
        for input_data in self.inputs.values():
            if input_data.result is not None:
                input_result = input_data.result
                break

        if input_result is None:
            raise ValueError("BackTransformNode requires a result in inputs")

        # Back-transform all solutions
        output_result = ResultStorage(
            list_solution_fits=[], mode_optim=input_result.mode_optim
        )

        for sol_target, _ in input_result:
            sol_source = self.transformation.back_transform_solution(
                sol_target, self.source_problem
            )
            # Use the problem's aggregation function (respects ParamsObjectiveFunction)
            fit_value = self.aggreg_from_sol(sol_source)
            output_result.append((sol_source, fit_value))

        # Return NodeData (solution auto-extracted in __post_init__)
        return NodeData(problem=self.source_problem, result=output_result)


class SolverGraph:
    """DAG-based solver workflow.

    Supports:
    - Branching (parallel strategies)
    - Merging (combine results)
    - Transformations (problem conversion)
    - Arbitrary directed acyclic graphs

    Example (linear, like SequentialMetasolver):
    #     >>> graph = SolverGraph(problem)
    #     >>> graph.add_solver("solver1", SubBrick(cls=Solver1, kwargs={}))
    #     >>> graph.add_solver("solver2", SubBrick(cls=Solver2, kwargs={}))
    #     >>> graph.add_edge("root", "solver1")
    #     >>> graph.add_edge("solver1", "solver2")
    #     >>> result = graph.run()
    #
    # Example (branching):
    #     >>> graph = SolverGraph(problem)
    #     >>> graph.add_solver("cpsat", SubBrick(cls=CPSat, kwargs={}))
    #     >>> graph.add_solver("lp", SubBrick(cls=LP, kwargs={}))
    #     >>> graph.add_merge("merge", strategy="best")
    #     >>> graph.add_edge("root", "cpsat")
    #     >>> graph.add_edge("root", "lp")
    #     >>> graph.add_edge("cpsat", "merge")
    #     >>> graph.add_edge("lp", "merge")
    #     >>> result = graph.run()

    """

    source_problem: Problem
    nodes: dict[str, GraphNode]
    edges: dict[str, list[str]]  # node_id -> list of downstream node_ids
    reverse_edges: dict[str, list[str]]  # node_id -> list of upstream node_ids

    # Execution state
    node_outputs: dict[str, NodeData]  # node_id -> NodeData

    def __init__(self, source_problem: Problem):
        """Initialize solver graph.

        Args:
            source_problem: The problem to solve

        """
        self.source_problem = source_problem
        self.nodes = {"root": RootNode("root", source_problem)}
        self.edges = defaultdict(list)
        self.reverse_edges = defaultdict(list)
        self.node_outputs = {}

    def add_transformation(
        self, node_id: str, transformation: ProblemTransformation
    ) -> str:
        """Add a transformation node.

        Args:
            node_id: Unique identifier for this node
            transformation: Transformation to apply

        Returns:
            Node ID (for chaining)

        """
        node = TransformationNode(node_id, transformation)
        self.nodes[node_id] = node
        return node_id

    def add_solver(self, node_id: str, solver_brick: SubBrick) -> str:
        """Add a solver node.

        Args:
            node_id: Unique identifier for this node
            solver_brick: Solver specification

        Returns:
            Node ID (for chaining)

        """
        node = SolverNode(node_id, solver_brick)
        self.nodes[node_id] = node
        return node_id

    def add_merge(self, node_id: str, strategy: str = "best") -> str:
        """Add a merge node.

        Args:
            node_id: Unique identifier for this node
            strategy: Merge strategy ("best" or "all")

        Returns:
            Node ID (for chaining)

        """
        node = MergeNode(node_id, strategy)
        self.nodes[node_id] = node
        return node_id

    def add_back_transform(
        self,
        node_id: str,
        transformation: ProblemTransformation,
        source_problem: Problem,
    ) -> str:
        """Add a back-transformation node.

        Args:
            node_id: Unique identifier for this node
            transformation: Transformation to reverse
            source_problem: Original problem

        Returns:
            Node ID (for chaining)

        """
        node = BackTransformNode(node_id, transformation, source_problem)
        self.nodes[node_id] = node
        return node_id

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge between nodes.

        Args:
            from_node: Source node ID
            to_node: Target node ID

        Raises:
            ValueError: If nodes don't exist

        """
        if from_node not in self.nodes:
            raise ValueError(f"Node {from_node} does not exist")
        if to_node not in self.nodes:
            raise ValueError(f"Node {to_node} does not exist")

        self.edges[from_node].append(to_node)
        self.reverse_edges[to_node].append(from_node)

    def topological_sort(self) -> list[str]:
        """Return nodes in topological order.

        Returns:
            List of node IDs in execution order

        Raises:
            ValueError: If graph contains a cycle

        """
        in_degree = {
            node_id: len(self.reverse_edges[node_id]) for node_id in self.nodes
        }
        queue = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
        result = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            for downstream in self.edges[node_id]:
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    queue.append(downstream)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle!")

        return result

    def run(self, **solve_kwargs: Any) -> ResultStorage:
        """Execute the graph and return final results.

        Args:
            **solve_kwargs: Keyword arguments passed to all solver nodes

        Returns:
            ResultStorage with final solutions

        Raises:
            RuntimeError: If execution fails

        """
        # Topological sort to determine execution order
        execution_order = self.topological_sort()

        print(f"Execution order: {' → '.join(execution_order)}")

        # Execute nodes in order
        for node_id in execution_order:
            node = self.nodes[node_id]

            # Collect inputs from upstream nodes
            for upstream_id in self.reverse_edges[node_id]:
                if upstream_id not in self.node_outputs:
                    raise RuntimeError(f"Node {upstream_id} has not been executed")

                # Pass outputs from upstream to this node's inputs
                # node.inputs is a dict: upstream_id -> NodeData
                node.inputs[upstream_id] = self.node_outputs[upstream_id]

            # Execute node
            if node.can_execute():
                print(f"Executing {node_id}: {node}")
                output = node.execute(**solve_kwargs)
                self.node_outputs[node_id] = output
            else:
                raise RuntimeError(f"Node {node_id} cannot execute (missing inputs)")

        # Find terminal nodes (nodes with no outgoing edges)
        terminal_nodes = [
            node_id
            for node_id in self.nodes
            if len(self.edges[node_id]) == 0 and node_id != "root"
        ]

        if len(terminal_nodes) == 0:
            raise ValueError("Graph has no terminal nodes")

        # Return output from terminal node(s)
        if len(terminal_nodes) == 1:
            terminal_output = self.node_outputs[terminal_nodes[0]]
            if terminal_output.result is not None:
                return terminal_output.result
            else:
                raise ValueError("Terminal node has no result")
        else:
            # Multiple terminal nodes - merge them
            results = []
            for terminal_id in terminal_nodes:
                terminal_output = self.node_outputs[terminal_id]
                if terminal_output.result is not None:
                    results.append(terminal_output.result)

            # Simple merge: combine all results
            if len(results) > 0:
                merged = ResultStorage(
                    list_solution_fits=[], mode_optim=results[0].mode_optim
                )
                for res in results:
                    merged.extend(res)
                return merged
            else:
                raise ValueError("No results from terminal nodes")

    def visualize(self) -> str:
        """Create ASCII art visualization of the graph.

        Returns:
            String representation of the graph

        """
        lines = ["SolverGraph:"]
        lines.append(f"  Source Problem: {type(self.source_problem).__name__}")
        lines.append("")
        lines.append("Nodes:")

        for node_id, node in self.nodes.items():
            if node_id == "root":
                continue
            lines.append(f"  - {node_id}: {type(node).__name__}")

        lines.append("")
        lines.append("Edges:")

        for from_node, to_nodes in self.edges.items():
            for to_node in to_nodes:
                lines.append(f"  {from_node} → {to_node}")

        return "\n".join(lines)
