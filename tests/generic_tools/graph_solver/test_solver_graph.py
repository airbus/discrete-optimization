#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Unit tests for SolverGraph (DAG-based solver workflow)."""

import pytest

from discrete_optimization.generic_tools.graph_solver.solver_graph import (
    BackTransformNode,
    MergeNode,
    NodeData,
    RootNode,
    SolverGraph,
    SolverNode,
    TransformationNode,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.knapsack.problem import Item, KnapsackProblem
from discrete_optimization.knapsack.solvers.greedy import GreedyBestKnapsackSolver
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.pile import PileRcpspSolver
from discrete_optimization.rcpsp.transformations.to_multiskill import (
    RcpspToMultiskillTransformation,
)
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)


@pytest.fixture
def simple_knapsack():
    """Create a simple knapsack problem for testing."""
    items = [
        Item(index=0, value=10, weight=5),
        Item(index=1, value=20, weight=10),
        Item(index=2, value=15, weight=8),
        Item(index=3, value=8, weight=4),
    ]
    return KnapsackProblem(list_items=items, max_capacity=15)


@pytest.fixture
def small_rcpsp():
    """Load a small RCPSP instance for testing."""
    files = get_data_available()
    file_path = [f for f in files if "j301_1.sm" in f][0]
    return parse_file(file_path)


class TestNodeData:
    """Test NodeData class."""

    def test_nodedata_init(self, simple_knapsack):
        """Test NodeData initialization."""
        data = NodeData(problem=simple_knapsack)
        assert data.problem is not None
        assert data.result is None
        assert data.solution is None

    def test_nodedata_solution_extraction(self, simple_knapsack):
        """Test automatic solution extraction from result."""
        from discrete_optimization.generic_tools.do_problem import ModeOptim
        from discrete_optimization.generic_tools.result_storage.result_storage import (
            ResultStorage,
        )
        from discrete_optimization.knapsack.problem import KnapsackSolution

        sol = KnapsackSolution(list_taken=[1, 0, 1, 0], problem=simple_knapsack)
        result = ResultStorage(
            list_solution_fits=[(sol, -25)], mode_optim=ModeOptim.MAXIMIZATION
        )
        data = NodeData(problem=simple_knapsack, result=result)
        assert data.solution is not None
        assert data.solution == sol


class TestGraphNodes:
    """Test individual graph node types."""

    def test_root_node(self, simple_knapsack):
        """Test RootNode behavior."""
        root = RootNode("root", simple_knapsack)
        assert root.can_execute()
        output = root.execute()
        assert output.problem == simple_knapsack

    def test_solver_node(self, simple_knapsack):
        """Test SolverNode execution."""
        root = RootNode("root", simple_knapsack)
        root_output = root.execute()

        solver_node = SolverNode(
            "greedy", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )
        solver_node.inputs["root"] = root_output

        assert solver_node.can_execute()
        output = solver_node.execute()
        assert output.result is not None
        assert len(output.result) > 0
        assert output.solution is not None

    def test_transformation_node(self, small_rcpsp):
        """Test TransformationNode."""
        transformation = RcpspToMultiskillTransformation()
        root = RootNode("root", small_rcpsp)
        root_output = root.execute()

        trans_node = TransformationNode("transform", transformation)
        trans_node.inputs["root"] = root_output

        assert trans_node.can_execute()
        output = trans_node.execute()
        assert output.problem is not None
        assert type(output.problem).__name__ == "MultiskillRcpspProblem"

    def test_back_transform_node(self, small_rcpsp):
        """Test BackTransformNode."""

        transformation = RcpspToMultiskillTransformation()

        # Transform problem
        multiskill_problem = transformation.transform_problem(small_rcpsp)

        # Create a fast solution in multiskill space
        solver = CpSatMultiskillRcpspSolver(multiskill_problem, time_limit=2)
        solver.init_model(time_limit=2)
        multiskill_result = solver.solve(time_limit=2)

        # Create BackTransformNode
        back_node = BackTransformNode("back_transform", transformation, small_rcpsp)
        back_node.inputs["solver"] = NodeData(
            problem=multiskill_problem, result=multiskill_result
        )

        assert back_node.can_execute()
        output = back_node.execute()
        assert output.problem == small_rcpsp
        assert output.result is not None
        assert len(output.result) > 0

        # Verify back-transformed solution is valid in source problem
        back_sol = output.result.get_best_solution()
        assert small_rcpsp.satisfy(back_sol)

    def test_merge_node_best_strategy(self, simple_knapsack):
        """Test MergeNode with 'best' strategy."""
        from discrete_optimization.generic_tools.do_problem import ModeOptim
        from discrete_optimization.generic_tools.result_storage.result_storage import (
            ResultStorage,
        )
        from discrete_optimization.knapsack.problem import KnapsackSolution

        # Create two result storages
        sol1 = KnapsackSolution(list_taken=[1, 0, 0, 1], problem=simple_knapsack)
        result1 = ResultStorage(
            list_solution_fits=[(sol1, -18)], mode_optim=ModeOptim.MAXIMIZATION
        )

        sol2 = KnapsackSolution(list_taken=[0, 1, 0, 1], problem=simple_knapsack)
        result2 = ResultStorage(
            list_solution_fits=[(sol2, -28)], mode_optim=ModeOptim.MAXIMIZATION
        )

        # Create merge node
        merge = MergeNode("merge", strategy="best")
        merge.inputs["solver1"] = NodeData(result=result1)
        merge.inputs["solver2"] = NodeData(result=result2)

        output = merge.execute()
        assert len(output.result) == 2  # Best from each
        # Just verify a solution exists, don't assert on fitness value
        best_sol, best_fit = output.result.get_best_solution_fit()
        assert best_sol is not None

    def test_merge_node_all_strategy(self, simple_knapsack):
        """Test MergeNode with 'all' strategy."""
        from discrete_optimization.generic_tools.do_problem import ModeOptim
        from discrete_optimization.generic_tools.result_storage.result_storage import (
            ResultStorage,
        )
        from discrete_optimization.knapsack.problem import KnapsackSolution

        # Create two result storages with multiple solutions each
        sol1a = KnapsackSolution(list_taken=[1, 0, 0, 1], problem=simple_knapsack)
        sol1b = KnapsackSolution(list_taken=[1, 0, 0, 0], problem=simple_knapsack)
        result1 = ResultStorage(
            list_solution_fits=[(sol1a, -18), (sol1b, -10)],
            mode_optim=ModeOptim.MAXIMIZATION,
        )

        sol2a = KnapsackSolution(list_taken=[0, 1, 0, 1], problem=simple_knapsack)
        result2 = ResultStorage(
            list_solution_fits=[(sol2a, -28)], mode_optim=ModeOptim.MAXIMIZATION
        )

        # Create merge node
        merge = MergeNode("merge", strategy="all")
        merge.inputs["solver1"] = NodeData(result=result1)
        merge.inputs["solver2"] = NodeData(result=result2)

        output = merge.execute()
        assert len(output.result) == 3  # All solutions


class TestSolverGraph:
    """Test SolverGraph orchestration."""

    def test_topological_sort_with_networkx(self, simple_knapsack):
        """Test that topological_sort using NetworkX is correct."""
        graph = SolverGraph(source_problem=simple_knapsack)

        # Build a complex DAG
        solver1 = graph.add_solver(
            "s1", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )
        solver2 = graph.add_solver(
            "s2", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )
        solver3 = graph.add_solver(
            "s3", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )
        merge = graph.add_merge("merge", strategy="best")

        # Build DAG: root -> s1 -> s3 -> merge
        #            root -> s2 -------> merge
        graph.add_edge("root", solver1)
        graph.add_edge("root", solver2)
        graph.add_edge(solver1, solver3)
        graph.add_edge(solver3, merge)
        graph.add_edge(solver2, merge)

        # Get topological order
        order = graph.topological_sort()

        # Verify properties of topological sort:
        # 1. All nodes are present
        assert set(order) == set(["root", "s1", "s2", "s3", "merge"])

        # 2. root comes first
        assert order[0] == "root"

        # 3. merge comes last
        assert order[-1] == "merge"

        # 4. s1 comes before s3 (dependency)
        assert order.index("s1") < order.index("s3")

        # 5. s3 comes before merge (dependency)
        assert order.index("s3") < order.index("merge")

        # 6. s2 comes before merge (dependency)
        assert order.index("s2") < order.index("merge")

        # 7. root comes before all others
        root_idx = order.index("root")
        for node in ["s1", "s2", "s3", "merge"]:
            assert root_idx < order.index(node)

    def test_networkx_graph_caching(self, simple_knapsack):
        """Test that NetworkX graph is cached and invalidated correctly."""
        graph = SolverGraph(source_problem=simple_knapsack)

        # Initially, cache should be None
        assert graph._nx_graph is None

        # First topological sort should build the graph
        graph.add_solver("s1", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={}))
        graph.add_edge("root", "s1")
        order1 = graph.topological_sort()
        assert graph._nx_graph is not None
        cached_graph = graph._nx_graph

        # Second call should use the cache
        order2 = graph.topological_sort()
        assert graph._nx_graph is cached_graph  # Same object
        assert order1 == order2

        # Adding an edge should invalidate the cache
        graph.add_solver("s2", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={}))
        graph.add_edge("root", "s2")
        assert graph._nx_graph is None  # Cache invalidated

        # Next topological sort should rebuild
        order3 = graph.topological_sort()
        assert graph._nx_graph is not None
        assert graph._nx_graph is not cached_graph  # New graph object

    def test_linear_pipeline(self, simple_knapsack):
        """Test linear solver pipeline (like SequentialMetasolver)."""
        graph = SolverGraph(source_problem=simple_knapsack)

        # Add two solvers in sequence
        solver1_id = graph.add_solver(
            "greedy1", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )
        solver2_id = graph.add_solver(
            "greedy2", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )

        # Connect: root -> solver1 -> solver2
        graph.add_edge("root", solver1_id)
        graph.add_edge(solver1_id, solver2_id)

        # Run graph
        result = graph.run()
        assert result is not None
        assert len(result) > 0
        best_sol = result.get_best_solution()
        assert simple_knapsack.satisfy(best_sol)

    def test_parallel_solving_with_merge(self, simple_knapsack):
        """Test parallel solving strategies with merge."""
        graph = SolverGraph(source_problem=simple_knapsack)

        # Add two parallel solvers
        solver1_id = graph.add_solver(
            "greedy1", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )
        solver2_id = graph.add_solver(
            "greedy2", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )

        # Add merge node
        merge_id = graph.add_merge("merge", strategy="best")

        # Connect: root -> solver1 -> merge
        #          root -> solver2 -> merge
        graph.add_edge("root", solver1_id)
        graph.add_edge("root", solver2_id)
        graph.add_edge(solver1_id, merge_id)
        graph.add_edge(solver2_id, merge_id)

        # Run graph
        result = graph.run()
        assert result is not None
        assert len(result) > 0

    def test_transformation_pipeline(self, small_rcpsp):
        """Test transformation and back-transformation pipeline."""
        graph = SolverGraph(source_problem=small_rcpsp)

        transformation = RcpspToMultiskillTransformation()

        # Build pipeline: root -> transform -> solve -> back-transform
        transform_id = graph.add_transformation("to_multiskill", transformation)
        solve_id = graph.add_solver(
            "solve_multiskill",
            SubBrick(cls=CpSatMultiskillRcpspSolver, kwargs={"time_limit": 2}),
        )
        back_id = graph.add_back_transform("back_to_rcpsp", transformation, small_rcpsp)

        graph.add_edge("root", transform_id)
        graph.add_edge(transform_id, solve_id)
        graph.add_edge(solve_id, back_id)

        # Run graph
        result = graph.run()
        assert result is not None
        assert len(result) > 0

        # Verify solution is in original problem space
        best_sol = result.get_best_solution()
        assert small_rcpsp.satisfy(best_sol)

    def test_complex_dag(self, small_rcpsp):
        """Test complex DAG with branching and transformation."""
        graph = SolverGraph(source_problem=small_rcpsp)

        # Branch 1: Direct greedy
        direct_id = graph.add_solver(
            "direct_greedy", SubBrick(cls=PileRcpspSolver, kwargs={})
        )

        # Branch 2: Transform -> solve -> back-transform
        transformation = RcpspToMultiskillTransformation()
        transform_id = graph.add_transformation("transform", transformation)
        solve_id = graph.add_solver(
            "solve_transformed",
            SubBrick(cls=CpSatMultiskillRcpspSolver, kwargs={"time_limit": 2}),
        )
        back_id = graph.add_back_transform(
            "back_transform", transformation, small_rcpsp
        )

        # Merge both branches
        merge_id = graph.add_merge("merge", strategy="best")

        # Build edges
        graph.add_edge("root", direct_id)
        graph.add_edge("root", transform_id)
        graph.add_edge(transform_id, solve_id)
        graph.add_edge(solve_id, back_id)
        graph.add_edge(direct_id, merge_id)
        graph.add_edge(back_id, merge_id)

        # Run graph
        result = graph.run()
        assert result is not None
        assert len(result) > 0
        best_sol = result.get_best_solution()
        assert small_rcpsp.satisfy(best_sol)

    def test_topological_sort(self, simple_knapsack):
        """Test topological sorting of graph."""
        graph = SolverGraph(source_problem=simple_knapsack)

        solver1 = graph.add_solver(
            "s1", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )
        solver2 = graph.add_solver(
            "s2", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )
        merge = graph.add_merge("merge", strategy="best")

        graph.add_edge("root", solver1)
        graph.add_edge("root", solver2)
        graph.add_edge(solver1, merge)
        graph.add_edge(solver2, merge)

        order = graph.topological_sort()
        assert order[0] == "root"
        assert "merge" == order[-1]
        assert set(order[1:3]) == {"s1", "s2"}

    def test_cycle_detection(self, simple_knapsack):
        """Test that cycles are detected."""
        graph = SolverGraph(source_problem=simple_knapsack)

        solver1 = graph.add_solver(
            "s1", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )
        solver2 = graph.add_solver(
            "s2", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )

        graph.add_edge("root", solver1)
        graph.add_edge(solver1, solver2)
        graph.add_edge(solver2, solver1)  # Create cycle

        with pytest.raises(ValueError, match="cycle"):
            graph.run()

    def test_missing_node_error(self, simple_knapsack):
        """Test error when adding edge to non-existent node."""
        graph = SolverGraph(source_problem=simple_knapsack)

        with pytest.raises(ValueError, match="does not exist"):
            graph.add_edge("root", "nonexistent")

    def test_duplicate_node_warning(self, simple_knapsack):
        """Test that duplicate node IDs are handled."""
        graph = SolverGraph(source_problem=simple_knapsack)

        graph.add_solver("solver1", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={}))
        # Try to add same ID again
        result = graph.add_solver(
            "solver1", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )
        assert result is None  # Should return None and log error

    def test_visualize(self, simple_knapsack):
        """Test graph visualization."""
        graph = SolverGraph(source_problem=simple_knapsack)

        solver1 = graph.add_solver(
            "s1", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )
        solver2 = graph.add_solver(
            "s2", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )
        merge = graph.add_merge("merge", strategy="best")

        graph.add_edge("root", solver1)
        graph.add_edge("root", solver2)
        graph.add_edge(solver1, merge)
        graph.add_edge(solver2, merge)

        viz = graph.visualize()
        assert "SolverGraph" in viz
        assert "KnapsackProblem" in viz
        assert "s1" in viz
        assert "s2" in viz
        assert "merge" in viz
        assert "root → s1" in viz

    def test_no_terminal_nodes_error(self, simple_knapsack):
        """Test error when graph has no terminal nodes."""
        graph = SolverGraph(source_problem=simple_knapsack)

        # Create a graph where all nodes point back (invalid, but tests the check)
        # Actually, this would create a cycle, so let's test empty graph instead
        with pytest.raises(ValueError, match="no terminal nodes"):
            graph.run()

    def test_multiple_terminal_nodes(self, simple_knapsack):
        """Test graph with multiple terminal nodes (auto-merge)."""
        graph = SolverGraph(source_problem=simple_knapsack)

        # Create two independent branches
        solver1 = graph.add_solver(
            "s1", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )
        solver2 = graph.add_solver(
            "s2", SubBrick(cls=GreedyBestKnapsackSolver, kwargs={})
        )

        graph.add_edge("root", solver1)
        graph.add_edge("root", solver2)

        # Both are terminal nodes - should auto-merge
        result = graph.run()
        assert result is not None
        assert len(result) > 0


class TestIntegrationExamples:
    """Test examples from documentation."""

    def test_example_04_sequential(self, small_rcpsp):
        """Test sequential solving pattern from example 04."""
        graph = SolverGraph(source_problem=small_rcpsp)

        # Sequential pipeline: greedy -> greedy (simulating warmstart)
        stage1 = graph.add_solver("greedy1", SubBrick(cls=PileRcpspSolver, kwargs={}))
        stage2 = graph.add_solver("greedy2", SubBrick(cls=PileRcpspSolver, kwargs={}))

        graph.add_edge("root", stage1)
        graph.add_edge(stage1, stage2)

        result = graph.run()
        assert len(result) > 0
        assert small_rcpsp.satisfy(result.get_best_solution())

    def test_example_05_parallel(self, small_rcpsp):
        """Test parallel solving pattern from example 05."""
        graph = SolverGraph(source_problem=small_rcpsp)

        # Strategy 1: Direct greedy
        greedy_id = graph.add_solver("greedy", SubBrick(cls=PileRcpspSolver, kwargs={}))

        # Strategy 2: Transform -> solve -> back
        transformation = RcpspToMultiskillTransformation()
        transform_id = graph.add_transformation("transform", transformation)
        solve_id = graph.add_solver(
            "solve_multiskill",
            SubBrick(cls=CpSatMultiskillRcpspSolver, kwargs={"time_limit": 2}),
        )
        back_id = graph.add_back_transform("back", transformation, small_rcpsp)

        # Merge strategies
        merge_id = graph.add_merge("merge", strategy="best")

        # Build DAG
        graph.add_edge("root", greedy_id)
        graph.add_edge("root", transform_id)
        graph.add_edge(transform_id, solve_id)
        graph.add_edge(solve_id, back_id)
        graph.add_edge(greedy_id, merge_id)
        graph.add_edge(back_id, merge_id)

        result = graph.run()
        assert len(result) > 0
        best_sol = result.get_best_solution()
        assert small_rcpsp.satisfy(best_sol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
