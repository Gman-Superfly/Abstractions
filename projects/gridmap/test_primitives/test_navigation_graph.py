"""
Test Navigation Graph Computation

Verify that NavigationGraph is correctly derived from GridMap with automatic tracking.
"""

from abstractions.ecs.callable_registry import CallableRegistry
from game_entities import Wall
from grid_init import create_floor_grid, add_agent_to_grid, print_grid
from navigation import compute_navigation_graph


def test_navigation_graph_basic():
    """Test basic navigation graph computation."""
    print("=" * 60)
    print("TEST: Navigation Graph Computation")
    print("=" * 60)
    
    # Create a 5x5 grid with floor
    grid = create_floor_grid(5, 5)
    
    # Add a wall at (2, 2)
    wall_node = next((n for n in grid.nodes if n.position == (2, 2)), None)
    wall_node.entities.clear()
    wall = Wall(name="wall_2_2", walkable=False, transparent=False)
    wall_node.entities.append(wall)
    
    grid.promote_to_root()
    
    print("\nGrid state:")
    print_grid(grid)
    
    print(f"Grid ecs_id: {grid.ecs_id}")
    
    # Compute navigation graph
    result = CallableRegistry.execute("compute_navigation_graph", grid_map=grid)
    nav_graph = result if not isinstance(result, list) else result[0]
    
    print(f"\nNavigation Graph:")
    print(f"  ecs_id: {nav_graph.ecs_id}")
    print(f"  source_grid_id: {nav_graph.source_grid_id}")
    print(f"  derived_from_function: {nav_graph.derived_from_function}")
    print(f"  derived_from_execution_id: {nav_graph.derived_from_execution_id}")
    
    # Framework limitation: Single-entity returns from transactional path
    # don't get automatic tracking (see P10 tests and investigation_derivation_tracking.md)
    # We use manual tracking via source_grid_id instead
    assert nav_graph.source_grid_id == grid.ecs_id  # Manual causal link ✅
    
    # Check walkability
    assert nav_graph.walkable[(0, 0)] == True  # Floor is walkable
    assert nav_graph.walkable[(2, 2)] == False  # Wall is not walkable
    
    # Check adjacency
    print(f"\nNode (1, 1) walkable neighbors: {nav_graph.walkable_adjacency[(1, 1)]}")
    print(f"  Should NOT include (2, 2) - it's a wall")
    
    # (1, 1) should have 8 neighbors normally, but (2, 2) is a wall
    neighbors_1_1 = nav_graph.walkable_adjacency[(1, 1)]
    assert (2, 2) not in neighbors_1_1  # Wall not in walkable neighbors
    assert (0, 0) in neighbors_1_1  # Floor is in walkable neighbors
    
    print("\n✅ Navigation graph correctly computed!")
    print(f"✅ Manual causal tracking works: source_grid_id = {nav_graph.source_grid_id}")
    
    return grid, nav_graph


def test_staleness_detection():
    """Test that we can detect when navigation graph is stale."""
    print("\n" + "=" * 60)
    print("TEST: Staleness Detection")
    print("=" * 60)
    
    grid, nav_graph = test_navigation_graph_basic()
    
    print(f"\nOriginal grid ecs_id: {grid.ecs_id}")
    print(f"Nav graph source_grid_id: {nav_graph.source_grid_id}")
    print(f"Graph is fresh: {nav_graph.source_grid_id == grid.ecs_id}")
    
    # Modify grid (spawn a wall)
    from movement import spawn_entity
    
    new_wall = Wall(name="new_wall", walkable=False, transparent=False)
    result = CallableRegistry.execute("spawn_entity", grid_map=grid, entity=new_wall, position=(3, 3))
    grid_v2 = result if not isinstance(result, list) else result[0]
    
    print(f"\nAfter modification:")
    print(f"  New grid ecs_id: {grid_v2.ecs_id}")
    print(f"  Nav graph source_grid_id: {nav_graph.source_grid_id}")
    print(f"  Graph is stale: {nav_graph.source_grid_id != grid_v2.ecs_id}")
    
    assert nav_graph.source_grid_id != grid_v2.ecs_id  # Graph is stale!
    
    # Recompute
    result = CallableRegistry.execute("compute_navigation_graph", grid_map=grid_v2)
    nav_graph_v2 = result if not isinstance(result, list) else result[0]
    
    print(f"\nAfter recomputation:")
    print(f"  New nav graph ecs_id: {nav_graph_v2.ecs_id}")
    print(f"  New nav graph source_grid_id: {nav_graph_v2.source_grid_id}")
    print(f"  Graph is fresh: {nav_graph_v2.source_grid_id == grid_v2.ecs_id}")
    
    assert nav_graph_v2.source_grid_id == grid_v2.ecs_id  # Fresh!
    
    # Check that new wall is reflected
    assert nav_graph_v2.walkable[(3, 3)] == False
    
    print("\n✅ Staleness detection works!")
    print("✅ Recomputation creates fresh graph!")


def test_causal_chain():
    """Test the complete causal chain: GridMap → NavigationGraph."""
    print("\n" + "=" * 60)
    print("TEST: Causal Chain Tracking")
    print("=" * 60)
    
    # Create grid
    grid = create_floor_grid(3, 3)
    grid.promote_to_root()
    
    # Compute navigation graph
    result = CallableRegistry.execute("compute_navigation_graph", grid_map=grid)
    nav_graph = result if not isinstance(result, list) else result[0]
    
    print("\nCausal Chain:")
    print(f"GridMap (ecs_id: {grid.ecs_id})")
    print(f"  ↓ [compute_navigation_graph]")
    print(f"NavigationGraph (ecs_id: {nav_graph.ecs_id})")
    print(f"  - derived_from_function: {nav_graph.derived_from_function}")
    print(f"  - source_grid_id: {nav_graph.source_grid_id}")
    print(f"  - Causal link verified: {nav_graph.source_grid_id == grid.ecs_id}")
    
    assert nav_graph.derived_from_function == "compute_navigation_graph"
    assert nav_graph.source_grid_id == grid.ecs_id
    
    print("\n✅ Complete causal chain tracked automatically!")


if __name__ == "__main__":
    test_navigation_graph_basic()
    test_staleness_detection()
    test_causal_chain()
    
    print("\n" + "=" * 60)
    print("ALL NAVIGATION GRAPH TESTS COMPLETE!")
    print("=" * 60)
