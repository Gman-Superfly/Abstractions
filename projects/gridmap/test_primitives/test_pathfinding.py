"""
Test Pathfinding

Comprehensive test for BFS pathfinding with verifiable non-trivial paths.
"""

from abstractions.ecs.callable_registry import CallableRegistry
from game_entities import GridMap, GridNode, Wall, Floor, Agent
from navigation import compute_navigation_graph
from pathfinding import compute_reachable_paths


def create_test_grid_with_walls():
    """
    Create a 7x7 grid with specific wall layout for testing.
    
    Layout:
      0 1 2 3 4 5 6
    0 . . . # . . .
    1 . A . # . . .
    2 . . . # . . .
    3 # # # . . . .
    4 . . . . . # .
    5 . . . . . # .
    6 . . . . . . .
    
    A = Agent at (1, 1) with speed 3
    # = Wall (non-walkable)
    . = Floor (walkable)
    
    Expected reachable from (1,1) with speed 3 (8-way movement):
    - Distance 0: (1,1) - start
    - Distance 1: All 8 neighbors (including diagonals)
    - Distance 2: (3,3) - through the gap in walls
    - Distance 3: 5 positions beyond (3,3)
    
    Total: 15 reachable positions (1 + 8 + 1 + 5)
    """
    width, height = 7, 7
    nodes = []
    
    # Wall positions
    walls = [
        (3, 0), (3, 1), (3, 2),  # Vertical wall blocking east
        (0, 3), (1, 3), (2, 3),  # Horizontal wall blocking south
        (5, 4), (5, 5)           # Extra walls on right
    ]
    
    for y in range(height):
        for x in range(width):
            pos = (x, y)
            node = GridNode(position=pos)
            
            if pos in walls:
                node.entities = [Wall(name=f"wall_{x}_{y}", walkable=False, transparent=False)]
            else:
                node.entities = [Floor(name=f"floor_{x}_{y}", walkable=True, transparent=True)]
            
            nodes.append(node)
    
    grid = GridMap(nodes=nodes, width=width, height=height)
    grid.promote_to_root()
    
    return grid


def print_grid_with_distances(grid, reachable_positions, agent_pos):
    """Print grid showing distances from agent position."""
    print("\nGrid with distances from agent:")
    
    # Build distance map
    dist_map = {pos: None for pos in [(x, y) for x in range(grid.width) for y in range(grid.height)]}
    for pos in reachable_positions:
        # Calculate Manhattan distance as approximation
        dist = abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1])
        dist_map[pos] = dist
    
    # Print header
    print("  ", end="")
    for x in range(grid.width):
        print(f"{x} ", end="")
    print()
    
    # Print grid
    for y in range(grid.height):
        print(f"{y} ", end="")
        for x in range(grid.width):
            pos = (x, y)
            node = next((n for n in grid.nodes if n.position == pos), None)
            
            # Check if wall
            is_wall = any(not e.walkable for e in node.entities)
            
            if is_wall:
                print("# ", end="")
            elif pos == agent_pos:
                print("A ", end="")
            elif dist_map[pos] is not None:
                print(f"{dist_map[pos]} ", end="")
            else:
                print(". ", end="")
        print()


def test_pathfinding_basic():
    """Test basic pathfinding with verifiable paths."""
    print("=" * 70)
    print("TEST: Pathfinding - Reachable Paths")
    print("=" * 70)
    
    # Create test grid
    grid = create_test_grid_with_walls()
    
    # Add agent at (1, 1) with speed 3
    agent_position = (1, 1)
    agent_node = next((n for n in grid.nodes if n.position == agent_position), None)
    agent = Agent(name="test_agent", speed=3, sight=5)
    agent_node.entities.append(agent)
    agent.promote_to_root()
    
    print(f"\nAgent: {agent.name}")
    print(f"  Position: {agent_position}")
    print(f"  Speed: {agent.speed}")
    print(f"  ECS ID: {agent.ecs_id}")
    
    # Compute navigation graph
    print("\nComputing navigation graph...")
    nav_graph = CallableRegistry.execute("compute_navigation_graph", grid_map=grid)
    
    print(f"Navigation graph computed:")
    print(f"  ECS ID: {nav_graph.ecs_id}")
    print(f"  Source grid: {nav_graph.source_grid_id}")
    print(f"  Walkable nodes: {sum(nav_graph.walkable.values())}")
    
    # Compute reachable paths
    print("\nComputing reachable paths...")
    path_collection = CallableRegistry.execute(
        "compute_reachable_paths",
        nav_graph=nav_graph,
        agent=agent,
        start_position=agent_position
    )
    
    print(f"\nPathCollection:")
    print(f"  ECS ID: {path_collection.ecs_id}")
    print(f"  Agent ID: {path_collection.agent_id}")
    print(f"  Agent position: {path_collection.agent_position}")
    print(f"  Max distance: {path_collection.max_distance}")
    print(f"  Total paths: {len(path_collection.paths)}")
    print(f"  Reachable positions: {len(path_collection.reachable_positions)}")
    
    # Print grid with distances
    print_grid_with_distances(grid, path_collection.reachable_positions, agent_position)
    
    # Print all paths for debugging
    print("\n--- All Paths Found ---")
    paths_by_distance = {}
    for path in path_collection.paths:
        dist = path.length
        if dist not in paths_by_distance:
            paths_by_distance[dist] = []
        paths_by_distance[dist].append(path)
    
    for dist in sorted(paths_by_distance.keys()):
        print(f"\nDistance {dist}: {len(paths_by_distance[dist])} paths")
        for path in paths_by_distance[dist]:
            steps_str = ' → '.join(str(s) for s in path.steps)
            print(f"  {path.end_position}: {steps_str}")
    
    # Verify automatic tracking
    print("\n--- Automatic Derivation Tracking ---")
    print(f"derived_from_function: {path_collection.derived_from_function}")
    print(f"derived_from_execution_id: {path_collection.derived_from_execution_id}")
    print(f"source_graph_id: {path_collection.source_graph_id}")
    
    assert path_collection.derived_from_function == "compute_reachable_paths", \
        f"Expected 'compute_reachable_paths', got '{path_collection.derived_from_function}'"
    assert path_collection.derived_from_execution_id is not None, \
        "derived_from_execution_id should not be None"
    assert path_collection.source_graph_id == nav_graph.ecs_id, \
        "source_graph_id should match nav_graph.ecs_id"
    
    print("✅ Automatic tracking verified!")
    
    # Verify path count
    print("\n--- Path Count Verification ---")
    expected_count = 15  # 1 start + 8 at dist 1 + 1 at dist 2 + 5 at dist 3
    assert len(path_collection.paths) == expected_count, \
        f"Expected {expected_count} paths, got {len(path_collection.paths)}"
    assert len(path_collection.reachable_positions) == expected_count, \
        f"Expected {expected_count} reachable positions, got {len(path_collection.reachable_positions)}"
    
    print(f"✅ Path count correct: {expected_count} paths")
    print(f"   Distance 0: 1 path (start)")
    print(f"   Distance 1: 8 paths (all neighbors including diagonals)")
    print(f"   Distance 2: 1 path (through gap to (3,3))")
    print(f"   Distance 3: 5 paths (beyond (3,3))")
    
    # Verify specific reachable positions
    print("\n--- Reachable Positions Verification ---")
    expected_reachable = [
        (1, 1),  # Start (dist 0)
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2),  # Distance 1 (8 neighbors)
        (3, 3),  # Distance 2 (through gap)
        (2, 4), (3, 4), (4, 2), (4, 3), (4, 4)  # Distance 3 (beyond gap)
    ]
    
    for pos in expected_reachable:
        assert pos in path_collection.reachable_positions, \
            f"Position {pos} should be reachable"
    
    print(f"✅ All expected positions reachable: {len(expected_reachable)}")
    
    # Verify unreachable positions (blocked by walls)
    print("\n--- Unreachable Positions Verification ---")
    unreachable = [
        (1, 3),  # Blocked by wall at (1, 3)
        (3, 1),  # Blocked by wall at (3, 1)
        (4, 1),  # Too far
        (1, 4),  # Blocked by wall at (1, 3)
    ]
    
    for pos in unreachable:
        assert pos not in path_collection.reachable_positions, \
            f"Position {pos} should NOT be reachable"
    
    print(f"✅ Blocked positions correctly excluded: {len(unreachable)}")
    
    # Verify specific paths
    print("\n--- Specific Path Verification ---")
    
    # Path to origin (0, 0) - diagonal neighbor, distance 1
    path_to_origin = next((p for p in path_collection.paths if p.end_position == (0, 0)), None)
    assert path_to_origin is not None, "Path to (0, 0) should exist"
    assert path_to_origin.length == 1, f"Path to (0, 0) should have length 1 (diagonal), got {path_to_origin.length}"
    assert path_to_origin.start_position == (1, 1), "Path should start at (1, 1)"
    assert len(path_to_origin.steps) == 2, f"Path should have 2 steps (start + 1), got {len(path_to_origin.steps)}"
    
    print(f"Path to (0, 0): {' → '.join(str(s) for s in path_to_origin.steps)}")
    print(f"  Length: {path_to_origin.length}")
    print(f"  Cost: {path_to_origin.cost}")
    
    # Path to corner (2, 2) - diagonal neighbor, distance 1
    path_to_corner = next((p for p in path_collection.paths if p.end_position == (2, 2)), None)
    assert path_to_corner is not None, "Path to (2, 2) should exist"
    assert path_to_corner.length == 1, f"Path to (2, 2) should have length 1 (diagonal), got {path_to_corner.length}"
    
    print(f"Path to (2, 2): {' → '.join(str(s) for s in path_to_corner.steps)}")
    print(f"  Length: {path_to_corner.length}")
    
    # Path through gap to (3, 3) - distance 2
    path_through_gap = next((p for p in path_collection.paths if p.end_position == (3, 3)), None)
    assert path_through_gap is not None, "Path to (3, 3) should exist"
    assert path_through_gap.length == 2, f"Path to (3, 3) should have length 2, got {path_through_gap.length}"
    
    print(f"Path to (3, 3): {' → '.join(str(s) for s in path_through_gap.steps)}")
    print(f"  Length: {path_through_gap.length}")
    
    # Path to adjacent (1, 0)
    path_adjacent = next((p for p in path_collection.paths if p.end_position == (1, 0)), None)
    assert path_adjacent is not None, "Path to (1, 0) should exist"
    assert path_adjacent.length == 1, f"Path to (1, 0) should have length 1, got {path_adjacent.length}"
    
    print(f"Path to (1, 0): {' → '.join(str(s) for s in path_adjacent.steps)}")
    print(f"  Length: {path_adjacent.length}")
    
    print("✅ Specific paths verified!")
    
    # Verify no paths go through walls
    print("\n--- Wall Collision Verification ---")
    wall_positions = [
        (3, 0), (3, 1), (3, 2),  # Vertical wall
        (0, 3), (1, 3), (2, 3),  # Horizontal wall
        (5, 4), (5, 5)           # Extra walls
    ]
    
    paths_through_walls = []
    for path in path_collection.paths:
        for step in path.steps:
            if step in wall_positions:
                paths_through_walls.append((path, step))
                break
    
    if paths_through_walls:
        print(f"❌ Found {len(paths_through_walls)} paths going through walls:")
        for path, wall_pos in paths_through_walls:
            steps_str = ' → '.join(str(s) for s in path.steps)
            print(f"  Path to {path.end_position} goes through wall at {wall_pos}: {steps_str}")
        assert False, "Paths should not go through walls!"
    else:
        print(f"✅ No paths go through walls (checked {len(path_collection.paths)} paths)")
    
    # Verify causal chain
    print("\n--- Causal Chain Verification ---")
    print("GridMap → NavigationGraph → PathCollection")
    print(f"  GridMap.ecs_id: {grid.ecs_id}")
    print(f"  NavGraph.source_grid_id: {nav_graph.source_grid_id}")
    print(f"  PathCollection.source_graph_id: {path_collection.source_graph_id}")
    
    assert nav_graph.source_grid_id == grid.ecs_id, "NavGraph should link to GridMap"
    assert path_collection.source_graph_id == nav_graph.ecs_id, "PathCollection should link to NavGraph"
    
    print("✅ Complete causal chain verified!")
    
    print("\n" + "=" * 70)
    print("ALL PATHFINDING TESTS PASSED!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_pathfinding_basic()
    exit(0 if success else 1)
