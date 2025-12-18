"""
Test Agent Movement

Test agent seeking apples through iterative pathfinding and movement.
"""

from abstractions.ecs.callable_registry import CallableRegistry
from game_entities import GridMap, GridNode, Floor, Agent, Apple
from navigation import compute_navigation_graph
from pathfinding import compute_reachable_paths, choose_path
from movement import move_agent_along_path, find_agent_position


def create_grid_with_apple():
    """
    Create a 5x5 grid with agent and apple.
    
    Layout:
      0 1 2 3 4
    0 . . . . .
    1 . A . . .
    2 . . . . .
    3 . . . @ .
    4 . . . . .
    
    A = Agent at (1, 1), speed 2
    @ = Apple at (3, 3)
    """
    width, height = 5, 5
    nodes = []
    
    for y in range(height):
        for x in range(width):
            pos = (x, y)
            node = GridNode(position=pos)
            node.entities = [Floor(name=f"floor_{x}_{y}", walkable=True, transparent=True)]
            nodes.append(node)
    
    grid = GridMap(nodes=nodes, width=width, height=height)
    grid.promote_to_root()
    
    return grid


def print_grid_state(grid: GridMap, step: int):
    """Print current grid state with agent and apple positions."""
    print(f"\n--- Step {step} ---")
    
    # Find agent and apple
    agent_pos = None
    apple_pos = None
    
    for node in grid.nodes:
        for entity in node.entities:
            if isinstance(entity, Agent):
                agent_pos = node.position
            elif isinstance(entity, Apple):
                apple_pos = node.position
    
    print(f"Grid {grid.width}x{grid.height} (ecs_id: {str(grid.ecs_id)[:8]}...)")
    
    # Print grid
    for y in range(grid.height):
        for x in range(grid.width):
            pos = (x, y)
            if pos == agent_pos:
                print("A ", end="")
            elif pos == apple_pos:
                print("@ ", end="")
            else:
                print(". ", end="")
        print()
    
    print(f"Agent at: {agent_pos}")
    print(f"Apple at: {apple_pos}")


def test_agent_reaches_apple():
    """Test that agent successfully navigates to apple."""
    print("=" * 70)
    print("TEST: Agent Seeks Apple")
    print("=" * 70)
    
    # Create grid
    grid = create_grid_with_apple()
    
    # Add agent at (1, 1)
    agent_pos = (1, 1)
    agent_node = next((n for n in grid.nodes if n.position == agent_pos), None)
    agent = Agent(name="seeker", speed=2, sight=5)
    agent_node.entities.append(agent)
    agent.promote_to_root()
    
    # Add apple at (3, 3)
    apple_pos = (3, 3)
    apple_node = next((n for n in grid.nodes if n.position == apple_pos), None)
    apple = Apple(name="apple1", nutrition=10)
    apple_node.entities.append(apple)
    apple.promote_to_root()
    
    print_grid_state(grid, 0)
    
    # Movement loop
    max_steps = 10
    for step in range(1, max_steps + 1):
        # Find current agent position
        current_pos = find_agent_position(grid, agent)
        
        if not current_pos:
            print("❌ Agent lost!")
            assert False, "Agent not found in grid"
        
        # Check if reached apple
        if current_pos == apple_pos:
            print(f"\n✅ Reached apple at {apple_pos} in {step - 1} steps!")
            break
        
        print(f"\nStep {step}: Agent at {current_pos}")
        
        # Compute navigation graph
        nav_graph = CallableRegistry.execute("compute_navigation_graph", grid_map=grid)
        print(f"  Nav graph: {str(nav_graph.ecs_id)[:8]}...")
        
        # Compute reachable paths
        path_collection = CallableRegistry.execute(
            "compute_reachable_paths",
            nav_graph=nav_graph,
            agent=agent,
            start_position=current_pos
        )
        print(f"  Paths: {len(path_collection.paths)} reachable")
        
        # Choose best path (towards apple)
        chosen_path = CallableRegistry.execute(
            "choose_path",
            path_collection=path_collection,
            grid_map=grid
        )
        print(f"  Chosen path: {' → '.join(str(s) for s in chosen_path.steps[:3])}...")
        
        # Verify tracking
        assert chosen_path.derived_from_function == "choose_path"
        assert chosen_path.derived_from_execution_id is not None
        
        # Move agent
        grid = CallableRegistry.execute(
            "move_agent_along_path",
            grid_map=grid,
            agent=agent,
            path=chosen_path
        )
        
        # Verify grid was versioned
        print(f"  New grid: {str(grid.ecs_id)[:8]}...")
        
        # Print state
        print_grid_state(grid, step)
    else:
        # Loop completed without reaching apple
        final_pos = find_agent_position(grid, agent)
        print(f"\n❌ Failed to reach apple in {max_steps} steps")
        print(f"   Final position: {final_pos}")
        print(f"   Apple position: {apple_pos}")
        assert False, f"Should reach apple in {max_steps} steps"
    
    # Verify final position
    final_pos = find_agent_position(grid, agent)
    assert final_pos == apple_pos, f"Final position {final_pos} != apple position {apple_pos}"
    
    print("\n" + "=" * 70)
    print("✅ AGENT MOVEMENT TEST PASSED!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_agent_reaches_apple()
    exit(0 if success else 1)
