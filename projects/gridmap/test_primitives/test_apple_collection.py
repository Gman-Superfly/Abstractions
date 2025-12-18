"""
Test Apple Collection

Test agent collecting multiple apples with random spawning.
"""

from abstractions.ecs.callable_registry import CallableRegistry
from game_entities import GridMap, GridNode, Floor, Agent, Apple
from navigation import compute_navigation_graph
from pathfinding import compute_reachable_paths, choose_path
from movement import (
    move_agent_along_path, find_agent_position, find_apple_at_position,
    get_latest_gridmap, get_latest_agent
)


def create_empty_grid(width=5, height=5):
    """Create an empty grid with just floor tiles."""
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


def print_grid_state(grid: GridMap, agent: Agent, step: int):
    """Print current grid state with agent, apples, and inventory."""
    print(f"\n--- Step {step} ---")
    
    # Find agent and apples
    agent_pos = find_agent_position(grid, agent)
    apple_positions = []
    
    for node in grid.nodes:
        for entity in node.entities:
            if isinstance(entity, Apple):
                apple_positions.append(node.position)
    
    print(f"Grid {grid.width}x{grid.height} (ecs_id: {str(grid.ecs_id)[:8]}...)")
    
    # Print grid
    for y in range(grid.height):
        for x in range(grid.width):
            pos = (x, y)
            if pos == agent_pos:
                print("A ", end="")
            elif pos in apple_positions:
                print("@ ", end="")
            else:
                print(". ", end="")
        print()
    
    print(f"Agent at: {agent_pos}")
    print(f"Apples on grid: {apple_positions}")
    print(f"Inventory: {len(agent.inventory)} apples")


def test_agent_collects_multiple_apples():
    """Test agent collecting 3 apples with random spawning."""
    print("=" * 70)
    print("TEST: Agent Collects Multiple Apples")
    print("=" * 70)
    
    # Create empty grid
    grid = create_empty_grid(5, 5)
    
    # Add agent at (2, 2) - center (agent is part of grid tree)
    agent_pos = (2, 2)
    agent_node = next((n for n in grid.nodes if n.position == agent_pos), None)
    agent = Agent(name="collector", speed=2, sight=5, inventory=[])
    agent_node.entities.append(agent)
    # Don't promote agent - it's part of the grid tree!
    
    # Spawn first apple
    grid = CallableRegistry.execute("spawn_random_apple", grid_map=grid)
    grid = get_latest_gridmap(grid)
    # Agent is part of grid tree, find it in the latest grid
    agent_node = next((n for n in grid.nodes if n.position == agent_pos), None)
    agent = next((e for e in agent_node.entities if isinstance(e, Agent)), None)
    
    print_grid_state(grid, agent, 0)
    
    # Collection loop
    target_apples = 3
    max_steps = 100
    
    for step in range(1, max_steps + 1):
        # Check if collected enough
        if len(agent.inventory) >= target_apples:
            print(f"\n🎉 Collected {target_apples} apples in {step - 1} steps!")
            break
        
        # Find current agent position
        current_pos = find_agent_position(grid, agent)
        
        if not current_pos:
            print("❌ Agent lost!")
            assert False, "Agent not found in grid"
        
        # Check if at apple position
        apple_at_pos = find_apple_at_position(grid, current_pos)
        if apple_at_pos:
            print(f"\n  Step {step}: Found apple '{apple_at_pos.name}' at {current_pos}")
            print(f"  Before collection - inventory: {[a.name for a in agent.inventory]}")
            
            # Collect apple
            grid = CallableRegistry.execute(
                "collect_apple",
                grid_map=grid,
                agent=agent,
                apple_position=current_pos
            )
            
            # Get latest grid version
            grid = get_latest_gridmap(grid)
            # Find agent in latest grid
            current_pos = find_agent_position(grid, agent)
            if current_pos:
                agent_node = next((n for n in grid.nodes if n.position == current_pos), None)
                agent = next((e for e in agent_node.entities if isinstance(e, Agent) and e.name == "collector"), None)
            
            print(f"  After collection - inventory: {[a.name for a in agent.inventory]}")
            
            # Verify apple removed from grid
            apple_still_there = find_apple_at_position(grid, current_pos)
            assert apple_still_there is None, f"Apple should be removed from grid"
            
            # Verify apple in inventory by lineage_id
            apple_lineages_in_inventory = [a.lineage_id for a in agent.inventory]
            assert apple_at_pos.lineage_id in apple_lineages_in_inventory, f"Apple should be in inventory"
            
            print(f"  Total collected: {len(agent.inventory)}/{target_apples}")
            
            # Spawn new apple if needed
            if len(agent.inventory) < target_apples:
                grid = CallableRegistry.execute("spawn_random_apple", grid_map=grid)
                grid = get_latest_gridmap(grid)
            
            print_grid_state(grid, agent, step)
            continue
        
        # Pathfind and move towards apple
        nav_graph = CallableRegistry.execute("compute_navigation_graph", grid_map=grid)
        
        path_collection = CallableRegistry.execute(
            "compute_reachable_paths",
            nav_graph=nav_graph,
            agent=agent,
            start_position=current_pos
        )
        
        chosen_path = CallableRegistry.execute(
            "choose_path",
            path_collection=path_collection,
            grid_map=grid
        )
        
        # Only move if path has more than 1 step
        if len(chosen_path.steps) > 1:
            grid = CallableRegistry.execute(
                "move_agent_along_path",
                grid_map=grid,
                agent=agent,
                path=chosen_path
            )
            grid = get_latest_gridmap(grid)
            # Find agent in latest grid
            current_pos = find_agent_position(grid, agent)
            if current_pos:
                agent_node = next((n for n in grid.nodes if n.position == current_pos), None)
                agent = next((e for e in agent_node.entities if isinstance(e, Agent) and e.name == "collector"), None)
    else:
        # Loop completed without collecting enough
        print(f"\n❌ Failed to collect {target_apples} apples in {max_steps} steps")
        print(f"   Collected: {len(agent.inventory)}")
        assert False, f"Should collect {target_apples} apples in {max_steps} steps"
    
    # Verify final state
    assert len(agent.inventory) == target_apples, f"Should have {target_apples} apples, got {len(agent.inventory)}"
    
    # Verify all apples are unique by lineage_id (proper entity identity)
    apple_lineages = [a.lineage_id for a in agent.inventory]
    print(f"\nDEBUG: Apple lineages in inventory: {[str(l)[:8] for l in apple_lineages]}")
    print(f"DEBUG: Total: {len(apple_lineages)}, Unique: {len(set(apple_lineages))}")
    assert len(apple_lineages) == len(set(apple_lineages)), f"All apples should be unique by lineage_id"
    
    # Print final inventory
    print("\n📦 Final Inventory:")
    for i, apple in enumerate(agent.inventory, 1):
        print(f"  {i}. {apple.name} (nutrition: {apple.nutrition})")
    
    print("\n" + "=" * 70)
    print("✅ APPLE COLLECTION TEST PASSED!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_agent_collects_multiple_apples()
    exit(0 if success else 1)
