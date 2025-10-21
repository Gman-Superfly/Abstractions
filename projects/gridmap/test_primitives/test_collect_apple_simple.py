"""Simple test for collect_apple function."""

from abstractions.ecs.callable_registry import CallableRegistry
from abstractions.ecs.entity import EntityRegistry
from game_entities import GridMap, GridNode, Floor, Agent, Apple
import movement  # Import to register functions


def test_collect_apple():
    """Test that collect_apple properly removes apple from grid and adds to inventory."""
    print("\nTEST: Collect Apple")
    print("=" * 50)
    
    # Create simple grid
    nodes = []
    for y in range(3):
        for x in range(3):
            node = GridNode(position=(x, y))
            node.entities = [Floor(name=f"floor_{x}_{y}")]
            nodes.append(node)
    
    grid = GridMap(nodes=nodes, width=3, height=3)
    grid.promote_to_root()
    
    # Add agent at (1, 1) - agent is part of grid tree, don't promote separately
    agent_node = next(n for n in grid.nodes if n.position == (1, 1))
    agent = Agent(name="test_agent", speed=2, inventory=[])
    agent_node.entities.append(agent)
    # Don't promote agent - it's part of the grid tree!
    
    # Add apple at (1, 1) - same position as agent
    apple = Apple(name="test_apple", nutrition=10)
    agent_node.entities.append(apple)
    # Don't promote apple - it's part of the grid tree!
    
    print(f"BEFORE collection:")
    print(f"  Grid ecs_id: {str(grid.ecs_id)[:8]}...")
    print(f"  Agent inventory: {len(agent.inventory)}")
    print(f"  Apples at (1,1): {sum(1 for e in agent_node.entities if isinstance(e, Apple))}")
    
    # Collect apple
    grid_after = CallableRegistry.execute(
        "collect_apple",
        grid_map=grid,
        agent=agent,
        apple_position=(1, 1)
    )
    
    print(f"\nAFTER collection (returned grid):")
    print(f"  Grid ecs_id: {str(grid_after.ecs_id)[:8]}...")
    
    # Get latest versions
    lineage_id = grid.lineage_id
    versions = EntityRegistry.lineage_registry.get(lineage_id, [])
    latest_root_id = versions[-1]
    latest_tree = EntityRegistry.get_stored_tree(latest_root_id)
    grid_latest = latest_tree.get_entity(latest_root_id)
    
    # Find agent in latest tree
    agent_latest = None
    for entity in latest_tree.nodes.values():
        if isinstance(entity, Agent) and entity.name == "test_agent":
            agent_latest = entity
            break
    
    # Find node at (1,1) in latest grid
    node_latest = next(n for n in grid_latest.nodes if n.position == (1, 1))
    apples_at_pos = [e for e in node_latest.entities if isinstance(e, Apple)]
    
    print(f"\nLATEST version:")
    print(f"  Grid ecs_id: {str(grid_latest.ecs_id)[:8]}...")
    print(f"  Agent inventory: {len(agent_latest.inventory)}")
    print(f"  Apples at (1,1): {len(apples_at_pos)}")
    
    # Verify
    assert len(agent_latest.inventory) == 1, f"Agent should have 1 apple, got {len(agent_latest.inventory)}"
    assert len(apples_at_pos) == 0, f"Should be 0 apples at position, got {len(apples_at_pos)}"
    
    print("\n✅ PASSED: Apple collected successfully!")
    return True


if __name__ == "__main__":
    success = test_collect_apple()
    exit(0 if success else 1)
