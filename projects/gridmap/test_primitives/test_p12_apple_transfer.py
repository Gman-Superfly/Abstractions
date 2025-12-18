"""
Test P12: Entity Transfer Between Container Fields

Tests that entities can be transferred from one container field to another
with proper versioning and tracking. This simulates apple collection behavior
using generic entities.
"""

from abstractions.ecs.entity import Entity, EntityRegistry
from abstractions.ecs.callable_registry import CallableRegistry
from typing import List
from pydantic import Field


class Item(Entity):
    """Simple item entity."""
    name: str
    value: int = 10


class Container(Entity):
    """Container with items in a list."""
    name: str
    items: List[Item] = Field(default_factory=list)


class Collector(Entity):
    """Entity that can collect items into inventory."""
    name: str
    inventory: List[Item] = Field(default_factory=list)


@CallableRegistry.register("transfer_item")
def transfer_item(
    container: Container,
    collector: Collector,
    item_name: str
) -> Container:
    """
    Transfer an item from container to collector's inventory.
    
    Simulates apple collection behavior.
    """
    # Find item in container
    item_to_transfer = None
    for item in container.items:
        if item.name == item_name:
            item_to_transfer = item
            break
    
    if not item_to_transfer:
        print(f"Item '{item_name}' not found in container")
        return container
    
    print(f"Transferring '{item_name}' to {collector.name}'s inventory")
    
    # Remove from container
    container.items.remove(item_to_transfer)
    
    # Add to collector's inventory
    collector.inventory.append(item_to_transfer)
    
    print(f"  Container now has {len(container.items)} items")
    print(f"  Collector now has {len(collector.inventory)} items")
    
    return container


def test_p12_entity_transfer():
    """Test entity transfer between container fields."""
    print("=" * 70)
    print("TEST P12: Entity Transfer Between Containers")
    print("=" * 70)
    
    # Create container with items
    item1 = Item(name="item1", value=10)
    item2 = Item(name="item2", value=20)
    
    container = Container(name="box", items=[item1, item2])
    container.promote_to_root()
    
    print(f"\nContainer created: {container.name}")
    print(f"  Items: {len(container.items)}")
    print(f"  ECS ID: {str(container.ecs_id)[:8]}...")
    print(f"  Lineage: {str(container.lineage_id)[:8]}...")
    
    # Create collector with empty inventory
    collector = Collector(name="agent", inventory=[])
    collector.promote_to_root()
    
    print(f"\nCollector created: {collector.name}")
    print(f"  Inventory: {len(collector.inventory)} items")
    agent_node.entities.append(agent)
    agent.promote_to_root()
    
    print(f"\nAgent added: {agent.name}")
    print(f"  Initial inventory: {len(agent.inventory)} apples")
    
    # Add apple at (1, 1) - same position as agent
    apple = Apple(name="test_apple", nutrition=10)
    agent_node.entities.append(apple)
    apple.promote_to_root()
    
    print(f"\nApple added: {apple.name} at (1, 1)")
    
    # Verify apple is on grid
    apple_on_grid = find_apple_at_position(grid, (1, 1))
    assert apple_on_grid is not None, "Apple should be on grid before collection"
    print(f"  Apple on grid: ✓")
    
    # Collect apple
    print(f"\nCollecting apple...")
    grid_v1 = CallableRegistry.execute(
        "collect_apple",
        grid_map=grid,
        agent=agent,
        apple_position=(1, 1)
    )
    
    print(f"  Grid versioned: {str(grid.ecs_id)[:8]}... → {str(grid_v1.ecs_id)[:8]}...")
    
    # Get latest versions
    grid_latest = get_latest_gridmap(grid)
    agent_latest = get_latest_agent(agent)
    
    print(f"\nLatest versions:")
    print(f"  Grid: {str(grid_latest.ecs_id)[:8]}...")
    print(f"  Agent: {str(agent_latest.ecs_id)[:8]}...")
    print(f"  Agent inventory: {len(agent_latest.inventory)} apples")
    
    # Verify apple removed from grid
    apple_still_on_grid = find_apple_at_position(grid_latest, (1, 1))
    assert apple_still_on_grid is None, "Apple should be removed from grid"
    print(f"  Apple removed from grid: ✓")
    
    # Verify apple in inventory
    assert len(agent_latest.inventory) == 1, f"Agent should have 1 apple, got {len(agent_latest.inventory)}"
    print(f"  Apple in inventory: ✓")
    
    # Verify apple properties preserved
    collected_apple = agent_latest.inventory[0]
    assert collected_apple.name == "test_apple", f"Apple name should be 'test_apple', got '{collected_apple.name}'"
    assert collected_apple.nutrition == 10, f"Apple nutrition should be 10, got {collected_apple.nutrition}"
    print(f"  Apple properties preserved: ✓")
    
    # Verify grid was versioned
    assert grid_v1.ecs_id != grid.ecs_id, "Grid should have new ecs_id"
    assert grid_v1.lineage_id == grid.lineage_id, "Grid should maintain same lineage"
    print(f"  Grid versioning correct: ✓")
    
    # Verify derivation tracking
    assert grid_v1.derived_from_function == "collect_apple", f"Expected 'collect_apple', got '{grid_v1.derived_from_function}'"
    assert grid_v1.derived_from_execution_id is not None, "Should have execution_id"
    print(f"  Derivation tracking: ✓")
    
    print("\n" + "=" * 70)
    print("✅ P12 TEST PASSED!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_p12_apple_transfer()
    exit(0 if success else 1)
