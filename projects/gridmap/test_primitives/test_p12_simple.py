"""
Simple test to verify entity versioning when List[Entity] fields are modified.

Tests that when an item is added to a List[Entity] field, the parent entity
gets a new ecs_id (is versioned).
"""

from abstractions.ecs.entity import Entity, EntityRegistry
from abstractions.ecs.callable_registry import CallableRegistry
from typing import List
from pydantic import Field


class Item(Entity):
    name: str
    value: int = 10


class Collector(Entity):
    name: str
    inventory: List[Item] = Field(default_factory=list)


class Container(Entity):
    name: str
    items: List[Item] = Field(default_factory=list)
    collector: Collector = None


@CallableRegistry.register("transfer_item")
def transfer_item(container: Container, item_name: str) -> Container:
    """Transfer an item from container.items to container.collector.inventory."""
    # Find and remove item from container
    item_to_transfer = None
    for item in container.items:
        if item.name == item_name:
            item_to_transfer = item
            break
    
    if item_to_transfer:
        container.items.remove(item_to_transfer)
        container.collector.inventory.append(item_to_transfer)
    
    return container


def test_list_entity_modification_triggers_versioning():
    """Test that modifying List[Entity] triggers parent versioning."""
    
    # Setup
    item1 = Item(name="apple", value=10)
    item2 = Item(name="banana", value=20)
    collector = Collector(name="agent", inventory=[])
    container = Container(name="box", items=[item1, item2], collector=collector)
    container.promote_to_root()
    
    # Record initial state
    container_id_before = container.ecs_id
    collector_id_before = container.collector.ecs_id
    
    print(f"BEFORE execution:")
    print(f"  Container ecs_id: {str(container_id_before)[:8]}...")
    print(f"  Collector ecs_id: {str(collector_id_before)[:8]}...")
    print(f"  Container.items: {len(container.items)} items")
    print(f"  Collector.inventory: {len(container.collector.inventory)} items")
    
    # Execute mutation
    container_after = CallableRegistry.execute(
        "transfer_item",
        container=container,
        item_name="apple"
    )
    
    # Get latest versions
    lineage_id = container.lineage_id
    versions = EntityRegistry.lineage_registry.get(lineage_id, [])
    latest_root_id = versions[-1]
    latest_tree = EntityRegistry.get_stored_tree(latest_root_id)
    container_latest = latest_tree.get_entity(latest_root_id)
    
    container_id_after = container_latest.ecs_id
    collector_id_after = container_latest.collector.ecs_id
    
    print(f"\nAFTER execution:")
    print(f"  Container ecs_id: {str(container_id_after)[:8]}...")
    print(f"  Collector ecs_id: {str(collector_id_after)[:8]}...")
    print(f"  Container.items: {len(container_latest.items)} items")
    print(f"  Collector.inventory: {len(container_latest.collector.inventory)} items")
    
    # Verify results
    print(f"\nVERIFICATION:")
    
    # 1. Container should be versioned (new ecs_id)
    container_versioned = container_id_after != container_id_before
    print(f"  Container versioned: {container_versioned} {'✓' if container_versioned else '✗'}")
    
    # 2. Collector should be versioned (new ecs_id) - THIS IS THE BUG
    collector_versioned = collector_id_after != collector_id_before
    print(f"  Collector versioned: {collector_versioned} {'✓' if collector_versioned else '✗'}")
    
    # 3. Item was transferred
    items_transferred = len(container_latest.items) == 1 and len(container_latest.collector.inventory) == 1
    print(f"  Item transferred: {items_transferred} {'✓' if items_transferred else '✗'}")
    
    # 4. Lineage has 2 versions
    has_two_versions = len(versions) == 2
    print(f"  Lineage has 2 versions: {has_two_versions} {'✓' if has_two_versions else '✗'}")
    
    # Assert
    print(f"\nRESULT:")
    if not collector_versioned:
        print(f"  ❌ FAILED: Collector was not versioned!")
        print(f"     Expected: Collector gets new ecs_id when inventory changes")
        print(f"     Actual: Collector kept same ecs_id {str(collector_id_before)[:8]}...")
        return False
    else:
        print(f"  ✅ PASSED: All entities properly versioned")
        return True


if __name__ == "__main__":
    success = test_list_entity_modification_triggers_versioning()
    exit(0 if success else 1)
