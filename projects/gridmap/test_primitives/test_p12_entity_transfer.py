"""
Test P12: Entity Transfer with Proper Versioning

Tests that when entities are transferred between List[Entity] fields,
ALL affected entities (including parents) get properly versioned with new ecs_ids.

This test verifies the fix for the bug where child entities weren't being
versioned when their List[Entity] fields were modified.
"""

from abstractions.ecs.entity import Entity, EntityRegistry
from abstractions.ecs.callable_registry import CallableRegistry
from typing import List
from pydantic import Field


class Item(Entity):
    """Simple item entity."""
    name: str
    value: int = 10


class Collector(Entity):
    """Entity that can collect items into inventory."""
    name: str
    inventory: List[Item] = Field(default_factory=list)


class Container(Entity):
    """Container with items and a collector (single tree)."""
    name: str
    items: List[Item] = Field(default_factory=list)
    collector: Collector = None


@CallableRegistry.register("transfer_item")
def transfer_item(container: Container, item_name: str) -> Container:
    """
    Transfer an item from container.items to container.collector.inventory.
    
    This function mutates the container tree directly.
    Framework will create a new version after execution.
    """
    # Find item in container
    item_to_transfer = None
    for item in container.items:
        if item.name == item_name:
            item_to_transfer = item
            break
    
    if not item_to_transfer:
        print(f"❌ Item '{item_name}' not found")
        return container
    
    print(f"🔄 Transferring '{item_name}' to collector's inventory")
    
    # Mutate: Remove from container.items
    container.items.remove(item_to_transfer)
    
    # Mutate: Add to collector.inventory
    container.collector.inventory.append(item_to_transfer)
    
    print(f"   Container.items: {len(container.items)}")
    
    return container


def test_p12_entity_transfer():
    """Test entity transfer with proper versioning of all affected entities."""
    print("\n" + "=" * 70)
    print("TEST: Entity Transfer with Versioning")
    print("=" * 70)
    
    # Create initial tree
    print("\n1. Setup")
    item1 = Item(name="apple", value=10)
    item2 = Item(name="banana", value=20)
    collector = Collector(name="agent", inventory=[])
    
    container = Container(name="box", items=[item1, item2], collector=collector)
    container.promote_to_root()
    
    # Record initial IDs
    container_id_before = container.ecs_id
    collector_id_before = container.collector.ecs_id
    item1_id_before = item1.ecs_id
    
    print(f"   Created container with 2 items and empty collector")
    print(f"   Container: {str(container_id_before)[:8]}...")
    print(f"   Collector: {str(collector_id_before)[:8]}...")
    print(f"   Items in container: {len(container.items)}")
    print(f"   Items in collector: {len(container.collector.inventory)}")
    
    # Step 2: Execute transfer (creates v1)
    print("\n--- Step 2: Execute Transfer ---")
    print("Calling: transfer_item(container, 'apple')")
    
    container_v1 = CallableRegistry.execute(
        "transfer_item",
        container=container_v0,
        item_name="apple"
    )
    
    print(f"\nReturned container_v1:")
    print(f"  ecs_id: {str(container_v1.ecs_id)[:8]}...")
    print(f"  lineage_id: {str(container_v1.lineage_id)[:8]}...")
    
    # Step 3: Check OLD version (should be unchanged)
    print("\n--- Step 3: Check OLD Version (v0) ---")
    print(f"container_v0.items: {len(container_v0.items)}")
    print(f"container_v0.collector.inventory: {len(container_v0.collector.inventory)}")
    
    assert len(container_v0.items) == 2, "OLD version should be unchanged!"
    assert len(container_v0.collector.inventory) == 0, "OLD version should be unchanged!"
    print("✓ OLD version unchanged (immutability verified)")
    
    # Step 4: Check NEW version (should have changes)
    print("\n--- Step 4: Check NEW Version (v1) ---")
    print(f"container_v1.items: {len(container_v1.items)}")
    print(f"container_v1.collector.inventory: {len(container_v1.collector.inventory)}")
    
    assert len(container_v1.items) == 1, f"NEW version should have 1 item, got {len(container_v1.items)}"
    assert len(container_v1.collector.inventory) == 1, f"NEW version should have 1 in inventory, got {len(container_v1.collector.inventory)}"
    print("✓ NEW version has changes")
    
    # Step 5: Verify item properties
    print("\n--- Step 5: Verify Item Properties ---")
    collected_item = container_v1.collector.inventory[0]
    print(f"Collected item: {collected_item.name} (value={collected_item.value})")
    
    assert collected_item.name == "apple"
    assert collected_item.value == 10
    print("✓ Item properties preserved")
    
    # Step 6: Verify lineage registry
    print("\n--- Step 6: Verify Lineage Registry ---")
    lineage_id = container_v0.lineage_id
    versions = EntityRegistry.lineage_registry.get(lineage_id, [])
    
    print(f"Lineage {str(lineage_id)[:8]}... has {len(versions)} versions:")
    for i, root_ecs_id in enumerate(versions):
        print(f"  v{i}: {str(root_ecs_id)[:8]}...")
    
    assert len(versions) == 2, f"Should have 2 versions, got {len(versions)}"
    assert versions[0] == container_v0.ecs_id
    assert versions[1] == container_v1.ecs_id
    print("✓ Lineage registry correct")
    
    # Step 6.5: Inspect tree edges
    print("\n--- Step 6.5: Inspect Tree Edges ---")
    tree_v0 = EntityRegistry.get_stored_tree(container_v0.ecs_id)
    tree_v1 = EntityRegistry.get_stored_tree(container_v1.ecs_id)
    
    print(f"Tree v0 edges: {len(tree_v0.edges)}")
    for (source_id, target_id), edge in tree_v0.edges.items():
        source_entity = tree_v0.get_entity(source_id)
        target_entity = tree_v0.get_entity(target_id)
        print(f"  {source_entity.__class__.__name__}({str(source_id)[:8]}...) → {target_entity.__class__.__name__}({str(target_id)[:8]}...) [field: {edge.field_name}]")
    
    print(f"\nTree v1 edges: {len(tree_v1.edges)}")
    for (source_id, target_id), edge in tree_v1.edges.items():
        source_entity = tree_v1.get_entity(source_id)
        target_entity = tree_v1.get_entity(target_id)
        print(f"  {source_entity.__class__.__name__}({str(source_id)[:8]}...) → {target_entity.__class__.__name__}({str(target_id)[:8]}...) [field: {edge.field_name}]")
    
    # Step 6.6: Call find_modified_entities with debug
    print("\n--- Step 6.6: Debug find_modified_entities ---")
    from abstractions.ecs.entity import find_modified_entities
    
    # Get the tree that was built BEFORE versioning (from live root)
    from abstractions.ecs.entity import build_entity_tree
    new_tree_before_version = build_entity_tree(container_v0.get_live_root_entity())
    
    modified, debug_info = find_modified_entities(
        new_tree=new_tree_before_version,
        old_tree=tree_v0,
        debug=True
    )
    
    print(f"Modified entities: {len(modified)}")
    for ecs_id in modified:
        entity = new_tree_before_version.get_entity(ecs_id)
        if entity:
            print(f"  {entity.__class__.__name__}({str(ecs_id)[:8]}...)")
    
    print(f"\nDebug info:")
    print(f"  Added entities: {len(debug_info['added_entities'])}")
    print(f"  Removed entities: {len(debug_info['removed_entities'])}")
    print(f"  Moved entities: {len(debug_info['moved_entities'])}")
    print(f"  Unchanged entities: {len(debug_info['unchanged_entities'])}")
    print(f"  Comparisons: {debug_info['comparison_count']}")
    
    # Step 7: Get latest using lineage
    print("\n--- Step 7: Get Latest Using Lineage ---")
    latest_root_ecs_id = versions[-1]
    latest_tree = EntityRegistry.get_stored_tree(latest_root_ecs_id)
    container_latest = latest_tree.get_entity(latest_root_ecs_id)
    
    print(f"Latest version:")
    print(f"  ecs_id: {str(container_latest.ecs_id)[:8]}...")
    print(f"  items: {len(container_latest.items)}")
    print(f"  collector.inventory: {len(container_latest.collector.inventory)}")
    
    assert container_latest.ecs_id == container_v1.ecs_id
    assert len(container_latest.collector.inventory) == 1
    print("✓ Retrieved latest version correctly")
    
    # Step 8: Verify derivation tracking
    print("\n--- Step 8: Verify Derivation Tracking ---")
    print(f"derived_from_function: {container_v1.derived_from_function}")
    print(f"derived_from_execution_id: {str(container_v1.derived_from_execution_id)[:8]}...")
    
    assert container_v1.derived_from_function == "transfer_item"
    assert container_v1.derived_from_execution_id is not None
    print("✓ Derivation tracking works")
    
    # Step 9: ASSERT THE BUG
    print("\n--- Step 9: VERIFY BUG EXISTS ---")
    
    print("\n🐛 BUG FOUND:")
    print(f"  find_modified_entities returned: {len(modified)} modified entities")
    print(f"  But Container DID get versioned: {container_v0.ecs_id} → {container_v1.ecs_id}")
    print(f"  And Collector did NOT get versioned: {container_v0.collector.ecs_id} → {container_v1.collector.ecs_id}")
    
    print("\n🔍 Edge changes detected:")
    v0_edges = set(tree_v0.edges.keys())
    v1_edges = set(tree_v1.edges.keys())
    added = v1_edges - v0_edges
    removed = v0_edges - v1_edges
    
    print(f"  Added edges: {len(added)}")
    for source_id, target_id in added:
        source = tree_v1.get_entity(source_id)
        target = tree_v1.get_entity(target_id)
        edge = tree_v1.edges[(source_id, target_id)]
        print(f"    {source.__class__.__name__} → {target.__class__.__name__} [field: {edge.field_name}]")
    
    print(f"  Removed edges: {len(removed)}")
    for source_id, target_id in removed:
        source = tree_v0.get_entity(source_id)
        target = tree_v0.get_entity(target_id)
        edge = tree_v0.edges[(source_id, target_id)]
        print(f"    {source.__class__.__name__} → {target.__class__.__name__} [field: {edge.field_name}]")
    
    print("\n❌ EXPECTED BEHAVIOR:")
    print("  1. Collector should get NEW ecs_id (it owns the list that changed)")
    print("  2. find_modified_entities should detect edge changes")
    print("  3. Collector's path should be marked for versioning")
    
    print("\n❌ ACTUAL BEHAVIOR:")
    print(f"  1. Collector kept SAME ecs_id: {container_v1.collector.ecs_id}")
    print(f"  2. find_modified_entities found: {len(modified)} entities")
    print(f"  3. Edge added but parent not versioned!")
    
    # FAIL THE TEST
    assert False, f"BUG: Collector should be versioned when item added to inventory! Modified entities: {len(modified)}, Collector ecs_id unchanged: {container_v0.collector.ecs_id}"
    
    return True


if __name__ == "__main__":
    success = test_p12_entity_transfer()
    exit(0 if success else 1)
