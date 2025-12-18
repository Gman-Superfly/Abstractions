"""
Test P1: Entity Creation & Lifecycle

Tests basic entity creation, promotion to root, and entities with collections.
"""

from typing import List
from abstractions.ecs.entity import Entity
from pydantic import Field


# Test entities
class SimpleEntity(Entity):
    name: str
    value: int = Field(default=0)


class Container(Entity):
    items: List[SimpleEntity] = Field(default_factory=list)


def test_p1_1_create_simple_entity():
    """P1.1: Create an entity with basic fields."""
    print("\n=== P1.1: Create Simple Entity ===")
    
    entity = SimpleEntity(name="test", value=42)
    
    print(f"Created entity: {entity.name}, value={entity.value}")
    print(f"ECS ID: {entity.ecs_id}")
    print(f"Lineage ID: {entity.lineage_id}")
    
    assert entity.name == "test"
    assert entity.value == 42
    assert entity.ecs_id is not None
    assert entity.lineage_id is not None
    
    print("✅ P1.1 PASSED")
    return True


def test_p1_2_promote_to_root():
    """P1.2: Promote entity to root for distributed addressing."""
    print("\n=== P1.2: Promote Entity to Root ===")
    
    entity = SimpleEntity(name="test", value=42)
    
    print(f"Before promotion:")
    print(f"  is_root_entity: {entity.is_root_entity()}")
    print(f"  root_ecs_id: {entity.root_ecs_id}")
    print(f"  root_live_id: {entity.root_live_id}")
    
    entity.promote_to_root()
    
    print(f"\nAfter promotion:")
    print(f"  is_root_entity: {entity.is_root_entity()}")
    print(f"  root_ecs_id: {entity.root_ecs_id}")
    print(f"  root_live_id: {entity.root_live_id}")
    print(f"  ecs_id: {entity.ecs_id}")
    
    assert entity.is_root_entity()
    assert entity.root_ecs_id == entity.ecs_id
    assert entity.root_live_id == entity.live_id
    
    print("✅ P1.2 PASSED")
    return True


def test_p1_3_entity_with_collections():
    """P1.3: Entity containing lists of other entities."""
    print("\n=== P1.3: Entity with Collection Fields ===")
    
    container = Container(items=[])
    item1 = SimpleEntity(name="item1", value=1)
    item2 = SimpleEntity(name="item2", value=2)
    
    print(f"Created container with {len(container.items)} items")
    
    container.items.append(item1)
    container.items.append(item2)
    
    print(f"After appending: {len(container.items)} items")
    print(f"  Item 1: {container.items[0].name}, value={container.items[0].value}")
    print(f"  Item 2: {container.items[1].name}, value={container.items[1].value}")
    
    assert len(container.items) == 2
    assert container.items[0].name == "item1"
    assert container.items[0].value == 1
    assert container.items[1].name == "item2"
    assert container.items[1].value == 2
    
    print("✅ P1.3 PASSED")
    return True


def run_all_tests():
    """Run all P1 tests."""
    print("=" * 60)
    print("TESTING P1: ENTITY CREATION & LIFECYCLE")
    print("=" * 60)
    
    tests = [
        test_p1_1_create_simple_entity,
        test_p1_2_promote_to_root,
        test_p1_3_entity_with_collections,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"P1 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
