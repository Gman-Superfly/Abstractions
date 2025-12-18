"""
Test P3: Direct Mutation & Versioning

Tests direct field mutation and how framework handles versioning.
"""

from typing import List
from abstractions.ecs.entity import Entity
from abstractions.ecs.callable_registry import CallableRegistry
from pydantic import Field


# Test entities
class SimpleEntity(Entity):
    name: str
    value: int = Field(default=0)


class Container(Entity):
    items: List[SimpleEntity] = Field(default_factory=list)


def test_p3_1_direct_field_mutation():
    """P3.1: Mutate entity field directly, framework handles versioning."""
    print("\n=== P3.1: Direct Field Mutation ===")
    
    entity = SimpleEntity(name="original", value=10)
    entity.promote_to_root()
    
    original_ecs_id = entity.ecs_id
    print(f"Original: name={entity.name}, value={entity.value}, ecs_id={original_ecs_id}")
    
    # Direct mutation
    entity.name = "modified"
    entity.value = 20
    
    print(f"After mutation: name={entity.name}, value={entity.value}, ecs_id={entity.ecs_id}")
    print(f"Same object: {entity.ecs_id == original_ecs_id}")
    
    # Entity mutated in place
    assert entity.name == "modified"
    assert entity.value == 20
    assert entity.ecs_id == original_ecs_id  # Same object, not versioned yet
    
    print("✅ P3.1 PASSED")
    return True


def test_p3_2_mutation_in_function_creates_version():
    """P3.2: Function that mutates entity creates new version."""
    print("\n=== P3.2: Mutation in Function Creates Version ===")
    
    @CallableRegistry.register("mutate_entity")
    def mutate_entity(entity: SimpleEntity, new_value: int) -> SimpleEntity:
        entity.value = new_value  # Direct mutation
        return entity
    
    entity = SimpleEntity(name="test", value=10)
    entity.promote_to_root()
    original_ecs_id = entity.ecs_id
    original_lineage_id = entity.lineage_id
    
    print(f"Before execute: value={entity.value}, ecs_id={original_ecs_id}")
    
    result = CallableRegistry.execute("mutate_entity", entity=entity, new_value=20)
    updated = result if not isinstance(result, list) else result[0]
    
    print(f"After execute:")
    print(f"  Original: value={entity.value}, ecs_id={entity.ecs_id}")
    print(f"  Updated: value={updated.value}, ecs_id={updated.ecs_id}")
    print(f"  Different ecs_id: {updated.ecs_id != original_ecs_id}")
    print(f"  Same lineage: {updated.lineage_id == original_lineage_id}")
    
    # New version created
    assert updated.value == 20
    assert updated.ecs_id != original_ecs_id
    assert updated.lineage_id == original_lineage_id
    assert entity.value == 10  # Original unchanged
    
    print("✅ P3.2 PASSED")
    return True


def test_p3_3_list_mutation_in_tree():
    """P3.3: Mutate list within entity tree, framework tracks changes."""
    print("\n=== P3.3: List Mutation in Tree ===")
    
    @CallableRegistry.register("add_to_container")
    def add_to_container(container: Container, item: SimpleEntity) -> Container:
        container.items.append(item)  # Direct list mutation
        return container
    
    container = Container(items=[])
    container.promote_to_root()
    original_ecs_id = container.ecs_id
    
    print(f"Before: {len(container.items)} items, ecs_id={original_ecs_id}")
    
    item = SimpleEntity(name="new_item", value=99)
    result = CallableRegistry.execute("add_to_container", container=container, item=item)
    updated = result if not isinstance(result, list) else result[0]
    
    print(f"After:")
    print(f"  Original: {len(container.items)} items, ecs_id={container.ecs_id}")
    print(f"  Updated: {len(updated.items)} items, ecs_id={updated.ecs_id}")
    if len(updated.items) > 0:
        print(f"    Item: {updated.items[0].name}, value={updated.items[0].value}")
    print(f"  Different ecs_id: {updated.ecs_id != original_ecs_id}")
    
    assert len(updated.items) == 1
    assert updated.items[0].name == "new_item"
    assert updated.ecs_id != original_ecs_id  # New version
    
    print("✅ P3.3 PASSED")
    return True


def run_all_tests():
    """Run all P3 tests."""
    print("=" * 60)
    print("TESTING P3: DIRECT MUTATION & VERSIONING")
    print("=" * 60)
    
    tests = [
        test_p3_1_direct_field_mutation,
        test_p3_2_mutation_in_function_creates_version,
        test_p3_3_list_mutation_in_tree,
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
    print(f"P3 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
