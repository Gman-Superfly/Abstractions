"""
Test P7: Entity Tree Operations

Tests attach(), detach(), and version detection in trees.
"""

from typing import List
from abstractions.ecs.entity import Entity, build_entity_tree, EntityRegistry
from pydantic import Field


# Test entities
class SimpleEntity(Entity):
    name: str
    value: int = Field(default=0)


class Parent(Entity):
    name: str
    children: List[SimpleEntity] = Field(default_factory=list)


def test_p7_1_detach_entity_from_tree():
    """P7.1: Remove entity from tree, call detach()."""
    print("\n=== P7.1: Detach Entity from Tree ===")
    
    parent = Parent(name="parent", children=[])
    child = SimpleEntity(name="child", value=42)
    parent.children.append(child)
    parent.promote_to_root()
    
    print(f"Before detach:")
    print(f"  Child is_root_entity: {child.is_root_entity()}")
    print(f"  Child root_ecs_id: {child.root_ecs_id}")
    
    # Child is part of tree
    assert not child.is_root_entity()
    
    # Remove and detach
    parent.children.remove(child)
    child.detach()
    
    print(f"After detach:")
    print(f"  Child is_root_entity: {child.is_root_entity()}")
    print(f"  Child root_ecs_id: {child.root_ecs_id}")
    
    # Child is now root
    assert child.is_root_entity()
    
    print("✅ P7.1 PASSED")
    return True


def test_p7_2_attach_root_entity_to_tree():
    """P7.2: Add external root entity to tree, call attach()."""
    print("\n=== P7.2: Attach Root Entity to Tree ===")
    
    parent = Parent(name="parent", children=[])
    parent.promote_to_root()
    
    # Create separate root entity
    external = SimpleEntity(name="external", value=99)
    external.promote_to_root()
    
    print(f"Before attach:")
    print(f"  External is_root_entity: {external.is_root_entity()}")
    print(f"  External root_ecs_id: {external.root_ecs_id}")
    
    assert external.is_root_entity()
    
    # Add to parent
    parent.children.append(external)
    external.attach(parent)
    
    print(f"After attach:")
    print(f"  External is_root_entity: {external.is_root_entity()}")
    print(f"  External root_ecs_id: {external.root_ecs_id}")
    print(f"  Parent ecs_id: {parent.ecs_id}")
    
    # External is now part of parent's tree
    assert not external.is_root_entity()
    assert external.root_ecs_id == parent.ecs_id
    
    print("✅ P7.2 PASSED")
    return True


def test_p7_3_version_detection_after_mutation():
    """P7.3: Versioning happens through CallableRegistry, not manual mutation."""
    print("\n=== P7.3: Version Detection After Mutation ===")
    
    from abstractions.ecs.callable_registry import CallableRegistry
    
    # Register a function that mutates the tree
    @CallableRegistry.register("mutate_child")
    def mutate_child(parent: Parent, new_value: int) -> Parent:
        if len(parent.children) > 0:
            parent.children[0].value = new_value
        return parent
    
    parent = Parent(name="parent", children=[])
    child = SimpleEntity(name="child", value=10)
    parent.children.append(child)
    parent.promote_to_root()
    
    original_parent_id = parent.ecs_id
    original_child_id = child.ecs_id
    
    print(f"Before mutation:")
    print(f"  Parent ecs_id: {original_parent_id}")
    print(f"  Child ecs_id: {original_child_id}")
    print(f"  Child value: {child.value}")
    
    # Mutate through CallableRegistry (this triggers versioning)
    result = CallableRegistry.execute("mutate_child", parent=parent, new_value=20)
    updated_parent = result if not isinstance(result, list) else result[0]
    
    print(f"After CallableRegistry.execute:")
    print(f"  Original parent ecs_id: {parent.ecs_id}")
    print(f"  Updated parent ecs_id: {updated_parent.ecs_id}")
    print(f"  Updated child value: {updated_parent.children[0].value if len(updated_parent.children) > 0 else 'N/A'}")
    print(f"  Different ecs_id: {updated_parent.ecs_id != original_parent_id}")
    
    # Versioning happens automatically through CallableRegistry
    assert updated_parent.ecs_id != original_parent_id
    assert parent.children[0].value == 10  # Original unchanged
    if len(updated_parent.children) > 0:
        assert updated_parent.children[0].value == 20
    
    print("✅ P7.3 PASSED")
    print("Note: Versioning happens through CallableRegistry.execute(), not manual EntityRegistry.version_entity()")
    return True


def run_all_tests():
    """Run all P7 tests."""
    print("=" * 60)
    print("TESTING P7: ENTITY TREE OPERATIONS")
    print("=" * 60)
    
    tests = [
        test_p7_1_detach_entity_from_tree,
        test_p7_2_attach_root_entity_to_tree,
        test_p7_3_version_detection_after_mutation,
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
    print(f"P7 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
