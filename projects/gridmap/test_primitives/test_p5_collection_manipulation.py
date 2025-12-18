"""
Test P5: Collection Manipulation

Tests list operations within entity trees (append, remove, move).
"""

from typing import List
from abstractions.ecs.entity import Entity
from abstractions.ecs.callable_registry import CallableRegistry
from pydantic import Field


# Test entities
class SimpleEntity(Entity):
    name: str
    value: int = Field(default=0)


class Parent(Entity):
    name: str
    children: List[SimpleEntity] = Field(default_factory=list)


class TwoContainers(Entity):
    list_a: List[SimpleEntity] = Field(default_factory=list)
    list_b: List[SimpleEntity] = Field(default_factory=list)


def test_p5_1_append_to_list_in_tree():
    """P5.1: Add entity to list within tree structure."""
    print("\n=== P5.1: Append to List in Tree ===")
    
    @CallableRegistry.register("append_item")
    def append_item(parent: Parent, item: SimpleEntity) -> Parent:
        parent.children.append(item)
        return parent
    
    parent = Parent(name="parent", children=[])
    parent.promote_to_root()
    
    print(f"Before: {len(parent.children)} children")
    
    item = SimpleEntity(name="child", value=42)
    result = CallableRegistry.execute("append_item", parent=parent, item=item)
    updated = result if not isinstance(result, list) else result[0]
    
    print(f"After: {len(updated.children)} children")
    if len(updated.children) > 0:
        print(f"  Child: name={updated.children[0].name}, value={updated.children[0].value}")
    
    assert len(updated.children) == 1
    assert updated.children[0].name == "child"
    assert updated.children[0].value == 42
    
    print("✅ P5.1 PASSED")
    return True


def test_p5_2_remove_from_list_in_tree():
    """P5.2: Remove entity from list within tree structure."""
    print("\n=== P5.2: Remove from List in Tree ===")
    
    @CallableRegistry.register("remove_item")
    def remove_item(parent: Parent, item_name: str) -> Parent:
        parent.children = [c for c in parent.children if c.name != item_name]
        return parent
    
    parent = Parent(name="parent", children=[])
    child1 = SimpleEntity(name="keep", value=1)
    child2 = SimpleEntity(name="remove", value=2)
    parent.children.append(child1)
    parent.children.append(child2)
    parent.promote_to_root()
    
    print(f"Before: {len(parent.children)} children")
    for child in parent.children:
        print(f"  - {child.name}")
    
    result = CallableRegistry.execute("remove_item", parent=parent, item_name="remove")
    updated = result if not isinstance(result, list) else result[0]
    
    print(f"After: {len(updated.children)} children")
    for child in updated.children:
        print(f"  - {child.name}")
    
    assert len(updated.children) == 1
    assert updated.children[0].name == "keep"
    
    print("✅ P5.2 PASSED")
    return True


def test_p5_3_move_item_between_lists():
    """P5.3: Remove from one list, add to another (within same tree)."""
    print("\n=== P5.3: Move Item Between Lists ===")
    
    @CallableRegistry.register("move_item")
    def move_item(container: TwoContainers, item_name: str) -> TwoContainers:
        # Find and remove from list_a
        item = None
        for i, entity in enumerate(container.list_a):
            if entity.name == item_name:
                item = container.list_a.pop(i)
                break
        
        # Add to list_b
        if item:
            container.list_b.append(item)
        
        return container
    
    container = TwoContainers(list_a=[], list_b=[])
    item = SimpleEntity(name="movable", value=42)
    container.list_a.append(item)
    container.promote_to_root()
    
    print(f"Before: list_a={len(container.list_a)}, list_b={len(container.list_b)}")
    
    result = CallableRegistry.execute("move_item", container=container, item_name="movable")
    updated = result if not isinstance(result, list) else result[0]
    
    print(f"After: list_a={len(updated.list_a)}, list_b={len(updated.list_b)}")
    if len(updated.list_b) > 0:
        print(f"  Item in list_b: {updated.list_b[0].name}")
    
    assert len(updated.list_a) == 0
    assert len(updated.list_b) == 1
    assert updated.list_b[0].name == "movable"
    
    print("✅ P5.3 PASSED")
    return True


def run_all_tests():
    """Run all P5 tests."""
    print("=" * 60)
    print("TESTING P5: COLLECTION MANIPULATION")
    print("=" * 60)
    
    tests = [
        test_p5_1_append_to_list_in_tree,
        test_p5_2_remove_from_list_in_tree,
        test_p5_3_move_item_between_lists,
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
    print(f"P5 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
