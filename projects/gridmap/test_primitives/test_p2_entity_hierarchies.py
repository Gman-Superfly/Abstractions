"""
Test P2: Entity Hierarchies & Trees

Tests nested entity structures and tree building.
"""

from typing import List
from abstractions.ecs.entity import Entity, build_entity_tree
from pydantic import Field


# Test entities
class SimpleEntity(Entity):
    name: str
    value: int = Field(default=0)


class Parent(Entity):
    name: str
    children: List[SimpleEntity] = Field(default_factory=list)


class Level3(Entity):
    value: int


class Level2(Entity):
    items: List[Level3] = Field(default_factory=list)


class Level1(Entity):
    containers: List[Level2] = Field(default_factory=list)


def test_p2_1_nested_entity_structure():
    """P2.1: Entity containing other entities (hierarchical tree)."""
    print("\n=== P2.1: Nested Entity Structure ===")
    
    parent = Parent(name="parent", children=[])
    child1 = SimpleEntity(name="child1", value=1)
    child2 = SimpleEntity(name="child2", value=2)
    
    parent.children.append(child1)
    parent.children.append(child2)
    parent.promote_to_root()
    
    print(f"Parent: {parent.name}")
    print(f"  is_root_entity: {parent.is_root_entity()}")
    print(f"  Children: {len(parent.children)}")
    for i, child in enumerate(parent.children):
        print(f"    Child {i+1}: {child.name}, value={child.value}")
    
    assert parent.is_root_entity()
    assert len(parent.children) == 2
    assert parent.children[0].name == "child1"
    assert parent.children[1].name == "child2"
    
    print("✅ P2.1 PASSED")
    return True


def test_p2_2_build_entity_tree():
    """P2.2: Framework can build tree from nested entities."""
    print("\n=== P2.2: Build Entity Tree ===")
    
    parent = Parent(name="parent", children=[])
    child = SimpleEntity(name="child", value=42)
    parent.children.append(child)
    parent.promote_to_root()
    
    print(f"Building tree from parent...")
    tree = build_entity_tree(parent)
    
    print(f"Tree built:")
    print(f"  Tree is None: {tree is None}")
    if tree:
        print(f"  Nodes in tree: {len(tree.nodes)}")
        print(f"  Parent in tree: {parent.ecs_id in tree.nodes}")
        print(f"  Child in tree: {child.ecs_id in tree.nodes}")
    
    assert tree is not None
    assert parent.ecs_id in tree.nodes
    assert child.ecs_id in tree.nodes
    
    print("✅ P2.2 PASSED")
    return True


def test_p2_3_three_level_hierarchy():
    """P2.3: Three-level hierarchy (GridMap depth)."""
    print("\n=== P2.3: Three-Level Hierarchy ===")
    
    root = Level1(containers=[])
    mid = Level2(items=[])
    leaf = Level3(value=42)
    
    mid.items.append(leaf)
    root.containers.append(mid)
    root.promote_to_root()
    
    print(f"Level 1 (root): {len(root.containers)} containers")
    print(f"Level 2 (mid): {len(mid.items)} items")
    print(f"Level 3 (leaf): value={leaf.value}")
    
    print(f"\nBuilding tree...")
    tree = build_entity_tree(root)
    
    if tree:
        print(f"Tree nodes: {len(tree.nodes)}")
        print(f"  Root in tree: {root.ecs_id in tree.nodes}")
        print(f"  Mid in tree: {mid.ecs_id in tree.nodes}")
        print(f"  Leaf in tree: {leaf.ecs_id in tree.nodes}")
    
    assert tree is not None
    assert root.ecs_id in tree.nodes
    assert mid.ecs_id in tree.nodes
    assert leaf.ecs_id in tree.nodes
    
    print("✅ P2.3 PASSED")
    return True


def run_all_tests():
    """Run all P2 tests."""
    print("=" * 60)
    print("TESTING P2: ENTITY HIERARCHIES & TREES")
    print("=" * 60)
    
    tests = [
        test_p2_1_nested_entity_structure,
        test_p2_2_build_entity_tree,
        test_p2_3_three_level_hierarchy,
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
    print(f"P2 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
