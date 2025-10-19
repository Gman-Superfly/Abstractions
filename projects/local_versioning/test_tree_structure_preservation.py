#!/usr/bin/env python3
"""
Test: Tree Structure Preservation in Transactional Execution

This test identifies the problem with 4 scenarios:

SCENARIO 1: GLOBAL with tree preservation (root + children in, root out)
  Input: gridmap + node1 + node2 + agent (all from same tree)
  Expected: Modifications to nodes should affect the returned gridmap
  Current: BROKEN - gridmap copy has original nodes, not modified copies

SCENARIO 2: GLOBAL without tree preservation (borrowing pattern)
  Input: student + course (from different trees)
  Expected: Create new entity borrowing data, originals unchanged
  Current: WORKS - this is the default behavior

SCENARIO 3: LOCAL with auto-reattachment (children in, children out)
  Input: node1 + node2 + agent (no root, but have root_ecs_id)
  Expected: Parent gridmap should be versioned, nodes reattached
  Current: BROKEN - nodes detached, parent not versioned

SCENARIO 4: LOCAL without reattachment (pure local processing)
  Input: node1 + node2 (orphan entities, no root_ecs_id)
  Expected: Process nodes independently, no parent involvement
  Current: WORKS - nodes are independent
"""

import sys
from pathlib import Path
from typing import List, Tuple
from pydantic import Field

abstractions_path = Path(__file__).parent.parent.parent / "abstractions"
sys.path.insert(0, str(abstractions_path))

from abstractions.ecs.entity import Entity, EntityRegistry
from abstractions.ecs.callable_registry import CallableRegistry


# Domain models
class Agent(Entity):
    name: str = ""


class Node(Entity):
    agents: List[Agent] = Field(default_factory=list)


class GridMap(Entity):
    nodes: List[Node] = Field(default_factory=list)


class Student(Entity):
    name: str = ""
    gpa: float = 0.0


class Course(Entity):
    name: str = ""
    credits: int = 0


class Report(Entity):
    student_name: str = ""
    course_name: str = ""
    summary: str = ""


# Test functions
@CallableRegistry.register("move_agent_global")
def move_agent_global(
    gridmap: GridMap,
    source_node: Node,
    target_node: Node,
    agent: Agent
) -> GridMap:
    """SCENARIO 1: Global with tree preservation"""
    source_node.agents.remove(agent)
    target_node.agents.append(agent)
    return gridmap


@CallableRegistry.register("create_report")
def create_report(student: Student, course: Course) -> Report:
    """SCENARIO 2: Borrowing pattern (no tree preservation needed)"""
    return Report(
        student_name=student.name,
        course_name=course.name,
        summary=f"{student.name} in {course.name}"
    )


@CallableRegistry.register("move_agent_local")
def move_agent_local(
    source_node: Node,
    target_node: Node,
    agent: Agent
) -> Tuple[Node, Node]:
    """SCENARIO 3: Local with auto-reattachment"""
    source_node.agents.remove(agent)
    target_node.agents.append(agent)
    return source_node, target_node


@CallableRegistry.register("process_nodes_orphan")
def process_nodes_orphan(node1: Node, node2: Node) -> Tuple[Node, Node]:
    """SCENARIO 4: Pure local processing (orphan entities)"""
    # Just swap agent counts or something
    return node1, node2


def test_scenario_1_global_with_tree_preservation():
    """
    SCENARIO 1: GLOBAL with tree preservation
    
    Problem: When we pass gridmap + nodes from same tree, the deep copy
    creates separate copies. The gridmap copy has ORIGINAL nodes, not the
    node copies that get modified.
    
    Tests both WITH and WITHOUT preserve_tree_structure flag.
    """
    print("\n" + "="*70)
    print("SCENARIO 1: GLOBAL with tree preservation")
    print("="*70)
    
    # Test WITHOUT flag (should fail)
    print("\n--- Test 1a: WITHOUT preserve_tree_structure (should fail) ---")
    gridmap1 = GridMap(
        nodes=[
            Node(agents=[Agent(name="agent1")]),
            Node(agents=[])
        ]
    )
    gridmap1.promote_to_root()
    
    node1 = gridmap1.nodes[0]
    node2 = gridmap1.nodes[1]
    agent = node1.agents[0]
    
    print(f"\nBEFORE:")
    print(f"  Agents in node1: {len(node1.agents)}")
    print(f"  Agents in node2: {len(node2.agents)}")
    
    result1 = CallableRegistry.execute(
        "move_agent_global",
        preserve_tree_structure=False,  # Default behavior
        gridmap=gridmap1,
        source_node=node1,
        target_node=node2,
        agent=agent
    )
    
    print(f"\nAFTER (without flag):")
    print(f"  Agents in result.nodes[0]: {len(result1.nodes[0].agents)}")
    print(f"  Agents in result.nodes[1]: {len(result1.nodes[1].agents)}")
    
    without_flag_works = (len(result1.nodes[0].agents) == 0 and len(result1.nodes[1].agents) == 1)
    if without_flag_works:
        print(f"  ⚠️  Unexpectedly worked without flag!")
    else:
        print(f"  ✓ Expected failure: gridmap has original nodes (not modified)")
    
    # Test WITH flag (should pass)
    print("\n--- Test 1b: WITH preserve_tree_structure (should pass) ---")
    gridmap2 = GridMap(
        nodes=[
            Node(agents=[Agent(name="agent2")]),
            Node(agents=[])
        ]
    )
    gridmap2.promote_to_root()
    
    node3 = gridmap2.nodes[0]
    node4 = gridmap2.nodes[1]
    agent2 = node3.agents[0]
    
    print(f"\nBEFORE:")
    print(f"  Agents in node3: {len(node3.agents)}")
    print(f"  Agents in node4: {len(node4.agents)}")
    
    result2 = CallableRegistry.execute(
        "move_agent_global",
        preserve_tree_structure=True,  # Enable tree preservation
        gridmap=gridmap2,
        source_node=node3,
        target_node=node4,
        agent=agent2
    )
    
    print(f"\nAFTER (with flag):")
    print(f"  Agents in result.nodes[0]: {len(result2.nodes[0].agents)}")
    print(f"  Agents in result.nodes[1]: {len(result2.nodes[1].agents)}")
    
    with_flag_works = (len(result2.nodes[0].agents) == 0 and len(result2.nodes[1].agents) == 1)
    if with_flag_works:
        print(f"  ✓ Works with flag: tree structure preserved!")
    else:
        print(f"  ❌ Failed even with flag!")
    
    # Overall result
    if not without_flag_works and with_flag_works:
        print(f"\n✅ SCENARIO 1 PASSED: Flag controls behavior correctly")
        return True
    else:
        print(f"\n❌ SCENARIO 1 FAILED:")
        if without_flag_works:
            print(f"   - Without flag worked (unexpected)")
        if not with_flag_works:
            print(f"   - With flag failed (expected to work)")
        return False


def test_scenario_2_borrowing_pattern():
    """
    SCENARIO 2: Borrowing pattern (no tree preservation)
    
    This should work fine - we're creating a NEW entity from data
    borrowed from separate entities.
    """
    print("\n" + "="*70)
    print("SCENARIO 2: Borrowing pattern (no tree preservation)")
    print("="*70)
    
    student = Student(name="Alice", gpa=3.8)
    student.promote_to_root()
    
    course = Course(name="Math", credits=3)
    course.promote_to_root()
    
    print(f"\nBEFORE:")
    print(f"  Student: {student.name} (root: {student.is_root_entity()})")
    print(f"  Course: {course.name} (root: {course.is_root_entity()})")
    
    # Execute
    report = CallableRegistry.execute(
        "create_report",
        student=student,
        course=course
    )
    
    print(f"\nAFTER:")
    print(f"  Report: {report.summary}")
    print(f"  Report is root: {report.is_root_entity()}")
    
    if report.student_name == "Alice" and report.course_name == "Math":
        print(f"\n✅ SCENARIO 2 PASSED: Borrowing pattern works")
        return True
    else:
        print(f"\n❌ SCENARIO 2 FAILED: Report data incorrect")
        return False


def test_scenario_3_local_with_reattachment():
    """
    SCENARIO 3: LOCAL with auto-reattachment
    
    Problem: When we pass only nodes (no root), they get copied independently.
    The parent gridmap should be versioned and nodes reattached, but currently
    the parent is not involved at all.
    """
    print("\n" + "="*70)
    print("SCENARIO 3: LOCAL with auto-reattachment")
    print("="*70)
    
    # Create gridmap with nodes
    gridmap = GridMap(
        nodes=[
            Node(agents=[Agent(name="agent1")]),
            Node(agents=[])
        ]
    )
    gridmap.promote_to_root()
    
    node1 = gridmap.nodes[0]
    node2 = gridmap.nodes[1]
    agent = node1.agents[0]
    
    original_gridmap_id = gridmap.ecs_id
    
    print(f"\nBEFORE:")
    print(f"  GridMap ID: {gridmap.ecs_id}")
    print(f"  Agents in node1: {len(node1.agents)}")
    print(f"  Agents in node2: {len(node2.agents)}")
    print(f"  node1.root_ecs_id: {node1.root_ecs_id}")
    
    # Execute (no gridmap passed, but with tree preservation!)
    result = CallableRegistry.execute(
        "move_agent_local",
        preserve_tree_structure=True,  # Auto-fetch parent tree and version it
        source_node=node1,
        target_node=node2,
        agent=agent
    )
    
    print(f"\nAFTER:")
    print(f"  Result type: {type(result)}")
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        result_node1, result_node2 = result[0], result[1]
        print(f"  Agents in result_node1: {len(result_node1.agents)}")
        print(f"  Agents in result_node2: {len(result_node2.agents)}")
        print(f"  result_node1.is_root: {result_node1.is_root_entity()}")
        print(f"  result_node2.is_root: {result_node2.is_root_entity()}")
    
    # Check if parent gridmap was versioned
    # The output nodes should point to a NEW version of the gridmap
    from abstractions.ecs.entity import EntityRegistry
    
    print(f"\n  Checking versioning:")
    print(f"    Original gridmap.ecs_id: {original_gridmap_id}")
    print(f"    Original gridmap.lineage_id: {gridmap.lineage_id}")
    
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        result_node1 = result[0]
        print(f"    Output node1.root_ecs_id: {result_node1.root_ecs_id}")
        print(f"    Output node1.lineage_id: {result_node1.lineage_id}")
        
        # Fetch the gridmap that the output nodes belong to
        if result_node1.root_ecs_id:
            stored_gridmap = EntityRegistry.get_stored_entity(result_node1.root_ecs_id, result_node1.root_ecs_id)
            
            print(f"\n  Stored GridMap (from output node's root_ecs_id):")
            if stored_gridmap:
                print(f"    GridMap ecs_id: {stored_gridmap.ecs_id}")
                print(f"    GridMap lineage_id: {stored_gridmap.lineage_id}")
                print(f"    Agents in nodes[0]: {len(stored_gridmap.nodes[0].agents)}")
                print(f"    Agents in nodes[1]: {len(stored_gridmap.nodes[1].agents)}")
                
                # Check if it's a NEW version (different ecs_id, same lineage)
                is_new_version = (stored_gridmap.ecs_id != original_gridmap_id and 
                                 stored_gridmap.lineage_id == gridmap.lineage_id)
                gridmap_was_versioned = (len(stored_gridmap.nodes[0].agents) == 0 and 
                                        len(stored_gridmap.nodes[1].agents) == 1 and
                                        is_new_version)
                
                if is_new_version:
                    print(f"    ✓ New version created (ecs_id changed, lineage preserved)")
                else:
                    print(f"    ✗ Not a new version")
            else:
                print(f"    GridMap not found in storage!")
                gridmap_was_versioned = False
        else:
            print(f"    Output node has no root_ecs_id!")
            gridmap_was_versioned = False
    else:
        gridmap_was_versioned = False
    
    # Expected: Parent gridmap should be versioned and have modified nodes
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        result_node1, result_node2 = result[0], result[1]
        nodes_modified = (len(result_node1.agents) == 0 and len(result_node2.agents) == 1)
        
        if nodes_modified and gridmap_was_versioned:
            print(f"\n✅ SCENARIO 3 PASSED: Local with reattachment works")
            print(f"   - Nodes modified correctly")
            print(f"   - Parent gridmap versioned in storage")
            return True
    
    print(f"\n❌ SCENARIO 3 FAILED:")
    if not gridmap_was_versioned:
        print(f"   Problem: Parent gridmap not versioned in storage")
    print(f"   Expected: Parent gridmap versioned, nodes reattached")
    return False


def test_scenario_3b_cross_tree_movement():
    """
    SCENARIO 3B: Moving entities between DIFFERENT trees
    
    Test moving an agent from one gridmap to another gridmap.
    Both WITH and WITHOUT preserve_tree_structure flag.
    """
    print("\n" + "="*70)
    print("SCENARIO 3B: Cross-tree movement")
    print("="*70)
    
    # Create two separate gridmaps
    gridmap_A = GridMap(
        nodes=[
            Node(agents=[Agent(name="agent1")]),
            Node(agents=[])
        ]
    )
    gridmap_A.promote_to_root()
    
    gridmap_B = GridMap(
        nodes=[
            Node(agents=[]),
            Node(agents=[])
        ]
    )
    gridmap_B.promote_to_root()
    
    node_A1 = gridmap_A.nodes[0]
    node_B1 = gridmap_B.nodes[0]
    agent = node_A1.agents[0]
    
    original_gridmap_A_id = gridmap_A.ecs_id
    original_gridmap_B_id = gridmap_B.ecs_id
    
    print(f"\nBEFORE:")
    print(f"  GridMap A ID: {gridmap_A.ecs_id}")
    print(f"  GridMap B ID: {gridmap_B.ecs_id}")
    print(f"  Agents in gridmap_A.nodes[0]: {len(node_A1.agents)}")
    print(f"  Agents in gridmap_B.nodes[0]: {len(node_B1.agents)}")
    print(f"  node_A1.root_ecs_id: {node_A1.root_ecs_id}")
    print(f"  node_B1.root_ecs_id: {node_B1.root_ecs_id}")
    
    # Test WITHOUT flag first
    print(f"\n--- Test WITHOUT preserve_tree_structure ---")
    result1 = CallableRegistry.execute(
        "move_agent_local",
        preserve_tree_structure=False,
        source_node=node_A1,
        target_node=node_B1,
        agent=agent
    )
    
    print(f"  Result: {len(result1[0].agents)} agents in source, {len(result1[1].agents)} in target")
    print(f"  Source root_ecs_id: {result1[0].root_ecs_id}")
    print(f"  Target root_ecs_id: {result1[1].root_ecs_id}")
    
    # Reset for second test
    gridmap_A2 = GridMap(
        nodes=[
            Node(agents=[Agent(name="agent2")]),
            Node(agents=[])
        ]
    )
    gridmap_A2.promote_to_root()
    
    gridmap_B2 = GridMap(
        nodes=[
            Node(agents=[]),
            Node(agents=[])
        ]
    )
    gridmap_B2.promote_to_root()
    
    node_A2 = gridmap_A2.nodes[0]
    node_B2 = gridmap_B2.nodes[0]
    agent2 = node_A2.agents[0]
    
    original_gridmap_A2_id = gridmap_A2.ecs_id
    original_gridmap_B2_id = gridmap_B2.ecs_id
    
    # Test WITH flag
    print(f"\n--- Test WITH preserve_tree_structure ---")
    result2 = CallableRegistry.execute(
        "move_agent_local",
        preserve_tree_structure=True,
        source_node=node_A2,
        target_node=node_B2,
        agent=agent2
    )
    
    print(f"  Result: {len(result2[0].agents)} agents in source, {len(result2[1].agents)} in target")
    print(f"  Source root_ecs_id: {result2[0].root_ecs_id}")
    print(f"  Target root_ecs_id: {result2[1].root_ecs_id}")
    
    # Verify both trees were versioned
    from abstractions.ecs.entity import EntityRegistry
    
    # Check if gridmap_A was versioned (agent removed)
    stored_A = EntityRegistry.get_stored_entity(result2[0].root_ecs_id, result2[0].root_ecs_id)
    # Check if gridmap_B was versioned (agent added)
    stored_B = EntityRegistry.get_stored_entity(result2[1].root_ecs_id, result2[1].root_ecs_id)
    
    print(f"\n  Stored GridMap A:")
    if stored_A:
        print(f"    ecs_id: {stored_A.ecs_id} (original: {original_gridmap_A2_id})")
        print(f"    Agents in nodes[0]: {len(stored_A.nodes[0].agents)}")
        gridmap_A_versioned = (stored_A.ecs_id != original_gridmap_A2_id and 
                               len(stored_A.nodes[0].agents) == 0)
    else:
        gridmap_A_versioned = False
    
    print(f"\n  Stored GridMap B:")
    if stored_B:
        print(f"    ecs_id: {stored_B.ecs_id} (original: {original_gridmap_B2_id})")
        print(f"    Agents in nodes[0]: {len(stored_B.nodes[0].agents)}")
        gridmap_B_versioned = (stored_B.ecs_id != original_gridmap_B2_id and 
                               len(stored_B.nodes[0].agents) == 1)
    else:
        gridmap_B_versioned = False
    
    # Check results
    nodes_modified = (len(result2[0].agents) == 0 and len(result2[1].agents) == 1)
    both_trees_versioned = gridmap_A_versioned and gridmap_B_versioned
    different_roots = result2[0].root_ecs_id != result2[1].root_ecs_id
    
    if nodes_modified and both_trees_versioned and different_roots:
        print(f"\n✅ SCENARIO 3B PASSED: Cross-tree movement works")
        print(f"   - Nodes modified correctly")
        print(f"   - Both source and target trees versioned")
        print(f"   - Nodes belong to different trees")
        return True
    else:
        print(f"\n❌ SCENARIO 3B FAILED:")
        if not nodes_modified:
            print(f"   - Nodes not modified correctly")
        if not gridmap_A_versioned:
            print(f"   - Source tree (A) not versioned")
        if not gridmap_B_versioned:
            print(f"   - Target tree (B) not versioned")
        if not different_roots:
            print(f"   - Nodes should belong to different trees")
        return False


def test_scenario_4_pure_local():
    """
    SCENARIO 4: Pure local processing (orphan entities)
    
    This should work - processing entities that have no parent.
    """
    print("\n" + "="*70)
    print("SCENARIO 4: Pure local processing (orphan entities)")
    print("="*70)
    
    # Create orphan nodes (no parent)
    node1 = Node(agents=[Agent(name="agent1")])
    node2 = Node(agents=[])
    
    node1.promote_to_root()
    node2.promote_to_root()
    
    print(f"\nBEFORE:")
    print(f"  node1.is_root: {node1.is_root_entity()}")
    print(f"  node2.is_root: {node2.is_root_entity()}")
    print(f"  node1.root_ecs_id: {node1.root_ecs_id}")
    
    # Execute
    result = CallableRegistry.execute(
        "process_nodes_orphan",
        node1=node1,
        node2=node2
    )
    
    print(f"\nAFTER:")
    print(f"  Result type: {type(result)}")
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        result_node1, result_node2 = result[0], result[1]
        print(f"  result_node1.is_root: {result_node1.is_root_entity()}")
        print(f"  result_node2.is_root: {result_node2.is_root_entity()}")
    
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        print(f"\n✅ SCENARIO 4 PASSED: Pure local processing works")
        return True
    else:
        print(f"\n❌ SCENARIO 4 FAILED: Result format incorrect")
        return False


def main():
    print("\n" + "="*70)
    print("TREE STRUCTURE PRESERVATION TEST SUITE")
    print("="*70)
    print("\nThis test identifies problems with tree structure preservation")
    print("in transactional execution across 5 scenarios.")
    
    results = []
    
    # Run all scenarios
    results.append(("Scenario 1: Global with tree preservation", test_scenario_1_global_with_tree_preservation()))
    results.append(("Scenario 2: Borrowing pattern", test_scenario_2_borrowing_pattern()))
    results.append(("Scenario 3: Local with reattachment", test_scenario_3_local_with_reattachment()))
    results.append(("Scenario 3B: Cross-tree movement", test_scenario_3b_cross_tree_movement()))
    results.append(("Scenario 4: Pure local processing", test_scenario_4_pure_local()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All scenarios working correctly!")
    else:
        print(f"\n⚠️  {total - passed} scenario(s) need fixing")
        print("\nProblems identified:")
        if not results[0][1]:
            print("  - SCENARIO 1: Need tree-aware copying for same-tree entities")
        if not results[2][1]:
            print("  - SCENARIO 3: Need auto-reattachment for local functions")


if __name__ == "__main__":
    main()
