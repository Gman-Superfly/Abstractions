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
    """
    print("\n" + "="*70)
    print("SCENARIO 1: GLOBAL with tree preservation")
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
    
    print(f"\nBEFORE:")
    print(f"  Agents in node1: {len(node1.agents)}")
    print(f"  Agents in node2: {len(node2.agents)}")
    print(f"  node1.root_ecs_id: {node1.root_ecs_id}")
    print(f"  node2.root_ecs_id: {node2.root_ecs_id}")
    
    # Execute
    result = CallableRegistry.execute(
        "move_agent_global",
        gridmap=gridmap,
        source_node=node1,
        target_node=node2,
        agent=agent
    )
    
    print(f"\nAFTER:")
    print(f"  Agents in result.nodes[0]: {len(result.nodes[0].agents)}")
    print(f"  Agents in result.nodes[1]: {len(result.nodes[1].agents)}")
    
    # Check if it worked
    expected_node1_agents = 0
    expected_node2_agents = 1
    actual_node1_agents = len(result.nodes[0].agents)
    actual_node2_agents = len(result.nodes[1].agents)
    
    if actual_node1_agents == expected_node1_agents and actual_node2_agents == expected_node2_agents:
        print(f"\n✅ SCENARIO 1 PASSED: Agent moved correctly")
        return True
    else:
        print(f"\n❌ SCENARIO 1 FAILED:")
        print(f"   Expected: node1={expected_node1_agents}, node2={expected_node2_agents}")
        print(f"   Got: node1={actual_node1_agents}, node2={actual_node2_agents}")
        print(f"   Problem: gridmap copy has original nodes, not modified copies")
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
    
    # Execute (no gridmap passed!)
    result = CallableRegistry.execute(
        "move_agent_local",
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
    
    # Check original gridmap
    print(f"\n  Original GridMap ID: {gridmap.ecs_id} (changed: {gridmap.ecs_id != original_gridmap_id})")
    print(f"  Agents in gridmap.nodes[0]: {len(gridmap.nodes[0].agents)}")
    print(f"  Agents in gridmap.nodes[1]: {len(gridmap.nodes[1].agents)}")
    
    # Expected: Parent gridmap should be versioned and have modified nodes
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        result_node1, result_node2 = result[0], result[1]
        if (len(result_node1.agents) == 0 and 
            len(result_node2.agents) == 1 and
            gridmap.ecs_id != original_gridmap_id):
            print(f"\n✅ SCENARIO 3 PASSED: Local with reattachment works")
            return True
    
    print(f"\n❌ SCENARIO 3 FAILED:")
    print(f"   Problem: Nodes processed independently, parent not versioned")
    print(f"   Expected: Parent gridmap versioned, nodes reattached")
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
    print("in transactional execution across 4 scenarios.")
    
    results = []
    
    # Run all scenarios
    results.append(("Scenario 1: Global with tree preservation", test_scenario_1_global_with_tree_preservation()))
    results.append(("Scenario 2: Borrowing pattern", test_scenario_2_borrowing_pattern()))
    results.append(("Scenario 3: Local with reattachment", test_scenario_3_local_with_reattachment()))
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
