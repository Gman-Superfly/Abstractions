"""
Test output filtering and selective tree versioning.

These tests verify that:
1. Trees are only versioned if they have outputs (root or sub-entities)
2. Within-tree modifications are visible in non-returned entities
3. Cross-tree movement only versions the target tree if source has no outputs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Tuple
from abstractions.ecs.entity import Entity, EntityRegistry
from abstractions.ecs.callable_registry import CallableRegistry
from pydantic import Field


# Test models
class Agent(Entity):
    name: str


class Node(Entity):
    agents: List[Agent] = Field(default_factory=list)


class GridMap(Entity):
    nodes: List[Node] = Field(default_factory=list)


# Test functions
@CallableRegistry.register
def move_agent_return_target_only(source_node: Node, target_node: Node, agent: Agent) -> Node:
    """
    Move agent from source to target, but ONLY return target node.
    
    This tests whether the source tree gets versioned when it has no outputs.
    """
    source_node.agents.remove(agent)
    target_node.agents.append(agent)
    return target_node  # Only return target!


@CallableRegistry.register
def move_agent_within_tree_return_one(gridmap: GridMap, source_idx: int, target_idx: int, agent: Agent) -> Node:
    """
    Move agent within same tree, only return one node.
    
    This tests whether modifications are visible in non-returned nodes.
    """
    source_node = gridmap.nodes[source_idx]
    target_node = gridmap.nodes[target_idx]
    
    source_node.agents.remove(agent)
    target_node.agents.append(agent)
    
    return target_node  # Only return target node


def test_cross_tree_output_filtering():
    """
    Test that source tree is NOT versioned when only target is returned.
    
    Setup:
    - Move agent from gridmap_A.node to gridmap_B.node
    - Only return target node (from gridmap_B)
    
    Expected:
    - gridmap_B is versioned (has output)
    - gridmap_A is NOT versioned (no outputs, should remain at original version)
    """
    print("\n" + "="*70)
    print("TEST: Cross-tree output filtering")
    print("="*70)
    
    # Create two gridmaps
    gridmap_A = GridMap(
        nodes=[Node(agents=[Agent(name="agent1")])]
    )
    gridmap_A.promote_to_root()
    
    gridmap_B = GridMap(
        nodes=[Node(agents=[])]
    )
    gridmap_B.promote_to_root()
    
    node_A = gridmap_A.nodes[0]
    node_B = gridmap_B.nodes[0]
    agent = node_A.agents[0]
    
    original_gridmap_A_id = gridmap_A.ecs_id
    original_gridmap_A_lineage = gridmap_A.lineage_id
    original_gridmap_B_id = gridmap_B.ecs_id
    original_gridmap_B_lineage = gridmap_B.lineage_id
    
    print(f"\nBEFORE:")
    print(f"  GridMap A: {original_gridmap_A_id}")
    print(f"  GridMap B: {original_gridmap_B_id}")
    print(f"  Agents in A.nodes[0]: {len(node_A.agents)}")
    print(f"  Agents in B.nodes[0]: {len(node_B.agents)}")
    
    # Execute - only returns target node
    result = CallableRegistry.execute(
        "move_agent_return_target_only",
        preserve_tree_structure=True,
        source_node=node_A,
        target_node=node_B,
        agent=agent
    )
    
    print(f"\nAFTER:")
    print(f"  Result type: {type(result).__name__}")
    print(f"  Result.root_ecs_id: {result.root_ecs_id}")
    print(f"  Agents in result: {len(result.agents)}")
    
    # Check gridmap_A (source) - should NOT be versioned
    latest_A = EntityRegistry.get_stored_entity(original_gridmap_A_id, original_gridmap_A_id)
    
    print(f"\n  GridMap A (source - no outputs):")
    if latest_A:
        print(f"    ecs_id: {latest_A.ecs_id}")
        print(f"    Same as original: {latest_A.ecs_id == original_gridmap_A_id}")
        print(f"    Agents in nodes[0]: {len(latest_A.nodes[0].agents)}")
        
        # Should still be at original version (NOT versioned)
        gridmap_A_not_versioned = (latest_A.ecs_id == original_gridmap_A_id)
    else:
        print(f"    Not found!")
        gridmap_A_not_versioned = False
    
    # Check gridmap_B (target) - should BE versioned
    stored_B = EntityRegistry.get_stored_entity(result.root_ecs_id, result.root_ecs_id)
    
    print(f"\n  GridMap B (target - has output):")
    if stored_B:
        print(f"    ecs_id: {stored_B.ecs_id}")
        print(f"    Different from original: {stored_B.ecs_id != original_gridmap_B_id}")
        print(f"    Agents in nodes[0]: {len(stored_B.nodes[0].agents)}")
        
        gridmap_B_versioned = (stored_B.ecs_id != original_gridmap_B_id and
                               len(stored_B.nodes[0].agents) == 1)
    else:
        print(f"    Not found!")
        gridmap_B_versioned = False
    
    # Verify
    if gridmap_A_not_versioned and gridmap_B_versioned:
        print(f"\n✅ TEST PASSED: Output filtering works")
        print(f"   - Source tree NOT versioned (no outputs)")
        print(f"   - Target tree versioned (has output)")
        return True
    else:
        print(f"\n❌ TEST FAILED:")
        if not gridmap_A_not_versioned:
            print(f"   - Source tree was versioned (should NOT be)")
        if not gridmap_B_versioned:
            print(f"   - Target tree not versioned (should be)")
        return False


def test_within_tree_partial_output():
    """
    Test that within-tree modifications are visible in non-returned entities.
    
    Setup:
    - Move agent within same gridmap
    - Only return target node
    
    Expected:
    - Entire tree is versioned
    - Modifications visible in both returned and non-returned nodes
    - Source node (not returned) shows agent removed
    """
    print("\n" + "="*70)
    print("TEST: Within-tree partial output")
    print("="*70)
    
    # Create gridmap with two nodes
    gridmap = GridMap(
        nodes=[
            Node(agents=[Agent(name="agent1")]),
            Node(agents=[])
        ]
    )
    gridmap.promote_to_root()
    
    original_gridmap_id = gridmap.ecs_id
    
    print(f"\nBEFORE:")
    print(f"  GridMap: {original_gridmap_id}")
    print(f"  Agents in nodes[0]: {len(gridmap.nodes[0].agents)}")
    print(f"  Agents in nodes[1]: {len(gridmap.nodes[1].agents)}")
    
    # Execute - only returns target node (nodes[1])
    result = CallableRegistry.execute(
        "move_agent_within_tree_return_one",
        preserve_tree_structure=True,
        gridmap=gridmap,
        source_idx=0,
        target_idx=1,
        agent=gridmap.nodes[0].agents[0]
    )
    
    print(f"\nAFTER:")
    print(f"  Result type: {type(result).__name__}")
    print(f"  Result.root_ecs_id: {result.root_ecs_id}")
    print(f"  Agents in result: {len(result.agents)}")
    
    # Fetch the versioned gridmap
    stored_gridmap = EntityRegistry.get_stored_entity(result.root_ecs_id, result.root_ecs_id)
    
    print(f"\n  Stored GridMap:")
    if stored_gridmap:
        print(f"    ecs_id: {stored_gridmap.ecs_id}")
        print(f"    Different from original: {stored_gridmap.ecs_id != original_gridmap_id}")
        print(f"    Agents in nodes[0] (source, not returned): {len(stored_gridmap.nodes[0].agents)}")
        print(f"    Agents in nodes[1] (target, returned): {len(stored_gridmap.nodes[1].agents)}")
        
        # Both nodes should show modifications
        tree_versioned = (stored_gridmap.ecs_id != original_gridmap_id)
        source_modified = (len(stored_gridmap.nodes[0].agents) == 0)
        target_modified = (len(stored_gridmap.nodes[1].agents) == 1)
        
        all_correct = tree_versioned and source_modified and target_modified
    else:
        print(f"    Not found!")
        all_correct = False
    
    # Verify
    if all_correct:
        print(f"\n✅ TEST PASSED: Within-tree modifications visible")
        print(f"   - Tree versioned")
        print(f"   - Source node modified (not returned)")
        print(f"   - Target node modified (returned)")
        return True
    else:
        print(f"\n❌ TEST FAILED:")
        if not tree_versioned:
            print(f"   - Tree not versioned")
        if not source_modified:
            print(f"   - Source node not modified")
        if not target_modified:
            print(f"   - Target node not modified")
        return False


def main():
    print("\n" + "="*70)
    print("OUTPUT FILTERING TEST SUITE")
    print("="*70)
    print("\nTests for selective tree versioning based on outputs.")
    
    results = []
    
    # Run tests
    results.append(("Cross-tree output filtering", test_cross_tree_output_filtering()))
    results.append(("Within-tree partial output", test_within_tree_partial_output()))
    
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
        print("\n🎉 All tests passing!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failing")


if __name__ == "__main__":
    main()
