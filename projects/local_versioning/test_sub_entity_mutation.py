#!/usr/bin/env python3
"""
Test: What happens when we pass sub-entities to CallableRegistry?

This test explores whether modifying a sub-entity triggers parent versioning.
"""

import sys
from pathlib import Path

# Add abstractions to path
abstractions_path = Path(__file__).parent.parent.parent / "abstractions"
sys.path.insert(0, str(abstractions_path))

from abstractions.ecs.entity import Entity, EntityRegistry
from abstractions.ecs.callable_registry import CallableRegistry
from pydantic import Field
from typing import List, Tuple


# ============================================================================
# Test Entities
# ============================================================================

class Agent(Entity):
    """An agent with a name."""
    name: str = ""
    speed: int = 1


class Node(Entity):
    """A node containing agents."""
    position: Tuple[int, int] = (0, 0)
    agents: List[Agent] = Field(default_factory=list)
    
    def add_agent(self, agent: Agent):
        self.agents.append(agent)
    
    def remove_agent(self, agent: Agent):
        if agent in self.agents:
            self.agents.remove(agent)


class GridMap(Entity):
    """A grid map containing nodes."""
    nodes: List[Node] = Field(default_factory=list)


# ============================================================================
# Test Functions
# ============================================================================

@CallableRegistry.register("move_agent_local")
def move_agent_local(
    source_node: Node,
    target_node: Node,
    agent: Agent
) -> Tuple[Node, Node]:
    """
    LOCAL mutation: Operates on two nodes directly.
    
    Question: What happens to the parent GridMap?
    - Are the nodes detached?
    - Does the parent get versioned?
    - Do we need to manually reattach?
    """
    # Remove from source
    source_node.remove_agent(agent)
    
    # Add to target
    target_node.add_agent(agent)
    
    return source_node, target_node


# ============================================================================
# Test Scenarios
# ============================================================================

def test_sub_entity_mutation():
    """Test what happens when we mutate sub-entities."""
    
    print("\n" + "="*70)
    print("TEST: Sub-Entity Mutation via CallableRegistry")
    print("="*70 + "\n")
    
    # Create scenario
    print("Step 1: Create GridMap with 2 nodes and 1 agent")
    gridmap = GridMap()
    
    node1 = Node(position=(0, 0))
    node2 = Node(position=(1, 0))
    
    agent = Agent(name="agent_1", speed=5)
    node1.add_agent(agent)
    
    gridmap.nodes.append(node1)
    gridmap.nodes.append(node2)
    
    # Register as root
    gridmap.promote_to_root()
    
    print(f"✓ GridMap created:")
    print(f"  - GridMap ID: {gridmap.ecs_id}")
    print(f"  - Node1 ID: {node1.ecs_id}, root_ecs_id: {node1.root_ecs_id}")
    print(f"  - Node2 ID: {node2.ecs_id}, root_ecs_id: {node2.root_ecs_id}")
    print(f"  - Agent ID: {agent.ecs_id}, root_ecs_id: {agent.root_ecs_id}")
    print(f"  - Node1 is_root: {node1.is_root_entity()}")
    print(f"  - Node2 is_root: {node2.is_root_entity()}")
    
    # Check tree
    tree = gridmap.get_tree()
    print(f"\n✓ Tree built:")
    print(f"  - Total entities: {tree.node_count}")
    print(f"  - Root: {tree.root_ecs_id}")
    
    # Store original IDs
    original_gridmap_id = gridmap.ecs_id
    original_node1_id = node1.ecs_id
    original_node2_id = node2.ecs_id
    
    print(f"\n{'='*70}")
    print("Step 2: Call move_agent_local with SUB-ENTITIES")
    print("="*70)
    print(f"Passing: source_node (sub-entity), target_node (sub-entity), agent")
    print(f"Node1 is_root_entity: {node1.is_root_entity()}")
    print(f"Node2 is_root_entity: {node2.is_root_entity()}")
    
    # Execute the local function (NO skip_divergence_check - let it do everything)
    result = CallableRegistry.execute(
        "move_agent_local",
        source_node=node1,
        target_node=node2,
        agent=agent
    )
    
    print(f"\n✓ Function executed, result type: {type(result)}")
    
    if isinstance(result, tuple):
        result_node1, result_node2 = result
        print(f"\nResult nodes:")
        print(f"  - Result Node1 ID: {result_node1.ecs_id}")
        print(f"  - Result Node2 ID: {result_node2.ecs_id}")
        print(f"  - Result Node1 is_root: {result_node1.is_root_entity()}")
        print(f"  - Result Node2 is_root: {result_node2.is_root_entity()}")
        print(f"  - Result Node1 root_ecs_id: {result_node1.root_ecs_id}")
        print(f"  - Result Node2 root_ecs_id: {result_node2.root_ecs_id}")
        
        # Check if IDs changed
        print(f"\nID Changes:")
        print(f"  - Node1: {original_node1_id} -> {result_node1.ecs_id} (changed: {original_node1_id != result_node1.ecs_id})")
        print(f"  - Node2: {original_node2_id} -> {result_node2.ecs_id} (changed: {original_node2_id != result_node2.ecs_id})")
    
    print(f"\n{'='*70}")
    print("Step 3: Check what happened to the PARENT GridMap")
    print("="*70)
    
    print(f"\nOriginal GridMap:")
    print(f"  - ID: {gridmap.ecs_id}")
    print(f"  - Changed: {gridmap.ecs_id != original_gridmap_id}")
    print(f"  - Node1 in gridmap.nodes: {node1 in gridmap.nodes}")
    print(f"  - Node2 in gridmap.nodes: {node2 in gridmap.nodes}")
    
    # Check if parent was versioned
    if gridmap.ecs_id != original_gridmap_id:
        print(f"\n✅ Parent GridMap WAS versioned!")
    else:
        print(f"\n❌ Parent GridMap was NOT versioned")
    
    # Check registry
    print(f"\n{'='*70}")
    print("Step 4: Check EntityRegistry")
    print("="*70)
    
    # Try to get the original gridmap from registry
    stored_tree = EntityRegistry.get_stored_tree(original_gridmap_id)
    if stored_tree:
        print(f"✓ Original GridMap tree still in registry:")
        print(f"  - Root ID: {stored_tree.root_ecs_id}")
        print(f"  - Node count: {stored_tree.node_count}")
    else:
        print(f"❌ Original GridMap tree NOT in registry")
    
    # Check if new version exists
    if gridmap.ecs_id != original_gridmap_id:
        new_tree = EntityRegistry.get_stored_tree(gridmap.ecs_id)
        if new_tree:
            print(f"✓ New GridMap tree in registry:")
            print(f"  - Root ID: {new_tree.root_ecs_id}")
            print(f"  - Node count: {new_tree.node_count}")
    
    # Check lineage
    print(f"\n{'='*70}")
    print("Step 5: Check Lineage")
    print("="*70)
    
    lineage_versions = EntityRegistry.lineage_registry.get(gridmap.lineage_id, [])
    print(f"Lineage ID: {gridmap.lineage_id}")
    print(f"Versions in lineage: {len(lineage_versions)}")
    for i, version_id in enumerate(lineage_versions):
        print(f"  Version {i}: {version_id}")
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print("="*70)
    
    if isinstance(result, tuple):
        result_node1, result_node2 = result
        
        if result_node1.is_root_entity() and result_node2.is_root_entity():
            print("❌ DETACHMENT: Nodes were DETACHED and promoted to root")
            print("   - They are no longer part of the GridMap tree")
            print("   - Parent GridMap was NOT updated")
            print("   - This is NOT what we want!")
        elif gridmap.ecs_id != original_gridmap_id:
            print("✅ PARENT VERSIONING: Parent GridMap WAS versioned")
            print("   - Nodes remain part of the tree")
            print("   - This is what we want!")
        else:
            print("⚠️  UNCLEAR: Need to investigate further")
    
    print("\n")


def test_detachment_scenario():
    """Test the detachment scenario explicitly."""
    
    print("\n" + "="*70)
    print("TEST: Explicit Detachment Scenario")
    print("="*70 + "\n")
    
    # Create scenario
    gridmap = GridMap()
    node1 = Node(position=(0, 0))
    agent = Agent(name="agent_1")
    node1.add_agent(agent)
    gridmap.nodes.append(node1)
    gridmap.promote_to_root()
    
    print(f"Initial state:")
    print(f"  - GridMap is_root: {gridmap.is_root_entity()}")
    print(f"  - Node1 is_root: {node1.is_root_entity()}")
    print(f"  - Node1.root_ecs_id: {node1.root_ecs_id}")
    
    # Now detach the node
    print(f"\nCalling node1.detach()...")
    node1.detach()
    
    print(f"\nAfter detach:")
    print(f"  - Node1 is_root: {node1.is_root_entity()}")
    print(f"  - Node1.root_ecs_id: {node1.root_ecs_id}")
    print(f"  - Node1.ecs_id: {node1.ecs_id}")
    
    # Check if parent was versioned
    print(f"\nParent GridMap:")
    print(f"  - GridMap.ecs_id: {gridmap.ecs_id}")
    print(f"  - Node1 still in gridmap.nodes: {node1 in gridmap.nodes}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    test_sub_entity_mutation()
    test_detachment_scenario()
