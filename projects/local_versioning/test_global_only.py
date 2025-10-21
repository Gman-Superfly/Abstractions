#!/usr/bin/env python3
"""
SIMPLE TEST: Just test GLOBAL function to see what's happening with versioning.
"""

import sys
from pathlib import Path
from typing import List, Tuple
from pydantic import Field

# Add abstractions to path
abstractions_path = Path(__file__).parent.parent.parent / "abstractions"
sys.path.insert(0, str(abstractions_path))

from abstractions.ecs.entity import Entity, EntityRegistry
from abstractions.ecs.callable_registry import CallableRegistry


# Define entities
class Agent(Entity):
    name: str = ""
    speed: int = 1


class Node(Entity):
    position: Tuple[int, int] = (0, 0)
    agents: List[Agent] = Field(default_factory=list)
    
    def add_agent(self, agent: Agent):
        self.agents.append(agent)
    
    def remove_agent(self, agent: Agent):
        if agent in self.agents:
            self.agents.remove(agent)


class GridMap(Entity):
    nodes: List[Node] = Field(default_factory=list)


# Register GLOBAL function
@CallableRegistry.register("move_agent_global")
def move_agent_global(
    gridmap: GridMap,
    source_node: Node,
    target_node: Node,
    agent: Agent
) -> GridMap:
    """GLOBAL: gridmap + nodes + agent -> gridmap"""
    source_node.remove_agent(agent)
    target_node.add_agent(agent)
    return gridmap


def test_global():
    print("\n" + "="*70)
    print("TEST: GLOBAL function versioning")
    print("="*70 + "\n")
    
    # Create scenario
    gridmap = GridMap()
    node1 = Node(position=(0, 0))
    node2 = Node(position=(1, 0))
    agent = Agent(name="test", speed=5)
    
    node1.add_agent(agent)
    gridmap.nodes.append(node1)
    gridmap.nodes.append(node2)
    
    print("BEFORE promote_to_root():")
    print(f"  gridmap.ecs_id: {gridmap.ecs_id}")
    print(f"  gridmap.root_ecs_id: {gridmap.root_ecs_id}")
    print(f"  node1.ecs_id: {node1.ecs_id}")
    print(f"  node1.root_ecs_id: {node1.root_ecs_id}")
    print(f"  node2.ecs_id: {node2.ecs_id}")
    print(f"  node2.root_ecs_id: {node2.root_ecs_id}")
    
    # Promote to root
    gridmap.promote_to_root()
    
    print("\nAFTER promote_to_root():")
    print(f"  gridmap.ecs_id: {gridmap.ecs_id}")
    print(f"  gridmap.root_ecs_id: {gridmap.root_ecs_id}")
    print(f"  gridmap.is_root_entity(): {gridmap.is_root_entity()}")
    print(f"  node1.ecs_id: {node1.ecs_id}")
    print(f"  node1.root_ecs_id: {node1.root_ecs_id}")
    print(f"  node1.is_root_entity(): {node1.is_root_entity()}")
    print(f"  node2.ecs_id: {node2.ecs_id}")
    print(f"  node2.root_ecs_id: {node2.root_ecs_id}")
    print(f"  node2.is_root_entity(): {node2.is_root_entity()}")
    
    # Check tree
    tree = gridmap.get_tree()
    if tree:
        print(f"\nTree info:")
        print(f"  root_ecs_id: {tree.root_ecs_id}")
        print(f"  node_count: {tree.node_count}")
        print(f"  Nodes in tree: {list(tree.nodes.keys())}")
    
    # Store original IDs
    original_gridmap_id = gridmap.ecs_id
    original_node1_id = node1.ecs_id
    original_node2_id = node2.ecs_id
    original_agent_id = agent.ecs_id
    
    print(f"\n{'='*70}")
    print("Calling GLOBAL function...")
    print("="*70)
    print(f"Agent in node1 BEFORE: {len(node1.agents)}")
    print(f"Agent in node2 BEFORE: {len(node2.agents)}")
    
    # Execute
    result = CallableRegistry.execute(
        "move_agent_global",
        gridmap=gridmap,
        source_node=node1,
        target_node=node2,
        agent=agent
    )
    
    print(f"\n{'='*70}")
    print("AFTER GLOBAL function:")
    print("="*70)
    print(f"Result type: {type(result)}")
    print(f"Result is gridmap: {result is gridmap}")
    print(f"Result == gridmap: {result == gridmap}")
    
    print(f"\nOBJECT IDENTITY CHECK:")
    print(f"  result.nodes[0] is node1: {result.nodes[0] is node1}")
    print(f"  result.nodes[1] is node2: {result.nodes[1] is node2}")
    print(f"  id(result.nodes[0]): {id(result.nodes[0])}")
    print(f"  id(node1): {id(node1)}")
    print(f"  id(result.nodes[1]): {id(result.nodes[1])}")
    print(f"  id(node2): {id(node2)}")
    
    print(f"\nID CHANGES:")
    print(f"  gridmap.ecs_id: {result.ecs_id}")
    print(f"    CHANGED? {result.ecs_id != original_gridmap_id}")
    print(f"    OLD: {original_gridmap_id}")
    print(f"    NEW: {result.ecs_id}")
    
    print(f"\n  node1.ecs_id: {result.nodes[0].ecs_id}")
    print(f"    CHANGED? {result.nodes[0].ecs_id != original_node1_id}")
    print(f"    OLD: {original_node1_id}")
    print(f"    NEW: {result.nodes[0].ecs_id}")
    
    print(f"\n  node2.ecs_id: {result.nodes[1].ecs_id}")
    print(f"    CHANGED? {result.nodes[1].ecs_id != original_node2_id}")
    print(f"    OLD: {original_node2_id}")
    print(f"    NEW: {result.nodes[1].ecs_id}")
    
    print(f"\n  agent.ecs_id: {result.nodes[1].agents[0].ecs_id if result.nodes[1].agents else 'NO AGENT'}")
    if result.nodes[1].agents:
        print(f"    CHANGED? {result.nodes[1].agents[0].ecs_id != original_agent_id}")
        print(f"    OLD: {original_agent_id}")
        print(f"    NEW: {result.nodes[1].agents[0].ecs_id}")
    
    print(f"\nFUNCTIONAL RESULT:")
    print(f"  Agent in node1: {len(result.nodes[0].agents)}")
    print(f"  Agent in node2: {len(result.nodes[1].agents)}")
    
    print(f"\n{'='*70}")
    print("CONCLUSION:")
    print("="*70)
    if result.ecs_id != original_gridmap_id:
        print("✅ GridMap WAS versioned (new ecs_id)")
    else:
        print("❌ GridMap was NOT versioned (same ecs_id)")
    
    if result.nodes[0].ecs_id != original_node1_id or result.nodes[1].ecs_id != original_node2_id:
        print("✅ Nodes WERE versioned")
    else:
        print("❌ Nodes were NOT versioned")
    
    print("\n")


if __name__ == "__main__":
    test_global()
