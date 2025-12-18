#!/usr/bin/env python3
"""
Simple SELF-CONTAINED test: What happens with sub-entity mutations?
NO imports from other test scripts!
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


# Define entities HERE - self-contained
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


# Register the GLOBAL function - takes gridmap, nodes, agent
@CallableRegistry.register("move_agent_global")
def move_agent_global(
    gridmap: GridMap,
    source_node: Node,
    target_node: Node,
    agent: Agent
) -> GridMap:
    """GLOBAL mutation: Takes gridmap + nodes + agent, returns gridmap."""
    source_node.remove_agent(agent)
    target_node.add_agent(agent)
    return gridmap


# Register the LOCAL function - operates on nodes directly
@CallableRegistry.register("move_agent_local")
def move_agent_local(
    source_node: Node,
    target_node: Node,
    agent: Agent
) -> Tuple[Node, Node]:
    """Move agent between two nodes - LOCAL mutation."""
    source_node.remove_agent(agent)
    target_node.add_agent(agent)
    return source_node, target_node


def test_what_happens():
    """Simple test - just see what happens."""
    
    print("\n" + "="*70)
    print("TEST: What happens with sub-entity mutations?")
    print("="*70 + "\n")
    
    # Create a simple scenario
    print("Creating GridMap with 2 nodes...")
    gridmap = GridMap()
    
    node1 = Node(position=(0, 0))
    node2 = Node(position=(1, 0))
    
    agent = Agent(name="test_agent", speed=5)
    node1.add_agent(agent)
    
    gridmap.nodes.append(node1)
    gridmap.nodes.append(node2)
    
    # Register
    gridmap.promote_to_root()
    
    print(f"✓ Created:")
    print(f"  GridMap ID: {gridmap.ecs_id}")
    print(f"  GridMap root_ecs_id: {gridmap.root_ecs_id}")
    print(f"  Node1 ID: {node1.ecs_id}, is_root: {node1.is_root_entity()}, root_ecs_id: {node1.root_ecs_id}")
    print(f"  Node2 ID: {node2.ecs_id}, is_root: {node2.is_root_entity()}, root_ecs_id: {node2.root_ecs_id}")
    print(f"  Agent in node1: {len(node1.agents)}")
    print(f"  Agent in node2: {len(node2.agents)}")
    
    # Store original IDs
    original_gridmap_id = gridmap.ecs_id
    original_node1_id = node1.ecs_id
    original_node2_id = node2.ecs_id
    
    print(f"\n{'='*70}")
    print("TEST 1: GLOBAL function (gridmap, nodes, agent -> gridmap)")
    print("="*70)
    
    gridmap = CallableRegistry.execute(
        "move_agent_global",
        gridmap=gridmap,
        source_node=node1,
        target_node=node2,
        agent=agent
    )
    
    print(f"\n✓ After GLOBAL move:")
    print(f"  GridMap ID: {gridmap.ecs_id} (changed: {gridmap.ecs_id != original_gridmap_id})")
    print(f"  Node1 ID: {gridmap.nodes[0].ecs_id} (changed: {gridmap.nodes[0].ecs_id != original_node1_id})")
    print(f"  Node2 ID: {gridmap.nodes[1].ecs_id} (changed: {gridmap.nodes[1].ecs_id != original_node2_id})")
    print(f"  Agent in node1: {len(gridmap.nodes[0].agents)}")
    print(f"  Agent in node2: {len(gridmap.nodes[1].agents)}")
    
    # Reset for local test
    print(f"\n{'='*70}")
    print("Resetting for LOCAL test...")
    print("="*70)
    
    gridmap2 = GridMap()
    node3 = Node(position=(0, 0))
    node4 = Node(position=(1, 0))
    agent2 = Agent(name="test_agent2", speed=5)
    node3.add_agent(agent2)
    gridmap2.nodes.append(node3)
    gridmap2.nodes.append(node4)
    gridmap2.promote_to_root()
    
    original_gridmap2_id = gridmap2.ecs_id
    original_node3_id = node3.ecs_id
    original_node4_id = node4.ecs_id
    
    print(f"\nCreated second scenario:")
    print(f"  GridMap2 ID: {gridmap2.ecs_id}")
    print(f"  Node3 ID: {node3.ecs_id}, is_root: {node3.is_root_entity()}")
    print(f"  Node4 ID: {node4.ecs_id}, is_root: {node4.is_root_entity()}")
    
    print(f"\n{'='*70}")
    print("TEST 2: LOCAL function (nodes, agent -> nodes)")
    print("="*70)
    
    result = CallableRegistry.execute(
        "move_agent_local",
        source_node=node3,
        target_node=node4,
        agent=agent2
    )
    
    print(f"\n✓ After LOCAL move:")
    print(f"  Result type: {type(result)}")
    if isinstance(result, tuple):
        result_node1, result_node2 = result
        print(f"  Result Node1 ID: {result_node1.ecs_id} (changed: {result_node1.ecs_id != original_node3_id})")
        print(f"  Result Node2 ID: {result_node2.ecs_id} (changed: {result_node2.ecs_id != original_node4_id})")
        print(f"  Result Node1 is_root: {result_node1.is_root_entity()}")
        print(f"  Result Node2 is_root: {result_node2.is_root_entity()}")
    
    print(f"\n  Original GridMap2 ID: {gridmap2.ecs_id} (changed: {gridmap2.ecs_id != original_gridmap2_id})")
    print(f"  Agent in node3: {len(gridmap2.nodes[0].agents)}")
    print(f"  Agent in node4: {len(gridmap2.nodes[1].agents)}")
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print("="*70)
    print("\nGLOBAL function: gridmap + nodes + agent -> gridmap")
    print("  - Did GridMap version? Check above")
    print("  - Did nodes version? Check above")
    print("\nLOCAL function: nodes + agent -> nodes")
    print("  - Did nodes detach? Check is_root above")
    print("  - Did parent GridMap version? Check above")
    print("\n")


if __name__ == "__main__":
    test_what_happens()
