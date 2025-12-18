#!/usr/bin/env python3
"""
Test: Proper entity initialization - create children INSIDE parent
"""

import sys
from pathlib import Path
from typing import List, Tuple
from pydantic import Field

abstractions_path = Path(__file__).parent.parent.parent / "abstractions"
sys.path.insert(0, str(abstractions_path))

from abstractions.ecs.entity import Entity, EntityRegistry
from abstractions.ecs.callable_registry import CallableRegistry


class Agent(Entity):
    name: str = ""


class Node(Entity):
    agents: List[Agent] = Field(default_factory=list)


class GridMap(Entity):
    nodes: List[Node] = Field(default_factory=list)


@CallableRegistry.register("move_agent")
def move_agent(gridmap: GridMap, source_idx: int, target_idx: int, agent_name: str) -> GridMap:
    """Move agent using indices"""
    agent = None
    for a in gridmap.nodes[source_idx].agents:
        if a.name == agent_name:
            agent = a
            break
    
    if agent:
        gridmap.nodes[source_idx].agents.remove(agent)
        gridmap.nodes[target_idx].agents.append(agent)
    
    return gridmap


print("\n" + "="*70)
print("TEST: Proper entity initialization")
print("="*70 + "\n")

# Create GridMap with children INSIDE
print("Creating GridMap with children initialized INSIDE...")
gridmap = GridMap(
    nodes=[
        Node(agents=[Agent(name="agent1")]),
        Node(agents=[])
    ]
)

print(f"BEFORE promote_to_root():")
print(f"  gridmap.root_ecs_id: {gridmap.root_ecs_id}")
print(f"  node[0].root_ecs_id: {gridmap.nodes[0].root_ecs_id}")
print(f"  node[1].root_ecs_id: {gridmap.nodes[1].root_ecs_id}")
print(f"  agent.root_ecs_id: {gridmap.nodes[0].agents[0].root_ecs_id}")

gridmap.promote_to_root()

print(f"\nAFTER promote_to_root():")
print(f"  gridmap.root_ecs_id: {gridmap.root_ecs_id}")
print(f"  gridmap.is_root_entity(): {gridmap.is_root_entity()}")
print(f"  node[0].root_ecs_id: {gridmap.nodes[0].root_ecs_id}")
print(f"  node[1].root_ecs_id: {gridmap.nodes[1].root_ecs_id}")
print(f"  agent.root_ecs_id: {gridmap.nodes[0].agents[0].root_ecs_id}")

# Check tree
tree = gridmap.get_tree()
print(f"\nTree info:")
print(f"  root_ecs_id: {tree.root_ecs_id}")
print(f"  node_count: {tree.node_count}")
print(f"  All entities in tree: {len(tree.nodes)}")

# Store IDs
orig_gridmap_id = gridmap.ecs_id
orig_node0_id = gridmap.nodes[0].ecs_id
orig_node1_id = gridmap.nodes[1].ecs_id

print(f"\n{'='*70}")
print("Executing move_agent...")
print("="*70)

gridmap = CallableRegistry.execute(
    "move_agent",
    gridmap=gridmap,
    source_idx=0,
    target_idx=1,
    agent_name="agent1"
)

print(f"\nAFTER execution:")
print(f"  gridmap.ecs_id changed: {gridmap.ecs_id != orig_gridmap_id}")
print(f"  node[0].ecs_id changed: {gridmap.nodes[0].ecs_id != orig_node0_id}")
print(f"  node[1].ecs_id changed: {gridmap.nodes[1].ecs_id != orig_node1_id}")
print(f"  Agent in node[0]: {len(gridmap.nodes[0].agents)}")
print(f"  Agent in node[1]: {len(gridmap.nodes[1].agents)}")

print(f"\n{'='*70}")
if gridmap.ecs_id != orig_gridmap_id:
    print("✅ Versioning WORKED!")
else:
    print("❌ Versioning FAILED!")
print("="*70 + "\n")
