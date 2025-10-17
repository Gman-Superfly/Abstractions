"""
Profile the framework functions to find the real bottleneck.
"""
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abstractions.ecs.entity import build_entity_tree, find_modified_entities, EntityRegistry
from game_entities import GridMap, GridNode, Floor, Agent, Apple


# Create a simple grid
nodes = []
for y in range(5):
    for x in range(5):
        node = GridNode(position=(x, y))
        node.entities = [Floor(name=f"floor_{x}_{y}")]
        nodes.append(node)

grid = GridMap(nodes=nodes, width=5, height=5)

# Add agent
agent_node = nodes[12]  # Center
agent = Agent(name="test", speed=2, inventory=[])
agent_node.entities.append(agent)

# Add some apples
for i in range(5):
    nodes[i].entities.append(Apple(name=f"apple_{i}", nutrition=10))

grid.promote_to_root()

print("Grid created with:")
print(f"  - 25 GridNodes")
print(f"  - 25 Floor entities")
print(f"  - 1 Agent")
print(f"  - 5 Apples")
print(f"  Total: 56 entities\n")

# Profile build_entity_tree
print("Profiling build_entity_tree()...")
times = []
for i in range(100):
    start = time.perf_counter()
    tree = build_entity_tree(grid)
    duration = time.perf_counter() - start
    times.append(duration)

avg_time = sum(times) / len(times)
print(f"  Average time: {avg_time*1000:.3f}ms")
print(f"  Min: {min(times)*1000:.3f}ms, Max: {max(times)*1000:.3f}ms")
print(f"  Entities in tree: {len(tree.nodes)}")
print(f"  Edges: {len(tree.edges)}\n")

# Profile find_modified_entities (no changes)
print("Profiling find_modified_entities() with NO changes...")
tree1 = build_entity_tree(grid)
tree2 = build_entity_tree(grid)

times = []
for i in range(100):
    start = time.perf_counter()
    modified = find_modified_entities(tree1, tree2)
    duration = time.perf_counter() - start
    times.append(duration)

avg_time = sum(times) / len(times)
print(f"  Average time: {avg_time*1000:.3f}ms")
print(f"  Modified entities: {len(modified)}\n")

# Profile find_modified_entities (with changes)
print("Profiling find_modified_entities() WITH changes...")
# Modify the agent
agent.inventory.append(Apple(name="collected", nutrition=5))
tree_modified = build_entity_tree(grid)

times = []
for i in range(100):
    start = time.perf_counter()
    modified = find_modified_entities(tree_modified, tree1)
    duration = time.perf_counter() - start
    times.append(duration)

avg_time = sum(times) / len(times)
print(f"  Average time: {avg_time*1000:.3f}ms")
print(f"  Modified entities: {len(modified)}\n")

# Profile version_entity
print("Profiling EntityRegistry.version_entity()...")
grid2 = GridMap(nodes=nodes, width=5, height=5)
grid2.promote_to_root()
EntityRegistry.register_entity(grid2)

# Modify it
grid2.nodes[0].entities.append(Apple(name="new_apple", nutrition=15))

times = []
for i in range(10):  # Fewer iterations since this mutates state
    start = time.perf_counter()
    EntityRegistry.version_entity(grid2)
    duration = time.perf_counter() - start
    times.append(duration)
    # Reset for next iteration
    grid2 = GridMap(nodes=nodes, width=5, height=5)
    grid2.promote_to_root()
    grid2.nodes[0].entities.append(Apple(name="new_apple", nutrition=15))

avg_time = sum(times) / len(times)
print(f"  Average time: {avg_time*1000:.3f}ms")
print(f"  Min: {min(times)*1000:.3f}ms, Max: {max(times)*1000:.3f}ms\n")

print("="*60)
print("SUMMARY")
print("="*60)
print(f"For 56 entities:")
print(f"  build_entity_tree:        {sum([t for t in times])/len(times)*1000:.3f}ms")
print(f"  find_modified_entities:   ~{avg_time*1000:.3f}ms")
print(f"  version_entity (full):    ~{avg_time*1000:.3f}ms")
