#!/usr/bin/env python3
"""
Isolated test to demonstrate Pydantic model_copy(deep=True) slowdown problem.

This script shows that Pydantic's deep copy gets slower as we add MORE entities
to a registry, EVEN THOUGH we're copying the SAME entity each time.

Expected behavior: Copy time should be constant (same entity size)
Actual behavior: Copy time grows linearly with registry size
Root cause: Python's deepcopy() scans entire object graph to detect cycles
"""

import sys
import time
from pathlib import Path
from typing import List, Dict
from uuid import UUID, uuid4

# Add abstractions to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from abstractions.ecs.entity import Entity, EntityRegistry
from pydantic import Field


# Match the ACTUAL GridMap structure
class Agent(Entity):
    """Agent entity matching the real test"""
    name: str = "agent"
    speed: float = 1.0
    position_x: int = 0
    position_y: int = 0


class Node(Entity):
    """Node entity matching the real test"""
    name: str = "node"
    capacity: int = 100
    agents: List[Agent] = Field(default_factory=list)


class GridMap(Entity):
    """GridMap entity matching the real test"""
    name: str = "gridmap"
    width: int = 100
    height: int = 100
    nodes: List[Node] = Field(default_factory=list)


def create_gridmap_like_entity(num_nodes: int, agents_per_node: int) -> GridMap:
    """Create a GridMap with the EXACT structure from the test"""
    gridmap = GridMap(name="test_gridmap", width=num_nodes * 10, height=num_nodes * 10)
    
    for i in range(num_nodes):
        node = Node(name=f"node_{i}", capacity=agents_per_node * 2)
        
        for j in range(agents_per_node):
            agent = Agent(
                name=f"agent_{i}_{j}",
                speed=1.0 + (i + j) * 0.1,
                position_x=i * 10,
                position_y=j * 10
            )
            node.agents.append(agent)
        
        gridmap.nodes.append(node)
    
    return gridmap


def populate_registry(num_trees: int):
    """Add simple entity trees to registry"""
    for i in range(num_trees):
        dummy = Entity()
        dummy.root_ecs_id = dummy.ecs_id
        
        from abstractions.ecs.entity import EntityTree
        
        tree = EntityTree(
            root_ecs_id=dummy.ecs_id,
            lineage_id=dummy.lineage_id,
            nodes={dummy.ecs_id: dummy},
            edges={},
            outgoing_edges={},
            incoming_edges={},
            ancestry_paths={dummy.ecs_id: [dummy.ecs_id]}
        )
        
        EntityRegistry.register_entity_tree(tree)


def test_copy_time_with_registry_growth():
    """Test that shows copy time grows with registry size"""
    print("\n" + "="*80)
    print("DEEP COPY TIMING TEST - Same entity, growing registry")
    print("="*80)
    
    # Reset profiling stats to avoid spam
    EntityRegistry.reset_profiling_stats()
    
    # Create ONE entity that we'll copy repeatedly - MATCH THE 20x20 TEST
    target_entity = create_gridmap_like_entity(num_nodes=20, agents_per_node=20)
    total_entities = 1 + len(target_entity.nodes) + sum(len(node.agents) for node in target_entity.nodes)
    print(f"Target entity: GridMap with {len(target_entity.nodes)} nodes, {total_entities} total entities")
    
    # Test at LARGE scale to see the effect
    registry_sizes = [0, 100, 500, 1000, 5000, 10000, 50000, 100000]
    results = []
    
    for registry_size in registry_sizes:
        # Clear registry and repopulate to target size
        EntityRegistry.tree_registry.clear()
        EntityRegistry.lineage_registry.clear()
        EntityRegistry.live_id_registry.clear()
        EntityRegistry.ecs_id_to_root_id.clear()
        EntityRegistry.reset_profiling_stats()
        
        if registry_size > 0:
            print(f"  Populating registry with {registry_size} entities...", end='', flush=True)
            populate_registry(registry_size)
            print(f" done. Registry size: {len(EntityRegistry.tree_registry)}")
        
        # Warm up
        for _ in range(3):
            _ = target_entity.model_copy(deep=True)
        
        # Run many copies to get stable average
        num_copies = 100
        times = []
        print(f"  Running {num_copies} deep copies...", end='', flush=True)
        for i in range(num_copies):
            start = time.perf_counter()
            copy = target_entity.model_copy(deep=True)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        print(" done.")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        results.append({
            'registry_size': registry_size,
            'avg_ms': avg_time,
            'min_ms': min_time,
            'max_ms': max_time
        })
        
        actual_size = len(EntityRegistry.tree_registry)
        print(f"  Result: avg={avg_time:.3f}ms, min={min_time:.3f}ms, max={max_time:.3f}ms\n")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY - Deep copy time vs Registry size")
    print("="*80)
    print(f"{'Registry Size':<15} {'Avg Copy Time':<15} {'Growth Factor':<15}")
    print("-"*80)
    
    baseline = results[0]['avg_ms']
    for r in results:
        growth = r['avg_ms'] / baseline if baseline > 0 else 1.0
        print(f"{r['registry_size']:<15} {r['avg_ms']:<15.3f} {growth:<15.2f}x")
    
    print("="*80)
    print("\n📊 OBSERVATIONS:")
    print(f"   Copy time: {results[0]['avg_ms']:.2f}ms → {results[-1]['avg_ms']:.2f}ms ({results[-1]['avg_ms']/results[0]['avg_ms']:.2f}x)")
    print(f"   Registry: {results[0]['registry_size']} → {results[-1]['registry_size']} trees")
    print(f"   Entity being copied: SAME GridMap (20 nodes, {total_entities} entities) every time")
    print(f"\n   Real test showed: 75ms → 694ms (9.2x) as registry went 11→101")
    print(f"   This test shows: {results[0]['avg_ms']:.1f}ms → {[r for r in results if r['registry_size']==100][0]['avg_ms']:.1f}ms at registry=100")
    print(f"\n   ⚠️  Isolated test does NOT fully reproduce the 9.2x slowdown")
    print(f"   Something else in the real scenario is making it worse")
    print("="*80 + "\n")


def test_shallow_vs_deep():
    """Compare shallow vs deep copy performance"""
    print("\n" + "="*80)
    print("SHALLOW vs DEEP COPY COMPARISON")
    print("="*80)
    
    # Create test entity - MATCH THE 20x20 TEST
    target = create_gridmap_like_entity(num_nodes=20, agents_per_node=20)
    
    # Populate large registry
    EntityRegistry.tree_registry.clear()
    EntityRegistry.lineage_registry.clear()
    EntityRegistry.live_id_registry.clear()
    EntityRegistry.ecs_id_to_root_id.clear()
    EntityRegistry.reset_profiling_stats()
    
    test_size = 10000
    print(f"Populating registry with {test_size} entities...")
    populate_registry(test_size)
    print(f"✓ Registry populated with {len(EntityRegistry.tree_registry)} entities")
    
    # Time shallow copy
    shallow_times = []
    for _ in range(100):
        start = time.perf_counter()
        copy = target.model_copy(deep=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        shallow_times.append(elapsed_ms)
    
    # Time deep copy
    deep_times = []
    for _ in range(100):
        start = time.perf_counter()
        copy = target.model_copy(deep=True)
        elapsed_ms = (time.perf_counter() - start) * 1000
        deep_times.append(elapsed_ms)
    
    shallow_avg = sum(shallow_times)/len(shallow_times)
    deep_avg = sum(deep_times)/len(deep_times)
    speedup = deep_avg / shallow_avg if shallow_avg > 0 else 0
    
    print(f"\nWith registry_size={test_size}:")
    print(f"  Shallow copy: {shallow_avg:.4f}ms avg")
    print(f"  Deep copy:    {deep_avg:.3f}ms avg")
    print(f"  Speedup:      {speedup:.0f}x faster!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PYDANTIC DEEP COPY PERFORMANCE INVESTIGATION")
    print("="*80 + "\n")
    
    # Run the main test
    test_copy_time_with_registry_growth()
    
    # Compare shallow vs deep
    test_shallow_vs_deep()
