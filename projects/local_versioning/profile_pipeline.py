#!/usr/bin/env python3
"""
Profile the CallableRegistry execution pipeline to identify bottlenecks.

This script instruments each step of the execution to measure:
- Divergence check time
- Function execution time  
- Semantic detection time
- Versioning time
- Tree building time
- Diff computation time
"""

import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field
from contextlib import contextmanager

# Add abstractions to path
abstractions_path = Path(__file__).parent.parent.parent / "abstractions"
sys.path.insert(0, str(abstractions_path))

from abstractions.ecs.entity import Entity, EntityRegistry, build_entity_tree, find_modified_entities
from abstractions.ecs.callable_registry import CallableRegistry
from test_scenario import GridMap, Node, Agent, create_test_scenario


# ============================================================================
# Timing Context Manager
# ============================================================================

@dataclass
class TimingData:
    """Store timing measurements."""
    name: str
    timings: List[float] = field(default_factory=list)
    
    def add(self, duration_ms: float):
        self.timings.append(duration_ms)
    
    def mean(self) -> float:
        return sum(self.timings) / len(self.timings) if self.timings else 0
    
    def total(self) -> float:
        return sum(self.timings)


class ProfilerContext:
    """Global profiler context."""
    
    def __init__(self):
        self.timings: Dict[str, TimingData] = {}
        self.enabled = True
    
    def reset(self):
        self.timings.clear()
    
    @contextmanager
    def timer(self, name: str):
        """Time a code block."""
        if not self.enabled:
            yield
            return
        
        if name not in self.timings:
            self.timings[name] = TimingData(name)
        
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.timings[name].add(duration_ms)
    
    def print_summary(self):
        """Print timing summary."""
        print(f"\n{'='*70}")
        print("PROFILING SUMMARY")
        print(f"{'='*70}")
        
        # Calculate total
        total_time = sum(t.total() for t in self.timings.values())
        
        # Sort by total time
        sorted_timings = sorted(
            self.timings.values(),
            key=lambda t: t.total(),
            reverse=True
        )
        
        print(f"{'Operation':<30} {'Count':<8} {'Total(ms)':<12} {'Mean(ms)':<12} {'%':<8}")
        print(f"{'-'*70}")
        
        for timing in sorted_timings:
            count = len(timing.timings)
            total = timing.total()
            mean = timing.mean()
            percentage = (total / total_time * 100) if total_time > 0 else 0
            
            print(f"{timing.name:<30} {count:<8} {total:<12.2f} {mean:<12.3f} {percentage:<8.1f}")
        
        print(f"{'-'*70}")
        print(f"{'TOTAL':<30} {'':<8} {total_time:<12.2f}")
        print(f"{'='*70}\n")


# Global profiler
profiler = ProfilerContext()


# ============================================================================
# Instrumented Functions
# ============================================================================

def profile_build_entity_tree(entity: Entity):
    """Profile tree building."""
    with profiler.timer("build_entity_tree"):
        return build_entity_tree(entity)


def profile_find_modified_entities(new_tree, old_tree):
    """Profile diff computation."""
    with profiler.timer("find_modified_entities"):
        return find_modified_entities(new_tree, old_tree)


def profile_get_stored_tree(root_ecs_id):
    """Profile tree retrieval."""
    with profiler.timer("get_stored_tree"):
        return EntityRegistry.get_stored_tree(root_ecs_id)


def profile_version_entity(entity):
    """Profile entity versioning."""
    with profiler.timer("version_entity_total"):
        # Break down versioning steps
        with profiler.timer("version_entity.get_stored_tree"):
            old_tree = EntityRegistry.get_stored_tree(entity.root_ecs_id)
        
        with profiler.timer("version_entity.build_new_tree"):
            new_tree = build_entity_tree(entity)
        
        with profiler.timer("version_entity.find_modified"):
            modified = find_modified_entities(new_tree, old_tree)
        
        with profiler.timer("version_entity.update_ids"):
            # Version entities (simplified)
            if modified:
                # This is where update_ecs_ids() is called
                pass
        
        return True


# ============================================================================
# Profiled Move Operation
# ============================================================================

def profile_move_operation(gridmap: GridMap, source_idx: int, target_idx: int, agent_name: str) -> GridMap:
    """Profile a complete move operation through CallableRegistry."""
    
    with profiler.timer("total_operation"):
        # Use the ACTUAL CallableRegistry.execute to get real behavior
        with profiler.timer("callable_registry_execute"):
            new_gridmap = CallableRegistry.execute(
                "move_agent_global",
                gridmap=gridmap,
                source_index=source_idx,
                agent_name=agent_name,
                target_index=target_idx
            )
    
    return new_gridmap  # ← Return the NEW versioned gridmap!


# ============================================================================
# Profiling Tests
# ============================================================================

def profile_single_move(num_nodes: int, agents_per_node: int):
    """Profile a single move operation in detail."""
    print(f"\n{'='*70}")
    print(f"PROFILING SINGLE MOVE: {num_nodes} nodes × {agents_per_node} agents/node")
    print(f"{'='*70}\n")
    
    # Create scenario
    print("Creating scenario...")
    gridmap = create_test_scenario(num_nodes, agents_per_node, seed=42)
    print(f"✓ Created {gridmap.total_entities()} entities\n")
    
    # Reset profiler
    profiler.reset()
    
    # Profile one move
    print("Profiling move operation...")
    source_idx = 0
    target_idx = num_nodes // 2
    agent_name = gridmap.nodes[source_idx].agents[0].name
    
    gridmap = profile_move_operation(gridmap, source_idx, target_idx, agent_name)  # ← Capture return value!
    
    # Print results
    profiler.print_summary()


def profile_multiple_moves(num_nodes: int, agents_per_node: int, num_moves: int):
    """Profile multiple move operations."""
    print(f"\n{'='*70}")
    print(f"PROFILING {num_moves} MOVES: {num_nodes} nodes × {agents_per_node} agents/node")
    print(f"{'='*70}\n")
    
    # Create scenario
    print("Creating scenario...")
    gridmap = create_test_scenario(num_nodes, agents_per_node, seed=42)
    print(f"✓ Created {gridmap.total_entities()} entities\n")
    
    # Reset profiler
    profiler.reset()
    
    # Profile multiple moves
    print(f"Profiling {num_moves} move operations...")
    import random
    random.seed(42)
    
    for i in range(num_moves):
        source_idx = random.randint(0, num_nodes - 1)
        target_idx = random.randint(0, num_nodes - 1)
        
        if source_idx == target_idx:
            continue
        
        source_node = gridmap.nodes[source_idx]
        if not source_node.agents:
            continue
        
        agent = random.choice(source_node.agents)
        
        gridmap = profile_move_operation(gridmap, source_idx, target_idx, agent.name)  # ← Capture return value!
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_moves}")
    
    # Print results
    profiler.print_summary()


def profile_tree_operations(num_nodes: int, agents_per_node: int):
    """Profile tree building and diff operations separately."""
    print(f"\n{'='*70}")
    print(f"PROFILING TREE OPERATIONS: {num_nodes} nodes × {agents_per_node} agents/node")
    print(f"{'='*70}\n")
    
    # Create scenario
    gridmap = create_test_scenario(num_nodes, agents_per_node, seed=42)
    total_entities = gridmap.total_entities()
    print(f"✓ Created {total_entities} entities\n")
    
    # Reset profiler
    profiler.reset()
    
    # Test 1: Tree building
    print("Test 1: Tree building (10 iterations)...")
    for i in range(10):
        with profiler.timer("build_tree_test"):
            tree = build_entity_tree(gridmap)
    
    # Test 2: Tree retrieval
    print("Test 2: Tree retrieval (10 iterations)...")
    for i in range(10):
        with profiler.timer("get_tree_test"):
            tree = EntityRegistry.get_stored_tree(gridmap.root_ecs_id)
    
    # Test 3: Diff computation (no changes)
    print("Test 3: Diff computation - no changes (10 iterations)...")
    tree1 = build_entity_tree(gridmap)
    tree2 = build_entity_tree(gridmap)
    for i in range(10):
        with profiler.timer("diff_no_changes"):
            modified = find_modified_entities(tree1, tree2)
    
    # Test 4: Diff computation (with changes)
    print("Test 4: Diff computation - with changes (10 iterations)...")
    # Make a small change
    agent = gridmap.nodes[0].agents[0]
    gridmap.nodes[0].remove_agent(agent)
    gridmap.nodes[1].add_agent(agent)
    tree3 = build_entity_tree(gridmap)
    
    for i in range(10):
        with profiler.timer("diff_with_changes"):
            modified = find_modified_entities(tree3, tree1)
    
    # Print results
    profiler.print_summary()
    
    # Calculate per-entity costs
    print(f"\n{'='*70}")
    print("PER-ENTITY COSTS")
    print(f"{'='*70}")
    
    if "build_tree_test" in profiler.timings:
        build_time = profiler.timings["build_tree_test"].mean()
        print(f"Build tree: {build_time:.3f} ms total = {build_time/total_entities:.6f} ms/entity")
    
    if "get_tree_test" in profiler.timings:
        get_time = profiler.timings["get_tree_test"].mean()
        print(f"Get tree: {get_time:.3f} ms total = {get_time/total_entities:.6f} ms/entity")
    
    if "diff_no_changes" in profiler.timings:
        diff_time = profiler.timings["diff_no_changes"].mean()
        print(f"Diff (no changes): {diff_time:.3f} ms total = {diff_time/total_entities:.6f} ms/entity")
    
    if "diff_with_changes" in profiler.timings:
        diff_time = profiler.timings["diff_with_changes"].mean()
        print(f"Diff (with changes): {diff_time:.3f} ms total = {diff_time/total_entities:.6f} ms/entity")
    
    print(f"{'='*70}\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PIPELINE PROFILING SUITE")
    print("="*70)
    
    # Test 1: Single move with detailed breakdown
    profile_single_move(10, 10)
    
    # Test 2: Multiple moves to see patterns
    profile_multiple_moves(10, 10, 20)
    
    # Test 3: Tree operations in isolation
    profile_tree_operations(10, 10)
    
    # Test 4: Scaling analysis
    print(f"\n{'='*70}")
    print("SCALING ANALYSIS")
    print(f"{'='*70}\n")
    
    configs = [
        (10, 10),   # 111 entities
        (20, 20),   # 421 entities
        (50, 20),   # 1,051 entities
    ]
    
    for nodes, agents in configs:
        profiler.reset()
        profile_tree_operations(nodes, agents)
    
    print("\n✓ Profiling complete!")
