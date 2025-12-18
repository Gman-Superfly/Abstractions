#!/usr/bin/env python3
"""
Test Scenario: GridMap with Nodes and Agents

This module creates a parameterized hierarchical entity structure for testing
local versioning optimizations:

Structure:
    GridMap (root)
    └── nodes: List[Node]
        └── agents: List[Agent]

Entity counts:
    - Small: 10 nodes × 10 agents = 100 entities
    - Medium: 50 nodes × 50 agents = 2,500 entities  
    - Large: 100 nodes × 100 agents = 10,000 entities
    - XLarge: 200 nodes × 100 agents = 20,000 entities
"""

import sys
import random
import time
from typing import List, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

# Add abstractions to path
abstractions_path = Path(__file__).parent.parent.parent / "abstractions"
sys.path.insert(0, str(abstractions_path))

from abstractions.ecs.entity import Entity, EntityRegistry
from abstractions.ecs.callable_registry import CallableRegistry
from pydantic import Field


# ============================================================================
# Entity Definitions
# ============================================================================

class Agent(Entity):
    """
    An agent in the simulation.
    
    Represents a movable entity with a name and speed attribute.
    """
    name: str = ""
    speed: int = 1


class Node(Entity):
    """
    A grid node containing agents.
    
    Represents a location in the grid that can contain multiple agents.
    """
    position: Tuple[int, int] = (0, 0)
    agents: List[Agent] = Field(default_factory=list)
    
    def add_agent(self, agent: Agent) -> None:
        """Add an agent to this node."""
        self.agents.append(agent)
    
    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from this node."""
        if agent in self.agents:
            self.agents.remove(agent)
    
    def get_agent_by_name(self, name: str) -> Agent | None:
        """Find an agent by name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None


class GridMap(Entity):
    """
    A grid map containing nodes.
    
    Represents the top-level entity in the hierarchy.
    """
    nodes: List[Node] = Field(default_factory=list)
    width: int = 10
    height: int = 10
    
    def get_node_at(self, x: int, y: int) -> Node | None:
        """Get the node at position (x, y)."""
        for node in self.nodes:
            if node.position == (x, y):
                return node
        return None
    
    def get_node_by_index(self, index: int) -> Node | None:
        """Get node by list index."""
        if 0 <= index < len(self.nodes):
            return self.nodes[index]
        return None
    
    def total_agents(self) -> int:
        """Count total agents across all nodes."""
        return sum(len(node.agents) for node in self.nodes)
    
    def total_entities(self) -> int:
        """Count total entities (map + nodes + agents)."""
        return 1 + len(self.nodes) + self.total_agents()


# ============================================================================
# Scenario Creation
# ============================================================================

def create_test_scenario(
    num_nodes: int,
    agents_per_node: int,
    grid_width: int = 10,
    seed: int | None = None
) -> GridMap:
    """
    Create a test scenario with specified parameters.
    
    Args:
        num_nodes: Number of grid nodes to create
        agents_per_node: Number of agents per node
        grid_width: Width of the grid (for position calculation)
        seed: Random seed for reproducibility
    
    Returns:
        GridMap entity with all nodes and agents registered
    
    Example:
        >>> gridmap = create_test_scenario(100, 100)
        >>> print(f"Total entities: {gridmap.total_entities()}")
        Total entities: 10001
    """
    if seed is not None:
        random.seed(seed)
    
    gridmap = GridMap(width=grid_width, height=(num_nodes // grid_width) + 1)
    
    print(f"Creating scenario: {num_nodes} nodes × {agents_per_node} agents/node")
    
    for i in range(num_nodes):
        # Calculate grid position
        x = i % grid_width
        y = i // grid_width
        
        node = Node(position=(x, y))
        
        # Create agents for this node
        for j in range(agents_per_node):
            agent = Agent(
                name=f"agent_{i}_{j}",
                speed=random.randint(1, 10)
            )
            node.add_agent(agent)
        
        gridmap.nodes.append(node)
    
    # Promote to root and register
    gridmap.promote_to_root()
    
    print(f"✓ Created GridMap with {gridmap.total_entities()} total entities")
    print(f"  - 1 GridMap")
    print(f"  - {len(gridmap.nodes)} Nodes")
    print(f"  - {gridmap.total_agents()} Agents")
    
    return gridmap


# ============================================================================
# Predefined Configurations
# ============================================================================

def create_small_scenario(seed: int | None = None) -> GridMap:
    """Create small test scenario (100 entities)."""
    return create_test_scenario(
        num_nodes=10,
        agents_per_node=10,
        grid_width=10,
        seed=seed
    )


def create_medium_scenario(seed: int | None = None) -> GridMap:
    """Create medium test scenario (2,500 entities)."""
    return create_test_scenario(
        num_nodes=50,
        agents_per_node=50,
        grid_width=10,
        seed=seed
    )


def create_large_scenario(seed: int | None = None) -> GridMap:
    """Create large test scenario (10,000 entities)."""
    return create_test_scenario(
        num_nodes=100,
        agents_per_node=100,
        grid_width=10,
        seed=seed
    )


def create_xlarge_scenario(seed: int | None = None) -> GridMap:
    """Create extra-large test scenario (20,000 entities)."""
    return create_test_scenario(
        num_nodes=200,
        agents_per_node=100,
        grid_width=20,
        seed=seed
    )


# ============================================================================
# Agent Movement Operations
# ============================================================================

@CallableRegistry.register("move_agent_global")
def move_agent_global(
    gridmap: GridMap,
    source_index: int,
    agent_name: str,
    target_index: int
) -> GridMap:
    """
    Move an agent between nodes using global gridmap mutation.
    
    This is the CURRENT approach - operates on the entire gridmap.
    Uses CallableRegistry for automatic versioning.
    
    Args:
        gridmap: The grid map entity
        source_index: Index of source node
        agent_name: Name of agent to move
        target_index: Index of target node
    
    Returns:
        Mutated gridmap (same object, will trigger full versioning)
    """
    source_node = gridmap.get_node_by_index(source_index)
    target_node = gridmap.get_node_by_index(target_index)
    
    if not source_node or not target_node:
        raise ValueError("Invalid node indices")
    
    agent = source_node.get_agent_by_name(agent_name)
    if not agent:
        raise ValueError(f"Agent {agent_name} not found in source node")
    
    # Perform the move (direct mutation, CallableRegistry handles versioning)
    source_node.remove_agent(agent)
    target_node.add_agent(agent)
    
    return gridmap


@CallableRegistry.register("move_agent_local")
def move_agent_local(
    source_node: Node,
    agent: Agent,
    target_node: Node
) -> Tuple[Node, Node]:
    """
    Move an agent between nodes using local node mutation.
    
    This is the OPTIMIZED approach - operates only on affected nodes.
    Uses CallableRegistry for automatic versioning.
    
    Args:
        source_node: Source node containing the agent
        agent: Agent to move
        target_node: Target node to receive the agent
    
    Returns:
        Tuple of (modified_source_node, modified_target_node)
    """
    # Remove from source
    source_node.remove_agent(agent)
    
    # Add to target
    target_node.add_agent(agent)
    
    return source_node, target_node


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_scenario(gridmap: GridMap) -> bool:
    """
    Validate that a scenario is correctly structured.
    
    Checks:
    - GridMap is registered as root
    - All nodes are in the tree
    - All agents are in the tree
    - Tree structure is consistent
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    if not gridmap.is_root_entity():
        raise ValueError("GridMap is not a root entity")
    
    tree = gridmap.get_tree()
    if tree is None:
        raise ValueError("GridMap has no tree")
    
    # Check node count
    expected_nodes = 1 + len(gridmap.nodes)  # map + nodes
    expected_agents = gridmap.total_agents()
    expected_total = expected_nodes + expected_agents
    
    if tree.node_count != expected_total:
        raise ValueError(
            f"Tree node count mismatch: expected {expected_total}, got {tree.node_count}"
        )
    
    # Check that all nodes are in tree
    for node in gridmap.nodes:
        if node.ecs_id not in tree.nodes:
            raise ValueError(f"Node {node.ecs_id} not in tree")
        
        # Check that all agents are in tree
        for agent in node.agents:
            if agent.ecs_id not in tree.nodes:
                raise ValueError(f"Agent {agent.ecs_id} not in tree")
    
    print("✓ Scenario validation passed")
    return True


def print_scenario_stats(gridmap: GridMap) -> None:
    """Print statistics about a scenario."""
    tree = gridmap.get_tree()
    
    print("\n" + "="*60)
    print("SCENARIO STATISTICS")
    print("="*60)
    print(f"GridMap ID: {gridmap.ecs_id}")
    print(f"Lineage ID: {gridmap.lineage_id}")
    print(f"Grid Size: {gridmap.width}×{gridmap.height}")
    print(f"\nEntity Counts:")
    print(f"  - Nodes: {len(gridmap.nodes)}")
    print(f"  - Agents: {gridmap.total_agents()}")
    print(f"  - Total: {gridmap.total_entities()}")
    
    if tree:
        print(f"\nTree Statistics:")
        print(f"  - Node Count: {tree.node_count}")
        print(f"  - Edge Count: {tree.edge_count}")
        print(f"  - Max Depth: {tree.max_depth}")
    
    print("="*60 + "\n")


# ============================================================================
# Performance Tracking
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Track performance metrics for operations."""
    operation_name: str
    num_nodes: int
    agents_per_node: int
    num_operations: int
    
    # Timing data
    timings: List[float] = field(default_factory=list)
    
    # Operation counts
    total_entities: int = 0
    entities_versioned: int = 0
    
    def add_timing(self, duration_ms: float):
        """Add a timing measurement."""
        self.timings.append(duration_ms)
    
    def get_stats(self) -> Dict[str, float]:
        """Calculate statistics."""
        if not self.timings:
            return {}
        
        return {
            "total_ms": sum(self.timings),
            "mean_ms": sum(self.timings) / len(self.timings),
            "min_ms": min(self.timings),
            "max_ms": max(self.timings),
            "median_ms": sorted(self.timings)[len(self.timings) // 2],
            "ops_per_sec": len(self.timings) / (sum(self.timings) / 1000) if sum(self.timings) > 0 else 0
        }
    
    def print_summary(self):
        """Print performance summary."""
        stats = self.get_stats()
        if not stats:
            print("No timing data collected")
            return
        
        print(f"\n{'='*70}")
        print(f"PERFORMANCE SUMMARY: {self.operation_name}")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  - Nodes: {self.num_nodes}")
        print(f"  - Agents/Node: {self.agents_per_node}")
        print(f"  - Total Entities: {self.total_entities}")
        print(f"  - Operations: {self.num_operations}")
        print(f"\nTiming Statistics:")
        print(f"  - Total Time: {stats['total_ms']:.2f} ms")
        print(f"  - Mean Time: {stats['mean_ms']:.3f} ms/op")
        print(f"  - Min Time: {stats['min_ms']:.3f} ms/op")
        print(f"  - Max Time: {stats['max_ms']:.3f} ms/op")
        print(f"  - Median Time: {stats['median_ms']:.3f} ms/op")
        print(f"  - Throughput: {stats['ops_per_sec']:.2f} ops/sec")
        print(f"{'='*70}\n")


class PerformanceTracker:
    """Track performance across multiple test runs."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
    
    def add_metrics(self, metrics: PerformanceMetrics):
        """Add metrics from a test run."""
        self.metrics.append(metrics)
    
    def print_comparison_table(self):
        """Print comparison table across all test runs."""
        if not self.metrics:
            print("No metrics to compare")
            return
        
        print(f"\n{'='*100}")
        print("PERFORMANCE COMPARISON TABLE")
        print(f"{'='*100}")
        print(f"{'Config':<15} {'Entities':<10} {'Ops':<6} {'Total(ms)':<12} {'Mean(ms)':<12} {'Throughput':<15}")
        print(f"{'-'*100}")
        
        for m in self.metrics:
            stats = m.get_stats()
            config = f"{m.num_nodes}×{m.agents_per_node}"
            print(f"{config:<15} {m.total_entities:<10} {m.num_operations:<6} "
                  f"{stats['total_ms']:<12.2f} {stats['mean_ms']:<12.3f} "
                  f"{stats['ops_per_sec']:<15.2f}")
        
        print(f"{'='*100}\n")


# ============================================================================
# Stress Testing
# ============================================================================

def stress_test_moves(
    num_nodes: int,
    agents_per_node: int,
    num_operations: int,
    seed: int | None = None
) -> PerformanceMetrics:
    """
    Stress test agent movements with performance tracking.
    
    Args:
        num_nodes: Number of nodes in the grid
        agents_per_node: Number of agents per node
        num_operations: Number of move operations to perform
        seed: Random seed for reproducibility
    
    Returns:
        PerformanceMetrics with timing data
    """
    if seed is not None:
        random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"STRESS TEST: {num_nodes} nodes × {agents_per_node} agents/node")
    print(f"{'='*70}")
    
    # Create scenario
    print("Creating scenario...")
    start = time.perf_counter()
    gridmap = create_test_scenario(num_nodes, agents_per_node, seed=seed)
    creation_time = (time.perf_counter() - start) * 1000
    print(f"✓ Scenario created in {creation_time:.2f} ms")
    
    # Initialize metrics
    metrics = PerformanceMetrics(
        operation_name=f"move_agent_global_{num_nodes}x{agents_per_node}",
        num_nodes=num_nodes,
        agents_per_node=agents_per_node,
        num_operations=num_operations,
        total_entities=gridmap.total_entities()
    )
    
    # Perform operations
    print(f"Performing {num_operations} move operations...")
    
    for i in range(num_operations):
        # Pick random source and target nodes
        source_idx = random.randint(0, num_nodes - 1)
        target_idx = random.randint(0, num_nodes - 1)
        
        # Skip if same node
        if source_idx == target_idx:
            continue
        
        # Pick random agent from source node
        source_node = gridmap.nodes[source_idx]
        if not source_node.agents:
            continue
        
        agent = random.choice(source_node.agents)
        
        # Time the move operation using CallableRegistry with optimizations
        start = time.perf_counter()
        gridmap = CallableRegistry.execute(
            "move_agent_global",
            skip_divergence_check=False,#(i > 0),  # Skip check after first move (entity is fresh)
            gridmap=gridmap,
            source_index=source_idx,
            agent_name=agent.name,
            target_index=target_idx
        )
        duration_ms = (time.perf_counter() - start) * 1000
        
        metrics.add_timing(duration_ms)
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_operations} operations "
                  f"(avg: {sum(metrics.timings[-10:]) / 10:.3f} ms/op)")
    
    print(f"✓ Completed {len(metrics.timings)} operations")
    
    # Validate final state
    print("Validating final state...")
    validate_scenario(gridmap)
    
    return metrics


# ============================================================================
# Detailed Operation Profiling
# ============================================================================

def profile_single_move(
    num_nodes: int,
    agents_per_node: int,
    seed: int | None = None
) -> Dict[str, float]:
    """
    Profile a single move operation with detailed timing breakdown.
    
    Returns:
        Dict with timing for each phase
    """
    if seed is not None:
        random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"DETAILED PROFILING: {num_nodes} nodes × {agents_per_node} agents/node")
    print(f"{'='*70}")
    
    timings = {}
    
    # Phase 1: Scenario creation
    print("Phase 1: Creating scenario...")
    start = time.perf_counter()
    gridmap = create_test_scenario(num_nodes, agents_per_node, seed=seed)
    timings['scenario_creation'] = (time.perf_counter() - start) * 1000
    print(f"  ✓ {timings['scenario_creation']:.2f} ms")
    
    # Phase 2: Tree building (implicit in promote_to_root)
    print("Phase 2: Initial tree registration...")
    # Already done in create_test_scenario
    tree = gridmap.get_tree()
    timings['initial_tree_nodes'] = tree.node_count if tree else 0
    timings['initial_tree_edges'] = tree.edge_count if tree else 0
    print(f"  ✓ Tree has {timings['initial_tree_nodes']} nodes, {timings['initial_tree_edges']} edges")
    
    # Phase 3: Prepare move
    print("Phase 3: Preparing move operation...")
    source_idx = 0
    target_idx = num_nodes // 2
    agent = gridmap.nodes[source_idx].agents[0]
    print(f"  ✓ Moving {agent.name} from node {source_idx} to node {target_idx}")
    
    # Phase 4: Execute move (with versioning via CallableRegistry)
    print("Phase 4: Executing move with versioning via CallableRegistry...")
    start = time.perf_counter()
    gridmap = CallableRegistry.execute(
        "move_agent_global",
        gridmap=gridmap,
        source_index=source_idx,
        agent_name=agent.name,
        target_index=target_idx
    )
    timings['move_with_versioning'] = (time.perf_counter() - start) * 1000
    print(f"  ✓ {timings['move_with_versioning']:.2f} ms")
    
    # Phase 5: Validate result
    print("Phase 5: Validating result...")
    start = time.perf_counter()
    validate_scenario(gridmap)
    timings['validation'] = (time.perf_counter() - start) * 1000
    print(f"  ✓ {timings['validation']:.2f} ms")
    
    # Print breakdown
    print(f"\n{'='*70}")
    print("TIMING BREAKDOWN")
    print(f"{'='*70}")
    total = sum(v for k, v in timings.items() if isinstance(v, float))
    for key, value in timings.items():
        if isinstance(value, float):
            percentage = (value / total * 100) if total > 0 else 0
            print(f"{key:<30}: {value:>10.2f} ms ({percentage:>5.1f}%)")
    print(f"{'-'*70}")
    print(f"{'TOTAL':<30}: {total:>10.2f} ms")
    print(f"{'='*70}\n")
    
    return timings


# ============================================================================
# Main Stress Test Suite
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPREHENSIVE STRESS TEST SUITE")
    print("="*70 + "\n")
    
    tracker = PerformanceTracker()
    
    # Test configurations: (nodes, agents_per_node, num_operations)
    test_configs = [
        (10, 10, 100),      # Small: 111 entities, 100 ops
        (20, 20, 100),      # Medium-Small: 421 entities, 100 ops
        (50, 20, 100),      # Medium: 1,051 entities, 100 ops
        (50, 50, 100),      # Medium-Large: 2,551 entities, 100 ops
        (100, 50, 100),     # Large: 5,101 entities, 100 ops
    ]
    
    # Run stress tests
    for nodes, agents, ops in test_configs:
        try:
            metrics = stress_test_moves(nodes, agents, ops, seed=42)
            metrics.print_summary()
            tracker.add_metrics(metrics)
        except Exception as e:
            print(f"❌ Test failed for {nodes}×{agents}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison table
    tracker.print_comparison_table()
    
    # Detailed profiling for one configuration
    print("\n" + "="*70)
    print("DETAILED PROFILING (Single Operation)")
    print("="*70)
    profile_single_move(100, 50, seed=42)
    
    print("\n✓ All tests completed!")
