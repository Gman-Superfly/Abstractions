"""
Apple Maximization - Complete Agentic Loop Example

An agent navigates a grid to collect apples and maximize nutrition value.
Demonstrates:
1. Multi-step agentic behavior (pathfinding, movement, collection)
2. Entity versioning with proper List[Entity] handling
3. Lineage tracking across multiple function executions
4. Full event reconstruction from the registry
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abstractions.ecs.callable_registry import CallableRegistry
from abstractions.ecs.entity import EntityRegistry
from game_entities import GridMap, GridNode, Floor, Agent, Apple
from navigation import compute_navigation_graph
from pathfinding import compute_reachable_paths, choose_path
from movement import (
    move_agent_along_path, find_agent_position, find_apple_at_position,
    get_latest_gridmap
)
import random
import time


def create_grid_with_apples(width=5, height=5, num_apples=5):
    """Create a grid with agent and randomly placed apples."""
    # Create empty grid
    nodes = []
    for y in range(height):
        for x in range(width):
            pos = (x, y)
            node = GridNode(position=pos)
            node.entities = [Floor(name=f"floor_{x}_{y}", walkable=True, transparent=True)]
            nodes.append(node)
    
    grid = GridMap(nodes=nodes, width=width, height=height)
    grid.promote_to_root()
    
    # Add agent at center
    center_x, center_y = width // 2, height // 2
    agent_node = next(n for n in grid.nodes if n.position == (center_x, center_y))
    agent = Agent(name="collector", speed=2, sight=5, inventory=[])
    agent_node.entities.append(agent)
    
    # Add apples at random positions
    available_positions = [
        n.position for n in grid.nodes 
        if n.position != (center_x, center_y)
    ]
    
    apple_positions = random.sample(available_positions, min(num_apples, len(available_positions)))
    
    for i, pos in enumerate(apple_positions):
        node = next(n for n in grid.nodes if n.position == pos)
        nutrition = random.randint(5, 20)
        apple = Apple(name=f"apple_{i+1}", nutrition=nutrition)
        node.entities.append(apple)
    
    return grid


def print_grid_state(grid: GridMap, step: int):
    """Print current grid state."""
    print(f"\n{'='*60}")
    print(f"Step {step} - Grid {str(grid.ecs_id)[:8]}... (lineage: {str(grid.lineage_id)[:8]}...)")
    print('='*60)
    
    # Find agent
    agent = None
    agent_pos = None
    for node in grid.nodes:
        for entity in node.entities:
            if isinstance(entity, Agent):
                agent = entity
                agent_pos = node.position
                break
        if agent:
            break
    
    # Find apples
    apple_positions = {}
    for node in grid.nodes:
        for entity in node.entities:
            if isinstance(entity, Apple):
                apple_positions[node.position] = entity
    
    # Print grid
    for y in range(grid.height):
        for x in range(grid.width):
            pos = (x, y)
            if pos == agent_pos:
                print("A ", end="")
            elif pos in apple_positions:
                print("@ ", end="")
            else:
                print(". ", end="")
        print()
    
    print(f"\nAgent: {agent.name} at {agent_pos}")
    print(f"Inventory: {len(agent.inventory)} apples (total nutrition: {sum(a.nutrition for a in agent.inventory)})")
    print(f"Apples on grid: {len(apple_positions)}")
    if apple_positions:
        for pos, apple in apple_positions.items():
            print(f"  - {apple.name} at {pos} (nutrition: {apple.nutrition})")


def run_apple_maximization(max_steps=50, target_apples=5):
    """Run the apple maximization loop."""
    print("\n" + "="*60)
    print("APPLE MAXIMIZATION - Agentic Loop")
    print("="*60)
    
    # Timing - track EVERYTHING
    total_start = time.time()
    phase_times = {}  # Will auto-populate with all function calls
    
    # Monkey-patch CallableRegistry.execute to track all function calls
    original_execute = CallableRegistry.execute
    
    def timed_execute(func_name, **kwargs):
        start = time.time()
        result = original_execute(func_name, **kwargs)
        duration = time.time() - start
        phase_times[func_name] = phase_times.get(func_name, 0) + duration
        return result
    
    CallableRegistry.execute = timed_execute
    
    # Track other operations
    phase_times['setup'] = 0
    phase_times['find_agent'] = 0
    phase_times['find_apple'] = 0
    phase_times['get_latest_gridmap'] = 0
    phase_times['print_grid'] = 0
    
    # Create initial grid
    setup_start = time.time()
    grid = create_grid_with_apples(width=5, height=5, num_apples=target_apples)
    phase_times['setup'] = time.time() - setup_start
    
    print_grid_state(grid, 0)
    
    # Track execution history
    execution_history = []
    
    # Cache agent name for faster lookups
    agent_name = "collector"
    
    # Main loop
    for step in range(1, max_steps + 1):
        # OPTIMIZATION: Build entity->position index once per step
        find_start = time.time()
        entity_positions = {}  # Maps entity.ecs_id -> position
        agent = None
        current_pos = None
        
        for node in grid.nodes:
            for entity in node.entities:
                entity_positions[entity.ecs_id] = node.position
                # Find agent while building index
                if isinstance(entity, Agent) and entity.name == agent_name:
                    agent = entity
                    current_pos = node.position
        
        phase_times['find_agent'] += time.time() - find_start
        
        if not agent:
            print("\nAgent lost!")
            break
        
        # Check if goal reached
        if len(agent.inventory) >= target_apples:
            print(f"\nGoal reached! Collected {len(agent.inventory)} apples in {step-1} steps")
            total_nutrition = sum(a.nutrition for a in agent.inventory)
            print(f"Total nutrition value: {total_nutrition}")
            break
        
        # Check if at apple position
        find_apple_start = time.time()
        apple_at_pos = find_apple_at_position(grid, current_pos)
        phase_times['find_apple'] += time.time() - find_apple_start
        
        if apple_at_pos:
            print(f"\nStep {step}: Collecting apple at {current_pos}")
            
            # Record execution
            execution_history.append({
                'step': step,
                'action': 'collect_apple',
                'grid_before': grid.ecs_id,
                'position': current_pos,
                'apple': apple_at_pos.name
            })
            
            # Collect apple (timed automatically)
            grid = CallableRegistry.execute(
                "collect_apple",
                grid_map=grid,
                agent=agent,
                apple_position=current_pos
            )
            
            # Get latest grid
            latest_start = time.time()
            grid = get_latest_gridmap(grid)
            phase_times['get_latest_gridmap'] = phase_times.get('get_latest_gridmap', 0) + (time.time() - latest_start)
            
            # Record result
            execution_history[-1]['grid_after'] = grid.ecs_id
            
            print_grid_state(grid, step)
            continue
        
        # Pathfind to nearest apple
        print(f"\nStep {step}: Pathfinding to apple")
        
        # Compute navigation graph (timed automatically)
        nav_graph = CallableRegistry.execute("compute_navigation_graph", grid_map=grid)
        
        # Compute reachable paths (timed automatically)
        path_collection = CallableRegistry.execute(
            "compute_reachable_paths",
            nav_graph=nav_graph,
            agent=agent,
            start_position=current_pos
        )
        
        # Choose best path (timed automatically)
        chosen_path = CallableRegistry.execute(
            "choose_path",
            path_collection=path_collection,
            grid_map=grid
        )
        
        # Move if path exists
        if len(chosen_path.steps) > 1:
            target_pos = chosen_path.steps[-1]
            
            # Record execution
            execution_history.append({
                'step': step,
                'action': 'move',
                'grid_before': grid.ecs_id,
                'from': current_pos,
                'to': target_pos
            })
            
            # Move agent (timed automatically)
            grid = CallableRegistry.execute(
                "move_agent_along_path",
                grid_map=grid,
                agent=agent,
                path=chosen_path
            )
            
            # Get latest grid
            latest_start = time.time()
            grid = get_latest_gridmap(grid)
            phase_times['get_latest_gridmap'] = phase_times.get('get_latest_gridmap', 0) + (time.time() - latest_start)
            
            # Record result
            execution_history[-1]['grid_after'] = grid.ecs_id
        else:
            print("No valid path found!")
            break
    else:
        print(f"\nMax steps ({max_steps}) reached")
    
    # Restore original execute
    CallableRegistry.execute = original_execute
    
    # Print timing summary
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print("PERFORMANCE TIMING - ALL FUNCTIONS")
    print("="*60)
    print(f"\nTotal execution time: {total_time:.3f}s")
    print(f"\nFunction call breakdown (sorted by time):")
    for phase, duration in sorted(phase_times.items(), key=lambda x: -x[1]):
        percentage = (duration / total_time) * 100
        print(f"  {phase:30s}: {duration:6.3f}s ({percentage:5.1f}%)")
    
    return grid, execution_history, phase_times


def reconstruct_event_timeline(initial_grid: GridMap, execution_history: list):
    """Reconstruct the full timeline of events from the registry."""
    print("\n" + "="*60)
    print("EVENT TIMELINE RECONSTRUCTION")
    print("="*60)
    
    lineage_id = initial_grid.lineage_id
    versions = EntityRegistry.lineage_registry.get(lineage_id, [])
    
    print(f"\nLineage: {str(lineage_id)[:8]}...")
    print(f"Total versions: {len(versions)}")
    
    print("\n" + "-"*60)
    print("Version History:")
    print("-"*60)
    
    for i, root_ecs_id in enumerate(versions):
        tree = EntityRegistry.get_stored_tree(root_ecs_id)
        grid = tree.get_entity(root_ecs_id)
        
        # Find agent in this version
        agent = None
        for entity in tree.nodes.values():
            if isinstance(entity, Agent):
                agent = entity
                break
        
        # Count apples on grid and in inventory
        apples_on_grid = sum(
            1 for entity in tree.nodes.values() 
            if isinstance(entity, Apple)
        )
        apples_in_inventory = len(agent.inventory) if agent else 0
        
        print(f"\nVersion {i}: {str(root_ecs_id)[:8]}...")
        print(f"  Entities in tree: {len(tree.nodes)}")
        print(f"  Edges: {len(tree.edges)}")
        print(f"  Apples on grid: {apples_on_grid}")
        print(f"  Apples collected: {apples_in_inventory}")
        
        if agent:
            print(f"  Agent inventory: {[a.name for a in agent.inventory]}")
            print(f"  Total nutrition: {sum(a.nutrition for a in agent.inventory)}")
        
        # Match with execution history
        matching_executions = [
            ex for ex in execution_history 
            if ex.get('grid_after') == root_ecs_id
        ]
        
        if matching_executions:
            for ex in matching_executions:
                print(f"  > Created by: {ex['action']} at step {ex['step']}")
                if ex['action'] == 'collect_apple':
                    print(f"     Collected: {ex['apple']} at {ex['position']}")
                elif ex['action'] == 'move':
                    print(f"     Moved: {ex['from']} -> {ex['to']}")
    
    print("\n" + "-"*60)
    print("Entity Lineages:")
    print("-"*60)
    
    # Track unique entity lineages
    final_tree = EntityRegistry.get_stored_tree(versions[-1])
    entity_lineages = {}
    
    for entity in final_tree.nodes.values():
        entity_type = type(entity).__name__
        if entity_type not in entity_lineages:
            entity_lineages[entity_type] = []
        entity_lineages[entity_type].append({
            'name': getattr(entity, 'name', 'unnamed'),
            'lineage_id': entity.lineage_id,
            'ecs_id': entity.ecs_id
        })
    
    for entity_type, entities in entity_lineages.items():
        print(f"\n{entity_type} ({len(entities)} instances):")
        for e in entities:
            print(f"  - {e['name']}: lineage {str(e['lineage_id'])[:8]}..., ecs_id {str(e['ecs_id'])[:8]}...")
    
    print("\n" + "-"*60)
    print("Execution Summary:")
    print("-"*60)
    
    actions_by_type = {}
    for ex in execution_history:
        action = ex['action']
        actions_by_type[action] = actions_by_type.get(action, 0) + 1
    
    print(f"\nTotal executions: {len(execution_history)}")
    for action, count in actions_by_type.items():
        print(f"  - {action}: {count}")
    
    print(f"\nVersions created: {len(versions)}")
    print(f"Version creation rate: {len(versions) / len(execution_history):.2f} versions per execution")


def main():
    """Run the complete example."""
    overall_start = time.time()
    
    # Run the agentic loop
    loop_start = time.time()
    final_grid, execution_history, phase_times = run_apple_maximization(max_steps=50, target_apples=5)
    loop_time = time.time() - loop_start
    
    # Reconstruct the timeline
    reconstruct_start = time.time()
    reconstruct_event_timeline(final_grid, execution_history)
    reconstruct_time = time.time() - reconstruct_start
    
    overall_time = time.time() - overall_start
    
    print("\n" + "="*60)
    print("OVERALL TIMING")
    print("="*60)
    print(f"Agentic loop:        {loop_time:.3f}s")
    print(f"Event reconstruction: {reconstruct_time:.3f}s")
    print(f"Total time:          {overall_time:.3f}s")
    
    print("\n" + "="*60)
    print("✅ EXAMPLE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
