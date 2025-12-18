"""
Pathfinding Functions

BFS-based pathfinding to compute all reachable paths from an agent's position.
"""

from typing import List, Tuple, Dict
from collections import deque
import random
from abstractions.ecs.callable_registry import CallableRegistry
from game_entities import NavigationGraph, Agent, Path, PathCollection, GridMap, Apple


@CallableRegistry.register("compute_reachable_paths")
def compute_reachable_paths(
    nav_graph: NavigationGraph,
    agent: Agent,
    start_position: Tuple[int, int]
) -> PathCollection:
    """
    Compute all paths reachable from agent's position within speed limit.
    
    Uses BFS to explore all positions within agent.speed steps.
    Returns PathCollection with all unique paths.
    
    Args:
        nav_graph: NavigationGraph with walkability data
        agent: Agent with speed attribute
        start_position: Starting position (agent's current position)
        
    Returns:
        PathCollection with all reachable paths and positions
    """
    start = start_position
    max_dist = agent.speed
    
    # BFS with path tracking
    # Queue: (current_position, path_so_far, distance_from_start)
    queue = deque([(start, [start], 0)])
    
    # Track visited positions with their shortest path
    visited: Dict[Tuple[int, int], List[Tuple[int, int]]] = {start: [start]}
    
    # Collect all paths
    all_paths: List[Path] = []
    
    while queue:
        pos, path, dist = queue.popleft()
        
        # Record this path
        all_paths.append(Path(
            start_position=start,
            end_position=pos,
            steps=path,
            length=dist,
            cost=dist
        ))
        
        # Explore neighbors if within range
        if dist < max_dist:
            # Get walkable neighbors from navigation graph
            neighbors = nav_graph.walkable_adjacency.get(pos, [])
            
            for neighbor in neighbors:
                # Only visit if we haven't been there, or found a shorter path
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    visited[neighbor] = new_path
                    queue.append((neighbor, new_path, dist + 1))
    
    # Extract unique reachable positions
    reachable_positions = list(visited.keys())
    
    print(f"Computed {len(all_paths)} paths from {start} with max distance {max_dist}")
    print(f"Reachable positions: {len(reachable_positions)}")
    
    return PathCollection(
        agent_id=agent.ecs_id,
        agent_position=start,
        max_distance=max_dist,
        paths=all_paths,
        reachable_positions=reachable_positions,
        source_graph_id=nav_graph.ecs_id
    )


@CallableRegistry.register("choose_path")
def choose_path(
    path_collection: PathCollection,
    grid_map: GridMap
) -> Path:
    """
    Choose the best path from a PathCollection.
    
    Strategy:
    1. Find all apples in the grid
    2. If apples exist, choose shortest path to nearest apple
    3. If no apples, choose random path
    
    Args:
        path_collection: All reachable paths
        grid_map: Current grid state
        
    Returns:
        Chosen path to follow
    """
    # Find all apple positions in the grid
    apple_positions = []
    for node in grid_map.nodes:
        for entity in node.entities:
            if isinstance(entity, Apple):
                apple_positions.append(node.position)
    
    print(f"Apples found: {apple_positions}")
    
    if apple_positions:
        # Find shortest path to any apple
        paths_to_apples = []
        for path in path_collection.paths:
            if path.end_position in apple_positions:
                paths_to_apples.append(path)
        
        if paths_to_apples:
            # Choose shortest path to apple
            shortest = min(paths_to_apples, key=lambda p: p.length)
            print(f"Chose path to apple at {shortest.end_position}: length={shortest.length}")
            return shortest
        else:
            # No reachable apples, choose path that gets closest
            best_path = None
            best_distance = float('inf')
            
            for path in path_collection.paths:
                for apple_pos in apple_positions:
                    # Manhattan distance from path end to apple
                    dist = abs(path.end_position[0] - apple_pos[0]) + abs(path.end_position[1] - apple_pos[1])
                    if dist < best_distance:
                        best_distance = dist
                        best_path = path
            
            print(f"No reachable apples, chose path towards {best_path.end_position}")
            return best_path
    else:
        # No apples, choose random path
        if len(path_collection.paths) > 1:
            # Don't choose the "stay in place" path if we have options
            non_zero_paths = [p for p in path_collection.paths if p.length > 0]
            if non_zero_paths:
                chosen = random.choice(non_zero_paths)
            else:
                chosen = path_collection.paths[0]
        else:
            chosen = path_collection.paths[0]
        
        print(f"No apples, chose random path to {chosen.end_position}")
        return chosen
