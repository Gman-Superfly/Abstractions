"""
Navigation Graph Computation

Functions to compute navigation graphs from GridMap state.
These create derived entities that track walkability and visibility.
"""

from typing import Tuple, List
from abstractions.ecs.callable_registry import CallableRegistry
from game_entities import GridMap, GridNode, GameEntity, NavigationGraph
from movement import get_node_at


def is_node_transparent(node: GridNode) -> bool:
    """Check if a node is transparent (all entities must be transparent).
    
    Args:
        node: The node to check
        
    Returns:
        True if all entities in the node are transparent, False otherwise
    """
    if not node.entities:
        return True  # Empty node is transparent
    
    # All entities must be transparent for the node to be transparent
    return all(entity.transparent for entity in node.entities)


def is_node_walkable(node: GridNode) -> bool:
    """Check if a node is walkable (all entities must be walkable).
    
    Args:
        node: The node to check
        
    Returns:
        True if all entities in the node are walkable, False otherwise
    """
    if not node.entities:
        return True  # Empty node is walkable
    
    # All entities must be walkable for the node to be walkable
    return all(entity.walkable for entity in node.entities)


@CallableRegistry.register("compute_navigation_graph")
def compute_navigation_graph(grid_map: GridMap) -> NavigationGraph:
    """Compute navigation graph with walkability and visibility data.
    
    Framework automatically sets:
    - result.derived_from_function = "compute_navigation_graph"
    - result.derived_from_execution_id = <UUID>
    
    This creates a causal link: NavigationGraph was derived from GridMap
    
    Args:
        grid_map: The grid to analyze
        
    Returns:
        NavigationGraph with adjacency and state data
    """
    walkable_adj = {}
    walkable_state = {}
    transparent_adj = {}
    transparent_state = {}
    
    # Process each node
    for node in grid_map.nodes:
        pos = node.position
        
        # Check walkability
        is_walkable = is_node_walkable(node)
        walkable_state[pos] = is_walkable
        
        # Check transparency
        is_transparent = is_node_transparent(node)
        transparent_state[pos] = is_transparent
        
        # Compute neighbors (8-directional)
        walkable_neighbors = []
        transparent_neighbors = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip self
                
                neighbor_pos = (pos[0] + dx, pos[1] + dy)
                neighbor_node = get_node_at(grid_map, neighbor_pos)
                
                if neighbor_node:
                    # Add to walkable adjacency if neighbor is walkable
                    if is_node_walkable(neighbor_node):
                        walkable_neighbors.append(neighbor_pos)
                    
                    # Add to transparent adjacency if neighbor is transparent
                    if is_node_transparent(neighbor_node):
                        transparent_neighbors.append(neighbor_pos)
        
        walkable_adj[pos] = walkable_neighbors
        transparent_adj[pos] = transparent_neighbors
    
    print(f"Computed navigation graph from GridMap {grid_map.ecs_id}")
    print(f"  Nodes: {len(walkable_state)}")
    print(f"  Walkable nodes: {sum(walkable_state.values())}")
    print(f"  Transparent nodes: {sum(transparent_state.values())}")
    
    return NavigationGraph(
        source_grid_id=grid_map.ecs_id,
        walkable_adjacency=walkable_adj,
        walkable=walkable_state,
        transparent_adjacency=transparent_adj,
        transparent=transparent_state
    )
