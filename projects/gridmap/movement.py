"""
GridMap Movement Functions

Basic movement operations for entities in the grid.
All functions use direct mutation and are executed via CallableRegistry for automatic versioning.
"""

from typing import Tuple, Optional, List
import random
from abstractions.ecs.callable_registry import CallableRegistry
from abstractions.ecs.entity import EntityRegistry
from game_entities import GridMap, GridNode, GameEntity, Agent, Path, Apple


def get_latest_gridmap(grid_map: GridMap) -> GridMap:
    """Get the latest version of a GridMap from its lineage.
    
    Args:
        grid_map: Any version of the GridMap
        
    Returns:
        Latest version in the lineage
    """
    # Get all versions in this lineage
    lineage_id = grid_map.lineage_id
    root_ecs_ids = EntityRegistry.lineage_registry.get(lineage_id, [])
    
    if not root_ecs_ids:
        return grid_map
    
    # Get the latest (last) version
    latest_root_ecs_id = root_ecs_ids[-1]
    latest_tree = EntityRegistry.get_stored_tree(latest_root_ecs_id)
    
    if latest_tree:
        return latest_tree.get_entity(latest_root_ecs_id)
    
    return grid_map


def get_latest_agent(agent: Agent) -> Agent:
    """Get the latest version of an Agent from the grid tree.
    
    Since agents are part of the grid tree (not separate roots),
    this function is now redundant - just use the agent from the latest grid.
    
    Args:
        agent: Any version of the Agent (not used, kept for compatibility)
        
    Returns:
        The same agent (caller should use agent from latest grid instead)
    """
    # Agent is part of grid tree, not a separate root
    # The caller should get the agent from get_latest_gridmap result
    # This function is kept for backward compatibility but does nothing
    return agent


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def chebyshev_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Chebyshev distance (8-directional, max of x/y diff)."""
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))


def get_node_at(grid_map: GridMap, position: Tuple[int, int]) -> Optional[GridNode]:
    """Find the node at the given position.
    
    Args:
        grid_map: The grid to search
        position: (x, y) coordinates
        
    Returns:
        GridNode if found, None otherwise
    """
    return next((node for node in grid_map.nodes if node.position == position), None)


def find_entity_and_node(grid_map: GridMap, entity_name: str) -> Tuple[Optional[GameEntity], Optional[GridNode]]:
    """Find an entity by name and return both the entity and its containing node.
    
    Args:
        grid_map: The grid to search
        entity_name: Name of the entity to find
        
    Returns:
        Tuple of (entity, node) or (None, None) if not found
    """
    for node in grid_map.nodes:
        for entity in node.entities:
            if entity.name == entity_name:
                return entity, node
    return None, None


def find_agent_position(grid_map: GridMap, agent: Agent) -> Optional[Tuple[int, int]]:
    """Find the position of an agent in the grid.
    
    Args:
        grid_map: The grid to search
        agent: Agent to find (searches by name since ecs_id changes with versioning)
        
    Returns:
        Position tuple or None if not found
    """
    for node in grid_map.nodes:
        for entity in node.entities:
            if isinstance(entity, Agent) and entity.name == agent.name:
                return node.position
    return None


def find_apple_at_position(grid_map: GridMap, position: Tuple[int, int]) -> Optional[Apple]:
    """Find an apple at the given position.
    
    Args:
        grid_map: The grid to search
        position: Position to check
        
    Returns:
        Apple if found, None otherwise
    """
    node = get_node_at(grid_map, position)
    if not node:
        return None
    
    for entity in node.entities:
        if isinstance(entity, Apple):
            return entity
    return None


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


@CallableRegistry.register("move_entity_one_step")
def move_entity_one_step(grid_map: GridMap, entity_name: str, target_pos: Tuple[int, int]) -> GridMap:
    """Move an entity one step to an adjacent position.
    
    Movement rules:
    - Target must be within distance 1 (Chebyshev distance, allows diagonals)
    - Target node must exist
    - Target node must be walkable (all entities in it must be walkable)
    - Uses direct mutation, framework handles versioning
    
    Args:
        grid_map: The grid containing the entity
        entity_name: Name of the entity to move
        target_pos: Target (x, y) position
        
    Returns:
        Updated GridMap (new version if move succeeded, original if failed)
    """
    # Find entity and current node
    entity, current_node = find_entity_and_node(grid_map, entity_name)
    
    if not entity or not current_node:
        print(f"Entity '{entity_name}' not found")
        return grid_map
    
    # Check distance (Chebyshev distance = 1 for adjacent cells including diagonals)
    distance = chebyshev_distance(current_node.position, target_pos)
    if distance != 1:
        print(f"Target {target_pos} is not adjacent to current position {current_node.position} (distance={distance})")
        return grid_map
    
    # Find target node
    target_node = get_node_at(grid_map, target_pos)
    if not target_node:
        print(f"No node exists at position {target_pos}")
        return grid_map
    
    # Check if target is walkable
    if not is_node_walkable(target_node):
        print(f"Target position {target_pos} is not walkable")
        return grid_map
    
    # Perform the move (direct mutation)
    print(f"Moving '{entity_name}' from {current_node.position} to {target_pos}")
    
    # 1. Remove from current node
    current_node.entities.remove(entity)
    
    # 2. Add to target node
    target_node.entities.append(entity)
    
    # Return mutated grid (framework creates new version automatically)
    return grid_map


@CallableRegistry.register("spawn_entity")
def spawn_entity(grid_map: GridMap, entity: GameEntity, position: Tuple[int, int]) -> GridMap:
    """Spawn an entity at the given position.
    
    Args:
        grid_map: The grid to spawn in
        entity: The entity to spawn (already created)
        position: Where to spawn the entity
        
    Returns:
        Updated GridMap with entity added
    """
    node = get_node_at(grid_map, position)
    
    if not node:
        print(f"No node exists at position {position}")
        return grid_map
    
    print(f"Spawning '{entity.name}' at {position}")
    
    # Add entity to node (direct mutation)
    node.entities.append(entity)
    
    return grid_map


@CallableRegistry.register("despawn_entity")
def despawn_entity(grid_map: GridMap, entity_name: str) -> GridMap:
    """Remove an entity from the grid.
    
    Args:
        grid_map: The grid containing the entity
        entity_name: Name of the entity to remove
        
    Returns:
        Updated GridMap with entity removed
    """
    entity, node = find_entity_and_node(grid_map, entity_name)
    
    if not entity or not node:
        print(f"Entity '{entity_name}' not found")
        return grid_map
    
    print(f"Despawning '{entity_name}' from {node.position}")
    
    # Remove from node (direct mutation)
    node.entities.remove(entity)
    
    # Detach from tree (entity becomes root)
    entity.detach()
    
    return grid_map


@CallableRegistry.register("move_agent_along_path")
def move_agent_along_path(
    grid_map: GridMap,
    agent: Agent,
    path: Path
) -> GridMap:
    """
    Move agent one step along the given path.
    
    Args:
        grid_map: Current grid state
        agent: Agent to move
        path: Path to follow (will move to steps[1] if length > 1)
        
    Returns:
        Updated GridMap with agent moved
    """
    # Get next position from path
    if len(path.steps) < 2:
        print(f"Path too short to move (length={path.length})")
        return grid_map
    
    current_pos = path.steps[0]
    next_pos = path.steps[1]
    
    # Find agent in grid
    current_node = None
    for node in grid_map.nodes:
        if node.position == current_pos:
            if agent in node.entities:
                current_node = node
                break
    
    if not current_node:
        print(f"Agent not found at expected position {current_pos}")
        return grid_map
    
    # Find target node
    target_node = get_node_at(grid_map, next_pos)
    if not target_node:
        print(f"No node exists at position {next_pos}")
        return grid_map
    
    # Check if target is walkable
    if not is_node_walkable(target_node):
        print(f"Target position {next_pos} is not walkable")
        return grid_map
    
    # Perform the move
    print(f"Moving agent from {current_pos} to {next_pos}")
    
    # Remove from current node
    current_node.entities.remove(agent)
    
    # Add to target node
    target_node.entities.append(agent)
    
    return grid_map


@CallableRegistry.register("collect_apple")
def collect_apple(
    grid_map: GridMap,
    agent: Agent,
    apple_position: Tuple[int, int]
) -> GridMap:
    """
    Collect apple from grid into agent's inventory.
    
    Args:
        grid_map: Current grid state
        agent: Agent collecting the apple (used to identify by name)
        apple_position: Position of the apple
        
    Returns:
        Updated GridMap with apple moved to inventory
    """
    # Find apple at position
    apple = find_apple_at_position(grid_map, apple_position)
    
    if not apple:
        print(f"No apple found at {apple_position}")
        return grid_map
    
    # Find node containing apple
    node = get_node_at(grid_map, apple_position)
    if not node:
        print(f"No node at {apple_position}")
        return grid_map
    
    # Find the agent IN THE GRID TREE (not the parameter copy)
    agent_in_tree = None
    for n in grid_map.nodes:
        for entity in n.entities:
            if isinstance(entity, Agent) and entity.name == agent.name:
                agent_in_tree = entity
                break
        if agent_in_tree:
            break
    
    if not agent_in_tree:
        print(f"Agent '{agent.name}' not found in grid")
        return grid_map
    
    print(f"Collecting apple '{apple.name}' at {apple_position}")
    
    # Remove apple from node
    node.entities.remove(apple)
    
    # Add apple to the agent that's IN THE TREE
    agent_in_tree.inventory.append(apple)
    
    print(f"   Agent inventory: {len(agent_in_tree.inventory)} apples")
    
    return grid_map


def apple_name_generator():
    """Generator for unique apple names."""
    count = 1
    while True:
        yield f"apple_{count}"
        count += 1

# Global generator instance
_apple_namer = apple_name_generator()

@CallableRegistry.register("spawn_random_apple")
def spawn_random_apple(grid_map: GridMap) -> GridMap:
    """
    Spawn a new apple at a random walkable position.
    
    Args:
        grid_map: Current grid state
        
    Returns:
        Updated GridMap with new apple
    """
    # Find available positions (walkable, no apple, no agent)
    available_positions = []
    
    for node in grid_map.nodes:
        # Check if walkable
        if not is_node_walkable(node):
            continue
        
        # Check if no apple or agent already there
        has_apple = any(isinstance(e, Apple) for e in node.entities)
        has_agent = any(isinstance(e, Agent) for e in node.entities)
        
        if not has_apple and not has_agent:
            available_positions.append(node.position)
    
    if not available_positions:
        print("No available positions to spawn apple")
        return grid_map
    
    # Choose random position
    spawn_pos = random.choice(available_positions)
    
    # Create new apple with unique name from generator
    apple = Apple(name=next(_apple_namer), nutrition=10)
    
    # Add to grid
    node = get_node_at(grid_map, spawn_pos)
    node.entities.append(apple)
    
    
    return grid_map
