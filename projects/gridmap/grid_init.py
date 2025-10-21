"""
GridMap Initialization Utilities

Functions to create and initialize grids with various configurations.
"""

from typing import Tuple
from game_entities import GridMap, GridNode, Floor, Wall, Water, Agent


def create_empty_grid(width: int, height: int) -> GridMap:
    """Create an empty grid with no entities.
    
    Args:
        width: Grid width
        height: Grid height
        
    Returns:
        GridMap with empty nodes at all positions
    """
    grid_map = GridMap(nodes=[], width=width, height=height)
    
    # Create nodes for all positions
    for y in range(height):
        for x in range(width):
            node = GridNode(position=(x, y), entities=[])
            grid_map.nodes.append(node)
    
    return grid_map


def create_floor_grid(width: int, height: int) -> GridMap:
    """Create a grid with floor tiles at all positions.
    
    Args:
        width: Grid width
        height: Grid height
        
    Returns:
        GridMap with floor entities at all positions
    """
    grid_map = GridMap(nodes=[], width=width, height=height)
    
    # Create nodes with floor tiles
    for y in range(height):
        for x in range(width):
            floor = Floor(name=f"floor_{x}_{y}", walkable=True, transparent=True)
            node = GridNode(position=(x, y), entities=[floor])
            grid_map.nodes.append(node)
    
    return grid_map


def create_walled_grid(width: int, height: int) -> GridMap:
    """Create a grid with floor tiles and walls around the perimeter.
    
    Args:
        width: Grid width
        height: Grid height
        
    Returns:
        GridMap with floor inside and walls on edges
    """
    grid_map = GridMap(nodes=[], width=width, height=height)
    
    for y in range(height):
        for x in range(width):
            # Check if on perimeter
            is_edge = x == 0 or x == width - 1 or y == 0 or y == height - 1
            
            if is_edge:
                # Create wall
                wall = Wall(name=f"wall_{x}_{y}", walkable=False, transparent=False)
                node = GridNode(position=(x, y), entities=[wall])
            else:
                # Create floor
                floor = Floor(name=f"floor_{x}_{y}", walkable=True, transparent=True)
                node = GridNode(position=(x, y), entities=[floor])
            
            grid_map.nodes.append(node)
    
    return grid_map


def add_agent_to_grid(grid_map: GridMap, agent_name: str, position: Tuple[int, int], 
                      speed: int = 1, sight: int = 5) -> GridMap:
    """Add an agent to the grid at the specified position.
    
    Note: This is a helper for initial setup, not a registered function.
    For runtime spawning, use the spawn_entity function via CallableRegistry.
    
    Args:
        grid_map: The grid to add the agent to
        agent_name: Name for the agent
        position: Where to place the agent
        speed: Movement speed (default 1)
        sight: Vision range (default 5)
        
    Returns:
        GridMap with agent added
    """
    # Find the node
    node = next((n for n in grid_map.nodes if n.position == position), None)
    
    if not node:
        print(f"Warning: No node at position {position}")
        return grid_map
    
    # Create and add agent
    agent = Agent(name=agent_name, walkable=True, transparent=True, speed=speed, sight=sight)
    node.entities.append(agent)
    
    return grid_map


def print_grid(grid_map: GridMap, show_entities: bool = True) -> None:
    """Print a text representation of the grid.
    
    Visualization priority (highest to lowest):
    - Agent: @
    - Wall: #
    - Water: ~
    - Floor: .
    
    Args:
        grid_map: The grid to print
        show_entities: If True, show entity symbols; if False, show coordinates
    """
    print(f"\nGrid {grid_map.width}x{grid_map.height}:")
    print(f"ECS ID: {grid_map.ecs_id}")
    
    for y in range(grid_map.height):
        row = []
        for x in range(grid_map.width):
            node = next((n for n in grid_map.nodes if n.position == (x, y)), None)
            
            if not node or not node.entities:
                row.append(".")
            elif show_entities:
                # Priority-based visualization: Agent > Wall > Water > Floor
                symbol = "."  # Default
                
                for entity in node.entities:
                    if isinstance(entity, Agent):
                        symbol = "@"
                        break  # Highest priority, stop searching
                    elif isinstance(entity, Wall):
                        symbol = "#"
                        # Don't break, keep looking for agents
                    elif isinstance(entity, Water):
                        if symbol == ".":  # Only if we haven't found wall
                            symbol = "~"
                    elif isinstance(entity, Floor):
                        if symbol == ".":  # Only if we haven't found anything else
                            symbol = "."
                
                row.append(symbol)
            else:
                row.append(f"{x},{y}")
        
        print(" ".join(row))
    
    print()
