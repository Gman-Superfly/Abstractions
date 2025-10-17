"""
GridMap Entity Definitions

Core entity classes for the grid-based game environment.
All entities follow the verified pattern: no position field, position is implicit from node containment.
"""

from typing import List, Dict, Any, Tuple
from abstractions.ecs.entity import Entity
from pydantic import Field
from uuid import UUID


class GameEntity(Entity):
    """Base class for all entities that can exist in a grid node.
    
    Key design: No position field! Position is determined by which GridNode contains this entity.
    """
    name: str  # Identifier for the entity
    walkable: bool  # Can entities move through this?
    transparent: bool  # Does this block line of sight?


class Wall(GameEntity):
    """Solid wall - blocks movement and sight."""
    walkable: bool = Field(default=False)
    transparent: bool = Field(default=False)


class Floor(GameEntity):
    """Open floor - allows movement and sight."""
    walkable: bool = Field(default=True)
    transparent: bool = Field(default=True)


class Water(GameEntity):
    """Water terrain - blocks movement but allows sight."""
    walkable: bool = Field(default=False)
    transparent: bool = Field(default=True)


class Agent(GameEntity):
    """Mobile agent with vision and movement capabilities.
    
    Agents don't block movement or sight (can stack with terrain).
    """
    walkable: bool = Field(default=True)
    transparent: bool = Field(default=True)
    speed: int = Field(default=1)  # Movement range per turn (tiles)
    sight: int = Field(default=5)  # Vision range (tiles)
    inventory: List['Apple'] = Field(default_factory=list)  # Collected apples


class Apple(GameEntity):
    """Collectible item that agents seek.
    
    Apples don't block movement or sight.
    """
    walkable: bool = Field(default=True)
    transparent: bool = Field(default=True)
    nutrition: int = Field(default=10)  # Value when collected


class GridNode(Entity):
    """A single cell in the grid, containing entities at that position.
    
    The node's position determines the position of all entities it contains.
    """
    position: Tuple[int, int]  # (x, y) coordinates
    entities: List[GameEntity] = Field(default_factory=list)  # All entities at this position


class GridMap(Entity):
    """Root entity containing the complete game state.
    
    The entire game state is one EntityTree that versions atomically.
    """
    nodes: List[GridNode] = Field(default_factory=list)  # Flat list of all nodes
    width: int
    height: int
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Game state, turn counter, etc.


# Navigation and pathfinding entities

class NavigationGraph(Entity):
    """Precomputed navigation graph with walkability and visibility data.
    
    Derived from GridMap via compute_navigation_graph().
    Framework automatically tracks:
    - derived_from_function = "compute_navigation_graph"
    - derived_from_execution_id = <UUID>
    
    Check staleness: nav_graph.source_grid_id != current_grid.ecs_id
    """
    source_grid_id: UUID  # Which GridMap version this was computed from
    
    # Walkability data for pathfinding
    walkable_adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = Field(default_factory=dict)
    walkable: Dict[Tuple[int, int], bool] = Field(default_factory=dict)
    
    # Visibility data for line of sight
    transparent_adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = Field(default_factory=dict)
    transparent: Dict[Tuple[int, int], bool] = Field(default_factory=dict)


class Path(Entity):
    """A single path from start to destination.
    
    Represents one specific route through the grid.
    """
    start_position: Tuple[int, int]
    end_position: Tuple[int, int]
    steps: List[Tuple[int, int]] = Field(default_factory=list)  # Ordered positions including start and end
    length: int = 0  # Number of steps (excluding start)
    cost: int = 0  # Movement cost (for now, same as length)


class PathCollection(Entity):
    """Collection of all reachable paths from an agent's position.
    
    Derived from NavigationGraph via compute_reachable_paths().
    Framework automatically tracks:
    - derived_from_function = "compute_reachable_paths"
    - derived_from_execution_id = <UUID>
    
    Check staleness: path_collection.source_graph_id != current_nav_graph.ecs_id
    """
    agent_id: UUID  # Which agent these paths are for
    agent_position: Tuple[int, int]  # Starting position
    max_distance: int  # Speed limit (from agent.speed)
    paths: List[Path] = Field(default_factory=list)  # All reachable paths
    reachable_positions: List[Tuple[int, int]] = Field(default_factory=list)  # Unique destinations
    
    # Manual tracking for staleness detection
    source_graph_id: UUID  # Which NavigationGraph was used


class VisibleArea(Entity):
    """Result of visibility/shadowcasting computation.
    
    Framework automatically tracks derivation from compute_visible_positions().
    """
    origin: Tuple[int, int]
    max_range: int
    visible_positions: List[Tuple[int, int]] = Field(default_factory=list)
