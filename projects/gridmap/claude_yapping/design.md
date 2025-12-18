# GridMap Game Environment Design

## Overview
A comprehensive grid-based game environment leveraging the Abstractions framework's entity system for maximum flexibility and provenance tracking.

**Status**: ✅ All framework primitives tested and verified (22/22 tests passed)

## Core Philosophy

**Write normal Python → Execute via CallableRegistry → Get automatic versioning**

- Functions use direct mutation (normal Python)
- CallableRegistry.execute() handles versioning automatically
- No manual `model_copy()` or version management
- Complete game state as single EntityTree
- Atomic versioning of entire tree on each action

## Core Architecture

### 1. Root Structure: GridMap
```python
from typing import List, Dict, Any, Tuple
from abstractions.ecs.entity import Entity
from pydantic import Field

class GridMap(Entity):
    """Root entity containing the complete game state."""
    nodes: List[GridNode] = Field(default_factory=list)  # Flat list of all nodes
    width: int
    height: int
    metadata: Dict[str, Any] = Field(default_factory=dict)  # game state, turn counter, etc.
```

**Design Decision**: Store nodes as a **flat list**, not a dictionary:
- Position is a property of GridNode, not an index
- Spatial queries use linear scan (simple, works fine for small grids)
- Easy to iterate over all nodes
- No complex dict key management

### 2. GridNode
```python
class GridNode(Entity):
    """A single cell in the grid, containing entities at that position."""
    position: Tuple[int, int]  # (x, y) coordinates - property, not index
    entities: List[GameEntity] = Field(default_factory=list)  # Flat list of entities at this node
```

**Design Decision**: Store entities as a **flat list**:
- Position is a property of GameEntity
- Linear scan to find entities at a position
- Simple, no indexing complexity

**Key Design Decision**: Store **actual nested entities**, not UUIDs. This creates a complete hierarchical tree:
- **GridMap** is the root entity
- **GridNodes** are sub-entities of GridMap
- **GameEntities** are sub-entities of GridNode
- The entire game state is one **EntityTree** that versions atomically
- `build_entity_tree()` captures the complete state
- `find_modified_entities()` detects exactly what changed
- **Complete provenance**: Every game state transition is fully tracked

### 3. GameEntity Hierarchy

**Base class:**
```python
class GameEntity(Entity):
    """Base class for all entities that can exist in the grid."""
    name: str  # Identifier for the entity
    walkable: bool  # Can entities move through this?
    transparent: bool  # Does this block line of sight?
```

**Key Design Decision**: **No position field on GameEntity!**
- Entity's position is determined by **which GridNode contains it**
- GridNode has the position, not the entity
- To move an entity: remove from one node's list, add to another node's list
- This is the true hierarchical structure

**Concrete types:**
```python
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
    """Mobile agent with vision and movement capabilities."""
    walkable: bool = Field(default=True)  # Agents don't block movement
    transparent: bool = Field(default=True)  # Agents don't block sight
    speed: int  # Movement range per turn (tiles)
    sight: int  # Vision range (tiles)
```

**Verified**: ✅ Three-level hierarchy works (P2.3 passed), ✅ Move between nodes works (P8.1 passed)

## Key Capabilities

### Line of Sight (LOS)
- **Bresenham's line algorithm** between two points
- Check `transparent` property of entities in each GridNode along path
- **8-directional**: All cardinal + diagonal directions
- Agent's `sight` property determines max range

### Pathfinding
- **A* algorithm** using `walkable` property
- **8-directional movement**: Cost 1.0 for cardinal, 1.4 for diagonal
- Agent's `speed` property determines max path length per turn
- Respects terrain walkability

### Spatial Queries (Linear Scan)
- `get_node_at(grid_map, position)` → GridNode | None
  - Linear scan through `grid_map.nodes` to find node with matching position
- `get_entities_at(grid_map, position)` → List[GameEntity]
  - Find node at position, return its entities list
- `get_entities_in_radius(grid_map, position, radius)` → List[GameEntity]
  - Linear scan, check distance for each entity
- `get_neighbors(grid_map, position)` → List[GridNode]
  - Check 8 adjacent positions, linear scan for each

**Performance**: O(n) for most queries, but simple and correct. Optimize later if needed.

## Leveraging Abstractions Framework

### Immutability & Versioning
Every game action creates a new version:
```python
# Move entity → new GridMap version
new_map = move_entity(grid_map, entity_id, target_pos)
# grid_map unchanged, new_map has new ecs_id, same lineage_id
```

### Provenance Tracking
- Every grid state change is versioned
- Complete game replay from lineage history

### Callable Registry Functions

**Verified Pattern**: Write normal Python, execute via registry, get automatic versioning.

```python
from abstractions.ecs.callable_registry import CallableRegistry

# ✅ VERIFIED: Move between nodes pattern (P8.1 passed)
@CallableRegistry.register("move_entity")
def move_entity(grid_map: GridMap, entity_name: str, target_pos: Tuple[int, int]) -> GridMap:
    """Move entity to target position by moving between nodes."""
    # Find entity and current node
    entity = None
    current_node = None
    
    for node in grid_map.nodes:
        for e in node.entities:
            if e.name == entity_name:
                entity = e
                current_node = node
                break
        if entity:
            break
    
    if entity and current_node:
        target_node = next((n for n in grid_map.nodes if n.position == target_pos), None)
        
        if target_node:
            # Remove from current node (direct mutation)
            current_node.entities.remove(entity)
            
            # Add to target node (direct mutation)
            target_node.entities.append(entity)
    
    return grid_map

# ✅ VERIFIED: Functions return new entities (P4.3 passed)
@CallableRegistry.register("check_los")
def check_los(from_pos: Tuple[int, int], to_pos: Tuple[int, int], grid_map: GridMap) -> LOS:
    """Check line of sight between two positions. Returns LOS entity."""
    # Run Bresenham algorithm
    # Return LOS entity with results
    return LOS(start=from_pos, end=to_pos, visible=True, blocked_by=[])
    
# ✅ VERIFIED: Tuple returns (P4.4 passed)
@CallableRegistry.register("find_path")
def find_path(start: Tuple[int, int], goal: Tuple[int, int], grid_map: GridMap) -> Tuple[Path, PathMetadata]:
    """Find shortest walkable path using A*. Returns Path + metadata."""
    # Run A* algorithm
    path = Path(waypoints=[(0,0), (1,1)])
    metadata = PathMetadata(cost=2.0, nodes_explored=10)
    return path, metadata  # Tuple unpacked to list automatically
```

**Execution patterns**:
```python
# ✅ VERIFIED: Direct execution (P4.1 passed)
result = CallableRegistry.execute("move_agent", agent=my_agent, position=(5, 5))
updated_agent = result if not isinstance(result, list) else result[0]

# ✅ VERIFIED: Distributed addressing (P6.2, P6.3 passed)
result = CallableRegistry.execute("move_agent", 
                                  agent=f"@{agent.ecs_id}",  # Address string
                                  position=(5, 5))  # Direct value

# Framework automatically:
# 1. Resolves @uuid addresses to actual entities
# 2. Calls function with resolved values
# 3. Creates new version (different ecs_id, same lineage_id)
# 4. Preserves original entity unchanged
# 5. Tracks complete provenance
```

### Event-Driven Observation
```python
EntityMovedEvent(subject_id=entity_id, context={"from": old_pos, "to": new_pos})
EntitySpawnedEvent(subject_id=entity_id, context={"parent": parent_id})
EntityDespawnedEvent(subject_id=entity_id, context={"parent": parent_id})
LOSBlockedEvent(subject_id=observer_id, context={"target": target_id, "blockers": [...]})
PathNotFoundEvent(subject_id=entity_id, context={"goal": goal_pos})
```

## Design Decisions

### Core Principles

1. **Entities are pure data** - No computed fields, no cached properties, no magic
2. **Functions transform state** - All logic lives in registered functions
3. **Explicit over implicit** - Run computations when needed, don't hide them
4. **Complete state trees** - GridMap contains full nested hierarchy

### Confirmed Specifications

1. **Grid Size**: Start with 10x10, easily configurable

2. **Entity Stacking**: Yes - GridNode contains `List[GameEntity]`
   - Terrain (Wall/Floor/Water) + Agents can coexist
   - Each node typically has 1 terrain + 0-N agents

3. **Movement System**: 
   - ✅ Turn-based
   - ✅ 8-directional (cardinal + diagonal)
   - ✅ Agent.speed determines max movement per turn
   - ✅ Agent.sight determines vision range

4. **Type Safety**:
   - `move_agent(agent: Agent, ...)` only accepts Agent
   - Wall/Floor/Water cannot move (type system enforces this)

5. **No Computed Fields**:
   - GridNode does NOT have `is_walkable` field
   - Instead: `is_node_walkable(node: GridNode) -> bool` function
   - Computed when needed, not cached in entity

## Implementation Roadmap

### Phase 1: Core Entities (Start Here)
1. **Define entity classes** (`entities.py`)
   - `GameEntity` base class
   - `Wall`, `Floor`, `Water` terrain types
   - `Agent` with speed/sight

2. **Define grid structures** (`grid.py`)
   - `GridNode` with position + entities list
   - `GridMap` with 2D dict of nodes

3. **Grid initialization** (`functions.py`)
   - `create_empty_grid(width, height) -> GridMap`
   - `create_grid_with_terrain(width, height, terrain_map) -> GridMap`

### Phase 2: Basic Operations
4. **Spatial queries** (linear scan)
   - `get_node_at(grid_map, position) -> GridNode | None`
   - `get_entities_at(grid_map, position) -> List[GameEntity]`
   - `get_neighbors(grid_map, position) -> List[GridNode]`
   - `is_position_walkable(grid_map, position) -> bool`
   - `is_position_transparent(grid_map, position) -> bool`

5. **Entity movement**
   - `move_entity(entity: GameEntity, position: Tuple[int, int]) -> GameEntity`
   - `move_agent(agent: Agent, position: Tuple[int, int]) -> Agent`
   - `validate_move(entity, position, grid_map) -> bool`

### Phase 3: Advanced Features
6. **Line of Sight**
   - `LOS` entity (result of LOS check)
   - `compute_los(from_pos, to_pos, grid_map) -> LOS`
   - Bresenham's algorithm + transparency checks

7. **Pathfinding**
   - `Path` entity (result of pathfinding)
   - `find_path(start, goal, grid_map) -> Path`
   - A* algorithm respecting walkability

8. **Visualization**
   - `render_grid_ascii(grid_map) -> str`
   - Simple ASCII representation

### Phase 4: Examples & Testing
9. **Example scenarios**
   - Simple room with walls
   - Maze navigation
   - Multi-agent interaction

10. **RL Gym integration** (optional)
    - Gym-compatible wrapper
    - Observation/action spaces

## Philosophy

**Start minimal, expand organically.** 

### Minimal First Implementation:
- 10x10 grid
- 4 entity types: Wall, Floor, Water, Agent
- 8-directional movement
- Basic spatial queries
- Simple movement validation

### Then Add:
- Line of sight (Bresenham)
- Pathfinding (A*)
- Visualization
- Multi-agent scenarios
- RL integration

### Key Advantages:
- **Immutability**: Every game state is versioned
- **Provenance**: Complete history of all changes
- **Experimentation**: Fork states, try alternatives, compare
- **Debugging**: Time-travel through game history
- **Distribution**: Send entire game state as one entity tree

---

## Entity Manipulation Patterns

### The Core Pattern: Direct Mutation + Framework Notification

**You mutate Python objects directly, then call framework methods to update metadata.**

```python
# 1. Physically modify the Python object
grid_map.nodes.append(new_node)

# 2. Notify framework (only if needed)
new_node.attach(grid_map)  # Only if new_node was previously a root entity
```

### Three Scenarios for GridMap

#### Scenario A: Create Entity In Place (Most Common)
```python
# ✅ VERIFIED: Append to list works (P5.1 passed, P8.2 passed)
@CallableRegistry.register("spawn_wall")
def spawn_wall(grid_map: GridMap, name: str, position: Tuple[int, int]) -> GridMap:
    """Spawn a wall at the given position."""
    node = next((n for n in grid_map.nodes if n.position == position), None)
    
    if node:
        # Create new entity directly (no position field!)
        wall = Wall(name=name, walkable=False, transparent=False)
        
        # Add to node's entity list (direct mutation - just normal Python!)
        node.entities.append(wall)
    
    # Return mutated grid (framework handles versioning automatically)
    return grid_map
```

**No attach/detach needed** - entity created as part of the tree.
**Verified**: ✅ List append works (P5.1 passed), ✅ Multiple entities per node (P8.2 passed)

#### Scenario B: Move Within Same Tree
```python
# ✅ VERIFIED: Move between nodes works (P8.1 passed, P5.3 passed)
@CallableRegistry.register("move_agent_between_nodes")
def move_agent_between_nodes(grid_map: GridMap, agent_name: str, target_pos: Tuple[int, int]) -> GridMap:
    """Move agent from one node to another within the same grid."""
    # Find agent and current node
    agent = None
    current_node = None
    
    for node in grid_map.nodes:
        for entity in node.entities:
            if entity.name == agent_name:
                agent = entity
                current_node = node
                break
        if agent:
            break
    
    if agent and current_node:
        target_node = next((n for n in grid_map.nodes if n.position == target_pos), None)
        
        if target_node:
            # 1. Remove from current node (direct mutation)
            current_node.entities.remove(agent)
            
            # 2. Add to target node (direct mutation)
            target_node.entities.append(agent)
            
            # No detach/attach needed - staying within same tree
            # Position is implicit: agent is now in target_node which has target_pos
    
    return grid_map
```

**No attach/detach needed** - moving within same tree.
**Verified**: ✅ Move between nodes works (P8.1 passed), ✅ Move between lists works (P5.3 passed)

#### Scenario C: Remove From Tree
```python
# ✅ VERIFIED: Remove from list works (P5.2 passed), detach works (P7.1 passed)
@CallableRegistry.register("despawn_entity")
def despawn_entity(grid_map: GridMap, entity_id: str) -> GridMap:
    """Remove entity from grid and detach it."""
    for node in grid_map.nodes:
        for entity in node.entities:
            if str(entity.ecs_id) == entity_id:
                # 1. Physical removal (direct mutation)
                node.entities.remove(entity)
                
                # 2. Notify framework (entity becomes root)
                entity.detach()
                
                return grid_map
    
    return grid_map
```

**detach() required** - entity leaves the tree, becomes root.
**Verified**: ✅ Remove from list works (P5.2 passed), ✅ detach() works (P7.1 passed)

### Framework Methods

#### `promote_to_root()`
Makes an entity a root entity. Required before using with CallableRegistry.

```python
# ✅ VERIFIED: P1.2 passed
entity = GameEntity(position=(0, 0), walkable=True, transparent=True)
entity.promote_to_root()  # Now addressable via @{ecs_id}
assert entity.is_root_entity() == True
```

#### `attach(new_root_entity)`
Attaches a root entity to a new parent tree. Only for root entities.

```python
# ✅ VERIFIED: P7.2 passed
# Agent is a root entity
agent.promote_to_root()

# Add to grid physically
node.entities.append(agent)

# Notify framework
agent.attach(grid_map)

# Agent is now part of grid's tree
assert agent.root_ecs_id == grid_map.ecs_id
```

#### `detach()`
Removes entity from parent tree. Call AFTER physical removal.

```python
# ✅ VERIFIED: P7.1 passed
# Remove physically
node.entities.remove(entity)

# Notify framework
entity.detach()  # Entity becomes root
assert entity.is_root_entity() == True
```

### Boundary Conditions

The framework handles:
- ✅ Empty collections (valid)
- ✅ Non-existent lookups (safe - loop completes)
- ✅ Moving within tree (no special handling)
- ✅ Moving between trees (attach/detach required)
- ✅ Duplicate prevention (via ecs_id tracking)

### GridMap-Specific Rules

Since GridMap is a **single entity tree**:
1. **Spawning entities** → Create in place, append to list (Scenario A)
2. **Moving entities** → List operations within same tree (Scenario B)
3. **Despawning** → Remove from list + detach() (Scenario C)
4. **No attach() needed** unless importing entities from outside

---

## Critical Framework Patterns Summary

### 1. Entity Mutation
```python
# ❌ WRONG
return entity.model_copy(update={"position": position})

# ✅ CORRECT
entity.position = position  # Direct mutation
return entity
```

### 2. Function Registration
```python
from abstractions.ecs.callable_registry import CallableRegistry
from pydantic import Field

@CallableRegistry.register("function_name")
def function_name(entity: Entity, param: type) -> Entity:
    entity.field = new_value  # Direct mutation
    return entity
```

### 3. Pydantic Field Usage
```python
# Simple required fields
position: Tuple[int, int]
speed: int

# Fields with scalar defaults
walkable: bool = Field(default=False)
transparent: bool = Field(default=True)

# Fields with collection defaults
nodes: List[GridNode] = Field(default_factory=list)
entities: List[GameEntity] = Field(default_factory=list)
metadata: Dict[str, Any] = Field(default_factory=dict)
```

### 4. Execute Return Handling
```python
result = CallableRegistry.execute("function_name", entity=entity)

# Handle Union[Entity, List[Entity]]
if isinstance(result, list):
    entity = result[0]  # Tuple returns come as list
else:
    entity = result
```

### 5. Distributed Addressing
```python
from abstractions.ecs.functional_api import get

# Access via @uuid.field
name = get(f"@{entity.ecs_id}.name")

# Functions accept addresses
result = CallableRegistry.execute("process",
    name=f"@{entity.ecs_id}.name",  # Address
    value=42                          # Direct value
)
```

---

## Test Results Summary

### All Primitives Verified ✅ (25/25 tests passed)

**Phase 1: Basics**
- ✅ P1.1: Create simple entity
- ✅ P1.2: Promote to root
- ✅ P1.3: Entity with collections
- ✅ P2.1: Nested entity structure
- ✅ P2.2: Build entity tree
- ✅ P2.3: Three-level hierarchy (GridMap depth!)
- ✅ P3.1: Direct field mutation
- ✅ P3.2: Mutation in function creates version
- ✅ P3.3: List mutation in tree

**Phase 2: Advanced**
- ✅ P4.1: Register simple function
- ✅ P4.2: Function with multiple parameters
- ✅ P4.3: Function returning new entity
- ✅ P4.4: Function returning tuple
- ✅ P5.1: Append to list in tree
- ✅ P5.2: Remove from list in tree
- ✅ P5.3: Move item between lists
- ✅ P6.1: Access field via @uuid.field
- ✅ P6.2: Function with address parameter
- ✅ P6.3: Mixed addresses and values
- ✅ P7.1: Detach entity from tree
- ✅ P7.2: Attach root entity to tree
- ✅ P7.3: Versioning through CallableRegistry

**Phase 3: GridMap-Specific**
- ✅ P8.1: Move entity between nodes (core GridMap operation!)
- ✅ P8.2: Multiple entities in same node
- ✅ P8.3: Entity position consistency

### Key Findings

1. **Direct mutation works** - Just write normal Python
2. **CallableRegistry.execute() handles versioning** - Automatic, no manual work
3. **Three-level hierarchies work** - Perfect for GridMap → GridNode → GameEntity
4. **List operations work** - append, remove, move within trees
5. **Distributed addressing works** - @uuid.field pattern fully functional
6. **attach/detach work** - For moving entities between trees
7. **Tuple returns work** - Multiple entities from one function
8. **Position is implicit** - Entity position = which node contains it (no position field needed!)

### Critical Pattern Confirmed

```python
# Write normal Python
@CallableRegistry.register("function_name")
def function_name(entity: Entity, param: type) -> Entity:
    entity.field = new_value  # Direct mutation
    return entity

# Execute via registry
result = CallableRegistry.execute("function_name", entity=entity, param=value)

# Framework automatically:
# ✅ Creates new version (different ecs_id)
# ✅ Preserves original (unchanged)
# ✅ Tracks lineage (same lineage_id)
# ✅ Records provenance (complete history)
```

---

## Ready for Implementation

All framework primitives verified. Next steps:
1. Create entity classes (`entities.py`)
2. Create grid initialization functions
3. Implement spatial queries
4. Implement movement functions
5. Add LOS and pathfinding
6. Create visualization
7. Build example scenarios
