# Pathfinding & Visibility Design

## Problem Statement

We need intermediate representations for:
1. **Graph adjacency** - Which nodes connect to which (8-directional)
2. **Walkability state** - Which connections are traversable
3. **Visibility state** - Which connections allow line of sight

These should be **entities** that track causal dependencies from the base GridMap.

## Key Questions

### 1. What entities do we need?

#### Option A: Single Graph Entity
```python
class GridGraph(Entity):
    """Computed graph representation of the grid."""
    grid_id: UUID  # Which GridMap this was computed from
    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]]  # pos -> neighbors
    walkable_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]]  # (from, to) pairs
    transparent_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]]  # (from, to) pairs
```

**Pros**: Single entity, simple
**Cons**: Large dict/set fields, might be hard to version efficiently

#### Option B: Per-Node Adjacency Entities
```python
class NodeAdjacency(Entity):
    """Adjacency info for a single node."""
    position: Tuple[int, int]
    walkable_neighbors: List[Tuple[int, int]]  # 8-directional neighbors that are walkable
    visible_neighbors: List[Tuple[int, int]]   # 8-directional neighbors that are transparent
```

Then have a container:
```python
class GridGraph(Entity):
    """Graph representation of the grid."""
    grid_id: UUID
    node_adjacencies: List[NodeAdjacency]  # One per grid position
```

**Pros**: Granular, each node's adjacency is separate entity
**Cons**: More entities, more complex

#### Option C: Separate Walkability and Visibility Graphs
```python
class WalkabilityGraph(Entity):
    """Walkable adjacency graph."""
    grid_id: UUID
    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]]  # pos -> walkable neighbors

class VisibilityGraph(Entity):
    """Transparent adjacency graph."""
    grid_id: UUID
    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]]  # pos -> visible neighbors
```

**Pros**: Separation of concerns
**Cons**: Two entities to manage

### 2. How do we compute these from GridMap?

```python
@CallableRegistry.register("compute_walkability_graph")
def compute_walkability_graph(grid_map: GridMap) -> WalkabilityGraph:
    """Compute which nodes are walkable from which positions.
    
    For each position, check all 8 neighbors:
    - If neighbor exists and is_node_walkable(neighbor) -> add to adjacency
    
    Returns new WalkabilityGraph entity.
    """
    adjacency = {}
    
    for node in grid_map.nodes:
        pos = node.position
        neighbors = []
        
        # Check all 8 directions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                neighbor_pos = (pos[0] + dx, pos[1] + dy)
                neighbor_node = get_node_at(grid_map, neighbor_pos)
                
                if neighbor_node and is_node_walkable(neighbor_node):
                    neighbors.append(neighbor_pos)
        
        adjacency[pos] = neighbors
    
    return WalkabilityGraph(
        grid_id=grid_map.ecs_id,
        adjacency=adjacency
    )
```

### 3. Causal Dependency Tracking

**Key insight**: The graph is **derived from** the GridMap.

```python
# Compute graph from grid
graph = CallableRegistry.execute("compute_walkability_graph", grid_map=grid_map)

# graph.derived_from_function = "compute_walkability_graph"
# graph.derived_from_execution_id = <execution_id>
# Framework tracks: graph was created from grid_map
```

**When grid changes**:
- Grid gets new ecs_id (new version)
- Graph becomes stale (its grid_id points to old version)
- Need to recompute graph from new grid version

**Question**: Do we need automatic invalidation tracking?
- Framework already tracks derivation via `derived_from_function`
- We can check if `graph.grid_id == current_grid.ecs_id`
- If not equal, graph is stale

### 4. Using the Graph for Pathfinding

```python
@CallableRegistry.register("find_path_astar")
def find_path_astar(
    start: Tuple[int, int], 
    goal: Tuple[int, int], 
    graph: WalkabilityGraph
) -> Path:
    """Find shortest path using A* on the precomputed graph.
    
    Args:
        start: Starting position
        goal: Goal position
        graph: Precomputed walkability graph
        
    Returns:
        Path entity with waypoints
    """
    # Standard A* using graph.adjacency
    # No need to check walkability - already in graph!
    
    # ... A* implementation ...
    
    return Path(
        start=start,
        goal=goal,
        waypoints=waypoints,
        cost=total_cost
    )
```

### 5. Do We Need New Primitives?

**Check against verified primitives**:

✅ **P4.3**: Function returning new entity - `compute_walkability_graph` returns `WalkabilityGraph`
✅ **P4.4**: Function returning tuple - Could return `(WalkabilityGraph, VisibilityGraph)`
✅ **P6.2**: Function with address parameter - Could pass `grid_map=f"@{grid_id}"`
✅ **Dicts/Sets in entities** - Need to verify if Dict/Set fields work in entities

**Potential new primitive needed**:
- **P9: Dict/Set fields in entities** - Can we have `Dict[Tuple[int, int], List[Tuple[int, int]]]` as a field?

Let me check existing entity examples...

## Proposed Architecture

### Entities

```python
class NavigationGraph(Entity):
    """Precomputed navigation graph with both walkability and visibility.
    
    Automatically tracks derivation:
    - derived_from_function = "compute_navigation_graph"
    - derived_from_execution_id = <execution UUID>
    
    Can check if stale by comparing source_grid_id with current grid.ecs_id
    """
    source_grid_id: UUID  # Which GridMap version this was computed from
    
    # Walkability data
    walkable_adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = Field(default_factory=dict)
    walkable: Dict[Tuple[int, int], bool] = Field(default_factory=dict)
    
    # Visibility data  
    transparent_adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = Field(default_factory=dict)
    transparent: Dict[Tuple[int, int], bool] = Field(default_factory=dict)

class Path(Entity):
    """Result of pathfinding.
    
    Automatically tracks:
    - derived_from_function = "find_path_astar"
    - derived_from_execution_id = <execution UUID>
    """
    start: Tuple[int, int]
    goal: Tuple[int, int]
    waypoints: List[Tuple[int, int]]  # Ordered list of positions
    cost: float  # Total path cost
    found: bool  # Whether path exists

class VisibleArea(Entity):
    """Result of visibility computation.
    
    Automatically tracks derivation from shadowcasting function.
    """
    origin: Tuple[int, int]
    max_range: int
    visible_positions: List[Tuple[int, int]]  # All visible positions
```

### Functions

```python
@CallableRegistry.register("compute_navigation_graph")
def compute_navigation_graph(grid_map: GridMap) -> NavigationGraph:
    """Compute both walkability and visibility adjacency from grid state.
    
    Framework automatically sets:
    - result.derived_from_function = "compute_navigation_graph"
    - result.derived_from_execution_id = <UUID>
    
    This creates a causal link: NavigationGraph was derived from GridMap
    """
    walkable_adj = {}
    walkable_state = {}
    transparent_adj = {}
    transparent_state = {}
    
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
                    continue
                
                neighbor_pos = (pos[0] + dx, pos[1] + dy)
                neighbor_node = get_node_at(grid_map, neighbor_pos)
                
                if neighbor_node:
                    if is_node_walkable(neighbor_node):
                        walkable_neighbors.append(neighbor_pos)
                    if is_node_transparent(neighbor_node):
                        transparent_neighbors.append(neighbor_pos)
        
        walkable_adj[pos] = walkable_neighbors
        transparent_adj[pos] = transparent_neighbors
    
    return NavigationGraph(
        source_grid_id=grid_map.ecs_id,
        walkable_adjacency=walkable_adj,
        walkable=walkable_state,
        transparent_adjacency=transparent_adj,
        transparent=transparent_state
    )

@CallableRegistry.register("find_path_astar")
def find_path_astar(start: Tuple[int, int], goal: Tuple[int, int], 
                    nav_graph: NavigationGraph) -> Path:
    """A* pathfinding on precomputed graph.
    
    Framework automatically sets:
    - result.derived_from_function = "find_path_astar"
    - result.derived_from_execution_id = <UUID>
    
    This creates a causal link: Path was derived from NavigationGraph
    """
    # Standard A* using nav_graph.walkable_adjacency
    # ... implementation ...
    
    return Path(
        start=start,
        goal=goal,
        waypoints=waypoints,
        cost=total_cost,
        found=True
    )

@CallableRegistry.register("compute_visible_positions")
def compute_visible_positions(origin: Tuple[int, int], max_range: int,
                             nav_graph: NavigationGraph) -> VisibleArea:
    """Shadowcasting on precomputed graph.
    
    Framework automatically tracks derivation.
    """
    # Shadowcasting using nav_graph.transparent_adjacency
    # ... implementation ...
    
    return VisibleArea(
        origin=origin,
        max_range=max_range,
        visible_positions=visible_list
    )
```

### Workflow with Automatic Derivation Tracking

```python
# 1. Create grid
grid = create_floor_grid(10, 10)
grid = add_agent_to_grid(grid, "agent1", (5, 5))
grid.promote_to_root()

# 2. Compute navigation graph (derived entity)
nav_graph = CallableRegistry.execute("compute_navigation_graph", grid_map=grid)

# Framework automatically set:
# nav_graph.derived_from_function = "compute_navigation_graph"
# nav_graph.derived_from_execution_id = <UUID>
# nav_graph.source_grid_id = grid.ecs_id

# 3. Use graph for pathfinding (creates another derived entity)
path = CallableRegistry.execute("find_path_astar", 
                               start=(5, 5), 
                               goal=(8, 8), 
                               nav_graph=nav_graph)

# Framework automatically set:
# path.derived_from_function = "find_path_astar"
# path.derived_from_execution_id = <UUID>

# 4. Use graph for visibility (creates another derived entity)
visible = CallableRegistry.execute("compute_visible_positions",
                                  origin=(5, 5),
                                  max_range=5,
                                  nav_graph=nav_graph)

# Framework automatically set:
# visible.derived_from_function = "compute_visible_positions"
# visible.derived_from_execution_id = <UUID>

# 5. Modify grid (creates new version)
grid_v2 = CallableRegistry.execute("spawn_entity", 
                                   grid_map=grid, 
                                   entity=Wall(name="wall", walkable=False, transparent=False), 
                                   position=(6, 6))

# grid_v2.ecs_id != grid.ecs_id (new version)
# nav_graph.source_grid_id == grid.ecs_id (points to old version - STALE!)

# 6. Check if graph is stale
if nav_graph.source_grid_id != grid_v2.ecs_id:
    print("Graph is stale, recomputing...")
    nav_graph_v2 = CallableRegistry.execute("compute_navigation_graph", grid_map=grid_v2)
    # nav_graph_v2.source_grid_id == grid_v2.ecs_id (fresh!)
```

### Causal Chain Example

```
GridMap (v1)
    ↓ [compute_navigation_graph]
NavigationGraph (derived_from GridMap v1)
    ↓ [find_path_astar]
Path (derived_from NavigationGraph)

GridMap (v2) ← modified
    ↓ [compute_navigation_graph]
NavigationGraph (v2) (derived_from GridMap v2)
    ↓ [find_path_astar]
Path (v2) (derived_from NavigationGraph v2)
```

Every entity knows:
- What function created it (`derived_from_function`)
- What execution created it (`derived_from_execution_id`)
- What it was derived from (via input parameters tracked in execution)

This is **automatic provenance tracking** - no manual work needed!

## Questions to Resolve

1. **Can entities have Dict/Set fields?** Need to test P9
2. **Should graphs be root entities or attached to grid?** Probably root - they're query results
3. **Do we need automatic invalidation?** Or manual recomputation is fine?
4. **Should we cache graphs?** Or recompute on demand?

## Summary

### Key Design Decisions

1. **Single NavigationGraph entity** - Contains both walkability and visibility data
2. **Flat Dict fields** - Fast lookup, no hierarchical nesting needed
3. **Automatic derivation tracking** - Framework handles `derived_from_function`, `derived_from_execution_id`
4. **Manual staleness checking** - Compare `nav_graph.source_grid_id` with `current_grid.ecs_id`
5. **Recompute on demand** - No automatic invalidation, explicit recomputation

### Causal Dependency Chain

```
GridMap 
  → [compute_navigation_graph] → 
NavigationGraph
  → [find_path_astar] → 
Path

NavigationGraph
  → [compute_visible_positions] →
VisibleArea
```

All derivations tracked automatically by framework!

## Next Steps

1. ✅ **Test P9**: Dict/Set fields in entities (test created, needs to run)
2. **Implement NavigationGraph entity** in `game_entities.py`
3. **Implement `compute_navigation_graph` function**
4. **Implement `is_node_transparent` helper**
5. **Implement A* pathfinding using the graph**
6. **Implement shadowcasting using the graph**
7. **Test the complete workflow**
