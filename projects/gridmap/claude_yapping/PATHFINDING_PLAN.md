# Pathfinding Implementation Plan

## Current State

### What We Have
1. ✅ **GridMap** - Grid with nodes, entities, walls
2. ✅ **NavigationGraph** - Precomputed walkability/transparency with neighbors
3. ✅ **Agent** - Has position and speed
4. ✅ **Automatic derivation tracking** - All functions track provenance

### What We Need
- **Pathfinding function** that computes reachable paths from agent position
- **Path entity** to represent individual paths
- **Comprehensive test** with verifiable, non-trivial paths

## Design

### Path Entity
```python
class Path(Entity):
    """Represents a single path from start to destination."""
    start_position: Tuple[int, int]
    end_position: Tuple[int, int]
    steps: List[Tuple[int, int]]  # Ordered list of positions
    length: int  # Number of steps
    cost: int  # Movement cost (for now, same as length)
    
    # Automatic tracking (from framework)
    derived_from_function: Optional[str] = None
    derived_from_execution_id: Optional[UUID] = None
```

### PathCollection Entity
```python
class PathCollection(Entity):
    """Collection of all reachable paths from an agent's position."""
    agent_id: UUID  # Which agent
    agent_position: Tuple[int, int]  # Starting position
    max_distance: int  # Speed limit
    paths: List[Path]  # All reachable paths
    reachable_positions: List[Tuple[int, int]]  # Unique destinations
    
    # Manual tracking (for staleness)
    source_graph_id: UUID  # Which NavigationGraph was used
    
    # Automatic tracking (from framework)
    derived_from_function: Optional[str] = None
    derived_from_execution_id: Optional[UUID] = None
```

### Pathfinding Function
```python
@CallableRegistry.register("compute_reachable_paths")
def compute_reachable_paths(
    nav_graph: NavigationGraph,
    agent: Agent
) -> PathCollection:
    """
    Compute all paths reachable from agent's position within speed limit.
    
    Uses BFS to explore all positions within agent.speed steps.
    Returns PathCollection with all unique paths.
    """
```

## Test Scenario Design

### Grid Layout (7x7)
```
  0 1 2 3 4 5 6
0 . . . # . . .
1 . A . # . . .
2 . . . # . . .
3 # # # . . . .
4 . . . . . # .
5 . . . . . # .
6 . . . . . . .
```

- **A** = Agent at (1, 1), speed = 3
- **#** = Walls (non-walkable)

### Expected Reachable Positions (distance ≤ 3)
From (1, 1) with speed 3:

**Distance 1:**
- (0, 1), (1, 0), (1, 2), (2, 1)

**Distance 2:**
- (0, 0), (0, 2), (2, 0), (2, 2)

**Distance 3:**
- (0, 3) - blocked by wall at (0, 3)
- (3, 0) - blocked by walls
- (3, 1) - blocked by walls
- (3, 2) - blocked by walls
- (2, 3) - blocked by wall
- (1, 3) - blocked by wall
- (0, 4), (1, 4), (2, 4) - too far or blocked

Actually, let me recalculate with walls:

**Reachable within 3 steps from (1, 1):**
```
  0 1 2 3 4 5 6
0 3 2 3 # . . .
1 2 A 2 # . . .
2 3 2 3 # . . .
3 # # # . . . .
4 . . . . . # .
5 . . . . . # .
6 . . . . . . .
```

Positions reachable:
- Distance 1: (0,1), (1,0), (1,2), (2,1) = 4 positions
- Distance 2: (0,0), (0,2), (2,0), (2,2) = 4 positions  
- Distance 3: None new (all blocked by wall column at x=3)

**Total: 8 reachable positions + starting position = 9 paths**

### Interesting Paths to Verify

1. **Straight line**: (1,1) → (1,0) - 1 step
2. **Diagonal equivalent**: (1,1) → (0,0) - 2 steps via (0,1) or (1,0)
3. **Corner path**: (1,1) → (2,2) - 2 steps via (1,2)+(2,2) or (2,1)+(2,2)
4. **Max distance**: (1,1) → (0,0) or (2,2) - 2 steps (not 3 because blocked)
5. **Blocked path**: (1,1) → (1,3) - UNREACHABLE (wall at x=3)

### Test Assertions

```python
def test_pathfinding():
    # Setup grid with walls
    grid = create_test_grid_with_walls()
    
    # Add agent at (1, 1) with speed 3
    agent = Agent(name="test_agent", position=(1, 1), speed=3)
    
    # Compute navigation graph
    nav_graph = compute_navigation_graph(grid)
    
    # Compute reachable paths
    path_collection = compute_reachable_paths(nav_graph, agent)
    
    # Verify automatic tracking
    assert path_collection.derived_from_function == "compute_reachable_paths"
    assert path_collection.derived_from_execution_id is not None
    assert path_collection.source_graph_id == nav_graph.ecs_id
    
    # Verify path count
    assert len(path_collection.paths) == 9  # Including start position
    assert len(path_collection.reachable_positions) == 9
    
    # Verify specific paths
    path_to_origin = next(p for p in path_collection.paths if p.end_position == (0, 0))
    assert path_to_origin.length == 2
    assert path_to_origin.start_position == (1, 1)
    assert len(path_to_origin.steps) == 3  # start + 2 steps
    
    # Verify unreachable positions
    unreachable = [(1, 3), (3, 1), (4, 1)]
    for pos in unreachable:
        assert pos not in path_collection.reachable_positions
    
    # Verify causal chain
    # GridMap → NavigationGraph → PathCollection
    assert path_collection.source_graph_id == nav_graph.ecs_id
    assert nav_graph.source_grid_id == grid.ecs_id
```

## Implementation Steps

1. **Create Path and PathCollection entities** in `game_entities.py`
2. **Implement BFS pathfinding** in new file `pathfinding.py`
3. **Create test grid setup** with specific wall layout
4. **Write comprehensive test** in `test_pathfinding.py`
5. **Verify**:
   - Correct path counts
   - Correct path lengths
   - Blocked paths excluded
   - Automatic derivation tracking works
   - Causal chain: GridMap → NavGraph → PathCollection

## Algorithm: BFS for All Reachable Paths

```python
def compute_reachable_paths(nav_graph, agent):
    start = agent.position
    max_dist = agent.speed
    
    # BFS with path tracking
    queue = [(start, [start], 0)]  # (position, path, distance)
    visited = {start: [start]}  # position → shortest path
    all_paths = []
    
    while queue:
        pos, path, dist = queue.pop(0)
        
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
            for neighbor in nav_graph.get_walkable_neighbors(pos):
                if neighbor not in visited or len(path) + 1 < len(visited[neighbor]):
                    new_path = path + [neighbor]
                    visited[neighbor] = new_path
                    queue.append((neighbor, new_path, dist + 1))
    
    return PathCollection(
        agent_id=agent.ecs_id,
        agent_position=start,
        max_distance=max_dist,
        paths=all_paths,
        reachable_positions=list(visited.keys()),
        source_graph_id=nav_graph.ecs_id
    )
```

## Expected Output

```
Computing reachable paths for Agent at (1, 1) with speed 3

Reachable positions: 9
  (1, 1) - distance 0
  (0, 1) - distance 1
  (1, 0) - distance 1
  (1, 2) - distance 1
  (2, 1) - distance 1
  (0, 0) - distance 2
  (0, 2) - distance 2
  (2, 0) - distance 2
  (2, 2) - distance 2

Sample paths:
  Path to (0, 0): (1,1) → (0,1) → (0,0) [length=2]
  Path to (2, 2): (1,1) → (2,1) → (2,2) [length=2]

Blocked positions (unreachable):
  (1, 3) - wall at (3, x)
  (3, 1) - wall at (3, y)

PathCollection:
  derived_from_function: compute_reachable_paths ✅
  derived_from_execution_id: <UUID> ✅
  source_graph_id: <nav_graph.ecs_id> ✅

Causal Chain:
  GridMap → NavigationGraph → PathCollection ✅
```

## Next Steps

1. Implement entities
2. Implement pathfinding function
3. Create test with verifiable grid
4. Run and verify all assertions pass
