# Agent Movement Plan

## Goal
Create a complete movement loop where an agent moves towards apples, recomputing paths each step.

## Components to Add

### 1. Apple Entity
```python
class Apple(GameEntity):
    """Collectible item that agents seek."""
    walkable: bool = True  # Agents can walk through apples
    transparent: bool = True
    nutrition: int = 10  # Value when collected
```

### 2. choose_path Function
```python
@CallableRegistry.register("choose_path")
def choose_path(path_collection: PathCollection, grid_map: GridMap) -> Path:
    """
    Choose the best path from a PathCollection.
    
    Strategy:
    1. Find all apples in the grid
    2. If apples exist, choose shortest path to nearest apple
    3. If no apples, choose random path
    
    Returns:
        Path to follow
    """
```

### 3. move_agent_one_step Function
```python
@CallableRegistry.register("move_agent_one_step")
def move_agent_one_step(
    grid_map: GridMap,
    agent: Agent,
    path: Path
) -> GridMap:
    """
    Move agent one step along the path.
    
    Steps:
    1. Get agent's current position from grid
    2. Get next position from path (steps[1] if len > 1)
    3. Remove agent from current node
    4. Add agent to next node
    5. Return updated GridMap (versioned)
    """
```

### 4. Test Scenario

```
Grid 5x5:
  0 1 2 3 4
0 . . . . .
1 . A . . .
2 . . . . .
3 . . . @ .
4 . . . . .

A = Agent at (1, 1), speed 2
@ = Apple at (3, 3)
```

**Expected behavior:**
- Step 0: Agent at (1,1), Apple at (3,3)
- Compute paths → Choose path to apple
- Step 1: Move to (2,2) - diagonal towards apple
- Step 2: Move to (3,3) - reach apple!

## Complete Loop (Single Iteration)

```python
def test_agent_seeks_apple():
    # Setup
    grid = create_grid_with_apple()  # Agent at (1,1), Apple at (3,3)
    agent_pos = (1, 1)
    
    # Step 1: Compute navigation graph
    nav_graph = compute_navigation_graph(grid)
    
    # Step 2: Compute reachable paths
    agent = find_agent(grid)
    path_collection = compute_reachable_paths(nav_graph, agent, agent_pos)
    
    # Step 3: Choose best path (towards apple)
    chosen_path = choose_path(path_collection, grid)
    
    # Verify: Path should lead towards apple at (3,3)
    assert (3, 3) in chosen_path.steps or chosen_path moves towards (3,3)
    
    # Step 4: Move agent one step
    updated_grid = move_agent_one_step(grid, agent, chosen_path)
    
    # Verify: Agent moved from (1,1) to (2,2)
    new_agent_pos = find_agent_position(updated_grid)
    assert new_agent_pos == (2, 2)
    
    # Verify: Grid was versioned
    assert updated_grid.ecs_id != grid.ecs_id
```

## Multi-Step Test (Reach Apple)

```python
def test_agent_reaches_apple():
    grid = create_grid_with_apple()
    agent_pos = (1, 1)
    apple_pos = (3, 3)
    
    max_steps = 10
    for step in range(max_steps):
        # Find agent
        agent = find_agent(grid)
        agent_pos = find_agent_position(grid)
        
        print(f"Step {step}: Agent at {agent_pos}")
        
        # Check if reached apple
        if agent_pos == apple_pos:
            print(f"✅ Reached apple in {step} steps!")
            break
        
        # Compute → Choose → Move
        nav_graph = compute_navigation_graph(grid)
        path_collection = compute_reachable_paths(nav_graph, agent, agent_pos)
        chosen_path = choose_path(path_collection, grid)
        grid = move_agent_one_step(grid, agent, chosen_path)
    
    # Verify reached apple
    final_pos = find_agent_position(grid)
    assert final_pos == apple_pos, f"Should reach apple, got {final_pos}"
```

## Causal Chain Per Step

```
Step 0:
  GridMap_v0 → NavGraph_v0 → PathCollection_v0 → Path_v0 → GridMap_v1

Step 1:
  GridMap_v1 → NavGraph_v1 → PathCollection_v1 → Path_v1 → GridMap_v2

...
```

Each step creates new versions with full provenance tracking!

## Implementation Order

1. ✅ Add Apple entity to game_entities.py
2. ✅ Implement choose_path in pathfinding.py
3. ✅ Implement move_agent_one_step in movement.py (already exists, may need update)
4. ✅ Add helper: find_agent_position(grid) -> Tuple[int, int]
5. ✅ Create test_agent_movement.py with both tests
6. ✅ Run and verify agent reaches apple

## Expected Output

```
Step 0: Agent at (1, 1)
  Apples found: [(3, 3)]
  Chose path to (3, 3): (1,1) → (2,2) → (3,3)
  Moving to (2, 2)

Step 1: Agent at (2, 2)
  Apples found: [(3, 3)]
  Chose path to (3, 3): (2,2) → (3,3)
  Moving to (3, 3)

Step 2: Agent at (3, 3)
  ✅ Reached apple in 2 steps!
```
