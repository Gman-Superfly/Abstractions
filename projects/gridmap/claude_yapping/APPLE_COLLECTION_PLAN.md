# Apple Collection System Plan

## Goal
Agent collects apples into inventory and new apples spawn randomly until agent has 3 apples.

## Changes Needed

### 1. Update Agent Entity
Add inventory field to Agent:
```python
class Agent(GameEntity):
    ...
    inventory: List[Apple] = Field(default_factory=list)  # Collected apples
```

### 2. New Functions

#### collect_apple
```python
@CallableRegistry.register("collect_apple")
def collect_apple(grid_map: GridMap, agent: Agent, apple_position: Tuple[int, int]) -> GridMap:
    """
    Collect apple from grid into agent's inventory.
    
    Steps:
    1. Find apple at position
    2. Remove apple from node
    3. Add apple to agent.inventory
    4. Return updated GridMap
    """
```

#### spawn_random_apple
```python
@CallableRegistry.register("spawn_random_apple")
def spawn_random_apple(grid_map: GridMap) -> GridMap:
    """
    Spawn a new apple at a random walkable position.
    
    Steps:
    1. Find all walkable empty positions (no apples, no agents)
    2. Choose random position
    3. Create new apple
    4. Add to grid
    5. Return updated GridMap
    """
```

### 3. Updated Game Loop

```python
def test_agent_collects_apples():
    grid = create_empty_grid()
    agent = Agent(name="collector", speed=2, inventory=[])
    
    # Spawn first apple
    grid = spawn_random_apple(grid)
    
    target_apples = 3
    max_steps = 50
    
    for step in range(max_steps):
        # Check if collected enough
        if len(agent.inventory) >= target_apples:
            print(f"✅ Collected {target_apples} apples in {step} steps!")
            break
        
        # Find agent position
        agent_pos = find_agent_position(grid, agent)
        
        # Check if at apple position
        apple_at_pos = find_apple_at_position(grid, agent_pos)
        if apple_at_pos:
            # Collect apple
            grid = collect_apple(grid, agent, agent_pos)
            print(f"  🍎 Collected apple! Total: {len(agent.inventory)}")
            
            # Spawn new apple if needed
            if len(agent.inventory) < target_apples:
                grid = spawn_random_apple(grid)
                print(f"  ✨ New apple spawned!")
        
        # Pathfind and move towards apple
        nav_graph = compute_navigation_graph(grid)
        path_collection = compute_reachable_paths(nav_graph, agent, agent_pos)
        chosen_path = choose_path(path_collection, grid)
        grid = move_agent_along_path(grid, agent, chosen_path)
```

## Expected Behavior

```
Step 0: Agent at (1,1), Apple at (3,3), Inventory: []

Step 1: Moving towards (3,3)...
Step 2: Reached apple at (3,3)
  🍎 Collected apple! Total: 1
  ✨ New apple spawned at (0,4)!

Step 3: Moving towards (0,4)...
Step 4: Moving towards (0,4)...
Step 5: Reached apple at (0,4)
  🍎 Collected apple! Total: 2
  ✨ New apple spawned at (4,0)!

Step 6: Moving towards (4,0)...
...
Step N: Reached apple at (4,0)
  🍎 Collected apple! Total: 3
  ✅ Collected 3 apples in N steps!
```

## Causal Chain Per Collection

```
GridMap_v0 (apple at (3,3))
  ↓ [collect_apple]
GridMap_v1 (apple in agent.inventory, not on grid)
  ↓ [spawn_random_apple]
GridMap_v2 (new apple at random position)
  ↓ [compute_navigation_graph]
NavGraph_v2
  ↓ ... (pathfind towards new apple)
```

## Test Assertions

1. ✅ Agent inventory starts empty
2. ✅ After collection, apple removed from grid
3. ✅ After collection, apple in agent.inventory
4. ✅ New apple spawns at different position
5. ✅ Agent eventually collects 3 apples
6. ✅ All operations create new grid versions
7. ✅ Full derivation tracking on all entities

## Implementation Order

1. Update Agent entity with inventory field
2. Implement collect_apple function
3. Implement spawn_random_apple function
4. Add helper: find_apple_at_position
5. Create test_apple_collection.py
6. Run and verify agent collects 3 apples
