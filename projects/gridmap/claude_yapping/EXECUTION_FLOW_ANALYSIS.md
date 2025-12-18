# Execution Flow Analysis: collect_apple

## Step-by-Step Trace

### 1. Initial State
```python
grid = GridMap(ecs_id=AAA, lineage_id=XXX)
├── agent = Agent(ecs_id=CCC, inventory=[])
└── apple = Apple(ecs_id=DDD)

# All registered in EntityRegistry
EntityRegistry.tree_registry[AAA] = tree with all entities
EntityRegistry.lineage_registry[XXX] = [AAA]
```

### 2. Function Call
```python
grid_v1 = CallableRegistry.execute("collect_apple", grid_map=grid, agent=agent, apple_position=(1,1))
```

### 3. Input Preparation (_prepare_transactional_inputs)
```python
# Line 1039: Get STORED copy from registry
grid_copy = EntityRegistry.get_stored_entity(grid.root_ecs_id, grid.ecs_id)
agent_copy = EntityRegistry.get_stored_entity(agent.root_ecs_id, agent.ecs_id)

# Execute with COPIES, not originals!
execution_kwargs = {
    "grid_map": grid_copy,
    "agent": agent_copy,
    "apple_position": (1, 1)
}
```

### 4. Function Execution
```python
def collect_apple(grid_map, agent, apple_position):
    # Find apple
    apple = find_apple_at_position(grid_map, apple_position)
    
    # Remove from grid
    node.entities.remove(apple)  # Mutates grid_copy
    
    # Add to inventory
    agent.inventory.append(apple)  # Mutates agent_copy
    
    return grid_map  # Returns grid_copy
```

### 5. Versioning (After Execution)
```python
# Framework detects grid_copy was mutated
# Creates NEW version of the tree
grid_v1 = new version (ecs_id=BBB, lineage_id=XXX)
├── agent_v1 = Agent(ecs_id=EEE, inventory=[apple])  # NEW ecs_id!
└── (no apple on grid)

# Registers new version
EntityRegistry.tree_registry[BBB] = new tree
EntityRegistry.lineage_registry[XXX] = [AAA, BBB]  # Appended!
```

### 6. Return
```python
# Returns grid_v1 (ecs_id=BBB)
# But our test still has references to:
# - grid (ecs_id=AAA)
# - agent (ecs_id=CCC)
```

## The Core Issue

**Child entities get NEW ecs_ids when tree is versioned!**

```python
# Before
agent.ecs_id = CCC
agent.inventory = []

# After (in new tree)
agent_v1.ecs_id = EEE  # DIFFERENT!
agent_v1.inventory = [apple]
```

## How to Get Latest Version

### For Root Entity (GridMap)
```python
lineage_id = grid.lineage_id
root_ecs_ids = EntityRegistry.lineage_registry[lineage_id]
latest_root_ecs_id = root_ecs_ids[-1]  # BBB
latest_tree = EntityRegistry.get_stored_tree(latest_root_ecs_id)
grid_latest = latest_tree.get_entity(latest_root_ecs_id)
```

### For Child Entity (Agent)
**Problem**: Agent's ecs_id changed from CCC to EEE!

**Solutions**:
1. Find by name (fragile)
2. Find by type + some identifier
3. Track lineage of child entities separately
4. Use live_id instead of ecs_id?

## Key Insight

The agent in the new tree is a **DIFFERENT ENTITY** with a different ecs_id!

We can't use `agent.ecs_id` to find it because that's the OLD ecs_id.

## What About live_id?

Let me check if live_id is preserved...

Looking at line 1048: `copy.live_id = uuid4()` - NO! Live_id is also changed!

## The Real Solution

**We need to find the agent in the new tree by some stable identifier:**
- Name (if unique)
- Position in tree structure
- Some custom ID field

OR

**Return the updated agent from the function:**
```python
def collect_apple(...) -> Tuple[GridMap, Agent]:
    ...
    return grid_map, agent
```

But this changes the function signature!

## Current Workaround in Tests

```python
# Get latest grid
grid_latest = get_latest_gridmap(grid)

# Find agent by name in latest tree
for entity in latest_tree.nodes.values():
    if isinstance(entity, Agent) and entity.name == agent.name:
        agent_latest = entity
        break
```

This works but is fragile (requires unique names).

## Better Solution?

Maybe we should track child entity lineages separately? Or use a different approach entirely?
