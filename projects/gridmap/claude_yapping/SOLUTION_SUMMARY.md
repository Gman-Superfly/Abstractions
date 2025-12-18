# Solution Summary: Entity Versioning and References

## The Problem
When a function mutates an entity tree, the framework creates a NEW version with NEW ecs_ids for all entities. Test code holding old references sees old data.

## Why It Happens
1. Functions execute on **stored copies** from EntityRegistry
2. After execution, framework **versions the entire tree**
3. All entities in new tree get **new ecs_ids**
4. Old references point to old versions

## The Solution (Current)

### For Root Entities
Use lineage registry:
```python
def get_latest_gridmap(grid_map):
    lineage_id = grid_map.lineage_id
    root_ecs_ids = EntityRegistry.lineage_registry[lineage_id]
    latest_root_ecs_id = root_ecs_ids[-1]
    latest_tree = EntityRegistry.get_stored_tree(latest_root_ecs_id)
    return latest_tree.get_entity(latest_root_ecs_id)
```

### For Child Entities
Find by stable identifier (name):
```python
def get_latest_agent(agent):
    # Get agent's root lineage
    lineage_id = agent.lineage_id  # Same as grid's lineage
    root_ecs_ids = EntityRegistry.lineage_registry[lineage_id]
    latest_root_ecs_id = root_ecs_ids[-1]
    latest_tree = EntityRegistry.get_stored_tree(latest_root_ecs_id)
    
    # Find agent by name
    for entity in latest_tree.nodes.values():
        if isinstance(entity, Agent) and entity.name == agent.name:
            return entity
    return agent
```

## Test Pattern

```python
# Execute function
grid = CallableRegistry.execute("collect_apple", grid_map=grid, agent=agent, ...)

# Get latest versions
grid = get_latest_gridmap(grid)
agent = get_latest_agent(agent)

# Now check state
assert len(agent.inventory) == 1  # Works!
```

## Why This Works
- Root entities: Track by lineage_id (stable across versions)
- Child entities: Find by name in latest tree (assumes unique names)

## Limitations
- Requires unique names for child entities
- Fragile if entities don't have stable identifiers
- Need to call get_latest_* after every mutation

## Future Improvements
- Add stable IDs to entities
- Track child entity lineages separately
- Return updated entities from functions
- Use live_id if it's preserved (currently it's not)
