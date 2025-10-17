# Entity Manipulation Patterns

## Understanding Entity Lifecycle

The framework has three key methods for managing entity relationships:
1. **`promote_to_root()`** - Make an entity a root entity
2. **`attach(new_root_entity)`** - Attach a root entity to a new parent tree
3. **`detach()`** - Remove entity from parent tree (after physical removal)

## The Pattern: Physical Mutation + Framework Notification

### Key Insight
**You mutate Python objects directly, then call framework methods to update metadata.**

```python
# 1. Physically modify the Python object
grid_map.nodes.append(new_node)

# 2. Notify framework (if needed)
new_node.attach(grid_map)  # Only if new_node was previously a root entity
```

## Adding Entities to Collections

### Pattern 1: Create New Entity in Place
```python
@CallableRegistry.register("add_wall_to_grid")
def add_wall_to_grid(grid_map: GridMap, position: Tuple[int, int]) -> GridMap:
    # Find the node
    node = next((n for n in grid_map.nodes if n.position == position), None)
    
    if node:
        # Create new entity directly
        wall = Wall(position=position, walkable=False, transparent=False)
        
        # Add to list (direct mutation)
        node.entities.append(wall)
    
    # Return mutated grid (framework handles versioning)
    return grid_map
```

**No attach/detach needed** - new entity is created as part of the tree.

### Pattern 2: Move Existing Root Entity
```python
@CallableRegistry.register("add_agent_to_grid")
def add_agent_to_grid(grid_map: GridMap, agent: Agent, position: Tuple[int, int]) -> GridMap:
    # Find target node
    node = next((n for n in grid_map.nodes if n.position == position), None)
    
    if node:
        # Agent must be a root entity first
        if not agent.is_root_entity():
            agent.promote_to_root()
        
        # Physically add to list
        node.entities.append(agent)
        
        # Notify framework: agent is now part of grid_map's tree
        agent.attach(grid_map)
    
    return grid_map
```

**attach() required** - moving a root entity into another tree.

## Removing Entities from Collections

### Pattern: Physical Removal + detach()
```python
@CallableRegistry.register("remove_entity_from_grid")
def remove_entity_from_grid(grid_map: GridMap, entity_id: str) -> GridMap:
    # Find and remove entity
    for node in grid_map.nodes:
        for entity in node.entities:
            if str(entity.ecs_id) == entity_id:
                # 1. Physically remove from list
                node.entities.remove(entity)
                
                # 2. Notify framework
                entity.detach()
                
                break
    
    return grid_map
```

**detach() required** - entity is no longer part of the tree.

## What detach() Does

From `entity.py` line 1795-1826:

```python
def detach(self) -> None:
    """
    This has to be called after the entity has been removed from its parent python object
    
    Scenarios:
    1) Entity is already root → just version it
    2) Entity has no root_ecs_id → promote to root
    3) Tree doesn't exist → promote to root
    4) Entity not in tree → promote to root
    5) Entity in tree → version the tree root
    """
```

**Key**: Call `detach()` AFTER physically removing from parent.

## What attach() Does

From `entity.py` line 1960-1991:

```python
def attach(self, new_root_entity: "Entity") -> None:
    """
    This has to be attached when a previously root entity is added as subentity to a new root parent entity
    """
    # Can only attach root entities
    # Updates root_ecs_id, root_live_id, lineage_id
    # Versions both old and new root entities
```

**Key**: Call `attach()` AFTER physically adding to parent, and ONLY for root entities.

## The add_to() Method (Stub)

From `entity.py` line 1830-1842:

```python
def add_to(self, new_entity: "Entity", field_name: str, copy=False, detach_target: bool = False) -> None:
    """
    just a stub for now
    This method will move the entity to a new root entity and update the attribute_source
    """
```

**Status**: Not implemented yet. Use manual pattern above.

## GridMap Application Patterns

### Adding a New Node
```python
@CallableRegistry.register("add_node_to_grid")
def add_node_to_grid(grid_map: GridMap, position: Tuple[int, int]) -> GridMap:
    # Create node with empty entities list
    new_node = GridNode(position=position, entities=[])
    
    # Add to grid (direct mutation)
    grid_map.nodes.append(new_node)
    
    # No attach needed - created in place
    return grid_map
```

### Adding Entity to Node
```python
@CallableRegistry.register("spawn_entity")
def spawn_entity(grid_map: GridMap, entity_type: str, position: Tuple[int, int]) -> GridMap:
    # Find node
    node = next((n for n in grid_map.nodes if n.position == position), None)
    
    if node:
        # Create entity based on type
        if entity_type == "wall":
            entity = Wall(position=position, walkable=False, transparent=False)
        elif entity_type == "floor":
            entity = Floor(position=position, walkable=True, transparent=True)
        elif entity_type == "agent":
            entity = Agent(position=position, speed=3, sight=5, walkable=True, transparent=True)
        
        # Add to node (direct mutation)
        node.entities.append(entity)
        
        # No attach needed - created in place
    
    return grid_map
```

### Moving Agent Between Nodes
```python
@CallableRegistry.register("move_agent_between_nodes")
def move_agent_between_nodes(grid_map: GridMap, agent_id: str, target_pos: Tuple[int, int]) -> GridMap:
    # Find agent and current node
    agent = None
    current_node = None
    
    for node in grid_map.nodes:
        for entity in node.entities:
            if str(entity.ecs_id) == agent_id and isinstance(entity, Agent):
                agent = entity
                current_node = node
                break
        if agent:
            break
    
    if agent and current_node:
        # Find target node
        target_node = next((n for n in grid_map.nodes if n.position == target_pos), None)
        
        if target_node:
            # 1. Remove from current node
            current_node.entities.remove(agent)
            
            # 2. Update agent position
            agent.position = target_pos
            
            # 3. Add to target node
            target_node.entities.append(agent)
            
            # No detach/attach needed - staying within same tree
    
    return grid_map
```

### Removing Entity from Grid
```python
@CallableRegistry.register("despawn_entity")
def despawn_entity(grid_map: GridMap, entity_id: str) -> GridMap:
    # Find and remove
    for node in grid_map.nodes:
        for entity in node.entities:
            if str(entity.ecs_id) == entity_id:
                # Physical removal
                node.entities.remove(entity)
                
                # Notify framework
                entity.detach()
                
                return grid_map
    
    return grid_map
```

## Boundary Conditions to Test

### 1. Empty Collections
```python
# Node with no entities
node = GridNode(position=(0, 0), entities=[])
# Should work fine - empty list is valid
```

### 2. Adding to Non-existent Node
```python
# What if position doesn't exist?
node = next((n for n in grid_map.nodes if n.position == (99, 99)), None)
if node is None:
    # Handle gracefully - maybe create node first?
    pass
```

### 3. Removing Non-existent Entity
```python
# What if entity_id doesn't exist?
# Loop completes, nothing happens - safe
```

### 4. Moving Within Same Tree
```python
# Agent moves from node A to node B (both in same grid)
# NO detach/attach needed - just list operations
```

### 5. Moving Between Trees
```python
# Agent moves from grid_map_1 to grid_map_2
# NEED detach from grid_map_1, attach to grid_map_2
```

### 6. Duplicate Entities
```python
# What if we append same entity twice?
# Framework will handle via ecs_id tracking
# But probably should prevent at function level
```

## Summary: The Rules

1. **Create in place** → No attach/detach
2. **Move within tree** → No attach/detach (just list ops)
3. **Move between trees** → detach() from old, attach() to new
4. **Remove from tree** → detach() after removal
5. **Always mutate directly** → Framework handles versioning

## For GridMap

Since GridMap is a single tree, most operations are **Pattern 1**:
- Spawn entity → create in place, append to list
- Move entity → remove from one list, append to another (same tree)
- Despawn entity → remove from list, detach()

**No attach() needed** unless importing entities from outside the grid.
