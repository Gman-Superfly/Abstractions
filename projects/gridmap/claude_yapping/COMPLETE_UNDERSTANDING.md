# Complete Understanding of Versioning System

## How Versioning Actually Works

### The Flow (from entity.py lines 1433-1506)

```python
def version_entity(cls, entity: "Entity", force_versioning: bool = False) -> bool:
    # 1. Get stored tree snapshot
    old_tree = cls.get_stored_tree(entity.root_ecs_id)
    
    # 2. Build new tree from CURRENT STATE of entity
    new_tree = build_entity_tree(entity)
    
    # 3. Find modified entities by comparing trees
    modified_entities = list(find_modified_entities(new_tree=new_tree, old_tree=old_tree))
    
    # 4. If entities modified, version them
    if len(typed_entities) > 0:
        # Version root
        root_entity.update_ecs_ids()
        
        # Version all modified children
        for modified_entity_id in typed_entities:
            modified_entity.update_ecs_ids(new_root_ecs_id, root_entity_live_id)
        
        # Register new tree
        cls.register_entity_tree(new_tree)
```

## The Critical Issue in Our Test

### What We're Testing
```python
# Step 6.6: Debug find_modified_entities
new_tree_before_version = build_entity_tree(container_v0.get_live_root_entity())

modified, debug_info = find_modified_entities(
    new_tree=new_tree_before_version,
    old_tree=tree_v0,
    debug=True
)
```

### The Problem
We're calling `container_v0.get_live_root_entity()` which returns (line 1676):
```python
def get_live_root_entity(self) -> Optional["Entity"]: 
    if self.is_root_entity():
        return self  # Returns container_v0 itself!
```

So we're building the tree from `container_v0`, which is the ORIGINAL UNMUTATED entity!

### Why This Happens

Looking at callable_registry line 1297:
```python
EntityRegistry.version_entity(original_entity)
```

It passes `original_entity`, which is the input BEFORE mutation.

But the function ALREADY EXECUTED and mutated a COPY. The `original_entity` is still pointing to the old state!

## The Real Question

**Where is the mutated copy?**

The function returns `entity` (the mutated copy), but `version_entity` is called with `original_entity` (the unmutated input).

This seems intentional! Let me check if `original_entity` is supposed to be the live entity that was mutated...

## Wait - I Need to Understand get_stored_entity

Line 1365-1367:
```python
def get_stored_entity(cls, root_ecs_id: UUID, ecs_id: UUID) -> Optional["Entity"]:
    tree = cls.get_stored_tree(root_ecs_id)  # Gets a DEEP COPY!
    ...
    new_tree.update_live_ids()  # Updates live_ids to NEW values
    return new_tree
```

So `get_stored_entity` returns a DEEP COPY with NEW live_ids!

This means the execution copy is COMPLETELY SEPARATE from the original!

## The Actual Flow

1. Input: `container_v0` (live entity, registered)
2. Prepare: `copy = EntityRegistry.get_stored_entity(...)` - NEW Python object, NEW live_id
3. Execute: Function mutates `copy`
4. Return: `copy` is returned
5. Detect: `original_entity = object_identity_map[id(copy)]` - Gets `container_v0`
6. Version: `EntityRegistry.version_entity(container_v0)` - Versions the ORIGINAL!

But `container_v0` was NEVER mutated! Only `copy` was mutated!

So when we build the tree from `container_v0`, we get the OLD state!

## The REAL Bug

The bug is that `version_entity` is called with the WRONG entity!

It should be called with the MUTATED copy (the result), not the original input!

OR

The framework expects that the original input IS the live entity that gets mutated, but that's not how transactional execution works!

## Conclusion

My original fix WAS correct! We need to pass `entity` (the mutated result) to `version_entity`, not `original_entity` (the unmutated input).

The current code is fundamentally broken for transactional execution.
