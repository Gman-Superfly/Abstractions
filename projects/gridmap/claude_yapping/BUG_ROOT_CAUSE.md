# ROOT CAUSE FOUND

## The Bug

In `entity.py` line 1455, `version_entity` does:

```python
def version_entity(cls, entity: "Entity", force_versioning: bool = False) -> bool:
    old_tree = cls.get_stored_tree(entity.root_ecs_id)  # Line 1450
    new_tree = build_entity_tree(entity)  # Line 1455 - BUG HERE!
    modified_entities = list(find_modified_entities(new_tree=new_tree, old_tree=old_tree))
```

## The Problem

`entity` is the `original_entity` from the input, which is IMMUTABLE!

### Execution Flow

1. **Input**: `container_v0` (original, unmutated)
2. **Prepare**: Create stored copy for execution
   ```python
   copy = EntityRegistry.get_stored_entity(value.root_ecs_id, value.ecs_id)
   object_identity_map[id(copy)] = container_v0  # Maps copy → original
   ```
3. **Execute**: Function mutates the COPY
   ```python
   transfer_item(container=copy, ...)  # Mutates copy
   ```
4. **Detect**: Check if result is same object as copy
   ```python
   original_entity = object_identity_map[id(result)]  # Gets container_v0
   ```
5. **Version**: Call version_entity with ORIGINAL
   ```python
   EntityRegistry.version_entity(original_entity)  # Passes container_v0!
   ```
6. **Build Tree**: Build from ORIGINAL (unmutated!)
   ```python
   new_tree = build_entity_tree(entity)  # entity = container_v0 (OLD STATE!)
   ```
7. **Compare**: Compare old tree vs "new" tree
   ```python
   old_tree = stored snapshot (OLD STATE)
   new_tree = built from container_v0 (ALSO OLD STATE!)
   modified_entities = find_modified_entities(new_tree, old_tree)  # Returns 0!
   ```

## Why Container Still Got Versioned

Even though `find_modified_entities` returned 0, the Container still got a new ecs_id!

Looking at line 1463-1465:
```python
if len(typed_entities) > 0:
    if new_tree.root_ecs_id not in typed_entities:
        raise ValueError("if any entity is modified the root entity must be modified")
```

But we didn't hit this error, so `typed_entities` must be empty (0 entities).

So why did Container get versioned? Let me check if there's FORCE versioning somewhere...

Actually, looking at the callable_registry code, after `version_entity` is called, there's more code that updates the entity!

Line 1295-1296 in callable_registry:
```python
entity.update_ecs_ids()
EntityRegistry.register_entity(entity)
EntityRegistry.version_entity(original_entity)
```

Wait! `entity` here is the RESULT (mutated copy), and it calls `update_ecs_ids()` on it BEFORE calling `version_entity`!

So the Container got a new ecs_id from `update_ecs_ids()`, not from `version_entity`!

## The Real Bug

`version_entity` is being called with the wrong entity! It should be called with the MUTATED entity (the result), not the original input!

Or, `version_entity` should get the live root entity instead of using the passed entity directly.

## Fix Options

### Option 1: Pass the mutated entity
```python
EntityRegistry.version_entity(entity)  # entity is the mutated result
```

### Option 2: Get live root in version_entity
```python
def version_entity(cls, entity: "Entity", force_versioning: bool = False) -> bool:
    live_root = entity.get_live_root_entity()  # Get current live state
    new_tree = build_entity_tree(live_root)  # Build from live state
```

### Option 3: Pass both old and new
```python
EntityRegistry.version_entity(original_entity, mutated_entity)
```

Need to investigate which approach is correct!
