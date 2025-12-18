# Fix Proposal: Pass Mutated Entity to version_entity

## The Bug
`version_entity` is called with `original_entity` (unmutated input), so it builds the "new" tree from the old state and detects no changes.

## The Fix
Pass the mutated entity to `version_entity` instead.

### Location
`callable_registry.py` line 1297

### Current Code
```python
if semantic == "mutation":
    if original_entity:
        entity.update_ecs_ids()
        EntityRegistry.register_entity(entity)
        EntityRegistry.version_entity(original_entity)  # ← BUG: Wrong entity!
```

### Fixed Code
```python
if semantic == "mutation":
    if original_entity:
        entity.update_ecs_ids()
        EntityRegistry.register_entity(entity)
        EntityRegistry.version_entity(entity)  # ← FIX: Use mutated entity!
```

## Why This Works

1. `entity` is the mutated execution copy
2. `entity.update_ecs_ids()` gives it a new root ecs_id
3. `EntityRegistry.register_entity(entity)` registers it in live_id_registry
4. `EntityRegistry.version_entity(entity)` will:
   - Get `old_tree` from the stored snapshot (before mutation)
   - Build `new_tree` from `entity` (after mutation)
   - Compare them and detect changes!
   - Version all modified entities

## Impact

This will make `find_modified_entities` actually work and detect:
- Edge changes (items moved between lists)
- Attribute changes
- All entities in the modified paths will get new ecs_ids

## Testing

Run `test_p12_entity_transfer.py` - it should:
- Detect modified entities > 0
- Version the Collector (new ecs_id)
- Pass all assertions

## Concerns

Need to verify this doesn't break other code paths. The `original_entity` parameter might be used elsewhere for tracking purposes.

Let me check if `original_entity` is used after this call...
