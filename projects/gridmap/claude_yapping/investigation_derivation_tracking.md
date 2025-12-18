# Investigation: Derivation Tracking for Single Entity Returns

## Problem Statement

When a function returns a single newly created entity, `derived_from_function` is `None`.
Need to understand if this is a bug or if there's another mechanism for tracking.

## Execution Pipeline for Single Entity Return

### 1. Entry Point
```
CallableRegistry.execute(func_name, **kwargs)
  ↓
asyncio.run(cls.aexecute(func_name, **kwargs))
  ↓
cls._execute_async(func_name, **kwargs)
```

### 2. Strategy Detection
```
_execute_async()
  ↓
strategy = cls._detect_execution_strategy(kwargs, metadata)
  ↓
Routes to: _execute_transactional() or _execute_borrowing()
```

### 3. Transactional Execution Path
```
_execute_transactional(metadata, kwargs, classification)
  ↓
execution_kwargs, original_entities, execution_copies, object_identity_map = 
    await cls._prepare_transactional_inputs(kwargs)
  ↓
result = metadata.original_function(**execution_kwargs)
  ↓
if single entity:
    return await cls._finalize_single_entity_result(result, metadata, object_identity_map)
else:
    return await cls._finalize_multi_entity_result(...)
```

### 4. Single Entity Finalization
```python
async def _finalize_single_entity_result(
    cls,
    result: Any,
    metadata: FunctionMetadata,
    object_identity_map: Dict[int, Entity]
) -> Entity:
    
    if isinstance(result, Entity):
        semantic, original_entity = cls._detect_execution_semantic(result, object_identity_map)
        
        if semantic == "mutation":
            result.update_ecs_ids()
            EntityRegistry.register_entity(result)
            EntityRegistry.version_entity(original_entity)
            # NO derived_from_function SET HERE!
                
        elif semantic == "creation":
            result.promote_to_root()
            # NO derived_from_function SET HERE!
            
        elif semantic == "detachment":
            result.detach()
            if original_entity:
                EntityRegistry.version_entity(original_entity)
            # NO derived_from_function SET HERE!
        
        return result
```

**BUG FOUND**: `derived_from_function` is never set in `_finalize_single_entity_result`!

### 5. Multi Entity Finalization (for comparison)
```python
async def _finalize_multi_entity_result(...):
    for entity in unpacking_result.primary_entities:
        if isinstance(entity, Entity):
            semantic, original_entity = cls._detect_execution_semantic(entity, object_identity_map)
            
            # Apply semantic actions - THIS SETS derived_from_function!
            processed_entity = await cls._apply_semantic_actions(
                entity, semantic, original_entity, metadata, execution_id
            )
```

### 6. _apply_semantic_actions (only called for multi-entity)
```python
async def _apply_semantic_actions(
    cls, 
    entity: Entity, 
    semantic: str, 
    original_entity: Optional[Entity], 
    metadata: FunctionMetadata, 
    execution_id: UUID
) -> Entity:
    
    if semantic == "mutation":
        # ... mutation logic ...
        entity.derived_from_function = metadata.name  # ✅ SET HERE
        entity.derived_from_execution_id = execution_id
        
    elif semantic == "creation":
        entity.derived_from_function = metadata.name  # ✅ SET HERE
        entity.derived_from_execution_id = execution_id
        if not entity.is_root_entity():
            entity.promote_to_root()
        
    elif semantic == "detachment":
        entity.detach()
        if original_entity:
            EntityRegistry.version_entity(original_entity)
        entity.derived_from_function = metadata.name  # ✅ SET HERE
        entity.derived_from_execution_id = execution_id
    
    return entity
```

## Findings

### Two Execution Paths

**Path 1: Transactional (has Entity inputs)**
- Goes through `_execute_transactional` → `_finalize_single_entity_result`
- Does NOT set `derived_from_function` or `derived_from_execution_id` ❌

**Path 2: Borrowing (no Entity inputs OR address-based)**
- Goes through `_execute_borrowing` → `_create_output_entity_with_provenance`
- Sets `derived_from_function` ✅
- Does NOT set `derived_from_execution_id` ❌

### Bug Confirmed
**Both paths fail to set `derived_from_execution_id`** for single-entity returns!

Only `_apply_semantic_actions` (called in multi-entity path) sets both fields correctly.

### Other Tracking Fields

Looking at Entity fields:
```python
class Entity(BaseModel):
    # Phase 4 sibling relationship tracking fields
    derived_from_function: Optional[str] = Field(default=None)
    derived_from_execution_id: Optional[UUID] = Field(default=None)
    sibling_output_entities: List[UUID] = Field(default_factory=list)
    output_index: Optional[int] = Field(default=None)
```

These are the ONLY automatic tracking fields. There's no alternative mechanism.

## Test Results (P10)

### P10.1: Single Entity Creation (Borrowing Path)
- Input: No Entity parameters
- Path: `_execute_borrowing` → `_create_output_entity_with_provenance`
- Result:
  - ✅ `derived_from_function` = "create_simple"
  - ❌ `derived_from_execution_id` = None

### P10.2: Multi Entity Creation (Borrowing Path)
- Input: No Entity parameters, tuple return
- Path: `_execute_borrowing` → `_finalize_multi_entity_result` → `_apply_semantic_actions`
- Result:
  - ✅ `derived_from_function` = "create_pair"
  - ✅ `derived_from_execution_id` = UUID
  - ✅ `output_index` set
  - ✅ `sibling_output_entities` set

### P10.3: Single Entity Mutation (Transactional Path)
- Input: Entity parameter
- Path: `_execute_transactional` → `_finalize_single_entity_result`
- Result:
  - ❌ `derived_from_function` = None
  - ❌ `derived_from_execution_id` = None
  - ✅ Versioning works (new ecs_id)

## Root Cause

**Only `_apply_semantic_actions` sets both tracking fields correctly.**

It's called:
- ✅ In `_finalize_multi_entity_result` (multi-entity path)
- ❌ NOT in `_finalize_single_entity_result` (single-entity transactional path)
- ❌ NOT in `_create_output_entity_with_provenance` (borrowing path)

## Fix Required

### Option 1: Add to _finalize_single_entity_result (Transactional Path)

```python
async def _finalize_single_entity_result(
    cls,
    result: Any,
    metadata: FunctionMetadata,
    object_identity_map: Dict[int, Entity],
    execution_id: Optional[UUID] = None  # NEED TO ADD THIS PARAMETER
) -> Entity:
    
    if execution_id is None:
        execution_id = uuid4()
    
    if isinstance(result, Entity):
        semantic, original_entity = cls._detect_execution_semantic(result, object_identity_map)
        
        # ADD THIS CALL
        result = await cls._apply_semantic_actions(
            result, semantic, original_entity, metadata, execution_id
        )
        
        return result
```

**Issue**: `_finalize_single_entity_result` currently doesn't receive `execution_id` parameter!
Need to update the call site in `_execute_transactional` (line 1010).

### Option 2: Add to _create_output_entity_with_provenance (Borrowing Path)

```python
async def _create_output_entity_with_provenance(
    cls,
    result: Any,
    output_entity_class: Type[Entity],
    input_entity: Entity,
    function_name: str,
    execution_id: Optional[UUID] = None  # ADD THIS
) -> Entity:
    
    if execution_id is None:
        execution_id = uuid4()
    
    if isinstance(result, Entity):
        output_entity = result
        
        # Set BOTH fields
        if hasattr(output_entity, 'derived_from_function'):
            output_entity.derived_from_function = function_name
        if hasattr(output_entity, 'derived_from_execution_id'):
            output_entity.derived_from_execution_id = execution_id  # ADD THIS
```

**Issue**: Need to thread `execution_id` through the borrowing path.

## Impact on GridMap

Our `compute_navigation_graph` function:
- Takes `GridMap` as input → Transactional path
- Returns single `NavigationGraph` → `_finalize_single_entity_result`
- **Currently gets NO tracking fields set** ❌

This means we MUST use manual tracking via `source_grid_id` field.
