# Proposed Fix for Derivation Tracking

## Problem Statement

Single-entity returns do NOT get proper derivation tracking (`derived_from_function` and `derived_from_execution_id`).

Only multi-entity returns get full tracking because only they call `_apply_semantic_actions`.

## Root Cause

The function `_apply_semantic_actions` is the ONLY place that sets both tracking fields correctly.

It's called:
- ✅ In `_finalize_multi_entity_result` (multi-entity path)
- ❌ NOT in `_finalize_single_entity_result` (single-entity transactional path)
- ❌ NOT in `_create_output_entity_with_provenance` (borrowing path)
- ❌ NOT in `_execute_no_inputs` (no-input path)

## Proposed Fixes

### Fix #1: PATH 3 - Transactional Single-Entity (CRITICAL for GridMap)

**File**: `abstractions/ecs/callable_registry.py`

**Location 1**: Line 1010
```python
# CURRENT (WRONG):
return await cls._finalize_single_entity_result(result, metadata, object_identity_map)

# FIXED:
return await cls._finalize_single_entity_result(result, metadata, object_identity_map, execution_id)
```

**Location 2**: Line 1146-1151 (Function signature)
```python
# CURRENT (WRONG):
async def _finalize_single_entity_result(
    cls,
    result: Any,
    metadata: FunctionMetadata,
    object_identity_map: Dict[int, Entity]
) -> Entity:

# FIXED:
async def _finalize_single_entity_result(
    cls,
    result: Any,
    metadata: FunctionMetadata,
    object_identity_map: Dict[int, Entity],
    execution_id: Optional[UUID] = None  # ADD THIS
) -> Entity:
```

**Location 3**: Line 1163-1190 (Function body)
```python
# CURRENT (WRONG):
if isinstance(result, Entity):
    semantic, original_entity = cls._detect_execution_semantic(result, object_identity_map)
    
    if semantic == "mutation":
        if original_entity:
            result.update_ecs_ids()
            EntityRegistry.register_entity(result)
            EntityRegistry.version_entity(original_entity)
        else:
            result.promote_to_root()
            
    elif semantic == "creation":
        result.promote_to_root()
        
    elif semantic == "detachment":
        result.detach()
        if original_entity:
            EntityRegistry.version_entity(original_entity)
    
    return result

# FIXED:
if isinstance(result, Entity):
    if execution_id is None:
        execution_id = uuid4()
    
    semantic, original_entity = cls._detect_execution_semantic(result, object_identity_map)
    
    # Call _apply_semantic_actions to set tracking fields
    result = await cls._apply_semantic_actions(
        result, semantic, original_entity, metadata, execution_id
    )
    
    return result
```

### Fix #2: PATH 4/5 - Borrowing Single-Entity

**File**: `abstractions/ecs/callable_registry.py`

**Location 1**: Line 923-941
```python
# CURRENT (WRONG):
if is_multi_entity:
    object_identity_map = {}
    execution_id = uuid4()
    
    return await cls._finalize_multi_entity_result(
        result, metadata, object_identity_map, input_entity, execution_id
    )
else:
    output_entity = await cls._create_output_entity_with_provenance(
        result, metadata.output_entity_class, input_entity, metadata.name
    )
    
    if not output_entity.is_root_entity():
        output_entity.promote_to_root()
    
    await cls._record_basic_execution(input_entity, output_entity, metadata.name)
    
    return output_entity

# FIXED:
if is_multi_entity:
    object_identity_map = {}
    execution_id = uuid4()
    
    return await cls._finalize_multi_entity_result(
        result, metadata, object_identity_map, input_entity, execution_id
    )
else:
    execution_id = uuid4()  # ADD THIS
    
    output_entity = await cls._create_output_entity_with_provenance(
        result, metadata.output_entity_class, input_entity, metadata.name, execution_id  # ADD execution_id
    )
    
    if not output_entity.is_root_entity():
        output_entity.promote_to_root()
    
    await cls._record_basic_execution(input_entity, output_entity, metadata.name)
    
    return output_entity
```

**Location 2**: Line 1458-1462 (Function signature)
```python
# CURRENT (WRONG):
async def _create_output_entity_with_provenance(
    cls,
    result: Any,
    output_entity_class: Type[Entity],
    input_entity: Entity,
    function_name: str
) -> Entity:

# FIXED:
async def _create_output_entity_with_provenance(
    cls,
    result: Any,
    output_entity_class: Type[Entity],
    input_entity: Entity,
    function_name: str,
    execution_id: Optional[UUID] = None  # ADD THIS
) -> Entity:
```

**Location 3**: Line 1472-1477
```python
# CURRENT (WRONG):
if isinstance(result, Entity):
    output_entity = result
    
    if hasattr(output_entity, 'derived_from_function'):
        output_entity.derived_from_function = function_name
    
    # ... rest of function

# FIXED:
if isinstance(result, Entity):
    output_entity = result
    
    if execution_id is None:
        execution_id = uuid4()
    
    if hasattr(output_entity, 'derived_from_function'):
        output_entity.derived_from_function = function_name
    if hasattr(output_entity, 'derived_from_execution_id'):
        output_entity.derived_from_execution_id = execution_id  # ADD THIS
    
    # ... rest of function
```

### Fix #3: PATH 2 - No Inputs

**File**: `abstractions/ecs/callable_registry.py`

**Location**: Line 1641-1647
```python
# CURRENT (WRONG):
if isinstance(result, Entity):
    output_entity = result
    output_entity.promote_to_root()
else:
    output_entity = cls._create_output_entity_from_result(result, metadata.output_entity_class, metadata.name)
    output_entity.promote_to_root()

# FIXED:
execution_id = uuid4()  # ADD THIS

if isinstance(result, Entity):
    output_entity = result
    output_entity.promote_to_root()
    
    # Set tracking fields
    if hasattr(output_entity, 'derived_from_function'):
        output_entity.derived_from_function = metadata.name
    if hasattr(output_entity, 'derived_from_execution_id'):
        output_entity.derived_from_execution_id = execution_id
else:
    output_entity = cls._create_output_entity_from_result(result, metadata.output_entity_class, metadata.name)
    output_entity.promote_to_root()
```

## Testing the Fix

After applying fixes, all P10 tests should pass:
- ✅ P10.1: Single entity creation (borrowing) - both fields set
- ✅ P10.2: Multi entity creation - both fields set (already works)
- ✅ P10.3: Single entity mutation (transactional) - both fields set

## Impact

This fix ensures that **ALL** entity returns get proper provenance tracking, enabling:
- Complete causal chains
- Automatic derivation tracking
- No need for manual `source_grid_id` fields
- Consistent behavior across all execution paths
