# Deep Dive: Complete Execution Path Analysis

## Goal
Understand EXACTLY where and how `derived_from_function` and `derived_from_execution_id` should be set for ALL execution paths.

## Entry Point
```
CallableRegistry.execute(func_name, **kwargs)
  ↓
asyncio.run(cls.aexecute(func_name, **kwargs))
  ↓
cls._execute_async(func_name, **kwargs)
```

## Path Detection in _execute_async

```python
async def _execute_async(cls, func_name: str, **kwargs):
    metadata = cls.get_metadata(func_name)
    strategy = cls._detect_execution_strategy(kwargs, metadata)
    
    if strategy == "single_entity_with_config":
        return await cls._execute_with_partial(metadata, kwargs)
    elif strategy == "no_inputs":
        return await cls._execute_no_inputs(metadata)
    elif strategy in ["multi_entity_composite", "single_entity_direct"]:
        pattern_type, classification = InputPatternClassifier.classify_kwargs(kwargs)
        if pattern_type in ["pure_transactional", "mixed"]:
            return await cls._execute_transactional(metadata, kwargs, classification)
        else:
            return await cls._execute_borrowing(metadata, kwargs, classification)
    else:  # pure_borrowing
        pattern_type, classification = InputPatternClassifier.classify_kwargs(kwargs)
        return await cls._execute_borrowing(metadata, kwargs, classification)
```

## Path 1: Transactional Execution

### When Used
- Has Entity inputs
- Pattern type is "pure_transactional" or "mixed"

### Flow
```
_execute_transactional(metadata, kwargs, classification)
  ↓
execution_id = uuid4()  # LINE 979 - EXECUTION ID CREATED HERE
  ↓
execution_kwargs, original_entities, execution_copies, object_identity_map = 
    await cls._prepare_transactional_inputs(kwargs)
  ↓
result = metadata.original_function(**execution_kwargs)
  ↓
if is_multi_entity:
    return await cls._finalize_multi_entity_result(
        result, metadata, object_identity_map, input_entity, execution_id  # ✅ PASSES execution_id
    )
else:
    return await cls._finalize_single_entity_result(
        result, metadata, object_identity_map  # ❌ DOES NOT PASS execution_id!
    )
```

### BUG FOUND #1
**Line 1010 in callable_registry.py:**
```python
return await cls._finalize_single_entity_result(result, metadata, object_identity_map)
```

Should be:
```python
return await cls._finalize_single_entity_result(result, metadata, object_identity_map, execution_id)
```

### _finalize_single_entity_result (Current)
```python
async def _finalize_single_entity_result(
    cls,
    result: Any,
    metadata: FunctionMetadata,
    object_identity_map: Dict[int, Entity]
    # ❌ NO execution_id PARAMETER!
) -> Entity:
    
    if isinstance(result, Entity):
        semantic, original_entity = cls._detect_execution_semantic(result, object_identity_map)
        
        if semantic == "mutation":
            result.update_ecs_ids()
            EntityRegistry.register_entity(result)
            EntityRegistry.version_entity(original_entity)
            # ❌ NO TRACKING SET!
                
        elif semantic == "creation":
            result.promote_to_root()
            # ❌ NO TRACKING SET!
            
        elif semantic == "detachment":
            result.detach()
            if original_entity:
                EntityRegistry.version_entity(original_entity)
            # ❌ NO TRACKING SET!
        
        return result
```

### BUG FOUND #2
`_finalize_single_entity_result` doesn't call `_apply_semantic_actions` which is where tracking is set!

## Path 2: Borrowing Execution

### When Used
- No Entity inputs OR address-based inputs
- Pattern type is "pure_borrowing"

### Flow
```
_execute_borrowing(metadata, kwargs, classification)
  ↓
# NO execution_id created at this level!
  ↓
result = metadata.original_function(**function_args)
  ↓
if is_multi_entity:
    object_identity_map = {}
    execution_id = uuid4()  # ✅ Created here
    return await cls._finalize_multi_entity_result(
        result, metadata, object_identity_map, input_entity, execution_id
    )
else:
    output_entity = await cls._create_output_entity_with_provenance(
        result, metadata.output_entity_class, input_entity, metadata.name
        # ❌ NO execution_id passed!
    )
```

### _create_output_entity_with_provenance (Current)
```python
async def _create_output_entity_with_provenance(
    cls,
    result: Any,
    output_entity_class: Type[Entity],
    input_entity: Entity,
    function_name: str
    # ❌ NO execution_id PARAMETER!
) -> Entity:
    
    if isinstance(result, Entity):
        output_entity = result
        
        if hasattr(output_entity, 'derived_from_function'):
            output_entity.derived_from_function = function_name  # ✅ SET
        
        # ❌ derived_from_execution_id NOT SET!
```

### BUG FOUND #3
`_create_output_entity_with_provenance` sets `derived_from_function` but not `derived_from_execution_id`!

## Path 3: Multi-Entity (WORKING CORRECTLY)

### _finalize_multi_entity_result
```python
async def _finalize_multi_entity_result(
    cls,
    result: Any,
    metadata: FunctionMetadata,
    object_identity_map: Dict[int, Entity],
    input_entity: Optional[Entity] = None,
    execution_id: Optional[UUID] = None  # ✅ HAS PARAMETER
) -> Union[Entity, List[Entity]]:
    
    if execution_id is None:
        execution_id = uuid4()
    
    for entity in unpacking_result.primary_entities:
        if isinstance(entity, Entity):
            semantic, original_entity = cls._detect_execution_semantic(entity, object_identity_map)
            
            # ✅ CALLS _apply_semantic_actions!
            processed_entity = await cls._apply_semantic_actions(
                entity, semantic, original_entity, metadata, execution_id
            )
```

### _apply_semantic_actions (THE CORRECT WAY)
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
        entity.derived_from_function = metadata.name  # ✅
        entity.derived_from_execution_id = execution_id  # ✅
        
    elif semantic == "creation":
        entity.derived_from_function = metadata.name  # ✅
        entity.derived_from_execution_id = execution_id  # ✅
        if not entity.is_root_entity():
            entity.promote_to_root()
        
    elif semantic == "detachment":
        entity.detach()
        if original_entity:
            EntityRegistry.version_entity(original_entity)
        entity.derived_from_function = metadata.name  # ✅
        entity.derived_from_execution_id = execution_id  # ✅
    
    return entity
```

## Summary of Bugs

### Bug #1: Transactional Path - Missing execution_id Parameter
**File**: `callable_registry.py` line ~1010
**Current**:
```python
return await cls._finalize_single_entity_result(result, metadata, object_identity_map)
```
**Fix**:
```python
return await cls._finalize_single_entity_result(result, metadata, object_identity_map, execution_id)
```

### Bug #2: _finalize_single_entity_result - Doesn't Call _apply_semantic_actions
**File**: `callable_registry.py` line ~1146
**Current**: Function signature has no `execution_id`, doesn't call `_apply_semantic_actions`
**Fix**: Add parameter and call `_apply_semantic_actions`

### Bug #3: Borrowing Path - Missing execution_id in Single-Entity Case
**File**: `callable_registry.py` line ~930
**Current**: Doesn't create or pass `execution_id`
**Fix**: Create `execution_id` and pass to `_create_output_entity_with_provenance`

### Bug #4: _create_output_entity_with_provenance - Doesn't Set execution_id
**File**: `callable_registry.py` line ~1477
**Current**: Only sets `derived_from_function`
**Fix**: Also set `derived_from_execution_id`

## Next Steps

1. Verify exact line numbers in actual file
2. Create minimal fix patches
3. Test each fix individually
4. Submit PR to framework if needed
