# Complete Execution Path Analysis

## All 5 Execution Paths from _execute_async (Line 568)

### PATH 1: single_entity_with_config
- **Trigger**: `strategy == "single_entity_with_config"`
- **Handler**: `_execute_with_partial(metadata, kwargs)` (Line 580)
- **Flow**: Creates partial function, then calls `_execute_transactional` (Line 701)
- **Final**: Routes to PATH 3 (Transactional)

### PATH 2: no_inputs  
- **Trigger**: `strategy == "no_inputs"`
- **Handler**: `_execute_no_inputs(metadata)` (Line 582)
- **Flow**:
  - Executes function (Line 1632-1635)
  - If result is Entity: `promote_to_root()` (Line 1643)
  - **NO `derived_from_function` SET!** ❌
  - **NO `derived_from_execution_id` SET!** ❌
  - Returns entity (Line 1658)

### PATH 3: Transactional (Entity inputs)
- **Trigger**: Has Entity inputs + pure_transactional/mixed pattern
- **Handler**: `_execute_transactional(metadata, kwargs, classification)` (Line 587)
- **Flow**:
  - Creates `execution_id = uuid4()` (Line 979) ✅
  - Executes function
  - If multi-entity: calls `_finalize_multi_entity_result(..., execution_id)` ✅
  - If single-entity: calls `_finalize_single_entity_result(...)` **WITHOUT execution_id** ❌
  
### PATH 4 & 5: Borrowing (No Entity inputs OR address-based)
- **Trigger**: No Entity inputs OR address-based pattern
- **Handler**: `_execute_borrowing(metadata, kwargs, classification)` (Line 589/592)
- **Flow**:
  - Executes function
  - If multi-entity: 
    - Creates `execution_id = uuid4()` (Line 923) ✅
    - Calls `_finalize_multi_entity_result(..., execution_id)` ✅
  - If single-entity:
    - Calls `_create_output_entity_with_provenance(...)` **WITHOUT execution_id** ❌
    - Sets `derived_from_function` (Line 1477) ✅
    - **NO `derived_from_execution_id` SET!** ❌

## Summary: What Currently Happens

| Path | Single Entity | Multi Entity |
|------|---------------|--------------|
| PATH 1 (with_partial) | Routes to PATH 3 | Routes to PATH 3 |
| PATH 2 (no_inputs) | ❌ NO tracking | N/A |
| PATH 3 (transactional) | ❌ NO tracking | ✅ Full tracking |
| PATH 4/5 (borrowing) | ⚠️ Partial (function only) | ✅ Full tracking |

## What SHOULD Happen

**ALL paths should set BOTH fields for ALL entity returns:**
- `derived_from_function` = function name
- `derived_from_execution_id` = execution UUID

## Our Test Cases Map To:

- **P10.1** (create_simple, no inputs): PATH 4/5 (Borrowing) → Partial tracking ⚠️
- **P10.2** (create_pair, tuple return): PATH 4/5 (Borrowing) → Full tracking ✅
- **P10.3** (mutate_simple, Entity input): PATH 3 (Transactional) → No tracking ❌
- **compute_navigation_graph** (GridMap input): PATH 3 (Transactional) → No tracking ❌
