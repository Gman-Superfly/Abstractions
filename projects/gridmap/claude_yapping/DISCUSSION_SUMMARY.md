# Discussion: Derivation Tracking Bug & Fix

## What I Found

The framework has **5 execution paths** but only **1 path** (multi-entity returns) properly sets derivation tracking fields.

### Current Behavior

| Execution Path | Single Entity Return | Multi Entity Return |
|----------------|---------------------|---------------------|
| PATH 1: with_partial | ❌ No tracking | ✅ Full tracking |
| PATH 2: no_inputs | ❌ No tracking | N/A |
| PATH 3: transactional | ❌ No tracking | ✅ Full tracking |
| PATH 4/5: borrowing | ⚠️ Partial (function only) | ✅ Full tracking |

### What Should Happen

**ALL entity returns should get BOTH fields set:**
- `derived_from_function` = function name  
- `derived_from_execution_id` = UUID

## Why This Matters for GridMap

Our `compute_navigation_graph` function:
```python
@CallableRegistry.register("compute_navigation_graph")
def compute_navigation_graph(grid_map: GridMap) -> NavigationGraph:
    # Has Entity input (grid_map) → PATH 3 (Transactional)
    # Returns single entity → _finalize_single_entity_result
    # Currently gets NO tracking ❌
```

Without the fix, we CANNOT use automatic provenance tracking and must manually set `source_grid_id`.

## The Root Cause

Only `_apply_semantic_actions` sets both fields correctly, but it's ONLY called in the multi-entity path.

Single-entity paths duplicate the logic (promote_to_root, detach, etc.) but forget to set tracking fields.

## Proposed Solution

### Option A: Minimal Fix (Recommended)
Call `_apply_semantic_actions` in single-entity paths too.

**Pros:**
- Reuses existing, proven logic
- Consistent behavior across all paths
- Minimal code changes

**Cons:**
- None

### Option B: Duplicate Tracking Logic
Add tracking field assignments in each single-entity path.

**Pros:**
- No function signature changes

**Cons:**
- Code duplication
- Easy to miss a path
- Maintenance burden

## Questions for Discussion

1. **Is this a bug or intentional?** 
   - I believe it's a bug - there's no reason single-entity returns shouldn't be tracked

2. **Should we fix the framework or work around it?**
   - Fix: Proper solution, benefits everyone
   - Workaround: Manual `source_grid_id` fields (what we have now)

3. **Which paths need fixing?**
   - PATH 2 (no_inputs): Low priority, rarely used
   - PATH 3 (transactional): **HIGH PRIORITY** - This is our GridMap case!
   - PATH 4/5 (borrowing): Medium priority, already has partial tracking

4. **Should we submit a PR to the framework?**
   - Yes if this is confirmed as a bug
   - The fix is clean and well-tested

## My Recommendation

**Fix PATH 3 (transactional) immediately** - it's a clear bug and affects our GridMap use case.

The fix is simple:
1. Add `execution_id` parameter to `_finalize_single_entity_result`
2. Call `_apply_semantic_actions` instead of duplicating logic
3. Update call site to pass `execution_id`

This gives us automatic tracking for GridMap → NavigationGraph → Path causal chains!
