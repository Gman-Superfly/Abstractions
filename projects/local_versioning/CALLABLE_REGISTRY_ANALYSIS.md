# CallableRegistry Performance Analysis

**Current Overhead**: 20-25ms per operation (51-64% of total time)  
**Target**: Reduce to 5-10ms (another 2-3x speedup)  
**Total potential**: 39ms → 15-20ms per operation

---

## 🔍 Current Execution Flow Analysis

### Entry Point: `execute()` → `aexecute()` → `_execute_async()`

```python
# Sync wrapper (minimal overhead)
def execute(cls, func_name: str, skip_divergence_check: bool = False, **kwargs):
    return asyncio.run(cls.aexecute(func_name, skip_divergence_check, **kwargs))
    # Cost: ~1-2ms (asyncio.run overhead)

# Async entry with event emission
@emit_events(creating_factory=..., created_factory=...)
async def aexecute(cls, func_name: str, skip_divergence_check: bool = False, **kwargs):
    return await cls._execute_async(func_name, skip_divergence_check, **kwargs)
    # Cost: Event emission overhead ~1-2ms
```

**Overhead so far**: ~2-4ms

---

## 📊 Step-by-Step Breakdown of `_execute_async()`

### Step 1: Get Function Metadata (Lines 577-580)

```python
metadata = cls.get_metadata(func_name)
if not metadata:
    raise ValueError(f"Function '{func_name}' not registered")
```

**Cost**: <0.1ms (dict lookup)  
**Optimization potential**: None (already O(1))

---

### Step 2: Detect Execution Strategy (Lines 582-583)

```python
strategy = cls._detect_execution_strategy(kwargs, metadata)
```

**What it does** (Lines 524-565):
```python
def _detect_execution_strategy(cls, kwargs, metadata):
    # Count entity parameters
    entity_params = [k for k, v in kwargs.items() if isinstance(v, Entity)]
    primitive_params = [k for k in kwargs if k not in entity_params]
    
    # Check for ConfigEntity
    function_expects_config_entity = ...  # Type hint inspection
    config_params = [k for k, v in kwargs.items() if isinstance(v, ConfigEntity)]
    
    # Decision tree with 7 branches:
    if len(entity_params) > 1:
        return "multi_entity_composite"
    elif len(entity_params) == 1 and not primitive_params and not config_params:
        return "single_entity_direct"
    elif function_expects_config_entity or config_params:
        return "single_entity_with_config"
    # ... 4 more branches
```

**Cost**: ~0.5-1ms
- Iterate kwargs multiple times
- Type checking with `isinstance()`
- Type hint inspection

**Optimization potential**: **HIGH**
- Metadata could cache expected strategy at registration time
- Most functions have predictable patterns
- Could skip detection if strategy is known

---

### Step 3: Route to Execution Strategy (Lines 585-606)

```python
if strategy == "single_entity_with_config":
    return await cls._execute_with_partial(metadata, kwargs, skip_divergence_check)
elif strategy == "no_inputs":
    return await cls._execute_no_inputs(metadata)
elif strategy in ["multi_entity_composite", "single_entity_direct"]:
    pattern_type, classification = InputPatternClassifier.classify_kwargs(kwargs)
    if pattern_type in ["pure_transactional", "mixed"]:
        return await cls._execute_transactional(metadata, kwargs, classification, skip_divergence_check)
    else:
        return await cls._execute_borrowing(metadata, kwargs, classification, skip_divergence_check)
else:  # pure_borrowing
    pattern_type, classification = InputPatternClassifier.classify_kwargs(kwargs)
    return await cls._execute_borrowing(metadata, kwargs, classification, skip_divergence_check)
```

**Additional classification** (InputPatternClassifier):
```python
def classify_kwargs(kwargs):
    # More type checking and analysis
    # Determines if transactional or borrowing pattern
    # Returns pattern_type and classification dict
```

**Cost**: ~0.5-1ms  
**Optimization potential**: **MEDIUM**
- Classification could be cached at registration
- Most functions follow same pattern every time

**Total routing overhead**: ~1-2ms

---

## 🎯 Deep Dive: `_execute_transactional()` (Our Main Path)

This is where `move_agent_global` goes. Let's break it down:

### Phase 1: Prepare Transactional Inputs (Lines 1000-1001)

```python
execution_kwargs, original_entities, execution_copies, object_identity_map = \
    await cls._prepare_transactional_inputs(kwargs, skip_divergence_check)
```

**What happens in `_prepare_transactional_inputs()`** (Lines 1031-1070):

```python
async def _prepare_transactional_inputs(cls, kwargs, skip_divergence_check):
    execution_kwargs = {}
    original_entities = []
    execution_copies = []
    object_identity_map = {}
    
    for param_name, value in kwargs.items():
        if isinstance(value, Entity):
            # 1. Check divergence (unless skipped)
            if not skip_divergence_check:
                await cls._check_entity_divergence(value)
                # Cost: ~10-15ms (build tree + diff)
                # NOW OPTIMIZED: ~0.003ms with early exit!
            
            # 2. Store original
            original_entities.append(value)
            
            # 3. Get execution copy from storage
            if value.root_ecs_id:
                copy = EntityRegistry.get_stored_entity(value.root_ecs_id, value.ecs_id)
                # Cost: ~2-3ms (get tree + extract entity)
                
                if copy:
                    execution_copies.append(copy)
                    execution_kwargs[param_name] = copy
                    object_identity_map[id(copy)] = value
                else:
                    # Deep copy fallback
                    copy = value.model_copy(deep=True)
                    copy.live_id = uuid4()
                    # Cost: ~5-10ms for deep copy
            else:
                # Orphan entity - deep copy
                copy = value.model_copy(deep=True)
                copy.live_id = uuid4()
        else:
            # Non-entity values pass through
            execution_kwargs[param_name] = value
    
    return execution_kwargs, original_entities, execution_copies, object_identity_map
```

**Cost breakdown**:
- Divergence check (if not skipped): ~0.003ms ✅ (optimized!)
- Get stored entity: ~2-3ms per entity
- Object identity tracking: ~0.1ms
- **Total**: ~2-5ms per entity input

**For `move_agent_global(gridmap, source_index, agent_name, target_index)`**:
- 1 entity (gridmap)
- 3 primitives (indices, name)
- **Cost**: ~2-5ms

**Optimization potential**: **MEDIUM**
- `get_stored_entity()` does full tree retrieval then extracts one entity
- Could optimize to extract entity without full tree copy
- Object identity map might be overkill for simple cases

---

### Phase 2: Execute Function (Lines 1006-1014)

```python
try:
    if metadata.is_async:
        result = await metadata.original_function(**execution_kwargs)
    else:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: metadata.original_function(**execution_kwargs))
except Exception as e:
    # Error handling
```

**Cost**: 
- Async function: ~0.1ms overhead
- Sync function: ~1-2ms (executor overhead)
- **Actual function execution**: ~0.01ms (just list operations)

**For our case**: Sync function, so ~1-2ms overhead

**Optimization potential**: **LOW**
- This is Python async machinery overhead
- Hard to optimize without breaking async support

---

### Phase 3: Semantic Detection (Lines 1016-1028)

```python
return await cls._finalize_single_entity_result(result, metadata, object_identity_map, execution_id)
```

**What happens in `_finalize_single_entity_result()`**:

```python
async def _finalize_single_entity_result(cls, result, metadata, object_identity_map, execution_id):
    # 1. Detect semantic (mutation/creation/detachment)
    semantic = await cls._detect_execution_semantic(
        result, metadata, object_identity_map, None, execution_id
    )
    # Cost: ~0.5-1ms (object identity checks)
    
    # 2. Apply semantic actions
    await cls._apply_semantic_actions(
        semantic, result, metadata, object_identity_map, None, execution_id
    )
    # Cost: ~5-10ms (versioning!)
    
    return result
```

**Semantic detection** (Lines 1102-1165):
```python
async def _detect_execution_semantic(cls, result, metadata, object_identity_map, input_entity, execution_id):
    # Check if result is in object_identity_map
    if id(result) in object_identity_map:
        original = object_identity_map[id(result)]
        
        # MUTATION: result is same object as input
        if result is original:
            return "mutation"
        
        # Check if result entity matches original by ecs_id
        if result.ecs_id == original.ecs_id:
            return "mutation"
    
    # CREATION: result not in inputs
    return "creation"
```

**Cost**: ~0.5-1ms (just identity checks)

**Optimization potential**: **LOW**
- Already very fast
- Necessary for framework semantics

---

### Phase 4: Apply Semantic Actions (Lines 1167-1240)

```python
async def _apply_semantic_actions(cls, semantic, result, metadata, object_identity_map, input_entity, execution_id):
    if semantic == "mutation":
        # Version the mutated entity
        if result.is_root_entity():
            EntityRegistry.version_entity(result)
            # Cost: ~5-10ms (build tree + diff + version)
    
    elif semantic == "creation":
        # Register new entity
        if result.is_root_entity():
            EntityRegistry.register_entity(result)
    
    elif semantic == "detachment":
        # Handle detachment
        pass
```

**For mutation (our case)**:
- Build new tree: ~2.3ms ✅ (optimized!)
- Get old tree (read-only): ~5.6ms (shallow copy)
- Diff: ~0.003ms ✅ (optimized!)
- Update IDs: ~1-2ms
- Register tree: ~1-2ms
- **Total**: ~10-12ms

**Optimization potential**: **MEDIUM**
- Tree retrieval (5.6ms) could be optimized further
- ID updates might be optimizable
- Tree registration might have overhead

---

## 📊 Total Overhead Breakdown (39ms operation)

| Phase | Operation | Cost | % | Optimized? |
|-------|-----------|------|---|------------|
| **Entry** | asyncio.run + events | 2-4ms | 5-10% | ❌ |
| **Routing** | Strategy detection + classification | 1-2ms | 3-5% | ❌ |
| **Prepare** | Get stored entity + object tracking | 2-5ms | 5-13% | ⚠️ |
| **Execute** | Async executor overhead | 1-2ms | 3-5% | ❌ |
| **Semantic** | Object identity detection | 0.5-1ms | 1-3% | ✅ |
| **Versioning** | Build tree + diff + register | 10-12ms | 26-31% | ⚠️ |
| **Function** | Actual work (list ops) | 0.01ms | 0% | ✅ |
| **Other** | Events, tracking, misc | 5-10ms | 13-26% | ❌ |
| **TOTAL** | | **22-36ms** | **56-92%** | |

**Actual function execution**: 0.01ms (0.03% of time!)  
**Framework overhead**: 22-36ms (56-92% of time!)

---

## 🎯 Optimization Opportunities (Ranked)

### P0: Optimize Tree Retrieval in Versioning (HIGH IMPACT)

**Current**: `get_stored_tree()` returns shallow copy, but still ~5.6ms

**Problem**:
```python
# In version_entity()
old_tree = EntityRegistry.get_stored_tree(root_ecs_id, read_only=True)
# Returns: stored_tree.model_copy(deep=False)
# Cost: 5.6ms for 111 entities
```

**Why slow?**
- Pydantic `model_copy()` even with `deep=False` has overhead
- Copies all dict references
- Validates model structure

**Solution**:
```python
# Option A: Direct reference for read-only
if read_only:
    # Just return the stored tree directly
    # It's immutable, so safe for read-only operations
    return stored_tree

# Option B: Custom shallow copy
if read_only:
    # Manually copy only what's needed for diff
    return EntityTreeView(
        nodes=stored_tree.nodes,  # Share reference
        edges=stored_tree.edges,  # Share reference
        # ... other fields
    )
```

**Expected speedup**: 5.6ms → 0.5ms (10x faster)  
**Impact on total**: 39ms → 34ms (1.15x faster)

---

### P1: Cache Execution Strategy at Registration (MEDIUM IMPACT)

**Current**: Detect strategy on every execution

**Problem**:
```python
# Every call does this:
strategy = cls._detect_execution_strategy(kwargs, metadata)
# Cost: ~0.5-1ms
```

**Why wasteful?**
- Most functions have predictable patterns
- `move_agent_global(gridmap, ...)` is ALWAYS "single_entity_with_config"
- Strategy never changes for a given function

**Solution**:
```python
# At registration time:
class FunctionMetadata:
    name: str
    original_function: Callable
    expected_strategy: str  # ← Cache this!
    expected_pattern: str   # ← Cache this!
    # ...

# At execution time:
if metadata.expected_strategy:
    strategy = metadata.expected_strategy  # ← Skip detection!
else:
    strategy = cls._detect_execution_strategy(kwargs, metadata)
```

**Expected speedup**: 0.5-1ms saved  
**Impact on total**: 39ms → 38ms (1.03x faster)

---

### P2: Optimize get_stored_entity() (MEDIUM IMPACT)

**Current**: Gets full tree, then extracts one entity

**Problem**:
```python
# In _prepare_transactional_inputs()
copy = EntityRegistry.get_stored_entity(value.root_ecs_id, value.ecs_id)

# Implementation:
def get_stored_entity(cls, root_ecs_id, ecs_id):
    tree = cls.get_stored_tree(root_ecs_id)  # ← Gets full tree!
    return tree.get_entity(ecs_id)           # ← Extracts one entity
```

**Cost**: ~2-3ms (full tree retrieval for one entity)

**Solution**:
```python
def get_stored_entity(cls, root_ecs_id, ecs_id):
    # Direct lookup without full tree copy
    stored_tree = cls.tree_registry.get(root_ecs_id)
    if stored_tree:
        entity = stored_tree.nodes.get(ecs_id)
        if entity:
            # Return shallow copy of just this entity
            return entity.model_copy(deep=False)
    return None
```

**Expected speedup**: 2-3ms → 0.2ms (10x faster)  
**Impact on total**: 39ms → 37ms (1.05x faster)

---

### P3: Reduce Event Emission Overhead (LOW-MEDIUM IMPACT)

**Current**: Multiple event decorators on execution path

**Problem**:
```python
@emit_events(creating_factory=..., created_factory=...)
async def aexecute(...):
    # Event overhead ~1-2ms

@emit_events(creating_factory=..., created_factory=...)
async def _execute_with_partial(...):
    # Event overhead ~1-2ms

@emit_events(creating_factory=..., created_factory=...)
async def _execute_transactional(...):
    # Event overhead ~1-2ms
```

**Total event overhead**: ~3-6ms

**Solution**:
```python
# Option A: Add flag to skip events for performance
CallableRegistry.execute(
    "move_agent_global",
    skip_events=True,  # ← Skip event emission
    **kwargs
)

# Option B: Batch events (emit once at end)
# Option C: Make events opt-in instead of always-on
```

**Expected speedup**: 3-6ms saved  
**Impact on total**: 39ms → 33-36ms (1.08-1.18x faster)

---

### P4: Eliminate Async Overhead for Sync Functions (LOW IMPACT)

**Current**: Sync functions go through async executor

**Problem**:
```python
if metadata.is_async:
    result = await metadata.original_function(**execution_kwargs)
else:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: metadata.original_function(**execution_kwargs))
    # Cost: ~1-2ms executor overhead
```

**Solution**:
```python
# Add fast path for sync functions
if metadata.is_async:
    result = await metadata.original_function(**execution_kwargs)
else:
    # Direct call for sync functions (no executor)
    result = metadata.original_function(**execution_kwargs)
    # Cost: ~0.01ms
```

**Expected speedup**: 1-2ms saved  
**Impact on total**: 39ms → 37-38ms (1.03-1.05x faster)

---

### P5: Optimize Object Identity Tracking (LOW IMPACT)

**Current**: Creates object_identity_map for every execution

**Problem**:
```python
object_identity_map = {}  # Maps id(execution_copy) -> original_entity

for param_name, value in kwargs.items():
    if isinstance(value, Entity):
        # ...
        object_identity_map[id(copy)] = value
```

**Why might be wasteful?**
- For simple mutations, we know the semantic upfront
- Object identity map is only used for semantic detection
- Could skip if semantic is predictable

**Solution**:
```python
# If function metadata indicates "always mutation":
if metadata.semantic_hint == "mutation":
    # Skip object identity tracking
    object_identity_map = None
```

**Expected speedup**: 0.1-0.5ms saved  
**Impact on total**: 39ms → 38.5-38.9ms (1.01x faster)

---

## 🚀 Combined Optimization Potential

### Conservative Estimate

| Optimization | Speedup | Cumulative |
|--------------|---------|------------|
| **Baseline** | 39ms | 1.0x |
| P0: Tree retrieval | -5ms | 34ms (1.15x) |
| P1: Cache strategy | -1ms | 33ms (1.18x) |
| P2: get_stored_entity | -2ms | 31ms (1.26x) |
| P3: Event overhead | -3ms | 28ms (1.39x) |
| P4: Async overhead | -1ms | 27ms (1.44x) |
| **TOTAL** | **-12ms** | **27ms (1.44x)** |

### Optimistic Estimate

| Optimization | Speedup | Cumulative |
|--------------|---------|------------|
| **Baseline** | 39ms | 1.0x |
| P0: Tree retrieval | -5ms | 34ms (1.15x) |
| P1: Cache strategy | -1ms | 33ms (1.18x) |
| P2: get_stored_entity | -3ms | 30ms (1.30x) |
| P3: Event overhead | -6ms | 24ms (1.63x) |
| P4: Async overhead | -2ms | 22ms (1.77x) |
| P5: Object tracking | -0.5ms | 21.5ms (1.81x) |
| **TOTAL** | **-17.5ms** | **21.5ms (1.81x)** |

---

## 🎓 Key Insights

### 1. Most Overhead is Framework Machinery, Not Algorithms

**The actual work** (list operations): 0.01ms (0.03%)  
**Framework overhead**: 22-36ms (56-92%)

This is the price of:
- Automatic versioning
- Event emission
- Semantic detection
- Isolation guarantees
- Async support

### 2. Tree Operations Are Now Optimized

- Build tree: 2.3ms ✅
- Diff: 0.003ms ✅
- These are no longer bottlenecks!

### 3. Remaining Bottlenecks Are Structural

1. **Tree retrieval** (5.6ms) - Pydantic overhead
2. **Event emission** (3-6ms) - Multiple decorators
3. **Async machinery** (3-4ms) - asyncio.run + executor
4. **Entity copying** (2-3ms) - get_stored_entity

### 4. Strategy Detection is Wasteful

- Same function always follows same pattern
- Detection happens every call
- Could be cached at registration

### 5. Trade-offs Are Necessary

**Fast path vs. Safe path**:
- Could add `fast_execute()` that skips safety checks
- But defeats purpose of framework
- Better to optimize the safe path

---

## 📋 Recommended Implementation Order

### Session 1: Low-Hanging Fruit (2-3 hours)
1. **P0**: Optimize tree retrieval (direct reference for read-only)
2. **P1**: Cache execution strategy at registration
3. **P4**: Direct call for sync functions

**Expected**: 39ms → 30-32ms (1.2-1.3x faster)

### Session 2: Structural Changes (4-6 hours)
1. **P2**: Optimize get_stored_entity (direct entity lookup)
2. **P3**: Add flag to skip events for performance
3. **P5**: Skip object tracking when semantic is known

**Expected**: 30-32ms → 22-25ms (1.5-1.8x faster)

### Session 3: Advanced (Optional, 6-8 hours)
1. Custom shallow copy for EntityTree
2. Batch event emission
3. Lazy object identity tracking
4. Profile and optimize remaining overhead

**Expected**: 22-25ms → 15-20ms (2.0-2.6x faster)

---

## 🎯 Final Target

**Current**: 39ms per operation  
**After Session 1**: 30-32ms (1.2-1.3x faster)  
**After Session 2**: 22-25ms (1.5-1.8x faster)  
**After Session 3**: 15-20ms (2.0-2.6x faster)

**Combined with current optimizations**:
- Original: 148ms
- Current: 39ms (3.8x faster)
- Final target: 15-20ms (7.4-9.9x total speedup!)

---

## ✅ Conclusion

**CallableRegistry overhead is real but optimizable!**

The framework provides immense value (versioning, events, semantics), but we can make it faster without sacrificing safety:

1. **Tree retrieval** can be 10x faster with direct references
2. **Strategy detection** can be eliminated with caching
3. **Event emission** can be made optional
4. **Entity copying** can be optimized

**Next session**: Implement P0-P4 for 1.5-1.8x additional speedup!
