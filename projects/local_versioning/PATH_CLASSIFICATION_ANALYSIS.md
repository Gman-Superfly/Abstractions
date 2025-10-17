# Path Classification Pre-Computation Analysis

**Goal**: Determine what can be cached at registration time vs. what must be computed at runtime.

**Current overhead**: ~1-2ms per call for strategy detection + pattern classification

---

## 🔍 Current Two-Stage Classification System

### Stage 1: Strategy Detection (`_detect_execution_strategy`)

**Input**: `kwargs` + `metadata`  
**Output**: Strategy string

**Strategies**:
1. `"single_entity_direct"` - Pure single entity, no config
2. `"multi_entity_composite"` - 2+ entities → composite
3. `"single_entity_with_config"` - Entity + config/primitives
4. `"no_inputs"` - No inputs
5. `"pure_borrowing"` - Fallback for address-based

**What it checks** (Lines 531-572):
```python
def _detect_execution_strategy(cls, kwargs, metadata):
    sig = signature(metadata.original_function)
    type_hints = get_type_hints(metadata.original_function)
    
    # Runtime checks on kwargs:
    entity_params = []      # Count Entity instances in kwargs
    config_params = []      # Count ConfigEntity instances in kwargs
    primitive_params = {}   # Count primitives in kwargs
    
    # Static check on function signature:
    function_expects_config_entity = any(
        is_top_level_config_entity(type_hints.get(param.name))
        for param in sig.parameters.values()
    )
    
    # Decision tree based on counts:
    if len(entity_params) == 1 and not primitive_params and not function_expects_config_entity:
        return "single_entity_direct"
    elif len(entity_params) >= 2:
        return "multi_entity_composite"
    elif function_expects_config_entity or config_params:
        return "single_entity_with_config"
    # ... etc
```

### Stage 2: Pattern Classification (`InputPatternClassifier.classify_kwargs`)

**Input**: `kwargs`  
**Output**: `(pattern_type, classification_dict)`

**Pattern Types**:
1. `"pure_transactional"` - All direct Entity objects
2. `"pure_borrowing"` - All @uuid.field address strings
3. `"mixed"` - Both entities and addresses
4. `"direct"` - Only primitives

**What it checks** (Lines 394-434):
```python
def classify_kwargs(cls, kwargs):
    entity_count = 0
    address_count = 0
    direct_count = 0
    
    for key, value in kwargs.items():
        if hasattr(value, 'ecs_id'):  # Entity instance
            entity_count += 1
        elif isinstance(value, str) and cls.is_ecs_address(value):  # @uuid.field
            address_count += 1
        else:  # Primitive
            direct_count += 1
    
    # Determine pattern
    if entity_count > 0 and address_count == 0:
        return "pure_transactional"
    elif address_count > 0 and entity_count == 0:
        return "pure_borrowing"
    elif entity_count > 0 and address_count > 0:
        return "mixed"
    else:
        return "direct"
```

---

## 📊 What Can Be Pre-Computed?

### ✅ STATIC: Can Be Cached at Registration

These depend ONLY on function signature, not runtime values:

#### 1. Function Signature Analysis
```python
sig = signature(metadata.original_function)
type_hints = get_type_hints(metadata.original_function)
```
**Cost**: ~0.1-0.2ms  
**Cacheable**: YES - signature never changes

#### 2. Expected Config Entity Check
```python
function_expects_config_entity = any(
    is_top_level_config_entity(type_hints.get(param.name))
    for param in sig.parameters.values()
)
```
**Cost**: ~0.05-0.1ms  
**Cacheable**: YES - type hints never change

#### 3. Parameter Names and Types
```python
param_names = list(sig.parameters.keys())
param_types = {name: type_hints.get(name) for name in param_names}
```
**Cost**: ~0.05ms  
**Cacheable**: YES - static metadata

---

### ❌ DYNAMIC: Must Be Runtime

These depend on actual values passed at call time:

#### 1. Entity Count in kwargs
```python
entity_params = [k for k, v in kwargs.items() if isinstance(v, Entity)]
```
**Why dynamic**: User can pass different numbers of entities each call
- Call 1: `func(entity1)` → 1 entity
- Call 2: `func(entity1, entity2)` → 2 entities

**BUT**: If function signature is strict, we can infer!
```python
def move_agent(gridmap: GridMap, source_index: int, ...) -> GridMap:
    # ↑ Always expects exactly 1 Entity (GridMap)
    # ↑ Always expects primitives (ints)
```

#### 2. Primitive Count in kwargs
```python
primitive_params = {k: v for k, v in kwargs.items() if not isinstance(v, Entity)}
```
**Why dynamic**: User can pass different primitives
- Call 1: `func(entity, x=1)` → 1 primitive
- Call 2: `func(entity, x=1, y=2)` → 2 primitives

**BUT**: If function signature is strict, we can infer!

#### 3. Address vs Entity Detection
```python
if isinstance(value, str) and cls.is_ecs_address(value):
    # Address string like "@uuid.field"
elif hasattr(value, 'ecs_id'):
    # Entity instance
```
**Why dynamic**: User can pass either:
- `func(entity_instance)` → transactional
- `func("@uuid.field")` → borrowing

**This CANNOT be pre-computed!**

---

## 🎯 Key Insight: Type Hints vs Runtime Values

### The Problem

Python's type system is **optional and not enforced**:

```python
def move_agent(gridmap: GridMap, source_index: int) -> GridMap:
    pass

# All of these are valid Python:
move_agent(gridmap, 5)                    # ✅ As expected
move_agent(gridmap, "5")                  # ⚠️ Wrong type, but Python allows it
move_agent("@uuid.field", 5)              # ⚠️ Address instead of entity
move_agent(gridmap, 5, extra_param=10)    # ⚠️ Extra param
```

**We CANNOT trust type hints alone!** We must inspect runtime values.

---

## 💡 What CAN We Pre-Compute?

### Strategy 1: Signature-Based Hints (Conservative)

Cache **expected** strategy based on signature, but still validate at runtime:

```python
class FunctionMetadata:
    # ... existing fields ...
    
    # NEW: Cached signature analysis
    expected_entity_count: int           # From type hints
    expected_primitive_count: int        # From type hints
    expects_config_entity: bool          # From type hints
    expected_strategy: Optional[str]     # Predicted strategy
    
    # NEW: Fast path flag
    has_strict_signature: bool           # True if signature is unambiguous
```

**At registration time**:
```python
def analyze_function_signature(func):
    sig = signature(func)
    type_hints = get_type_hints(func)
    
    entity_params = [
        name for name, param in sig.parameters.items()
        if is_entity_type(type_hints.get(name))
    ]
    
    primitive_params = [
        name for name, param in sig.parameters.items()
        if not is_entity_type(type_hints.get(name))
    ]
    
    expects_config = any(
        is_top_level_config_entity(type_hints.get(name))
        for name in sig.parameters
    )
    
    # Predict strategy
    if len(entity_params) == 1 and not expects_config and not primitive_params:
        expected_strategy = "single_entity_direct"
    elif len(entity_params) >= 2:
        expected_strategy = "multi_entity_composite"
    elif expects_config:
        expected_strategy = "single_entity_with_config"
    elif len(entity_params) == 0:
        expected_strategy = "no_inputs"
    else:
        expected_strategy = None  # Ambiguous
    
    return FunctionMetadata(
        expected_entity_count=len(entity_params),
        expected_primitive_count=len(primitive_params),
        expects_config_entity=expects_config,
        expected_strategy=expected_strategy,
        has_strict_signature=(expected_strategy is not None)
    )
```

**At execution time**:
```python
def _detect_execution_strategy_fast(cls, kwargs, metadata):
    # Fast path: Use cached strategy if signature is strict
    if metadata.has_strict_signature:
        # Quick validation: check actual kwargs match expectations
        actual_entity_count = sum(1 for v in kwargs.values() if isinstance(v, Entity))
        
        if actual_entity_count == metadata.expected_entity_count:
            # Signature matches! Use cached strategy
            return metadata.expected_strategy
    
    # Slow path: Full runtime detection (fallback)
    return cls._detect_execution_strategy_full(kwargs, metadata)
```

**Speedup**: 
- Fast path: ~0.1ms (just count entities)
- Slow path: ~1ms (full analysis)
- **Savings**: ~0.9ms when fast path works

---

### Strategy 2: Pattern Classification Hints

For `InputPatternClassifier`, we **CANNOT** pre-compute because:
- Address strings (`"@uuid.field"`) vs Entity instances is runtime-only
- User can pass either on any call

**BUT** we can optimize the check:

```python
# Current: Check every value
for key, value in kwargs.items():
    if hasattr(value, 'ecs_id'):
        entity_count += 1
    elif isinstance(value, str) and cls.is_ecs_address(value):
        address_count += 1

# Optimized: Early exit
entity_count = 0
address_count = 0

for value in kwargs.values():
    if hasattr(value, 'ecs_id'):
        entity_count += 1
    elif isinstance(value, str):
        # Only check if string (most values aren't strings)
        if value.startswith('@'):  # Quick pre-check
            if cls.is_ecs_address(value):
                address_count += 1

# Early exit: If we found entities and no addresses, we're done
if entity_count > 0 and address_count == 0:
    return "pure_transactional", {}
```

**Speedup**: ~0.1-0.2ms (early exit + faster string check)

---

## 🚨 Critical Safety Considerations

### Why We MUST Validate at Runtime

**Scenario 1: Wrong type passed**
```python
@register_function
def move_agent(gridmap: GridMap, index: int) -> GridMap:
    pass

# User passes wrong type:
CallableRegistry.execute("move_agent", gridmap="@uuid.field", index=5)
# ↑ Type hint says Entity, but user passed address string!
```

**If we trusted type hints**:
- Cached strategy: `"single_entity_direct"` (1 entity, no config)
- Actual runtime: Address string → should be `"pure_borrowing"`
- **WRONG PATH!** → Crash or incorrect behavior

### Why We MUST Keep Runtime Checks

**Scenario 2: Dynamic entity count**
```python
@register_function
def process(*entities: Entity) -> Entity:
    # Variadic args - can accept any number of entities
    pass

# Different calls:
process(entity1)           # 1 entity
process(entity1, entity2)  # 2 entities
```

**Cannot pre-compute**: Number of entities varies per call

### Why We MUST Check for Addresses

**Scenario 3: Borrowing pattern**
```python
@register_function
def get_agent(gridmap: GridMap, agent_address: str) -> Agent:
    pass

# User can pass either:
get_agent(gridmap, "@uuid.agents[0]")  # Borrowing pattern
get_agent(gridmap, "some_name")        # Just a string name
```

**Cannot pre-compute**: Whether string is address or just data

---

## 📊 Realistic Optimization Potential

### What We CAN Do (Safe)

#### 1. Cache Signature Analysis (~0.2ms saved)
```python
class FunctionMetadata:
    # Cache at registration:
    signature: Signature                    # ← Cache this
    type_hints: Dict[str, Type]            # ← Cache this
    expects_config_entity: bool            # ← Cache this
    param_entity_types: List[str]          # ← Cache this
    param_primitive_types: List[str]       # ← Cache this
```

**Speedup**: ~0.2ms (avoid repeated `signature()` and `get_type_hints()` calls)

#### 2. Fast Path for Common Cases (~0.3-0.5ms saved)
```python
# At execution:
if metadata.expected_strategy == "single_entity_with_config":
    # Quick check: Do we have 1 entity + primitives?
    entity_count = sum(1 for v in kwargs.values() if isinstance(v, Entity))
    if entity_count == 1:
        # Fast path confirmed!
        return await cls._execute_with_partial(...)
```

**Speedup**: ~0.3-0.5ms (skip full strategy detection for common case)

#### 3. Optimize Pattern Classification (~0.2ms saved)
```python
# Early exit optimization
for value in kwargs.values():
    if hasattr(value, 'ecs_id'):
        entity_count += 1
    elif isinstance(value, str) and value.startswith('@'):
        if cls.is_ecs_address(value):
            address_count += 1

# Early exit
if entity_count > 0 and address_count == 0:
    return "pure_transactional", {}
```

**Speedup**: ~0.2ms (early exit + faster checks)

---

### What We CANNOT Do (Unsafe)

#### ❌ Skip Runtime Type Checking
**Why**: Type hints are not enforced in Python

#### ❌ Pre-compute Pattern Classification
**Why**: Address vs Entity is runtime-only

#### ❌ Cache Strategy Per Function
**Why**: Same function can be called with different patterns:
```python
# Same function, different patterns:
move_agent(gridmap, 5)              # Transactional
move_agent("@uuid.gridmap", 5)      # Borrowing
```

---

## 🎯 Recommended Implementation

### Phase 1: Safe Caching (Low Risk, ~0.2ms saved)

```python
class FunctionMetadata(BaseModel):
    # ... existing fields ...
    
    # NEW: Cached signature analysis
    cached_signature: Optional[Signature] = None
    cached_type_hints: Optional[Dict[str, Type]] = None
    expects_config_entity: bool = False
    
    @classmethod
    def from_function(cls, func, name):
        sig = signature(func)
        type_hints = get_type_hints(func)
        
        expects_config = any(
            is_top_level_config_entity(type_hints.get(param.name))
            for param in sig.parameters.values()
        )
        
        return cls(
            name=name,
            original_function=func,
            cached_signature=sig,           # ← Cache
            cached_type_hints=type_hints,   # ← Cache
            expects_config_entity=expects_config,  # ← Cache
            # ... other fields
        )
```

**Usage**:
```python
def _detect_execution_strategy(cls, kwargs, metadata):
    # Use cached values instead of recomputing
    sig = metadata.cached_signature
    type_hints = metadata.cached_type_hints
    function_expects_config_entity = metadata.expects_config_entity
    
    # Rest of logic unchanged...
```

**Impact**: ~0.2ms saved per call

---

### Phase 2: Fast Path Optimization (Medium Risk, ~0.3-0.5ms saved)

```python
class FunctionMetadata(BaseModel):
    # ... Phase 1 fields ...
    
    # NEW: Expected strategy hint
    expected_strategy_hint: Optional[str] = None
    
    @classmethod
    def from_function(cls, func, name):
        # ... Phase 1 logic ...
        
        # Analyze signature to predict common case
        entity_param_count = sum(
            1 for param in sig.parameters.values()
            if is_entity_type(type_hints.get(param.name))
        )
        
        if entity_param_count == 1 and expects_config:
            expected_hint = "single_entity_with_config"
        elif entity_param_count >= 2:
            expected_hint = "multi_entity_composite"
        else:
            expected_hint = None
        
        return cls(
            # ... other fields ...
            expected_strategy_hint=expected_hint
        )
```

**Usage**:
```python
def _detect_execution_strategy_fast(cls, kwargs, metadata):
    # Try fast path first
    if metadata.expected_strategy_hint:
        # Quick validation
        entity_count = sum(1 for v in kwargs.values() if isinstance(v, Entity))
        
        if metadata.expected_strategy_hint == "single_entity_with_config":
            if entity_count == 1:
                return "single_entity_with_config"
        elif metadata.expected_strategy_hint == "multi_entity_composite":
            if entity_count >= 2:
                return "multi_entity_composite"
    
    # Fallback to full detection
    return cls._detect_execution_strategy_full(kwargs, metadata)
```

**Impact**: ~0.3-0.5ms saved per call (when fast path works)

---

### Phase 3: Pattern Classification Optimization (Low Risk, ~0.2ms saved)

```python
@classmethod
def classify_kwargs_fast(cls, kwargs):
    entity_count = 0
    address_count = 0
    
    for value in kwargs.values():
        if hasattr(value, 'ecs_id'):
            entity_count += 1
            # Early exit: If we have entities and no addresses yet, keep checking
        elif isinstance(value, str):
            # Quick pre-filter: Most addresses start with @
            if value and value[0] == '@':
                if cls.is_ecs_address(value):
                    address_count += 1
    
    # Early determination
    if entity_count > 0 and address_count == 0:
        return "pure_transactional", {}
    elif address_count > 0 and entity_count == 0:
        return "pure_borrowing", {}
    elif entity_count > 0 and address_count > 0:
        return "mixed", {}
    else:
        return "direct", {}
```

**Impact**: ~0.2ms saved per call

---

## 📊 Total Optimization Potential

| Phase | Speedup | Risk | Effort |
|-------|---------|------|--------|
| Phase 1: Cache signature | 0.2ms | LOW | 1 hour |
| Phase 2: Fast path hints | 0.3-0.5ms | MEDIUM | 2-3 hours |
| Phase 3: Pattern optimization | 0.2ms | LOW | 1 hour |
| **TOTAL** | **0.7-0.9ms** | **LOW-MEDIUM** | **4-5 hours** |

---

## ⚠️ Important Caveats

### 1. Speedup is Small
- Current overhead: ~1-2ms
- Potential savings: ~0.7-0.9ms
- **Remaining overhead**: ~0.3-1.1ms

### 2. Complexity Cost
- Adds caching logic
- Adds fast path validation
- More code to maintain

### 3. Safety Trade-offs
- Fast path assumes type hints are correct
- Must still validate to catch misuse
- Cannot eliminate runtime checks entirely

---

## 🎓 Conclusion

### What We Learned

1. **Most classification MUST be runtime** because:
   - Python type hints are not enforced
   - Address vs Entity detection is runtime-only
   - Users can pass unexpected types

2. **Limited pre-computation possible**:
   - Signature analysis can be cached (~0.2ms)
   - Fast path hints can help (~0.3-0.5ms)
   - Pattern checks can be optimized (~0.2ms)

3. **Total realistic speedup: 0.7-0.9ms**
   - From 1-2ms → 0.3-1.1ms
   - Not a game-changer, but worthwhile

### Recommendation

**Implement Phase 1 only** (cache signature analysis):
- **Low risk** - just caching static data
- **Easy to implement** - 1 hour of work
- **Guaranteed safe** - no behavior changes
- **0.2ms speedup** - small but free

**Skip Phase 2 & 3 for now**:
- **Complexity not worth it** for 0.5-0.7ms
- **Better to focus on** other optimizations
- **Come back later** if needed

---

## 🚀 Next Steps

1. **Implement Phase 1** (cache signature) - 0.2ms saved, low risk
2. **Focus on bigger wins**:
   - Event emission overhead (3-6ms potential)
   - Other CallableRegistry overhead
3. **Revisit classification** only if it becomes a bottleneck again
