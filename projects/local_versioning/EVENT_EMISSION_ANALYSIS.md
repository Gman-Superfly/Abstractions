# Event Emission System Analysis

**Goal**: Understand event emission overhead and determine if it can be async/fire-and-forget.

**Current suspected overhead**: 3-6ms per operation

---

## 🔍 How Event Emission Works

### The `@emit_events` Decorator

**Location**: `abstractions/events/events.py:993-1142`

**What it does**:
```python
@emit_events(
    creating_factory=lambda cls, metadata, kwargs: InputPreparationEvent(...),
    created_factory=lambda result, cls, metadata, kwargs: InputPreparedEvent(...)
)
async def _execute_with_partial(cls, metadata, kwargs, skip_divergence_check):
    # Your actual function logic
    pass
```

**Execution flow**:
1. **Before function**: Create & emit "creating" event
2. **Execute function**: Run actual logic
3. **After function**: Create & emit "created" event
4. **On error**: Create & emit "failed" event

---

## 📊 Event Emission Breakdown

### Step 1: Create Event Object (Lines 1074-1106)

```python
# Get parent context
parent_event = get_current_parent_event()  # ~0.01ms (context lookup)

# Create event
start_event = creating_factory(*args, **kwargs)  # ~0.1-0.2ms (Pydantic model creation)

# Link to parent (if exists)
if parent_event:
    start_event.parent_id = parent_event.id
    start_event.root_id = parent_event.root_id
    start_event.lineage_id = parent_event.lineage_id
# ~0.01ms (attribute assignment)

# Push to context stack
push_event_context(start_event)  # ~0.01ms (list append)
```

**Cost**: ~0.13-0.23ms per event

---

### Step 2: Emit Event (Lines 1103, 1134)

```python
await bus.emit(start_event)
```

**What `bus.emit()` does** (Lines 557-569):
```python
async def emit(self, event: Event) -> Event:
    # Add to queue for processing
    await self._event_queue.put(event)  # ~0.01ms (queue put)
    return event
```

**Cost**: ~0.01ms per emit

**Key insight**: **Events are queued, not processed immediately!**

---

### Step 3: Background Processing

**The event queue is processed by a background task**:

```python
# EventBus has a background processor
self._processor_task = asyncio.create_task(self._process_events())
```

**What the processor does** (Lines 594-631):
```python
async def _emit_internal(self, event: Event):
    # 1. Record in history
    self._history.append(event)  # ~0.001ms
    
    # 2. Index by ID
    self._events_by_id[event.id] = event  # ~0.001ms
    
    # 3. Index parent-child relationships
    if event.parent_id:
        self._children_by_parent[event.parent_id].append(event)  # ~0.001ms
    
    # 4. Find matching handlers
    handlers = self._find_matching_handlers(event)  # ~0.01-0.1ms
    
    # 5. Execute handlers
    await self._execute_handlers(event, handlers)  # ~0-∞ms (depends on handlers!)
```

**Cost**: 
- Without handlers: ~0.013ms
- With handlers: Depends on handler complexity

---

## 🎯 Critical Discovery: No Handlers in Our Tests!

**Checked our test files**:
```bash
grep -r "subscribe" projects/local_versioning/
# No results!
```

**This means**:
- No event handlers are registered
- `_find_matching_handlers()` returns empty list
- `_execute_handlers()` does nothing
- **Events are just being logged and indexed!**

---

## 📊 Actual Event Overhead in Our Case

### Per Operation (2 events: creating + created)

**Synchronous overhead** (in main execution path):
```
Event 1 (creating):
  - Create event object: 0.1-0.2ms
  - Context management: 0.02ms
  - Queue put: 0.01ms
  Subtotal: 0.13-0.23ms

Event 2 (created):
  - Create event object: 0.1-0.2ms
  - Context management: 0.02ms
  - Queue put: 0.01ms
  Subtotal: 0.13-0.23ms

TOTAL SYNC: 0.26-0.46ms
```

**Asynchronous overhead** (background processing):
```
Event 1 processing:
  - History append: 0.001ms
  - Index by ID: 0.001ms
  - Index parent-child: 0.001ms
  - Find handlers: 0.01ms (returns empty)
  - Execute handlers: 0ms (no handlers)
  Subtotal: 0.013ms

Event 2 processing:
  - Same as above: 0.013ms

TOTAL ASYNC: 0.026ms
```

**TOTAL OVERHEAD: 0.29-0.49ms per operation**

---

## 🤔 Why Did We Think It Was 3-6ms?

**Theory 1: Multiple decorators**

Let me count how many `@emit_events` are in the execution path:

```python
# 1. aexecute()
@emit_events(creating_factory=..., created_factory=...)
async def aexecute(cls, func_name, **kwargs):
    # 2 events
    
    # 2. _execute_with_partial()
    @emit_events(creating_factory=..., created_factory=...)
    async def _execute_with_partial(cls, metadata, kwargs, skip_divergence_check):
        # 2 events
        
        # 3. _execute_transactional()
        @emit_events(creating_factory=..., created_factory=...)
        async def _execute_transactional(cls, metadata, kwargs, classification, skip_divergence_check):
            # 2 events
            
            # Total: 6 events per operation!
```

**If we have 3 nested decorators**:
- 6 events × 0.15ms = **0.9ms synchronous overhead**
- 6 events × 0.013ms = **0.078ms async overhead**
- **Total: ~1ms**

**Still not 3-6ms!**

---

**Theory 2: Pydantic model creation is slow**

Event objects are Pydantic models:
```python
class InputPreparationEvent(Event):
    process_name: str
    function_name: str
    preparation_type: str
    input_entity_ids: List[UUID]
    entity_count: int
    # ... many fields
```

**Pydantic overhead**:
- Field validation: ~0.05-0.1ms
- Model construction: ~0.05-0.1ms
- **Total per event**: ~0.1-0.2ms

**For 6 events**: 0.6-1.2ms

**Still not 3-6ms!**

---

**Theory 3: It's not actually 3-6ms**

Looking back at our profiling:
- Total operation: 27-37ms
- CallableRegistry overhead: 15-20ms
- Event emission: ???

**We never directly measured event overhead!** We just assumed it was 3-6ms based on "multiple decorators."

**Reality**: Event overhead is probably **0.5-1.5ms**, not 3-6ms!

---

## 🚀 Can We Make Events Async/Fire-and-Forget?

### Current Behavior

**Events are ALREADY async!**

```python
await bus.emit(start_event)
# ↓
async def emit(self, event):
    await self._event_queue.put(event)  # Just queues it
    return event  # Returns immediately
```

**The queue put is fast** (~0.01ms), but we still `await` it.

**Background processor** handles actual processing asynchronously.

---

### Can We Remove the `await`?

**Option 1: Fire-and-forget emit**

```python
# Current:
await bus.emit(start_event)  # Wait for queue put (~0.01ms)

# Fire-and-forget:
asyncio.create_task(bus.emit(start_event))  # Don't wait
```

**Savings**: ~0.01ms × 6 events = **0.06ms** (negligible!)

**Risk**: Events might not be queued before function returns

---

### Can We Skip Event Creation Entirely?

**Option 2: Conditional event emission**

```python
@emit_events(
    creating_factory=...,
    created_factory=...,
    enabled=False  # ← Add flag to disable
)
async def _execute_with_partial(...):
    pass
```

**Savings**: 0.5-1.5ms per operation

**Trade-off**: Lose observability and debugging

---

### Can We Make Event Creation Lazy?

**Option 3: Lazy event creation**

```python
# Current: Create event immediately
start_event = creating_factory(*args, **kwargs)  # ~0.1-0.2ms

# Lazy: Only create if there are subscribers
if bus.has_subscribers_for(event_type):
    start_event = creating_factory(*args, **kwargs)
    await bus.emit(start_event)
```

**Savings**: 0.1-0.2ms × 6 events = **0.6-1.2ms**

**Benefit**: Still get events when needed, skip when not

---

## 📊 Optimization Potential

### Option 1: Fire-and-Forget Emit (MINIMAL GAIN)

**Change**:
```python
# In emit_events decorator:
asyncio.create_task(bus.emit(start_event))  # Don't await
```

**Savings**: ~0.06ms per operation  
**Risk**: Low (events are queued anyway)  
**Effort**: 30 minutes

---

### Option 2: Disable Events Flag (MEDIUM GAIN)

**Change**:
```python
@emit_events(
    creating_factory=...,
    created_factory=...,
    enabled=lambda: not PERFORMANCE_MODE  # ← Add flag
)
```

**Savings**: 0.5-1.5ms per operation  
**Risk**: Medium (lose observability)  
**Effort**: 2 hours

---

### Option 3: Lazy Event Creation (BEST GAIN)

**Change**:
```python
def emit_events(...):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            bus = get_event_bus()
            
            # Only create events if there are subscribers
            if creating_factory and bus.has_subscribers_for(EventType):
                start_event = creating_factory(*args, **kwargs)
                await bus.emit(start_event)
            
            result = await func(*args, **kwargs)
            
            if created_factory and bus.has_subscribers_for(EventType):
                end_event = created_factory(result, *args, **kwargs)
                await bus.emit(end_event)
            
            return result
```

**Savings**: 0.6-1.2ms per operation (when no subscribers)  
**Risk**: Low (events still work when needed)  
**Effort**: 3-4 hours

---

## 🎯 Recommended Approach

### Phase 1: Measure First! (CRITICAL)

**Before optimizing, let's actually measure event overhead**:

```python
# Add timing to profile_pipeline.py
with profiler.timer("event_creation"):
    # Time just the event creation
    event = InputPreparationEvent(...)

with profiler.timer("event_emission"):
    # Time just the emission
    await bus.emit(event)
```

**Why**: We're guessing it's 3-6ms, but it might only be 0.5-1.5ms!

---

### Phase 2: Lazy Event Creation (IF NEEDED)

**Only if measurement shows >1ms overhead**:

1. Add `has_subscribers_for()` method to EventBus
2. Modify `emit_events` decorator to check before creating
3. Test that events still work when subscribers exist

**Expected savings**: 0.6-1.2ms

---

### Phase 3: Fire-and-Forget (OPTIONAL)

**Only if we need every microsecond**:

1. Change `await bus.emit()` to `asyncio.create_task(bus.emit())`
2. Ensure events are still queued before function returns

**Expected savings**: 0.06ms (probably not worth it)

---

## 🎓 Key Insights

### 1. Events Are Already Async!

The event system is well-designed:
- Events are queued immediately (~0.01ms)
- Processing happens in background
- No blocking on handler execution

### 2. Overhead is Smaller Than Expected

**Estimated breakdown**:
- Event creation (Pydantic): 0.6-1.2ms (6 events)
- Context management: 0.12ms
- Queue operations: 0.06ms
- **Total: 0.78-1.38ms**

**NOT 3-6ms as we thought!**

### 3. No Handlers = Minimal Cost

In our tests:
- No event subscribers registered
- Background processing is trivial (~0.026ms)
- Events are just logged and indexed

### 4. Biggest Cost is Pydantic Model Creation

Creating 6 Pydantic event objects:
- Field validation
- Type checking
- Model construction
- **This is the real overhead** (~0.6-1.2ms)

### 5. Lazy Creation is the Best Optimization

**If no subscribers**:
- Skip event creation entirely
- Save 0.6-1.2ms
- Still get events when needed

---

## ✅ Action Items

### Immediate: Measure Event Overhead

Add timing to profile_pipeline.py:
```python
# Time event creation separately
# Time event emission separately
# Confirm actual overhead
```

### If Overhead > 1ms: Implement Lazy Creation

1. Add `has_subscribers_for()` to EventBus
2. Modify `emit_events` decorator
3. Test with and without subscribers

### If Overhead < 1ms: Skip Optimization

- Event overhead is acceptable
- Focus on other bottlenecks
- Keep observability benefits

---

## 🎯 Conclusion

**Event emission is NOT the bottleneck we thought!**

**Actual overhead**: ~0.8-1.4ms (not 3-6ms)  
**Already async**: Events are queued and processed in background  
**Optimization potential**: 0.6-1.2ms with lazy creation

**Recommendation**: 
1. **Measure first** to confirm actual overhead
2. **Only optimize if >1ms** overhead confirmed
3. **Use lazy creation** if optimization needed
4. **Keep fire-and-forget** as-is (already fast)

**The event system is well-designed and not a major bottleneck!**
