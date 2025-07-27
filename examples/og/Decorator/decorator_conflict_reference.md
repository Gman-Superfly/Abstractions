# Decorator Conflict Resolution Reference

## Overview

The Abstractions framework provides **declarative conflict resolution decorators** that integrate with the proven staging area and priority-based conflict resolution patterns validated in `dynamic_stress_test.py` and `total_brutality_test.py`.

These decorators follow the framework's **architectural philosophy**:
- **90% of operations** are naturally conflict-free through immutable entity design
- **10% that need protection** get robust two-stage conflict resolution
- **Surgical application** only where the natural immutability model is insufficient

## Architecture

### Two-Stage Conflict Resolution System

```
Operations Submitted → [Pre-ECS Staging] → Priority Resolution → [ECS Winners] → [OCC Protection] → Success
     ↓                      ↓                       ↓               ↓              ↓
  Multiple ops         Shared staging area    Priority-based    Data-level     Version validation
  per target          Global coordinator      winner selection   race detection  Retry until success
```

### Core Components

1. **Global Staging Coordinator** - Background process that resolves conflicts using proven patterns
2. **Priority-Based Resolution** - Higher priority operations win Pre-ECS conflicts
3. **Clean Rejection** - Losers never get ECS IDs, no cleanup required
4. **OCC Protection** - Winners get optional optimistic concurrency control
5. **Event Integration** - Complete lifecycle monitoring through event system

## Decorator Usage

### Basic Usage

```python
from abstractions.ecs.conflict_decorators import with_conflict_resolution, no_conflict_resolution
from abstractions.ecs.entity_hierarchy import OperationPriority

# Explicit conflict resolution for specific operations
@with_conflict_resolution(
    pre_ecs=PreECSConfig(enabled=True, priority=OperationPriority.HIGH),
    occ=OCCConfig(enabled=True)
)
async def process_student_cohort(cohort: List[Student]) -> List[AnalysisResult]:
    """Process a batch of students with conflict protection."""
    # Two-stage protection applied automatically
    results = []
    for student in cohort:
        # Complex processing that could conflict with other operations
        analysis = perform_complex_analysis(student)
        results.append(analysis)
    return results

# Normal operations use standard framework behavior
async def update_individual_student(student: Student, new_data: Dict) -> Student:
    """Update single student - no conflict resolution needed."""
    # Natural immutability handles this safely
    student.gpa = new_data.get('gpa', student.gpa)
    return student

# Explicit opt-out for performance
@no_conflict_resolution  
async def read_only_batch_analysis(students: List[Student]) -> Statistics:
    """Read-only operation, skip protection for performance."""
    total_gpa = sum(s.gpa for s in students)
    return Statistics(average_gpa=total_gpa / len(students))
```

### Advanced Configuration

```python
from abstractions.ecs.conflict_decorators import (
    ConflictResolutionConfig, ConflictResolutionMode, 
    PreECSConfig, OCCConfig
)

# Custom configuration for complex operations
@with_conflict_resolution(config=ConflictResolutionConfig(
    mode=ConflictResolutionMode.BOTH,
    pre_ecs=PreECSConfig(
        priority=OperationPriority.CRITICAL,
        staging_timeout_ms=150.0  # Longer accumulation window
    ),
    occ=OCCConfig(
        max_retries=15,
        backoff_factor=2.0,
        base_delay_ms=10.0
    )
))
async def optimize_schedules(students: List[Student]) -> List[Schedule]:
    """Critical operation with custom protection parameters."""
    # Custom two-stage protection with tuned parameters
    pass
```

### Configuration Options

#### ConflictResolutionMode
- `NONE` - No conflict resolution (direct execution)
- `PRE_ECS` - Pre-ECS staging area only
- `OCC` - Optimistic concurrency control only  
- `BOTH` - Two-stage protection (recommended)

#### PreECSConfig
- `priority: int` - Operation priority (higher wins conflicts)
- `staging_timeout_ms: float` - Conflict accumulation window
- `enabled: bool` - Enable/disable Pre-ECS resolution

#### OCCConfig  
- `max_retries: int` - Maximum retry attempts
- `base_delay_ms: float` - Base retry delay
- `backoff_factor: float` - Exponential backoff multiplier
- `version_field: str` - Entity version field name
- `modified_field: str` - Last modified field name

#### Priority Levels
- `OperationPriority.CRITICAL = 10` - Critical system operations
- `OperationPriority.HIGH = 8` - Important operations
- `OperationPriority.NORMAL = 5` - Standard operations (default)
- `OperationPriority.LOW = 2` - Background operations

## When to Apply Conflict Resolution

### ✅ Apply Protection For:

1. **Shared Collections/Arrays**
   ```python
   @with_conflict_resolution(
       pre_ecs=PreECSConfig(enabled=True),
       occ=OCCConfig(enabled=True)
   )
   async def process_student_batch(cohort: List[Student]) -> List[Result]:
       # Multiple operations targeting same data structure
   ```

2. **Read-Process-Write Cycles**
   ```python
   @with_conflict_resolution(
       pre_ecs=PreECSConfig(enabled=True),
       occ=OCCConfig(enabled=True)
   )  
   async def complex_optimization(data: List[Entity]) -> List[Entity]:
       snapshot = read_current_state(data)  # READ
       # Complex processing time...          # PROCESS (conflict window)
       return apply_optimizations(snapshot)  # WRITE
   ```

3. **Stateful Aggregations**
   ```python
   @with_conflict_resolution(
       pre_ecs=PreECSConfig(enabled=True),
       occ=OCCConfig(enabled=True)
   )
   async def calculate_running_totals(transactions: List[Transaction]) -> Summary:
       total = 0  # Stateful accumulation
       for tx in transactions:
           total += tx.amount  # State modification
       return Summary(total=total)
   ```

4. **Cross-Entity Dependencies**
   ```python
   @with_conflict_resolution(
       pre_ecs=PreECSConfig(enabled=True, priority=OperationPriority.HIGH),
       occ=OCCConfig(enabled=True)
   )
   async def allocate_limited_resources(requests: List[Request], pool: ResourcePool) -> List[Allocation]:
       # Resource allocation affects subsequent allocations
   ```

### ❌ Skip Protection For:

1. **Single Entity Transformations**
   ```python
   # Natural framework design handles this
   async def update_student_grade(student: Student, grade: float) -> Student:
       student.gpa = grade  # Direct modification through framework
       return student
   ```

2. **Read-Only Operations**
   ```python
   @no_conflict_resolution  # Explicit opt-out for performance
   async def get_student_statistics(students: List[Student]) -> Statistics:
       # Pure data access, no conflicts possible
   ```

3. **Independent Parallel Operations**
   ```python
   # Each operation targets different entities
   results = await asyncio.gather(*[
       process_individual_student(s) for s in students  # No shared state
   ])
   ```

## Implementation Details

### Staging Coordinator Lifecycle

The staging coordinator is automatically managed:

```python
# Automatically started when first decorator operation is submitted
await _start_staging_coordinator()

# Runs in background, processing conflicts every 50ms
async def _staging_coordinator_worker():
    while _staging_coordinator_running:
        for target_entity_id in list(_staging_area.keys()):
            await _resolve_conflicts_for_target_stress_test_pattern(target_entity_id)
        await asyncio.sleep(0.05)

# Automatically stopped when test/application shuts down
await _stop_staging_coordinator()
```

### Conflict Resolution Flow

1. **Operation Submission**
   ```python
   # Operation created but NOT promoted to ECS
   operation = ConflictResolutionOperation(
       target_entity_id=primary_target_id,
       op_type=f"conflict_protected_{func.__name__}",
       priority=config.pre_ecs.priority,
       # Function and arguments stored for later execution
   )
   
   # Added to staging area WITHOUT ECS promotion
   _staging_area[primary_target_id].append(operation)
   ```

2. **Conflict Detection and Resolution**
   ```python
   # Background coordinator processes staging areas
   pending_ops = _staging_area.get(target_entity_id, [])
   
   if len(pending_ops) > 1:
       # Sort by priority (higher wins), then timestamp
       pending_ops.sort(
           key=lambda op: (op.priority, -op.created_at.timestamp()), 
           reverse=True
       )
       
       winner = pending_ops[0]
       losers = pending_ops[1:]
       
       # Only winner gets ECS ID
       winner.promote_to_root()
       
       # Losers cleanly rejected
       for loser in losers:
           # Signal rejection, emit events
   ```

3. **Winner Execution**
   ```python
   # Execute with original function arguments
   stored_function = operation.get_stored_function()
   original_args = operation.function_args
   original_kwargs = operation.function_kwargs
   
   result = await operation._execute_function(
       stored_function, *original_args, **original_kwargs
   )
   ```

### Event Integration

The decorators emit standard operation lifecycle events:

```python
# Operation started
OperationStartedEvent(
    process_name="conflict_protected_execution",
    op_id=operation.ecs_id,
    op_type=operation.op_type,
    priority=operation.priority,
    target_entity_id=operation.target_entity_id
)

# Operation completed successfully  
OperationCompletedEvent(
    process_name="conflict_protected_execution",
    op_id=operation.ecs_id,
    op_type=operation.op_type,
    target_entity_id=operation.target_entity_id,
    execution_duration_ms=duration
)

# Operation rejected (Pre-ECS or execution failure)
OperationRejectedEvent(
    op_id=operation.ecs_id,
    op_type=operation.op_type,
    target_entity_id=operation.target_entity_id,
    from_state="staged" | "executing",
    to_state="rejected" | "failed",
    rejection_reason="Lost Pre-ECS priority conflict" | "Function execution failed",
    retry_count=operation.retry_count
)
```

## Testing

### Test Files

#### `decorator_conflict_test.py` - Comprehensive Validation

**Purpose**: Side-by-side comparison of decorator vs manual conflict resolution

**Key Features**:
- **Real concurrent conflicts** - Multiple operations targeting same entities simultaneously
- **Mixed approach testing** - 60% decorator operations, 40% manual operations
- **Pattern validation** - Ensures decorators follow proven stress test patterns
- **Performance comparison** - Success rates, execution times, entity modifications
- **Event emission verification** - Complete lifecycle monitoring

**Configuration**:
```python
config = DecoratorTestConfig(
    test_duration_seconds=15,       # 15 seconds test
    operations_per_second=100.0,    # 100 ops/sec
    num_targets=3,                  # 3 targets = guaranteed conflicts
    batch_size=6,                   # 6 ops per batch
    decorator_vs_manual_ratio=0.6   # 60% decorator, 40% manual
)
```

**Expected Results**:
```
⚔️  PRE-ECS CONFLICT: 4 operations competing for target a31369b8
🏆 PRE-ECS RESOLUTION: 1 winner, 3 rejected
✅ WINNER: conflict_protected_decorator_complex_update (Priority: 10)
🚀 OPERATION: Started conflict_protected_decorator_complex_update operation
✅ OPERATION: Completed conflict_protected_decorator_complex_update operation
   └─ Duration: 10.0ms
```

#### Example Decorator Operations Used in Tests

```python
@with_conflict_resolution(
    pre_ecs=PreECSConfig(enabled=True, priority=OperationPriority.HIGH),
    occ=OCCConfig(enabled=True)
)
async def decorator_increment_operation(target: DecoratorTestEntity, amount: float) -> bool:
    """Simple increment with conflict protection."""
    await asyncio.sleep(0.005)  # Simulate processing where conflicts can occur
    new_value = target.value + amount
    target.record_modification("decorator_increment", new_value)
    return True

@with_conflict_resolution(
    config=ConflictResolutionConfig(
        mode=ConflictResolutionMode.BOTH,
        pre_ecs=PreECSConfig(priority=OperationPriority.CRITICAL, staging_timeout_ms=50.0),
        occ=OCCConfig(max_retries=15, backoff_factor=2.0)
    )
)
async def decorator_complex_update(target: DecoratorTestEntity, multiplier: float, offset: float) -> bool:
    """Complex operation with custom configuration."""
    await asyncio.sleep(0.008)  # More processing time
    intermediate = target.value * multiplier
    await asyncio.sleep(0.002)  # Additional conflict window
    new_value = intermediate + offset
    target.record_modification("decorator_complex", new_value)
    return True

@no_conflict_resolution
async def decorator_read_only_operation(target: DecoratorTestEntity) -> float:
    """Read-only operation with explicit opt-out."""
    await asyncio.sleep(0.001)
    return target.value + target.modification_count
```

### Running Tests

```bash
# Run comprehensive decorator vs manual comparison
python examples/og/decorator_conflict_test.py

# Expected output includes:
# - Real Pre-ECS conflicts detected and resolved
# - Priority-based winner selection
# - Successful function execution with proper arguments
# - Event emission for complete lifecycle
# - Comparative metrics between approaches
```

### Test Metrics

The test validates several key metrics:

1. **Conflict Detection**
   - Decorator conflicts detected and resolved
   - Manual conflicts detected and resolved
   - Both should show significant conflict activity

2. **Success Patterns**  
   - Decorator operations completing successfully
   - Manual operations completing successfully
   - Both should show reasonable success rates despite conflicts

3. **Performance Comparison**
   - Execution times should be comparable
   - Both approaches should modify entities
   - Event emission should be consistent

4. **Pattern Compliance**
   - Pre-ECS conflicts resolved by priority
   - Only winners get ECS IDs
   - Clean rejection of losers
   - Proper argument passing to functions

## Integration with Existing Systems

### CallableRegistry Integration

```python
# Decorators work seamlessly with CallableRegistry
@CallableRegistry.register("process_cohort_protected")
@with_conflict_resolution(
    pre_ecs=PreECSConfig(enabled=True, priority=OperationPriority.HIGH),
    occ=OCCConfig(enabled=True)
)
async def process_student_cohort(cohort: List[Student]) -> List[AnalysisResult]:
    # Function available through both decorator and registry
    pass

# Can be called through registry or directly
result1 = await CallableRegistry.aexecute("process_cohort_protected", cohort=students)
result2 = await process_student_cohort(students)  # Direct call with protection
```

### Event System Integration

```python
# Decorators emit events that integrate with existing handlers
@on(OperationStartedEvent)
async def log_protected_operations(event: OperationStartedEvent):
    if event.process_name == "conflict_protected_execution":
        print(f"Protected operation started: {event.op_type}")

@on(OperationRejectedEvent)  
async def handle_conflict_rejections(event: OperationRejectedEvent):
    if "Pre-ECS priority conflict" in event.rejection_reason:
        # Handle Pre-ECS rejections differently than execution failures
        pass
```

### Manual vs Decorator Patterns

You can mix approaches within the same application:

```python
# Use decorators for standard cases
@with_conflict_resolution(
    pre_ecs=PreECSConfig(enabled=True),
    occ=OCCConfig(enabled=True)
)
async def standard_batch_operation(entities: List[Entity]) -> List[Result]:
    # Standard two-stage protection
    pass

# Use manual implementation for complex custom logic
class AdvancedConflictManager:
    async def submit_with_custom_resolution(self, operation):
        # Custom business logic for conflict resolution
        # Integrates with same staging areas and event system
        pass
```

## Performance Considerations

### Framework-Native Operations (Optimal Performance)

Most operations should use the natural entity framework design:
- **Zero synchronization overhead** - Immutable model eliminates locks
- **Efficient versioning** - Copy-on-write semantics  
- **Lazy evaluation** - String addresses resolved only when needed
- **Event system efficiency** - UUID-based references, no data copying

### Conflict Resolution Overhead  

Two-stage protection adds minimal overhead only where needed:
- **Pre-ECS filtering** - Microseconds for priority-based resolution
- **OCC validation** - Nanoseconds for version/timestamp comparison
- **Retry logic** - Only on actual conflicts (rare in well-designed systems)
- **Memory efficiency** - Only winning operations consume ECS resources

### Best Practices

1. **Use sparingly** - Only for operations that violate natural immutability
2. **Choose appropriate priorities** - Critical system operations get highest priority
3. **Tune parameters** - Adjust staging timeout and retry settings for your use case
4. **Monitor events** - Use event system to track conflict patterns and optimize
5. **Test thoroughly** - Use `decorator_conflict_test.py` patterns to validate behavior

## Troubleshooting

### Common Issues

1. **No conflicts detected**
   - Check if operations are actually concurrent
   - Verify staging coordinator is running
   - Ensure multiple operations target same entities

2. **Operations failing with argument errors**
   - Verify function signatures match decorator usage
   - Check that stored function is set correctly
   - Ensure original arguments are preserved

3. **Poor performance**  
   - Consider if conflict resolution is actually needed
   - Tune staging timeout and retry parameters
   - Monitor event patterns for optimization opportunities

4. **Inconsistent results**
   - Verify priority settings are appropriate
   - Check for race conditions in entity modification
   - Ensure proper OCC field handling

### Debugging

```python
# Enable staging area monitoring
from abstractions.ecs.conflict_decorators import get_staging_area_status

status = get_staging_area_status()
print(f"Staging area status: {status}")

# Clear staging area for testing
from abstractions.ecs.conflict_decorators import clear_staging_area
clear_staging_area()

# Manual coordinator control
from abstractions.ecs.conflict_decorators import start_staging_coordinator, stop_staging_coordinator
await start_staging_coordinator()
await stop_staging_coordinator()
```

## Conclusion

The decorator conflict resolution system provides **surgical conflict protection** for the small subset of operations that require it, while preserving the **elegance and performance** of the entity framework for normal operations.

By following the proven patterns from `dynamic_stress_test.py` and `total_brutality_test.py`, the decorators offer:
- **Production-ready reliability** - Battle-tested conflict resolution patterns
- **Resource efficiency** - Only winners consume ECS resources  
- **Priority-based coordination** - Important operations get precedence
- **Declarative simplicity** - Just add `@with_conflict_resolution`
- **Framework integration** - Works seamlessly with existing entity and event systems

Use this system as a **surgical tool** for specific scenarios, not a general requirement for entity operations. This maintains the framework's core principles while handling edge cases that require explicit coordination. 