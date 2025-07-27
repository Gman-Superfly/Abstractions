# Why and When to Apply Conflict Resolution in the Abstractions Framework

## Overview

The Abstractions framework is designed around **entity-native functional data processing** where entities flow through pure functional transformations with automatic versioning, provenance tracking, and distributed addressing. The framework's **immutable entity model** naturally prevents most concurrency issues.

However, certain classes of operations - particularly those involving **shared collections, arrays, or complex read-process-write cycles** - can still experience race conditions that require explicit conflict resolution.

This document explains **when conflict resolution is necessary** versus when the framework's natural immutability is sufficient.

## Framework Foundation: Naturally Conflict-Free Operations

### Core Entity Model Prevents Most Conflicts

The Abstractions framework is built on principles that eliminate common concurrency problems:

```python
# Entities are immutable snapshots
original = Student(name="Alice", gpa=3.5)
original.promote_to_root()

# Transformations modify entities through registered functions
@CallableRegistry.register("update_gpa")
def update_gpa(student: Student, new_gpa: float) -> Student:
    student.gpa = new_gpa  # Direct field modification
    return student

updated = CallableRegistry.execute("update_gpa", student=original, new_gpa=3.8)

# Entity has been modified in-place through the callable registry
assert updated.gpa == 3.8   # Entity field updated
assert updated is original  # Same entity object (unless explicitly copied)
assert original.lineage_id == updated.lineage_id  # Same lineage
```

**Why this is generally conflict-free:**
- **Structured modifications**: Changes go through registered callable functions
- **Entity lifecycle management**: ECS versioning available when needed
- **Provenance tracking**: Complete audit trail automatically maintained
- **Functional design**: Operations designed as pure transformations where possible

### String Addressing is Read-Only

Distributed addressing through string references doesn't create conflicts:

```python
# String addressing is pure data access
student_name = get(f"@{student.ecs_id}.name")
student_gpa = get(f"@{student.ecs_id}.gpa")

# Multiple concurrent reads are safe
concurrent_reads = await asyncio.gather(*[
    asyncio.create_task(get(f"@{student.ecs_id}.name")),
    asyncio.create_task(get(f"@{student.ecs_id}.gpa")),
    asyncio.create_task(get(f"@{student.ecs_id}.lineage_id"))
])
```

**Why this is conflict-free:**
- **Read-only operations**: No modification of entity state
- **Immutable snapshots**: Data cannot change during access
- **Address resolution**: Returns copies, not references to mutable state

### Event System is Decoupled

Events contain only UUID references and metadata - no shared mutable state:

```python
@on(StudentCreatedEvent)
async def log_student_creation(event: StudentCreatedEvent):
    print(f"Student created: {event.subject_id}")

# Events are lightweight signals, not data containers
# Multiple handlers can process same event without interference
```

**Why this is conflict-free:**
- **UUID-based references**: Events contain IDs, not data copies
- **Immutable event objects**: Events themselves cannot be modified
- **Decoupled handlers**: Each handler operates independently

## When Conflicts CAN Occur: Shared State Scenarios

### Scenario 1: Concurrent Array/Collection Operations

**Problem**: Multiple operations targeting the same collection with read-process-write cycles:

```python
@CallableRegistry.register("normalize_grades")
def normalize_grades(cohort: List[Student]) -> List[Student]:
    # READ PHASE: Calculate average from current cohort
    avg_gpa = sum(s.gpa for s in cohort) / len(cohort)
    
    # PROCESS PHASE: Complex calculation (time for others to interfere)
    normalized = []
    for student in cohort:
        # During this processing, another operation might modify the cohort
        new_gpa = student.gpa / avg_gpa * 3.0
        normalized.append(Student(name=student.name, gpa=new_gpa))
    
    # WRITE PHASE: Return transformed collection
    return normalized

# CONCURRENT EXECUTION - CONFLICT POTENTIAL
students = [Student(name=f"S{i}", gpa=3.0+i*0.2) for i in range(100)]

# These could interfere with each other:
results = await asyncio.gather(*[
    CallableRegistry.aexecute("normalize_grades", cohort=students),      # Operation A
    CallableRegistry.aexecute("calculate_class_rank", cohort=students),  # Operation B  
    CallableRegistry.aexecute("analyze_distribution", cohort=students)   # Operation C
])
```

**Why conflicts occur:**
- **Shared collection reference**: All operations target same `students` list
- **Read-process-write pattern**: Gap between reading data and using calculations
- **Concurrent execution**: Multiple async operations running simultaneously
- **Calculation dependencies**: Results depend on consistent view of collection state

### Scenario 2: Stateful Aggregations

**Problem**: Operations that accumulate state during processing:

```python
@CallableRegistry.register("calculate_department_stats")
def calculate_department_stats(students: List[Student], department: str) -> DepartmentReport:
    # Stateful accumulation - vulnerable to interference
    total_gpa = 0.0
    count = 0
    grade_distribution = {}
    
    for student in students:
        if student.department == department:
            total_gpa += student.gpa  # State accumulation
            count += 1
            
            # Complex processing window
            grade_category = categorize_grade(student.gpa)
            if grade_category in grade_distribution:
                grade_distribution[grade_category] += 1
            else:
                grade_distribution[grade_category] = 1
    
    # Final calculation based on accumulated state
    avg_gpa = total_gpa / count if count > 0 else 0.0
    
    return DepartmentReport(
        department=department,
        average_gpa=avg_gpa,
        student_count=count,
        grade_distribution=grade_distribution
    )

# CONFLICT: Multiple departments processed concurrently on same student list
departments = ["CS", "Math", "Physics"] 
reports = await asyncio.gather(*[
    CallableRegistry.aexecute("calculate_department_stats", students=large_student_list, department=dept)
    for dept in departments
])
```

**Why conflicts occur:**
- **Shared data iteration**: All operations iterate over same student collection
- **Stateful processing**: Operations maintain internal state during execution
- **Extended execution time**: Complex processing creates larger conflict windows
- **Inconsistent snapshots**: Operations may see partially modified data

### Scenario 3: Cross-Entity Dependencies

**Problem**: Operations that depend on relationships between multiple entities:

```python
@CallableRegistry.register("optimize_course_assignments")
def optimize_course_assignments(students: List[Student], courses: List[Course]) -> List[Assignment]:
    # Complex optimization with cross-entity dependencies
    assignments = []
    course_capacities = {course.id: course.max_students for course in courses}
    
    # Sort students by priority (creates dependency on student ordering)
    prioritized_students = sorted(students, key=lambda s: s.gpa, reverse=True)
    
    for student in prioritized_students:
        best_course = None
        best_score = 0.0
        
        for course in courses:
            if course_capacities[course.id] > 0:
                # Complex scoring based on both student and course state
                compatibility_score = calculate_compatibility(student, course)
                
                if compatibility_score > best_score:
                    best_score = compatibility_score
                    best_course = course
        
        if best_course:
            course_capacities[best_course.id] -= 1  # Modifies shared state
            assignments.append(Assignment(student_id=student.id, course_id=best_course.id))
    
    return assignments

# CONFLICT: Multiple optimization algorithms running simultaneously  
optimization_results = await asyncio.gather(*[
    CallableRegistry.aexecute("optimize_course_assignments", students=students, courses=courses),
    CallableRegistry.aexecute("optimize_by_preferences", students=students, courses=courses),
    CallableRegistry.aexecute("optimize_by_capacity", students=students, courses=courses)
])
```

**Why conflicts occur:**
- **Cross-entity state**: Operations modify state that affects other operations
- **Order dependencies**: Results depend on the sequence of entity processing
- **Shared capacity tracking**: Multiple operations competing for limited resources
- **Complex interdependencies**: Changes to one entity affect others

## Framework Analysis: Natural vs. Artificial Conflicts

### Natural Entity Framework Operations (No Conflict Resolution Needed)

These operations align with the framework's immutable design and don't create conflicts:

```python
# ✅ Single entity transformations
student = Student(name="Alice", gpa=3.5)
updated_student = CallableRegistry.execute("update_gpa", student=student, new_gpa=3.8)

# ✅ Pure functional compositions  
@CallableRegistry.register("calculate_grade_letter")
def calculate_grade_letter(student: Student) -> GradeReport:
    letter = "A" if student.gpa >= 3.7 else "B" if student.gpa >= 3.0 else "C"
    return GradeReport(student_id=str(student.ecs_id), letter_grade=letter)

# ✅ Multi-entity outputs (sibling relationships)
@CallableRegistry.register("analyze_performance") 
def analyze_performance(student: Student) -> Tuple[Assessment, Recommendation]:
    assessment = Assessment(student_id=str(student.ecs_id), level="high" if student.gpa > 3.5 else "standard")
    recommendation = Recommendation(student_id=str(student.ecs_id), action="advanced" if student.gpa > 3.5 else "standard")
    return assessment, recommendation

# ✅ Independent concurrent execution
students = [Student(name=f"S{i}", gpa=3.0+i*0.1) for i in range(10)]
results = await asyncio.gather(*[
    CallableRegistry.aexecute("analyze_performance", student=s) for s in students
])

# ✅ String addressing and distributed access
student_data = get(f"@{student.ecs_id}.name")
course_info = get(f"@{course.ecs_id}.credits")

# ✅ Event-driven coordination
@on(StudentCreatedEvent)
async def process_new_student(event: StudentCreatedEvent):
    student = EntityRegistry.get_stored_entity(event.subject_id)
    await CallableRegistry.aexecute("assign_advisor", student=student)
```

**Why these don't need conflict resolution:**
- **Immutable inputs**: Each operation gets its own copy of entity data
- **Independent execution**: Operations don't share mutable state
- **Versioned outputs**: New entities created, no modification of existing ones
- **Event decoupling**: Handlers process events independently

### Operations Requiring Conflict Resolution

These operations violate the natural immutability model and need protection:

```python
# ❌ Shared collection modifications
@CallableRegistry.register("rebalance_class_sizes")
def rebalance_class_sizes(students: List[Student], target_class_size: int) -> List[Student]:
    # Multiple operations modifying same collection = conflicts

# ❌ Read-process-write with extended processing
@CallableRegistry.register("optimize_schedules") 
def optimize_schedules(students: List[Student], constraints: ScheduleConstraints) -> List[Schedule]:
    # Long processing time + shared data = race conditions

# ❌ Stateful aggregations with side effects
@CallableRegistry.register("allocate_resources")
def allocate_resources(requests: List[ResourceRequest], available: ResourcePool) -> List[Allocation]:
    # Shared resource pool state = conflicts

# ❌ Order-dependent transformations
@CallableRegistry.register("assign_priorities")
def assign_priorities(tasks: List[Task]) -> List[Task]:
    # Order of processing affects results = inconsistent outcomes
```

## When to Apply the Two-Stage Conflict Resolution System

### Apply Pre-ECS + OCC Protection When:

1. **Shared Collections/Arrays**:
   ```python
   # Multiple operations targeting same data structure
   process_student_batch(cohort: List[Student]) -> List[Result]
   analyze_student_batch(cohort: List[Student]) -> Analysis 
   normalize_student_batch(cohort: List[Student]) -> List[Student]
   ```

2. **Read-Process-Write Cycles**:
   ```python
   # Operations that read, compute for extended time, then write
   def complex_optimization(data: List[Entity]) -> List[Entity]:
       snapshot = read_current_state(data)  # READ
       # Complex processing time...          # PROCESS (conflict window)
       return apply_optimizations(snapshot)  # WRITE
   ```

3. **Stateful Aggregations**:
   ```python
   # Operations that accumulate state during processing
   def calculate_running_totals(transactions: List[Transaction]) -> Summary:
       total = 0  # Stateful accumulation
       for tx in transactions:
           total += tx.amount  # State modification
       return Summary(total=total)
   ```

4. **Cross-Entity Dependencies**:
   ```python
   # Operations where entity changes affect other entities
   def allocate_limited_resources(requests: List[Request], pool: ResourcePool) -> List[Allocation]:
       # Resource allocation affects subsequent allocations
   ```

5. **Order-Dependent Processing**:
   ```python
   # Operations where processing order affects results
   def assign_rankings(competitors: List[Competitor]) -> List[Competitor]:
       # Ranking assignment depends on processing sequence
   ```

### Skip Conflict Resolution When:

1. **Single Entity Transformations**:
   ```python
   # Pure functional entity operations
   def update_student_grade(student: Student, grade: float) -> Student:
       return Student(name=student.name, gpa=grade)  # Immutable transformation
   ```

2. **Independent Parallel Operations**:
   ```python
   # Operations on different entities
   results = await asyncio.gather(*[
       process_individual_student(s) for s in students  # No shared state
   ])
   ```

3. **Read-Only Operations**:
   ```python
   # Data access without modification
   student_name = get(f"@{student.ecs_id}.name")
   course_credits = get(f"@{course.ecs_id}.credits")
   ```

4. **Event Handling**:
   ```python
   # Event handlers processing independent signals
   @on(StudentCreatedEvent)
   async def log_creation(event: StudentCreatedEvent):
       # No shared mutable state
   ```

5. **Multi-Entity Outputs with Sibling Relationships**:
   ```python
   # Functions that create multiple related entities
   def analyze_student(student: Student) -> Tuple[Assessment, Recommendation]:
       # Creates new entities, doesn't modify shared state
   ```

## Implementation Patterns

### Pattern 1: Manual Implementation (Full Control)

For cases requiring custom logic or integration with existing systems, implement conflict resolution manually using the validated patterns from the stress tests:

```python
from abstractions.ecs.entity_hierarchy import (
    OperationEntity, OperationStatus, OperationPriority,
    resolve_operation_conflicts
)
from abstractions.events.events import emit, OperationConflictEvent, OperationRejectedEvent

# Manual Pre-ECS conflict resolution
class CustomOperationEntity(OperationEntity):
    """Custom operation with manual conflict resolution."""
    
    operation_type: str = "custom_operation"
    operation_data: Dict[str, Any] = {}
    
    async def execute_with_manual_occ(self, target: Entity) -> bool:
        """Manual OCC implementation with custom retry logic."""
        retry_count = 0
        max_retries = 10
        
        while retry_count <= max_retries:
            # READ: Snapshot current state
            read_version = target.version
            read_modified = target.last_modified
            
            # PROCESS: Your custom processing logic
            await asyncio.sleep(0.005)  # Simulate work
            new_value = self.compute_new_value(target.data)
            
            # WRITE: Check for conflicts before committing
            if (target.version != read_version or 
                target.last_modified != read_modified):
                
                # Emit conflict event
                await emit(OperationConflictEvent(
                    op_id=self.ecs_id,
                    op_type=self.op_type,
                    target_entity_id=target.ecs_id,
                    priority=self.priority,
                    conflict_details={"retry_count": retry_count}
                ))
                
                retry_count += 1
                await asyncio.sleep(0.002 * (2 ** retry_count))  # Exponential backoff
                continue
            
            # COMMIT: Safe to write
            target.data = new_value
            target.mark_modified()
            return True
        
        return False

# Manual Pre-ECS staging area management
class ConflictManager:
    """Manual conflict resolution manager."""
    
    def __init__(self):
        self.pending_operations: Dict[UUID, List[OperationEntity]] = {}
    
    async def submit_operation(self, operation: OperationEntity, target_id: UUID):
        """Submit operation to staging area."""
        if target_id not in self.pending_operations:
            self.pending_operations[target_id] = []
        
        self.pending_operations[target_id].append(operation)
        # Don't promote_to_root() yet - keep in staging
    
    async def resolve_conflicts_for_target(self, target_id: UUID):
        """Resolve conflicts using existing conflict resolution system."""
        pending_ops = self.pending_operations.get(target_id, [])
        
        if len(pending_ops) > 1:
            print(f"⚔️  CONFLICT: {len(pending_ops)} operations for target {target_id}")
            
            # Use existing conflict resolution system
            winners = resolve_operation_conflicts(target_id, pending_ops)
            
            # Promote winners to ECS
            for winner in winners:
                winner.promote_to_root()
            
            # Emit rejection events for losers
            for op in pending_ops:
                if op not in winners:
                    await emit(OperationRejectedEvent(
                        op_id=op.ecs_id,
                        op_type=op.op_type,
                        target_entity_id=target_id,
                        from_state="pending",
                        to_state="rejected",
                        rejection_reason="preempted_by_higher_priority",
                        retry_count=op.retry_count
                    ))
            
            self.pending_operations[target_id] = []
            return winners
        
        elif len(pending_ops) == 1:
            # No conflict - promote single operation
            winner = pending_ops[0]
            winner.promote_to_root()
            self.pending_operations[target_id] = []
            return [winner]
        
        return []

# Manual usage pattern
async def manual_conflict_resolution_example():
    """Example of manual conflict resolution implementation."""
    manager = ConflictManager()
    target_entity = MyEntity(data="initial")
    
    # Create competing operations
    ops = []
    for i in range(5):
        op = CustomOperationEntity(
            op_type=f"operation_{i}",
            priority=random.choice([2, 5, 8, 10]),
            target_entity_id=target_entity.ecs_id,
            operation_data={"value": i}
        )
        ops.append(op)
        await manager.submit_operation(op, target_entity.ecs_id)
    
    # Resolve conflicts manually
    winners = await manager.resolve_conflicts_for_target(target_entity.ecs_id)
    
    # Execute winners with OCC protection
    for winner in winners:
        success = await winner.execute_with_manual_occ(target_entity)
        print(f"Operation {winner.op_type} {'succeeded' if success else 'failed'}")
```

### Pattern 2: Decorator-Based Opt-In

```python
from abstractions.ecs.conflict_decorators import with_conflict_resolution, no_conflict_resolution
from abstractions.ecs.entity_hierarchy import OperationPriority

# Explicit conflict resolution for specific operations
@CallableRegistry.register("process_student_cohort")
@with_conflict_resolution(pre_ecs=True, occ=True, priority=OperationPriority.HIGH)
def process_student_cohort(cohort: List[Student]) -> List[AnalysisResult]:
    # Two-stage protection applied automatically
    pass

# Normal operations use standard framework behavior
@CallableRegistry.register("update_individual_student")  
def update_individual_student(student: Student, new_data: Dict) -> Student:
    # No conflict resolution needed - functional design handles this
    pass
```

### Pattern 2: Type-Based Detection

```python
# Framework automatically detects collection parameters
def analyze_cohort(students: List[Student]) -> Report:  # Auto-protected
    pass

def update_student(student: Student) -> Student:        # No protection needed
    pass

# Configuration override available
@no_conflict_resolution  # Explicit opt-out for performance
def read_only_batch_analysis(students: List[Student]) -> Statistics:
    # Read-only operation, skip protection
    pass
```

### Pattern 3: Hybrid Approach

Combine manual and declarative patterns for maximum flexibility:

```python
# Use decorators for standard cases
@with_conflict_resolution(pre_ecs=True, occ=True)
def standard_batch_operation(entities: List[Entity]) -> List[Result]:
    # Standard two-stage protection
    pass

# Use manual implementation for complex custom logic
class AdvancedConflictManager(ConflictManager):
    """Extended manager with custom conflict resolution logic."""
    
    async def resolve_conflicts_with_custom_logic(self, target_id: UUID):
        """Custom conflict resolution with business logic."""
        pending_ops = self.pending_operations.get(target_id, [])
        
        if len(pending_ops) > 1:
            # Custom resolution logic based on operation types
            critical_ops = [op for op in pending_ops if op.priority >= 10]
            if critical_ops:
                # Critical operations always win
                winners = critical_ops[:1]  # Take first critical op
            else:
                # Use standard resolution for non-critical
                winners = resolve_operation_conflicts(target_id, pending_ops)
            
            # Custom logging and metrics
            self.log_conflict_resolution(target_id, pending_ops, winners)
            
            return winners
        
        return pending_ops

# Integration with existing CallableRegistry
@CallableRegistry.register("complex_analysis")
async def complex_analysis_with_manual_conflict_resolution(
    entities: List[Entity], 
    analysis_type: str
) -> AnalysisResult:
    """Function that uses manual conflict resolution for specific scenarios."""
    
    if analysis_type == "high_contention":
        # Use manual conflict resolution for high-contention scenarios
        manager = AdvancedConflictManager()
        
        # Create operation with custom priority logic
        priority = OperationPriority.CRITICAL if len(entities) > 100 else OperationPriority.NORMAL
        operation = CustomOperationEntity(
            op_type="complex_analysis",
            priority=priority,
            target_entity_id=entities[0].ecs_id,
            operation_data={"analysis_type": analysis_type, "entity_count": len(entities)}
        )
        
        # Manual staging and conflict resolution
        await manager.submit_operation(operation, entities[0].ecs_id)
        winners = await manager.resolve_conflicts_with_custom_logic(entities[0].ecs_id)
        
        if operation in winners:
            # Execute the actual analysis
            return perform_complex_analysis(entities, analysis_type)
        else:
            raise RuntimeError("Operation rejected by conflict resolution")
    else:
        # Standard execution for low-contention scenarios
        return perform_complex_analysis(entities, analysis_type)
```

### Pattern 4: Advanced Configuration

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
        staging_timeout_ms=150.0
    ),
    occ=OCCConfig(
        max_retries=15,
        backoff_factor=2.0
    )
))
async def optimize_schedules(students: List[Student]) -> List[Schedule]:
    # Custom two-stage protection with tuned parameters
    pass
```

## Performance Considerations

### Framework-Native Operations (Optimal Performance)

Entity framework operations are highly optimized:
- **Zero synchronization overhead**: Immutable model eliminates locks
- **Efficient versioning**: Copy-on-write semantics
- **Lazy evaluation**: String addresses resolved only when needed
- **Event system efficiency**: UUID-based references, no data copying

### Conflict Resolution Overhead

Two-stage protection adds minimal overhead only where needed:
- **Pre-ECS filtering**: Microseconds for priority-based resolution
- **OCC validation**: Nanoseconds for version/timestamp comparison  
- **Retry logic**: Only on actual conflicts (rare in well-designed systems)
- **Memory efficiency**: Only winning operations consume ECS resources

## Conclusion

The Abstractions framework's **entity-native functional design** naturally prevents concurrency issues through:
- **Immutable entities** that cannot be corrupted by concurrent access
- **Versioned transformations** that preserve complete history
- **Pure functional operations** that operate on independent data copies
- **Event-driven coordination** that decouples system components

**Conflict resolution should only be applied** to the small subset of operations that involve:
- **Shared collections** with concurrent modifications
- **Complex read-process-write cycles** with extended processing windows  
- **Stateful aggregations** that accumulate data during execution
- **Cross-entity dependencies** where changes affect other entities

This approach preserves the **elegance and performance** of the entity framework for normal operations while providing **robust protection** only where the natural immutability model is insufficient.

The two-stage conflict resolution system (Pre-ECS + OCC) serves as a **surgical tool** for specific scenarios, not a general requirement for entity operations. This maintains the framework's core principles while handling edge cases that require explicit coordination. 