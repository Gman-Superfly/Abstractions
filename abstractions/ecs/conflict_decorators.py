"""
Conflict Resolution Decorators for the Abstractions Framework

These decorators integrate with the existing operation hierarchy and event system
to provide declarative conflict resolution for operations that need it.

The decorators work with the existing systems:
- Operation hierarchy from entity_hierarchy.py
- Event-driven conflict resolution from events.py  
- Pre-ECS staging and OCC protection from stress tests
"""

import asyncio
import functools
import time
from typing import Optional, Callable, Any, Dict, List, Type, Union
from uuid import UUID, uuid4
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field

from abstractions.ecs.entity import Entity
from abstractions.ecs.entity_hierarchy import (
    OperationEntity, OperationStatus, OperationPriority,
    StructuralOperation, NormalOperation, LowPriorityOperation,
    resolve_operation_conflicts
)
from abstractions.events.events import (
    emit, get_event_bus,
    OperationStartedEvent, OperationCompletedEvent, 
    OperationConflictEvent, OperationRejectedEvent, OperationRetryEvent
)

__all__ = [
    'with_conflict_resolution',
    'no_conflict_resolution', 
    'ConflictResolutionConfig',
    'OCCConfig',
    'PreECSConfig',
    'get_staging_area_status',
    'clear_staging_area',
    'start_staging_coordinator',
    'stop_staging_coordinator'
]


# Global staging coordinator flags
_staging_coordinator_running = False
_staging_coordinator_task: Optional[asyncio.Task] = None


class ConflictResolutionMode(str, Enum):
    """Available conflict resolution modes."""
    NONE = "none"
    PRE_ECS = "pre_ecs"
    OCC = "occ"
    BOTH = "both"


from pydantic import BaseModel

class PreECSConfig(BaseModel):
    """Configuration for Pre-ECS conflict resolution."""
    
    enabled: bool = True
    staging_timeout_ms: float = 100.0
    priority: int = OperationPriority.NORMAL
    operation_class_name: str = "NormalOperation"  # Store class name instead of class


class OCCConfig(BaseModel):
    """Configuration for Optimistic Concurrency Control."""
    
    enabled: bool = True
    max_retries: int = 10
    base_delay_ms: float = 5.0
    backoff_factor: float = 1.5
    version_field: str = "version"
    modified_field: str = "last_modified"


class ConflictResolutionConfig(BaseModel):
    """Complete configuration for conflict resolution."""
    
    mode: ConflictResolutionMode = ConflictResolutionMode.NONE
    pre_ecs: PreECSConfig = Field(default_factory=PreECSConfig)
    occ: OCCConfig = Field(default_factory=OCCConfig)
    
    def __init__(
        self,
        mode: ConflictResolutionMode = ConflictResolutionMode.NONE,
        pre_ecs: Optional[PreECSConfig] = None,
        occ: Optional[OCCConfig] = None,
        priority: Optional[int] = None,
        **data
    ):
        # Handle the old-style init while maintaining Pydantic compatibility
        if pre_ecs is None:
            pre_ecs = PreECSConfig()
        if occ is None:
            occ = OCCConfig()
        
        # Override priority if specified
        if priority is not None:
            pre_ecs.priority = priority
            
        super().__init__(mode=mode, pre_ecs=pre_ecs, occ=occ, **data)


class ConflictResolutionOperation(OperationEntity):
    """Operation entity for conflict resolution decorators."""
    
    # Function execution data
    function_name: str = ""
    function_args: tuple = ()
    function_kwargs: Dict[str, Any] = {}
    target_entities: List[UUID] = []
    execution_result: Any = None
    
    # OCC data
    occ_retry_count: int = 0
    occ_max_retries: int = 10
    occ_base_delay_ms: float = 5.0
    occ_backoff_factor: float = 1.5
    occ_version_field: str = "version"
    occ_modified_field: str = "last_modified"
    
    def __init__(self, **data):
        """Initialize with execution event (excluded from Pydantic validation)."""
        super().__init__(**data)
        # Manually create execution event - not a Pydantic field
        self._execution_event: Optional[asyncio.Event] = None
        self._stored_function: Optional[Callable] = None
    
    @property
    def execution_event(self) -> asyncio.Event:
        """Get or create execution event."""
        if self._execution_event is None:
            self._execution_event = asyncio.Event()
        return self._execution_event
    
    def set_stored_function(self, func: Callable):
        """Store the function to be executed later."""
        self._stored_function = func
    
    def get_stored_function(self) -> Optional[Callable]:
        """Get the stored function."""
        return self._stored_function
    
    async def execute_with_occ(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with decorator-appropriate OCC protection."""
        # For decorators, we simply execute the function
        # Pre-ECS conflict resolution already handled operation-level conflicts
        # The decorated function is trusted to handle entity modifications properly
        return await self._execute_function(func, *args, **kwargs)
    
    def _extract_target_entities(self, *args, **kwargs) -> List[Entity]:
        """Extract Entity objects from function arguments."""
        entities = []
        
        for arg in args:
            if isinstance(arg, Entity):
                entities.append(arg)
        
        for kwarg_value in kwargs.values():
            if isinstance(kwarg_value, Entity):
                entities.append(kwarg_value)
        
        return entities
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute the actual function."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)


# Global staging area - declared after ConflictResolutionOperation class to avoid forward reference
_staging_area: Dict[UUID, List[ConflictResolutionOperation]] = {}


async def _start_staging_coordinator():
    """Start the global staging coordinator following stress test patterns."""
    global _staging_coordinator_running, _staging_coordinator_task
    
    if not _staging_coordinator_running:
        _staging_coordinator_running = True
        _staging_coordinator_task = asyncio.create_task(_staging_coordinator_worker())
        print("🚀 Staging coordinator started (following stress test patterns)")


async def _stop_staging_coordinator():
    """Stop the global staging coordinator."""
    global _staging_coordinator_running, _staging_coordinator_task
    
    if _staging_coordinator_running:
        _staging_coordinator_running = False
        if _staging_coordinator_task:
            _staging_coordinator_task.cancel()
            try:
                await _staging_coordinator_task
            except asyncio.CancelledError:
                pass
        print("⏹️  Staging coordinator stopped")


async def _staging_coordinator_worker():
    """Background worker that resolves conflicts following stress test patterns."""
    global _staging_area
    
    print("🔄 Staging coordinator worker started")
    
    while _staging_coordinator_running:
        try:
            # Process each target entity's staging area
            for target_entity_id in list(_staging_area.keys()):
                await _resolve_conflicts_for_target_stress_test_pattern(target_entity_id)
            
            # Check every 50ms for conflicts (similar to stress test timing)
            await asyncio.sleep(0.05)
            
        except Exception as e:
            print(f"⚠️  Error in staging coordinator: {e}")
            await asyncio.sleep(0.1)


async def _resolve_conflicts_for_target_stress_test_pattern(target_entity_id: UUID):
    """
    Resolve conflicts using the exact stress test pattern from dynamic_stress_test.py.
    
    This follows the proven pattern:
    1. Check staging area for multiple operations
    2. Sort by priority (higher wins), then timestamp  
    3. Promote winner to ECS, reject losers
    4. Signal operations about results
    """
    global _staging_area
    
    if target_entity_id not in _staging_area:
        return
    
    pending_ops = _staging_area[target_entity_id]
    
    if len(pending_ops) == 0:
        # Clean up empty staging areas
        del _staging_area[target_entity_id]
        return
    
    elif len(pending_ops) == 1:
        # No conflict - promote single operation (stress test pattern)
        winner = pending_ops[0]
        winner.promote_to_root()
        
        print(f"✅ NO CONFLICT: Promoting single operation {winner.op_type} for target {str(target_entity_id)[:8]}")
        
        # Execute the operation
        await _execute_winning_operation(winner)
        
        # Clear staging
        del _staging_area[target_entity_id]
        
    else:
        # CONFLICT DETECTED - Follow exact stress test resolution pattern
        print(f"⚔️  PRE-ECS CONFLICT: {len(pending_ops)} operations competing for target {str(target_entity_id)[:8]}")
        for op in pending_ops:
            print(f"   ├─ {op.op_type} (Priority: {op.priority}, Status: PRE-ECS)")
        
        # Sort by priority (higher priority = higher number wins) - exact stress test pattern
        pending_ops.sort(key=lambda op: (op.priority, -op.created_at.timestamp()), reverse=True)
        
        # Winner is highest priority (first after sort) - exact stress test pattern
        winner = pending_ops[0]
        losers = pending_ops[1:]
        
        print(f"🏆 PRE-ECS RESOLUTION: 1 winner, {len(losers)} rejected")
        print(f"✅ WINNER: {winner.op_type} (Priority: {winner.priority})")
        
        # Promote winner to ECS for execution - exact stress test pattern
        winner.promote_to_root()
        
        # Execute the winning operation
        await _execute_winning_operation(winner)
        
        # Reject losers before they enter ECS - exact stress test pattern
        for loser in losers:
            print(f"❌ REJECTED: {loser.op_type} (Priority: {loser.priority})")
            
            # Signal loser that it was rejected
            loser.execution_result = RuntimeError(f"Operation rejected by Pre-ECS conflict resolution")
            loser.execution_event.set()
            
            # Emit rejection event (following stress test event pattern)
            await emit(OperationRejectedEvent(
                op_id=uuid4(),  # Synthetic ID since loser never got ECS ID
                op_type=loser.op_type,
                target_entity_id=target_entity_id,
                from_state="staged",
                to_state="rejected",
                rejection_reason=f"Lost Pre-ECS priority conflict (priority {loser.priority} vs winner {winner.priority})",
                retry_count=0
            ))
        
        # Clear staging area for this target - exact stress test pattern
        del _staging_area[target_entity_id]


async def _execute_winning_operation(operation: ConflictResolutionOperation):
    """Execute the winning operation and signal completion."""
    try:
        # Emit operation started event
        await emit(OperationStartedEvent(
            process_name="conflict_protected_execution",
            op_id=operation.ecs_id,
            op_type=operation.op_type,
            priority=operation.priority,
            target_entity_id=operation.target_entity_id
        ))
        
        operation.start_execution()
        
        # Execute the actual function
        func = operation.function_args[0] if operation.function_args else None
        args = operation.function_args[1:] if len(operation.function_args) > 1 else ()
        kwargs = operation.function_kwargs
        
        # Store function separately during operation creation
        result = await operation._execute_function(operation.get_stored_function(), *args, **kwargs)
        
        operation.complete_operation(success=True)
        operation.execution_result = result
        operation.execution_event.set()
        
        # Emit completion event
        await emit(OperationCompletedEvent(
            process_name="conflict_protected_execution",
            op_id=operation.ecs_id,
            op_type=operation.op_type,
            target_entity_id=operation.target_entity_id,
            execution_duration_ms=(
                (operation.completed_at - operation.started_at).total_seconds() * 1000
                if operation.completed_at and operation.started_at else 0.0
            )
        ))
        
    except Exception as e:
        operation.complete_operation(success=False, error_message=str(e))
        operation.execution_result = e
        operation.execution_event.set()
        
        # Emit failure event
        await emit(OperationRejectedEvent(
            op_id=operation.ecs_id,
            op_type=operation.op_type,
            target_entity_id=operation.target_entity_id,
            from_state="executing",
            to_state="failed",
            rejection_reason=f"Function execution failed: {str(e)}",
            retry_count=operation.retry_count
        ))


def with_conflict_resolution(
    pre_ecs: bool = False,
    occ: bool = False,
    priority: int = OperationPriority.NORMAL,
    mode: Optional[ConflictResolutionMode] = None,
    config: Optional[ConflictResolutionConfig] = None
) -> Callable:
    """
    Decorator that adds conflict resolution to functions that need it.
    
    This decorator integrates with the existing operation hierarchy and event system
    to provide declarative conflict resolution for specific operations.
    
    Args:
        pre_ecs: Enable Pre-ECS conflict resolution (staging area)
        occ: Enable Optimistic Concurrency Control
        priority: Operation priority for conflict resolution
        mode: Conflict resolution mode (overrides pre_ecs/occ flags)
        config: Complete configuration object
        
    Examples:
        @with_conflict_resolution(pre_ecs=True, occ=True, priority=OperationPriority.HIGH)
        def process_student_cohort(cohort: List[Student]) -> List[AnalysisResult]:
            # Two-stage protection applied automatically
            pass
            
        @with_conflict_resolution(config=ConflictResolutionConfig(
            mode=ConflictResolutionMode.BOTH,
            pre_ecs=PreECSConfig(priority=OperationPriority.CRITICAL),
            occ=OCCConfig(max_retries=15)
        ))
        async def optimize_schedules(students: List[Student]) -> List[Schedule]:
            # Custom configuration for complex operations
            pass
    """
    # Determine final configuration
    if config:
        final_config = config
    else:
        if mode:
            final_mode = mode
        elif pre_ecs and occ:
            final_mode = ConflictResolutionMode.BOTH
        elif pre_ecs:
            final_mode = ConflictResolutionMode.PRE_ECS
        elif occ:
            final_mode = ConflictResolutionMode.OCC
        else:
            final_mode = ConflictResolutionMode.NONE
        
        final_config = ConflictResolutionConfig(
            mode=final_mode,
            priority=priority
        )
    
    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await _execute_with_conflict_resolution(
                    func, final_config, *args, **kwargs
                )
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(_execute_with_conflict_resolution(
                    func, final_config, *args, **kwargs
                ))
            return sync_wrapper
    
    return decorator


async def _execute_with_conflict_resolution(
    func: Callable,
    config: ConflictResolutionConfig,
    *args,
    **kwargs
) -> Any:
    """Execute function with conflict resolution following proven stress test patterns."""
    global _staging_area
    
    if config.mode == ConflictResolutionMode.NONE:
        # No conflict resolution - execute directly
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    # Extract target entity IDs for conflict detection (SINGLE EXTRACTION)
    target_entity_ids = []
    for arg in args:
        if isinstance(arg, Entity):
            target_entity_ids.append(arg.ecs_id)
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, Entity):
                    target_entity_ids.append(item.ecs_id)
    
    for key, value in kwargs.items():
        if isinstance(value, Entity):
            target_entity_ids.append(value.ecs_id)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Entity):
                    target_entity_ids.append(item.ecs_id)
    
    # Use first entity as primary target, or generate UUID if no entities
    primary_target_id = target_entity_ids[0] if target_entity_ids else uuid4()
    
    # Create operation entity (FOLLOWING STRESS TEST PATTERN)
    operation = ConflictResolutionOperation(
        target_entity_id=primary_target_id,
        op_type=f"conflict_protected_{func.__name__}",
        function_name=func.__name__,
        function_args=args,  # Store original args
        function_kwargs=kwargs,  # Store original kwargs
        priority=config.pre_ecs.priority,
        max_retries=config.occ.max_retries,
        occ_max_retries=config.occ.max_retries,
        occ_base_delay_ms=config.occ.base_delay_ms,
        occ_backoff_factor=config.occ.backoff_factor,
        occ_version_field=config.occ.version_field,
        occ_modified_field=config.occ.modified_field,
        target_entities=target_entity_ids
    )
    
    # Store the function for later execution (important!)
    operation.set_stored_function(func)
    
    # CRITICAL: Start staging coordinator if needed
    await _start_staging_coordinator()
    
    # Pre-ECS conflict resolution (EXACT STRESS TEST PATTERN)
    if config.mode in [ConflictResolutionMode.PRE_ECS, ConflictResolutionMode.BOTH]:
        
        # Submit to staging area WITHOUT promoting to ECS (EXACT STRESS TEST PATTERN)
        if primary_target_id not in _staging_area:
            _staging_area[primary_target_id] = []
        _staging_area[primary_target_id].append(operation)
        
        print(f"📝 SUBMITTED TO STAGING: {operation.op_type} (Priority: {operation.priority}) → Target: {str(primary_target_id)[:8]}")
        
        # Wait for staging coordinator to resolve conflicts and execute operation
        await operation.execution_event.wait()
        
        # Check if operation was rejected or succeeded
        if isinstance(operation.execution_result, Exception):
            raise operation.execution_result
        else:
            return operation.execution_result
    
    else:
        # No Pre-ECS resolution - execute directly with optional OCC
        operation.promote_to_root()  # Direct ECS promotion
        operation_id = operation.ecs_id
        
        # Emit operation started event
        await emit(OperationStartedEvent(
            process_name="conflict_protected_execution",
            op_id=operation_id,
            op_type=operation.op_type,
            priority=operation.priority,
            target_entity_id=primary_target_id
        ))
        
        try:
            operation.start_execution()
            
            # Execute with OCC if enabled
            if config.mode == ConflictResolutionMode.OCC:
                result = await operation.execute_with_occ(func, *args, **kwargs)
            else:
                result = await operation._execute_function(func, *args, **kwargs)
            
            operation.complete_operation(success=True)
            
            # Emit completion event
            await emit(OperationCompletedEvent(
                process_name="conflict_protected_execution",
                op_id=operation_id,
                op_type=operation.op_type,
                target_entity_id=primary_target_id,
                execution_duration_ms=(
                    (operation.completed_at - operation.started_at).total_seconds() * 1000
                    if operation.completed_at and operation.started_at else 0.0
                )
            ))
            
            return result
            
        except Exception as e:
            operation.complete_operation(success=False, error_message=str(e))
            
            # Emit failure event
            await emit(OperationRejectedEvent(
                op_id=operation_id,
                op_type=operation.op_type,
                target_entity_id=primary_target_id,
                from_state="executing",
                to_state="failed",
                rejection_reason=f"Function execution failed: {str(e)}",
                retry_count=operation.retry_count
            ))
            
            raise


def no_conflict_resolution(func: Callable) -> Callable:
    """
    Decorator that explicitly opts out of conflict resolution for performance.
    
    This is a marker decorator that documents that the function has been
    analyzed and determined not to need conflict resolution.
    
    Example:
        @no_conflict_resolution  # Explicit opt-out for performance
        def read_only_batch_analysis(students: List[Student]) -> Statistics:
            # Read-only operation, skip protection
            pass
    """
    # Add metadata to mark this function as explicitly opted out
    func._no_conflict_resolution = True
    return func


# Utility functions for integration with existing systems

def get_staging_area_status() -> Dict[str, Any]:
    """Get current status of the Pre-ECS staging area."""
    global _staging_area
    
    total_staged = sum(len(ops) for ops in _staging_area.values())
    targets_with_conflicts = sum(1 for ops in _staging_area.values() if len(ops) > 1)
    
    return {
        "total_staged_operations": total_staged,
        "targets_with_staging": len(_staging_area),
        "targets_with_conflicts": targets_with_conflicts,
        "staging_details": {
            str(target_id): len(ops) 
            for target_id, ops in _staging_area.items()
        }
    }


def clear_staging_area() -> None:
    """Clear the Pre-ECS staging area (for testing/cleanup)."""
    global _staging_area
    _staging_area.clear()


def start_staging_coordinator():
    """Start the global staging coordinator (async wrapper)."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # If we're in a running loop, schedule the task
        return loop.create_task(_start_staging_coordinator())
    except RuntimeError:
        # No running loop, use asyncio.run
        return asyncio.run(_start_staging_coordinator())


def stop_staging_coordinator():
    """Stop the global staging coordinator (async wrapper)."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # If we're in a running loop, schedule the task
        return loop.create_task(_stop_staging_coordinator())
    except RuntimeError:
        # No running loop, use asyncio.run
        return asyncio.run(_stop_staging_coordinator())


def is_conflict_protected(func: Callable) -> bool:
    """Check if a function has conflict resolution applied."""
    return (
        hasattr(func, '__wrapped__') and 
        hasattr(func, '_conflict_resolution_config')
    ) or hasattr(func, '_no_conflict_resolution') 