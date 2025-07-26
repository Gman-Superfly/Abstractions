"""
Entity Hierarchy for Operation Management

Hierarchical operation entities with priority-based conflict resolution,
retry logic, and event-driven coordination. Operations are first-class
entities with full ECS lifecycle support.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import Field, model_validator
from uuid import UUID, uuid4
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio

from abstractions.ecs.entity import Entity, EntityRegistry
from abstractions.events.events import emit_events, ProcessingEvent, ProcessedEvent, StateTransitionEvent

# Import additional ECS components for proper integration
try:
    from abstractions.ecs.functional_api import get, promote_to_root
except ImportError:
    # Fallback imports if functional_api is not available
    def get(address: str):
        """Fallback get function."""
        raise NotImplementedError("ECS functional API not available")
    
    def promote_to_root(entity):
        """Fallback promote function."""
        entity.promote_to_root()

from abstractions.events.entity_events import (
    EntityRegistrationEvent, EntityRegisteredEvent,
    EntityVersioningEvent, EntityVersionedEvent
)

__all__ = [
    'OperationStatus', 'OperationPriority', 'OperationEntity', 
    'StructuralOperation', 'NormalOperation', 'LowPriorityOperation',
    'OperationHierarchyError', 'OperationConflictError',
    'get_conflicting_operations', 'get_operations_by_status', 
    'cleanup_completed_operations', 'get_operation_stats',
    'create_operation_hierarchy', 'resolve_operation_conflicts'
]


class OperationStatus(str, Enum):
    """Status states for operation lifecycle."""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class OperationPriority(int, Enum):
    """Standard priority levels for operations."""
    CRITICAL = 10
    HIGH = 8
    NORMAL = 5
    LOW = 2
    BACKGROUND = 1


class OperationHierarchyError(Exception):
    """Raised when operation hierarchy constraints are violated."""
    pass


class OperationConflictError(Exception):
    """Raised when operation conflicts cannot be resolved."""
    pass


class OperationEntity(Entity):
    """
    Base entity for operations with priority-based execution and retry logic.
    
    Features: priority-based conflict resolution, automatic retry with exponential
    backoff, hierarchical parent-child relationships, complete lifecycle tracking.
    """
    
    op_type: str = Field(default="", description="Operation type")
    priority: int = Field(default=OperationPriority.NORMAL, ge=1, le=10, description="Priority (1-10)")
    target_entity_id: UUID = Field(description="Target entity ID")
    retry_count: int = Field(default=0, ge=0, description="Current retry count")
    max_retries: int = Field(default=5, ge=0, description="Max retries before rejection")
    parent_op_id: Optional[UUID] = Field(default=None, description="Parent operation ID")
    status: OperationStatus = Field(default=OperationStatus.PENDING, description="Current status")
    error_message: Optional[str] = Field(default=None, description="Error details")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    
    @model_validator(mode='after')
    def validate_operation_constraints(self) -> 'OperationEntity':
        """Validate operation entity constraints."""
        assert self.priority >= 1 and self.priority <= 10, f"Priority must be 1-10, got {self.priority}"
        assert self.retry_count >= 0, f"Retry count cannot be negative, got {self.retry_count}"
        assert self.max_retries >= 0, f"Max retries cannot be negative, got {self.max_retries}"
        assert self.retry_count <= self.max_retries, f"Retry count {self.retry_count} exceeds max {self.max_retries}"
        
        if self.status in [OperationStatus.SUCCEEDED, OperationStatus.FAILED, OperationStatus.REJECTED]:
            assert self.completed_at is not None, f"Completed operations must have completion timestamp"
        
        if self.status == OperationStatus.EXECUTING:
            assert self.started_at is not None, f"Executing operations must have start timestamp"
        
        return self
    
    def get_effective_priority(self) -> int:
        """Calculate effective priority considering parent hierarchy."""
        visited = set()
        current_op = self
        max_priority = self.priority
        
        while current_op.parent_op_id:
            if current_op.ecs_id in visited:
                raise OperationHierarchyError(f"Circular dependency detected in operation hierarchy")
            
            visited.add(current_op.ecs_id)
            
            # Find parent in EntityRegistry
            parent_op = None
            for root_id in EntityRegistry.tree_registry.keys():
                tree = EntityRegistry.tree_registry.get(root_id)
                if tree and current_op.parent_op_id in tree.nodes:
                    parent_op = tree.nodes[current_op.parent_op_id]
                    break
            
            if not parent_op or not isinstance(parent_op, OperationEntity):
                raise OperationHierarchyError(f"Parent operation {current_op.parent_op_id} not found or invalid")
            
            max_priority = max(max_priority, parent_op.priority)
            current_op = parent_op
        
        return max_priority
    
    @emit_events(
        creating_factory=lambda self: StateTransitionEvent(
            subject_id=self.ecs_id, 
            subject_type=type(self),
            from_state=str(self.status),
            to_state="executing"
        ),
        created_factory=lambda result, self: StateTransitionEvent(
            subject_id=self.ecs_id, 
            subject_type=type(self),
            from_state="pending",
            to_state="executing"
        )
    )
    def start_execution(self) -> None:
        """Mark operation as started and update timestamps."""
        assert self.status == OperationStatus.PENDING, f"Can only start pending operations, current status: {self.status}"
        
        self.status = OperationStatus.EXECUTING
        self.started_at = datetime.now(timezone.utc)
        self.update_ecs_ids()
    
    @emit_events(
        creating_factory=lambda self, success, error_message=None: StateTransitionEvent(
            subject_id=self.ecs_id, 
            subject_type=type(self),
            from_state=str(self.status),
            to_state="succeeded" if success else "failed"
        ),
        created_factory=lambda result, self, success, error_message=None: StateTransitionEvent(
            subject_id=self.ecs_id, 
            subject_type=type(self),
            from_state="executing",
            to_state="succeeded" if success else "failed"
        )
    )
    def complete_operation(self, success: bool, error_message: Optional[str] = None) -> None:
        """Mark operation as completed with success/failure status."""
        assert self.status == OperationStatus.EXECUTING, f"Can only complete executing operations, current status: {self.status}"
        
        self.status = OperationStatus.SUCCEEDED if success else OperationStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error_message = error_message if not success else None
        self.update_ecs_ids()
    
    def increment_retry(self) -> bool:
        """Increment retry count and check if max retries exceeded."""
        assert self.status in [OperationStatus.PENDING, OperationStatus.FAILED], f"Can only retry pending/failed operations, current status: {self.status}"
        
        self.retry_count += 1
        
        if self.retry_count > self.max_retries:
            self.status = OperationStatus.REJECTED
            self.completed_at = datetime.now(timezone.utc)
            self.error_message = f"Max retries ({self.max_retries}) exceeded"
            self.update_ecs_ids()
            return False
        
        self.status = OperationStatus.PENDING
        self.started_at = None
        self.update_ecs_ids()
        return True
    
    def get_backoff_delay(self) -> float:
        """Calculate exponential backoff delay based on retry count."""
        assert self.retry_count > 0, "Backoff delay only applicable after retries"
        return 0.01 * (2 ** self.retry_count)
    
    def get_hierarchy_chain(self) -> List['OperationEntity']:
        """Get complete hierarchy chain from this operation to root."""
        chain = [self]
        visited = {self.ecs_id}
        current_op = self
        
        while current_op.parent_op_id:
            # Find parent in EntityRegistry
            parent_op = None
            for root_id in EntityRegistry.tree_registry.keys():
                tree = EntityRegistry.tree_registry.get(root_id)
                if tree and current_op.parent_op_id in tree.nodes:
                    parent_op = tree.nodes[current_op.parent_op_id]
                    break
            
            if not parent_op or not isinstance(parent_op, OperationEntity):
                raise OperationHierarchyError(f"Parent operation {current_op.parent_op_id} not found")
            
            if parent_op.ecs_id in visited:
                raise OperationHierarchyError("Circular dependency in operation hierarchy")
            
            chain.append(parent_op)
            visited.add(parent_op.ecs_id)
            current_op = parent_op
        
        return chain
    
    def register_operation(self) -> None:
        """Register this operation in the ECS system."""
        if not self.ecs_id:
            self.update_ecs_ids()
        
        # Promote to root if not already done
        if not self.is_root_entity():
            try:
                promote_to_root(self)
            except NotImplementedError:
                # Fallback to direct promotion
                self.promote_to_root()
    
    def update_operation_status(self, new_status: OperationStatus, error_message: Optional[str] = None) -> None:
        """Update operation status with proper ECS integration."""
        old_status = self.status
        self.status = new_status
        
        if new_status in [OperationStatus.SUCCEEDED, OperationStatus.FAILED, OperationStatus.REJECTED]:
            self.completed_at = datetime.now(timezone.utc)
            if error_message and new_status != OperationStatus.SUCCEEDED:
                self.error_message = error_message
        
        # Update ECS IDs to reflect changes
        self.update_ecs_ids()
    
    @classmethod
    def create_and_register(cls, **kwargs) -> 'OperationEntity':
        """Create and register a new operation entity."""
        operation = cls(**kwargs)
        operation.register_operation()
        return operation


class StructuralOperation(OperationEntity):
    """High-priority operations for core structural changes."""
    
    priority: int = Field(default=OperationPriority.CRITICAL, description="Critical priority")
    max_retries: int = Field(default=20, description="High retry count for persistence")
    
    def __init__(self, **data):
        super().__init__(**data)
        assert self.priority >= OperationPriority.HIGH, f"Structural operations must have high/critical priority, got {self.priority}"


class NormalOperation(OperationEntity):
    """Standard operations with balanced priority and retry behavior."""
    
    priority: int = Field(default=OperationPriority.NORMAL, description="Normal priority")
    max_retries: int = Field(default=5, description="Standard retry count")


class LowPriorityOperation(OperationEntity):
    """Low-priority operations that yield easily to higher-priority work."""
    
    priority: int = Field(default=OperationPriority.LOW, description="Low priority")
    max_retries: int = Field(default=3, description="Limited retries to avoid contention")
    
    def __init__(self, **data):
        super().__init__(**data)
        assert self.priority <= OperationPriority.NORMAL, f"Low priority operations cannot exceed normal priority, got {self.priority}"


def get_conflicting_operations(target_entity_id: UUID) -> List[OperationEntity]:
    """Find active operations targeting the same entity."""
    assert target_entity_id is not None, "Target entity ID required"
    
    conflicting_ops = []
    
    # Use EntityRegistry to find all operation entities
    for root_id in EntityRegistry.tree_registry.keys():
        tree = EntityRegistry.tree_registry.get(root_id)
        if tree:
            for entity_id, entity in tree.nodes.items():
                if (isinstance(entity, OperationEntity) and 
                    entity.target_entity_id == target_entity_id and
                    entity.status in [OperationStatus.PENDING, OperationStatus.EXECUTING]):
                    conflicting_ops.append(entity)
    
    return conflicting_ops


def get_operations_by_status(status: OperationStatus) -> List[OperationEntity]:
    """Find operations with specific status."""
    assert isinstance(status, OperationStatus), f"Expected OperationStatus, got {type(status)}"
    
    operations = []
    
    # Use EntityRegistry to find all operation entities
    for root_id in EntityRegistry.tree_registry.keys():
        tree = EntityRegistry.tree_registry.get(root_id)
        if tree:
            for entity_id, entity in tree.nodes.items():
                if isinstance(entity, OperationEntity) and entity.status == status:
                    operations.append(entity)
    
    return operations


def cleanup_completed_operations(retention_hours: int = 24, dry_run: bool = False) -> Dict[str, int]:
    """Clean up old completed operations."""
    assert retention_hours > 0, f"Retention hours must be positive, got {retention_hours}"
    
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
    cleaned = 0
    total = 0
    
    # Clean from EntityRegistry
    for root_id in list(EntityRegistry.tree_registry.keys()):
        tree = EntityRegistry.tree_registry.get(root_id)
        if tree:
            entities_to_remove = []
            for entity_id, entity in tree.nodes.items():
                if isinstance(entity, OperationEntity):
                    total += 1
                    if (entity.status in [OperationStatus.SUCCEEDED, OperationStatus.FAILED, OperationStatus.REJECTED] 
                        and entity.completed_at and entity.completed_at < cutoff_time):
                        
                        if not dry_run:
                            entities_to_remove.append(entity_id)
                        cleaned += 1
            
            # Remove entities outside the iteration
            if not dry_run:
                for entity_id in entities_to_remove:
                    if entity_id in tree.nodes:
                        del tree.nodes[entity_id]
                        tree.node_count = len(tree.nodes)
                        
                        # Clean up from other registry mappings
                        if entity_id in EntityRegistry.ecs_id_to_root_id:
                            del EntityRegistry.ecs_id_to_root_id[entity_id]
    
    return {"total": total, "cleaned": cleaned, "retention_hours": retention_hours}


def get_operation_stats() -> Dict[str, Any]:
    """Get operation statistics."""
    stats = {"total": 0, "by_status": {}, "by_type": {}, "by_priority": {}}
    
    # Collect from EntityRegistry
    for root_id in EntityRegistry.tree_registry.keys():
        tree = EntityRegistry.tree_registry.get(root_id)
        if tree:
            for entity_id, entity in tree.nodes.items():
                if isinstance(entity, OperationEntity):
                    stats["total"] += 1
                    stats["by_status"][entity.status.value] = stats["by_status"].get(entity.status.value, 0) + 1
                    stats["by_type"][entity.op_type] = stats["by_type"].get(entity.op_type, 0) + 1
                    stats["by_priority"][entity.priority] = stats["by_priority"].get(entity.priority, 0) + 1
    
    return stats


def create_operation_hierarchy(operations: List[OperationEntity], parent_op: Optional[OperationEntity] = None) -> List[OperationEntity]:
    """Create hierarchical operation chain with proper parent-child relationships."""
    assert len(operations) > 0, "Cannot create hierarchy from empty operation list"
    
    for op in operations:
        assert isinstance(op, OperationEntity), f"All items must be OperationEntity, got {type(op)}"
        assert op.ecs_id is not None, "All operations must have valid ECS IDs"
    
    if parent_op:
        operations[0].parent_op_id = parent_op.ecs_id
    
    for i in range(1, len(operations)):
        operations[i].parent_op_id = operations[i-1].ecs_id
    
    for op in operations:
        op.promote_to_root()
    
    return operations


def resolve_operation_conflicts(target_entity_id: UUID, current_operations: List[OperationEntity], grace_tracker=None) -> List[OperationEntity]:
    """
    Resolve conflicts between operations targeting the same entity.
    
    EXECUTING operations are protected from preemption - they cannot be rejected
    once they have started execution. Operations within their grace period are
    also protected from preemption.
    """
    assert len(current_operations) > 0, "Cannot resolve conflicts for empty operation list"
    
    # Separate executing operations (protected) from pending operations (can be preempted)
    executing_ops = [op for op in current_operations if op.status == OperationStatus.EXECUTING]
    pending_ops = [op for op in current_operations if op.status == OperationStatus.PENDING]
    
    # EXECUTING operations are automatically winners - they cannot be preempted
    winning_ops = executing_ops.copy()
    
    # Check for grace period protection among pending operations
    grace_protected_ops = []
    if grace_tracker:
        protected_ids = grace_tracker.get_protected_operations()
        grace_protected_ops = [op for op in pending_ops if op.ecs_id in protected_ids]
        # Grace protected operations are also winners
        winning_ops.extend(grace_protected_ops)
        # Remove grace protected operations from pending competition
        pending_ops = [op for op in pending_ops if op.ecs_id not in protected_ids]
    
    # Only resolve conflicts among non-protected PENDING operations
    if pending_ops:
        priority_groups: Dict[int, List[OperationEntity]] = {}
        
        for op in pending_ops:
            effective_priority = op.get_effective_priority()
            if effective_priority not in priority_groups:
                priority_groups[effective_priority] = []
            priority_groups[effective_priority].append(op)
        
        # Find highest priority among pending operations
        highest_priority = max(priority_groups.keys())
        highest_priority_pending = priority_groups[highest_priority]
        
        # If multiple operations have same highest priority, take earliest created
        if len(highest_priority_pending) > 1:
            highest_priority_pending.sort(key=lambda op: op.created_at)
            highest_priority_pending = [highest_priority_pending[0]]
        
        # Add winning pending operation to winners
        winning_ops.extend(highest_priority_pending)
        
        # Reject losing pending operations (cannot preempt executing or grace-protected operations)
        losing_pending_ops = [op for ops in priority_groups.values() for op in ops if op not in highest_priority_pending]
        
        for op in losing_pending_ops:
            op.status = OperationStatus.REJECTED
            op.completed_at = datetime.now(timezone.utc)
            if executing_ops:
                op.error_message = f"Cannot preempt executing operation(s). Protected operations: {[str(eo.ecs_id)[:8] for eo in executing_ops]}"
            elif grace_protected_ops:
                op.error_message = f"Cannot preempt grace-protected operation(s). Protected operations: {[str(gp.ecs_id)[:8] for gp in grace_protected_ops]}"
            elif highest_priority_pending:
                op.error_message = f"Preempted by higher priority pending operation {highest_priority_pending[0].ecs_id}"
            op.update_ecs_ids()
    
    return winning_ops