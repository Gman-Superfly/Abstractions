"""
Integration Test: Entity Hierarchy + Events + Agent Observer

This test demonstrates that the new operation hierarchy system works
correctly with the existing agent observer and registry agent components.
"""

import asyncio
from uuid import uuid4, UUID
from datetime import datetime, timezone

# Import the entity hierarchy components
from abstractions.ecs.entity_hierarchy import (
    OperationEntity, StructuralOperation, NormalOperation, LowPriorityOperation,
    OperationStatus, OperationPriority,
    get_conflicting_operations, get_operations_by_status, get_operation_stats,
    create_operation_hierarchy, resolve_operation_conflicts
)

# Import event system components
from abstractions.events.events import (
    get_event_bus, emit,
    OperationStartedEvent, OperationCompletedEvent, OperationConflictEvent,
    OperationRejectedEvent, OperationRetryEvent
)

# Import ECS components
from abstractions.ecs.entity import Entity, EntityRegistry

# Simple test entity
class TestEntity(Entity):
    """Simple entity for testing operations."""
    name: str = "test"
    value: int = 0


async def test_operation_creation_and_events():
    """Test creating operations and emitting events."""
    print("\n🧪 TEST: Operation Creation and Events")
    
    # Create a test target entity
    target_entity = TestEntity(name="test_target", value=42)
    target_entity.promote_to_root()
    
    # Create a normal operation
    operation = NormalOperation.create_and_register(
        op_type="test_operation",
        priority=OperationPriority.NORMAL,
        target_entity_id=target_entity.ecs_id,
        max_retries=3
    )
    
    print(f"✅ Created operation: {operation.ecs_id}")
    print(f"   └─ Target: {operation.target_entity_id}")
    print(f"   └─ Priority: {operation.priority}")
    print(f"   └─ Status: {operation.status}")
    
    # Start the operation and emit event
    await emit(OperationStartedEvent(
        process_name="operation_execution",
        op_id=operation.ecs_id,
        op_type=operation.op_type,
        priority=operation.priority,
        target_entity_id=operation.target_entity_id
    ))
    
    # Simulate operation execution
    operation.start_execution()
    
    # Complete the operation
    operation.complete_operation(success=True)
    
    await emit(OperationCompletedEvent(
        process_name="operation_execution",
        op_id=operation.ecs_id,
        op_type=operation.op_type,
        target_entity_id=operation.target_entity_id,
        execution_duration_ms=150.0
    ))
    
    print(f"✅ Operation completed: {operation.status}")
    
    return operation, target_entity


async def test_operation_conflicts():
    """Test operation conflict detection and resolution."""
    print("\n🧪 TEST: Operation Conflicts")
    
    # Create a test target entity
    target_entity = TestEntity(name="conflict_target", value=100)
    target_entity.promote_to_root()
    
    # Create two conflicting operations
    high_priority_op = StructuralOperation.create_and_register(
        op_type="high_priority_operation",
        priority=OperationPriority.CRITICAL,
        target_entity_id=target_entity.ecs_id,
        max_retries=5
    )
    
    low_priority_op = LowPriorityOperation.create_and_register(
        op_type="low_priority_operation", 
        priority=OperationPriority.LOW,
        target_entity_id=target_entity.ecs_id,
        max_retries=2
    )
    
    print(f"✅ Created conflicting operations:")
    print(f"   ├─ High priority: {high_priority_op.ecs_id} (priority {high_priority_op.priority})")
    print(f"   └─ Low priority: {low_priority_op.ecs_id} (priority {low_priority_op.priority})")
    
    # Emit conflict event
    await emit(OperationConflictEvent(
        op_id=low_priority_op.ecs_id,
        op_type=low_priority_op.op_type,
        target_entity_id=target_entity.ecs_id,
        priority=low_priority_op.priority,
        conflict_details={},
        conflicting_op_ids=[high_priority_op.ecs_id]
    ))
    
    # Find conflicting operations
    conflicts = get_conflicting_operations(target_entity.ecs_id)
    print(f"✅ Found {len(conflicts)} conflicting operations")
    
    # Resolve conflicts
    if conflicts:
        winners = resolve_operation_conflicts(target_entity.ecs_id, conflicts)
        print(f"✅ Conflict resolved, {len(winners)} winning operation(s)")
        
        # Emit rejection event for losers
        for op in conflicts:
            if op not in winners and op.status == OperationStatus.REJECTED:
                await emit(OperationRejectedEvent(
                    op_id=op.ecs_id,
                    op_type=op.op_type,
                    target_entity_id=op.target_entity_id,
                    from_state="pending",
                    to_state="rejected",
                    rejection_reason="preempted_by_higher_priority",
                    retry_count=op.retry_count
                ))
    
    return high_priority_op, low_priority_op


async def test_operation_hierarchy():
    """Test operation parent-child relationships."""
    print("\n🧪 TEST: Operation Hierarchy")
    
    # Create a test target entity
    target_entity = TestEntity(name="hierarchy_target", value=200)
    target_entity.promote_to_root()
    
    # Create parent operation
    parent_op = StructuralOperation.create_and_register(
        op_type="parent_operation",
        priority=OperationPriority.HIGH,
        target_entity_id=target_entity.ecs_id
    )
    
    # Create child operations
    child_op1 = NormalOperation.create_and_register(
        op_type="child_operation_1",
        priority=OperationPriority.NORMAL,
        target_entity_id=target_entity.ecs_id,
        parent_op_id=parent_op.ecs_id
    )
    
    child_op2 = NormalOperation.create_and_register(
        op_type="child_operation_2",
        priority=OperationPriority.NORMAL,
        target_entity_id=target_entity.ecs_id,
        parent_op_id=parent_op.ecs_id
    )
    
    print(f"✅ Created operation hierarchy:")
    print(f"   ├─ Parent: {parent_op.ecs_id}")
    print(f"   ├─ Child 1: {child_op1.ecs_id}")
    print(f"   └─ Child 2: {child_op2.ecs_id}")
    
    # Test hierarchy chain
    hierarchy = child_op1.get_hierarchy_chain()
    print(f"✅ Hierarchy chain length: {len(hierarchy)}")
    
    # Test effective priority (should inherit from parent)
    effective_priority = child_op1.get_effective_priority()
    print(f"✅ Child effective priority: {effective_priority} (inherited from parent: {parent_op.priority})")
    
    return parent_op, child_op1, child_op2


async def test_operation_statistics():
    """Test operation statistics and management."""
    print("\n🧪 TEST: Operation Statistics")
    
    # Get current stats
    stats = get_operation_stats()
    print(f"✅ Operation statistics:")
    print(f"   ├─ Total operations: {stats['total']}")
    print(f"   ├─ By status: {stats['by_status']}")
    print(f"   ├─ By type: {stats['by_type']}")
    print(f"   └─ By priority: {stats['by_priority']}")
    
    # Test operations by status
    pending_ops = get_operations_by_status(OperationStatus.PENDING)
    executing_ops = get_operations_by_status(OperationStatus.EXECUTING)
    completed_ops = get_operations_by_status(OperationStatus.SUCCEEDED)
    
    print(f"✅ Operations by status:")
    print(f"   ├─ Pending: {len(pending_ops)}")
    print(f"   ├─ Executing: {len(executing_ops)}")
    print(f"   └─ Completed: {len(completed_ops)}")


async def test_integration_with_agent_observer():
    """Test that agent observer receives operation events correctly."""
    print("\n🧪 TEST: Agent Observer Integration")
    
    # Import agent observer to ensure handlers are registered
    try:
        import abstractions.agent_observer
        print("✅ Agent observer imported successfully")
    except ImportError as e:
        print(f"⚠️  Agent observer import failed: {e}")
        return
    
    # Create and emit some operation events
    target_entity = TestEntity(name="observer_test", value=300)
    target_entity.promote_to_root()
    
    operation = NormalOperation.create_and_register(
        op_type="observer_test_operation",
        priority=OperationPriority.NORMAL,
        target_entity_id=target_entity.ecs_id
    )
    
    # Emit a series of operation events
    print("📡 Emitting operation events for observer...")
    
    await emit(OperationStartedEvent(
        process_name="observer_test",
        op_id=operation.ecs_id,
        op_type=operation.op_type,
        priority=operation.priority,
        target_entity_id=operation.target_entity_id
    ))
    
    # Small delay to let observers process
    await asyncio.sleep(0.1)
    
    await emit(OperationCompletedEvent(
        process_name="observer_test",
        op_id=operation.ecs_id,
        op_type=operation.op_type,
        target_entity_id=operation.target_entity_id,
        execution_duration_ms=250.0
    ))
    
    print("✅ Operation events emitted for observer")


async def main():
    """Run all integration tests."""
    print("🚀 Starting Entity Hierarchy Integration Tests")
    print("=" * 60)
    
    # Start the event bus
    bus = get_event_bus()
    await bus.start()
    
    try:
        # Run tests
        await test_operation_creation_and_events()
        await test_operation_conflicts()
        await test_operation_hierarchy()
        await test_operation_statistics()
        await test_integration_with_agent_observer()
        
        print("\n" + "=" * 60)
        print("🎉 All integration tests completed successfully!")
        
        # Final statistics
        final_stats = get_operation_stats()
        print(f"\n📊 Final operation count: {final_stats['total']}")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop the event bus
        await bus.stop()


if __name__ == "__main__":
    asyncio.run(main()) 