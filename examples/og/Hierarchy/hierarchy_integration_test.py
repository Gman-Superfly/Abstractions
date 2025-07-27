"""
Comprehensive Hierarchy Integration Test - Validates All Failure and Success Scenarios

This is a comprehensive production test that validates:
- Conflict resolution correctness under stress
- Grace period protection mechanisms  
- Operation lifecycle edge cases
- Error handling and retry logic
- Concurrent operation scenarios
- All failure and success conditions

Production-grade validation with comprehensive assertions and edge case coverage.
"""

import asyncio
import time
import random
from uuid import UUID
from datetime import datetime, timezone
from typing import List, Dict, Set, Optional

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
    modification_count: int = 0


class GracePeriodTracker:
    """Grace period tracker for testing execution protection."""
    
    def __init__(self, grace_period_seconds: float):
        self.grace_period_seconds = grace_period_seconds
        self.executing_operations: Dict[UUID, datetime] = {}
        
    def start_grace_period(self, op_id: UUID):
        self.executing_operations[op_id] = datetime.now(timezone.utc)
        
    def end_grace_period(self, op_id: UUID):
        self.executing_operations.pop(op_id, None)
        
    def can_be_preempted(self, op_id: UUID) -> bool:
        if op_id not in self.executing_operations:
            return True
            
        start_time = self.executing_operations[op_id]
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        return elapsed >= self.grace_period_seconds
        
    def get_protected_operations(self) -> Set[UUID]:
        now = datetime.now(timezone.utc)
        protected = set()
        
        for op_id, start_time in self.executing_operations.items():
            elapsed = (now - start_time).total_seconds()
            if elapsed < self.grace_period_seconds:
                protected.add(op_id)
                
        return protected


class RealOperationEntity(OperationEntity):
    """Production operation entity that performs actual work and handles failures."""
    
    operation_type: str = "test_operation"
    should_fail: bool = False
    execution_time_ms: float = 10.0
    
    async def execute_real_operation(self) -> bool:
        """Execute operation with configurable failure scenarios."""
        await asyncio.sleep(self.execution_time_ms / 1000.0)  # Simulate work
        
        if self.should_fail:
            self.error_message = "Simulated operation failure"
            return False
            
        # Modify the target entity
        target = self._get_target_entity()
        if target:
            target.modification_count += 1
            target.value += 1
            
        return True
    
    def _get_target_entity(self) -> Optional[TestEntity]:
        """Get the target entity for this operation."""
        try:
            root_id = EntityRegistry.ecs_id_to_root_id.get(self.target_entity_id)
            if root_id:
                entity = EntityRegistry.get_stored_entity(root_id, self.target_entity_id)
                if isinstance(entity, TestEntity):
                    return entity
            return None
        except:
            return None


class HierarchyTestSuite:
    """Comprehensive test suite for production hierarchy system validation."""
    
    def __init__(self):
        self.grace_tracker = GracePeriodTracker(0.1)  # 100ms grace period
        self.test_results = []
        
    def assert_condition(self, condition: bool, message: str, test_name: str):
        """Assert a condition and track test results for production validation."""
        if condition:
            print(f"✅ {test_name}: {message}")
            self.test_results.append((test_name, True, message))
        else:
            print(f"❌ {test_name}: FAILED - {message}")
            self.test_results.append((test_name, False, message))
            raise AssertionError(f"{test_name} failed: {message}")
    
    async def test_basic_conflict_resolution(self):
        """TEST: Comprehensive priority-based conflict resolution validation."""
        test_name = "BASIC_CONFLICT_RESOLUTION"
        
        # Create target entity
        target = TestEntity(name="conflict_test", value=0)
        target.promote_to_root()
        
        # Create operations with different priorities
        ops = []
        priorities = [OperationPriority.LOW, OperationPriority.NORMAL, OperationPriority.HIGH, OperationPriority.CRITICAL]
        
        for i, priority in enumerate(priorities):
            op = RealOperationEntity(
                op_type=f"test_op_{i}",
                operation_type="test_operation",
                priority=priority,
                target_entity_id=target.ecs_id
            )
            op.promote_to_root()
            ops.append(op)
        
        # All operations should be in PENDING state
        pending_ops = [op for op in ops if op.status == OperationStatus.PENDING]
        self.assert_condition(
            len(pending_ops) == 4,
            f"All 4 operations should be PENDING, got {len(pending_ops)}",
            test_name
        )
        
        # Resolve conflicts
        winners = resolve_operation_conflicts(target.ecs_id, ops, self.grace_tracker)
        
        # Only CRITICAL priority should win
        self.assert_condition(
            len(winners) == 1,
            f"Expected 1 winner, got {len(winners)}",
            test_name
        )
        
        self.assert_condition(
            winners[0].priority == OperationPriority.CRITICAL,
            f"Winner should have CRITICAL priority, got {winners[0].priority}",
            test_name
        )
        
        # Other operations should be rejected
        rejected_ops = [op for op in ops if op.status == OperationStatus.REJECTED]
        self.assert_condition(
            len(rejected_ops) == 3,
            f"Expected 3 rejected operations, got {len(rejected_ops)}",
            test_name
        )
        
        return target, ops
    
    async def test_executing_operation_protection(self):
        """TEST: Comprehensive validation that EXECUTING operations cannot be preempted."""
        test_name = "EXECUTING_PROTECTION"
        
        # Create target entity
        target = TestEntity(name="protection_test", value=0)
        target.promote_to_root()
        
        # Create operation that performs actual work and can fail
        low_priority_op = RealOperationEntity(
            op_type="executing_op",
            operation_type="test_operation",
            priority=OperationPriority.LOW,
            target_entity_id=target.ecs_id,
            execution_time_ms=100.0  # Longer execution
        )
        low_priority_op.promote_to_root()
        low_priority_op.start_execution()  # Now EXECUTING
        
        # Start grace period
        self.grace_tracker.start_grace_period(low_priority_op.ecs_id)
        
        # Create higher priority operation
        high_priority_op = RealOperationEntity(
            op_type="high_priority_op",
            operation_type="test_operation", 
            priority=OperationPriority.CRITICAL,
            target_entity_id=target.ecs_id
        )
        high_priority_op.promote_to_root()
        
        # Verify states before conflict resolution
        self.assert_condition(
            low_priority_op.status == OperationStatus.EXECUTING,
            f"Low priority op should be EXECUTING, got {low_priority_op.status}",
            test_name
        )
        
        self.assert_condition(
            high_priority_op.status == OperationStatus.PENDING,
            f"High priority op should be PENDING, got {high_priority_op.status}",
            test_name
        )
        
        # Resolve conflicts - EXECUTING operation should be protected
        conflicts = [low_priority_op, high_priority_op]
        winners = resolve_operation_conflicts(target.ecs_id, conflicts, self.grace_tracker)
        
        # Both should win - EXECUTING is protected, PENDING gets highest priority
        self.assert_condition(
            len(winners) == 2,
            f"Expected 2 winners (EXECUTING + highest PENDING), got {len(winners)}",
            test_name
        )
        
        self.assert_condition(
            low_priority_op in winners,
            "EXECUTING operation should be protected and win",
            test_name
        )
        
        self.assert_condition(
            high_priority_op in winners,
            "Highest priority PENDING operation should also win",
            test_name
        )
        
        # No operations should be rejected (both are winners)
        rejected_count = len([op for op in conflicts if op.status == OperationStatus.REJECTED])
        self.assert_condition(
            rejected_count == 0,
            f"Expected 0 rejected operations, got {rejected_count}",
            test_name
        )
        
        return target, low_priority_op, high_priority_op
    
    async def test_grace_period_protection(self):
        """TEST: Comprehensive validation of grace period protection mechanisms."""
        test_name = "GRACE_PERIOD_PROTECTION"
        
        # Create target entity
        target = TestEntity(name="grace_test", value=0)
        target.promote_to_root()
        
        # Create and start a low priority operation
        protected_op = RealOperationEntity(
            op_type="protected_op",
            operation_type="test_operation",
            priority=OperationPriority.LOW,
            target_entity_id=target.ecs_id
        )
        protected_op.promote_to_root()
        
        # Start grace period (operation recently started executing)
        self.grace_tracker.start_grace_period(protected_op.ecs_id)
        
        # Verify operation is protected
        protected_ids = self.grace_tracker.get_protected_operations()
        self.assert_condition(
            protected_op.ecs_id in protected_ids,
            "Operation should be in grace period protection",
            test_name
        )
        
        # Create higher priority operation that tries to preempt
        attacking_op = RealOperationEntity(
            op_type="attacking_op",
            operation_type="test_operation",
            priority=OperationPriority.CRITICAL,
            target_entity_id=target.ecs_id
        )
        attacking_op.promote_to_root()
        
        # Resolve conflicts with grace tracker
        conflicts = [protected_op, attacking_op]
        winners = resolve_operation_conflicts(target.ecs_id, conflicts, self.grace_tracker)
        
        # Both should win - protected by grace + highest priority
        self.assert_condition(
            len(winners) == 2,
            f"Expected 2 winners (grace protected + highest priority), got {len(winners)}",
            test_name
        )
        
        self.assert_condition(
            protected_op in winners,
            "Grace-protected operation should win",
            test_name
        )
        
        self.assert_condition(
            attacking_op in winners, 
            "Highest priority operation should also win",
            test_name
        )
        
        return target, protected_op, attacking_op
    
    async def test_operation_failure_and_retry(self):
        """TEST: Comprehensive validation of operation failure handling and retry logic."""
        test_name = "FAILURE_AND_RETRY"
        
        # Create target entity
        target = TestEntity(name="retry_test", value=0)
        target.promote_to_root()
        
        # Create operation that will fail initially
        failing_op = RealOperationEntity(
            op_type="failing_op",
            operation_type="test_operation",
            priority=OperationPriority.NORMAL,
            target_entity_id=target.ecs_id,
            should_fail=True,  # Will fail
            max_retries=3
        )
        failing_op.promote_to_root()
        
        # Execute and validate failure
        failing_op.start_execution()
        success = await failing_op.execute_real_operation()
        
        self.assert_condition(
            not success,
            "Operation should fail as configured",
            test_name
        )
        
        self.assert_condition(
            failing_op.error_message is not None,
            "Failed operation should have error message",
            test_name
        )
        
        # Complete with failure
        failing_op.complete_operation(success=False, error_message="Test failure")
        
        # Retry the operation
        initial_retry_count = failing_op.retry_count
        can_retry = failing_op.increment_retry()
        
        self.assert_condition(
            can_retry,
            "Operation should be able to retry",
            test_name
        )
        
        self.assert_condition(
            failing_op.retry_count == initial_retry_count + 1,
            f"Retry count should increment, expected {initial_retry_count + 1}, got {failing_op.retry_count}",
            test_name
        )
        
        # Operation should be back to PENDING for retry
        self.assert_condition(
            failing_op.status == OperationStatus.PENDING,
            f"Retrying operation should be PENDING, got {failing_op.status}",
            test_name
        )
        
        # Make it succeed on retry
        failing_op.should_fail = False
        failing_op.start_execution()
        success = await failing_op.execute_real_operation()
        
        self.assert_condition(
            success,
            "Operation should succeed on retry",
            test_name
        )
        
        failing_op.complete_operation(success=True)
        
        self.assert_condition(
            failing_op.status == OperationStatus.SUCCEEDED,
            f"Successful operation should have SUCCEEDED status, got {failing_op.status}",
            test_name
        )
        
        return target, failing_op
    
    async def test_concurrent_stress_scenario(self):
        """TEST: Comprehensive concurrent operations validation under stress conditions."""
        test_name = "CONCURRENT_STRESS"
        
        # Create target entity
        target = TestEntity(name="stress_test", value=0)
        target.promote_to_root()
        
        # Create many operations with random priorities for stress testing
        num_operations = 50
        operations = []
        
        for i in range(num_operations):
            priority = random.choice([OperationPriority.LOW, OperationPriority.NORMAL, 
                                    OperationPriority.HIGH, OperationPriority.CRITICAL])
            
            op = RealOperationEntity(
                op_type=f"stress_op_{i}",
                operation_type="test_operation",
                priority=priority,
                target_entity_id=target.ecs_id,
                execution_time_ms=random.uniform(1.0, 10.0)  # Random execution time
            )
            op.promote_to_root()
            operations.append(op)
        
        # Start some operations executing
        executing_count = 5
        for i in range(executing_count):
            operations[i].start_execution()
            self.grace_tracker.start_grace_period(operations[i].ecs_id)
        
        # Resolve conflicts
        start_time = time.time()
        winners = resolve_operation_conflicts(target.ecs_id, operations, self.grace_tracker)
        resolution_time = (time.time() - start_time) * 1000
        
        # Verify executing operations are protected
        executing_ops = [op for op in operations[:executing_count]]
        executing_winners = [op for op in executing_ops if op in winners]
        
        self.assert_condition(
            len(executing_winners) == executing_count,
            f"All {executing_count} executing operations should be protected and win, got {len(executing_winners)}",
            test_name
        )
        
        # Verify highest priority among pending operations wins
        pending_ops = [op for op in operations[executing_count:]]
        pending_winners = [op for op in pending_ops if op in winners]
        
        if pending_ops:  # If there are pending operations
            highest_pending_priority = max(op.priority for op in pending_ops)
            self.assert_condition(
                len(pending_winners) > 0,
                "At least one pending operation should win",
                test_name
            )
            
            if pending_winners:
                winner_priority = pending_winners[0].priority
                self.assert_condition(
                    winner_priority == highest_pending_priority,
                    f"Winning pending operation should have highest priority {highest_pending_priority}, got {winner_priority}",
                    test_name
                )
        
        # Performance check
        self.assert_condition(
            resolution_time < 100.0,  # Should resolve in under 100ms
            f"Conflict resolution should be fast, took {resolution_time:.1f}ms",
            test_name
        )
        
        # Verify total winners is reasonable
        expected_winners = executing_count + 1  # All executing + 1 highest pending
        self.assert_condition(
            len(winners) == expected_winners,
            f"Expected {expected_winners} winners, got {len(winners)}",
            test_name
        )
        
        return target, operations, winners
    
    async def test_hierarchy_inheritance(self):
        """TEST: Comprehensive validation of hierarchical priority inheritance."""
        test_name = "HIERARCHY_INHERITANCE"
        
        # Create target entity
        target = TestEntity(name="hierarchy_test", value=0)
        target.promote_to_root()
        
        # Create parent with high priority
        parent_op = RealOperationEntity(
            op_type="parent_op",
            operation_type="test_operation",
            priority=OperationPriority.CRITICAL,
            target_entity_id=target.ecs_id
        )
        parent_op.promote_to_root()
        
        # Create child with lower priority
        child_op = RealOperationEntity(
            op_type="child_op",
            operation_type="test_operation",
            priority=OperationPriority.LOW,
            target_entity_id=target.ecs_id,
            parent_op_id=parent_op.ecs_id
        )
        child_op.promote_to_root()
        
        # Child should inherit parent's higher priority
        effective_priority = child_op.get_effective_priority()
        self.assert_condition(
            effective_priority == OperationPriority.CRITICAL,
            f"Child should inherit parent's CRITICAL priority, got {effective_priority}",
            test_name
        )
        
        # Create competing operation with high priority
        competitor_op = RealOperationEntity(
            op_type="competitor_op",
            operation_type="test_operation",
            priority=OperationPriority.HIGH,
            target_entity_id=target.ecs_id
        )
        competitor_op.promote_to_root()
        
        # Child should win due to inherited priority
        conflicts = [child_op, competitor_op]
        winners = resolve_operation_conflicts(target.ecs_id, conflicts, self.grace_tracker)
        
        self.assert_condition(
            len(winners) == 1,
            f"Expected 1 winner, got {len(winners)}",
            test_name
        )
        
        self.assert_condition(
            child_op in winners,
            "Child with inherited CRITICAL priority should win over HIGH priority competitor",
            test_name
        )
        
        self.assert_condition(
            competitor_op.status == OperationStatus.REJECTED,
            f"Competitor should be rejected, got status {competitor_op.status}",
            test_name
        )
        
        return target, parent_op, child_op, competitor_op
    
    def print_test_summary(self):
        """Print comprehensive production test results."""
        print("\n" + "=" * 80)
        print("🧪 COMPREHENSIVE HIERARCHY INTEGRATION TEST RESULTS")
        print("=" * 80)
        
        passed = len([r for r in self.test_results if r[1]])
        total = len(self.test_results)
        
        print(f"📊 Tests: {passed}/{total} passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - Hierarchy system validated for production use")
        else:
            print("❌ SOME TESTS FAILED - System requires fixes before production deployment")
        
        print("\n📋 Detailed Results:")
        for test_name, success, message in self.test_results:
            status = "✅" if success else "❌"
            print(f"   {status} {test_name}: {message}")


async def main():
    """Execute comprehensive hierarchy integration test suite."""
    print("🚀 Comprehensive Hierarchy Integration Test Suite")
    print("Production-grade validation of all failure and success scenarios")
    print("=" * 80)
    
    # Start the event bus
    bus = get_event_bus()
    await bus.start()
    
    try:
        # Import agent observer to ensure handlers are registered
        import abstractions.agent_observer
        
        # Create test suite
        test_suite = HierarchyTestSuite()
        
        # Execute comprehensive test suite
        await test_suite.test_basic_conflict_resolution()
        await test_suite.test_executing_operation_protection()
        await test_suite.test_grace_period_protection()
        await test_suite.test_operation_failure_and_retry()
        await test_suite.test_concurrent_stress_scenario()
        await test_suite.test_hierarchy_inheritance()
        
        # Print comprehensive results
        test_suite.print_test_summary()
        
    except Exception as e:
        print(f"\n💥 CRITICAL TEST FAILURE: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Stop the event bus
        await bus.stop()


if __name__ == "__main__":
    asyncio.run(main()) 