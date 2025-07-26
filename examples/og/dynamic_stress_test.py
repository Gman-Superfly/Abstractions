"""
Conflict Resolution Algorithm Test

Production-ready test for validating conflict resolution algorithms.
Submits operations to the system and measures actual conflict resolution performance.

This test does NOT simulate fake data - it submits real operations
and measures how the conflict resolution system handles them.
"""

import asyncio
import time
import statistics
import psutil
from typing import List, Dict, Any, Set, Optional
from collections import deque, defaultdict
from datetime import datetime, timezone
from uuid import UUID

# Core imports
from abstractions.ecs.entity_hierarchy import (
    OperationEntity, StructuralOperation, NormalOperation, LowPriorityOperation,
    OperationStatus, OperationPriority,
    get_conflicting_operations, get_operation_stats, resolve_operation_conflicts
)
from abstractions.events.events import (
    get_event_bus, emit,
    OperationStartedEvent, OperationCompletedEvent, OperationRejectedEvent, OperationConflictEvent, OperationRetryEvent
)
from abstractions.ecs.entity import Entity, EntityRegistry

# Enable operation observers
import abstractions.agent_observer


class TestConfig:
    """Configuration for conflict resolution test."""
    
    def __init__(self,
                 duration_seconds: int,
                 num_targets: int,
                 operation_rate_per_second: float,
                 priority_distribution: Dict[OperationPriority, float],
                 target_completion_rate: float,
                 max_memory_mb: float,
                 grace_period_seconds: float):
        """
        Configure test parameters.
        
        Args:
            duration_seconds: How long to run the test
            num_targets: Number of target entities to create
            operation_rate_per_second: Rate of operation creation
            priority_distribution: Distribution of operation priorities
            target_completion_rate: Expected completion rate for system health
            max_memory_mb: Memory threshold for system health
            grace_period_seconds: Grace period for executing operations
        """
        self.duration_seconds = duration_seconds
        self.num_targets = num_targets
        self.operation_rate_per_second = operation_rate_per_second
        self.priority_distribution = priority_distribution
        self.target_completion_rate = target_completion_rate
        self.max_memory_mb = max_memory_mb
        self.grace_period_seconds = grace_period_seconds
        
        # Validate configuration
        assert duration_seconds > 0, "Duration must be positive"
        assert num_targets > 0, "Must have at least one target"
        assert operation_rate_per_second > 0, "Operation rate must be positive"
        assert abs(sum(priority_distribution.values()) - 1.0) < 1e-6, "Priority distribution must sum to 1.0"
        assert 0.0 <= target_completion_rate <= 1.0, "Target completion rate must be between 0 and 1"
        assert max_memory_mb > 0, "Memory threshold must be positive"
        assert grace_period_seconds >= 0, "Grace period must be non-negative"


class ConflictResolutionMetrics:
    """Metrics for conflict resolution test."""
    
    def __init__(self):
        self.start_time = time.time()
        
        # Core metrics
        self.operations_submitted = 0
        self.operations_started = 0
        self.operations_completed = 0
        self.operations_rejected = 0
        self.operations_failed = 0
        self.operations_retried = 0
        self.operations_in_progress = 0
        self.conflicts_detected = 0
        self.conflicts_resolved = 0
        
        # Priority-based tracking
        self.submitted_by_priority = defaultdict(int)
        self.started_by_priority = defaultdict(int)
        self.completed_by_priority = defaultdict(int)
        self.rejected_by_priority = defaultdict(int)
        self.failed_by_priority = defaultdict(int)
        self.retried_by_priority = defaultdict(int)
        
        # Grace period metrics
        self.grace_period_saves = 0
        self.operations_protected = 0
        
        # Performance tracking
        self.resolution_times = []
        self.conflict_sizes = []
        self.memory_samples = []
        self.cpu_samples = []
        
    def _get_priority_name(self, priority):
        """Convert priority to consistent string representation."""
        if hasattr(priority, 'name'):
            return priority.name
        else:
            priority_map = {2: 'LOW', 5: 'NORMAL', 8: 'HIGH', 10: 'CRITICAL'}
            return priority_map.get(priority, str(priority))
    
    def record_operation_submitted(self, priority: OperationPriority):
        self.operations_submitted += 1
        priority_name = self._get_priority_name(priority)
        self.submitted_by_priority[priority_name] += 1
        
    def record_operation_started(self, priority: OperationPriority):
        self.operations_started += 1
        priority_name = self._get_priority_name(priority)
        self.started_by_priority[priority_name] += 1
        
    def record_operation_completed(self, priority: OperationPriority):
        self.operations_completed += 1
        priority_name = self._get_priority_name(priority)
        self.completed_by_priority[priority_name] += 1
        
    def record_operation_rejected(self, priority: OperationPriority):
        self.operations_rejected += 1
        priority_name = self._get_priority_name(priority)
        self.rejected_by_priority[priority_name] += 1
        
    def record_operation_failed(self, priority: OperationPriority):
        self.operations_failed += 1
        priority_name = self._get_priority_name(priority)
        self.failed_by_priority[priority_name] += 1
        
    def record_operation_retried(self, priority: OperationPriority):
        self.operations_retried += 1
        priority_name = self._get_priority_name(priority)
        self.retried_by_priority[priority_name] += 1
        
    def update_operations_in_progress(self, count: int):
        self.operations_in_progress = count
        
    def record_conflict_detected(self, num_conflicts: int):
        self.conflicts_detected += 1
        self.conflict_sizes.append(num_conflicts)
        
    def record_conflict_resolved(self, resolution_time_ms: float):
        self.conflicts_resolved += 1
        self.resolution_times.append(resolution_time_ms)
        
    def record_grace_period_save(self):
        self.grace_period_saves += 1
        
    def record_operation_protected(self):
        self.operations_protected += 1
        
    def record_system_stats(self):
        try:
            process = psutil.Process()
            self.memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
            self.cpu_samples.append(process.cpu_percent())
        except:
            pass
            
    def get_current_throughput(self) -> float:
        elapsed = time.time() - self.start_time
        return self.operations_submitted / elapsed if elapsed > 0 else 0
        
    def get_current_completion_rate(self) -> float:
        return self.operations_completed / self.operations_submitted if self.operations_submitted > 0 else 0


class GracePeriodTracker:
    """Tracks grace periods for executing operations."""
    
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


class TestTarget(Entity):
    """Test target entity."""
    name: str
    capacity: int = 1


class ConflictResolutionTest:
    """
    Production-ready test for conflict resolution algorithms.
    
    Submits operations to the system and measures actual performance.
    """
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.metrics = ConflictResolutionMetrics()
        self.grace_tracker = GracePeriodTracker(config.grace_period_seconds)
        
        # Test entities
        self.target_entities: List[TestTarget] = []
        self.submitted_operations: Set[UUID] = set()
        self.stop_flag = False
        
        # Operation ID counter for unique operation names
        self.operation_counter = 0
        
    async def setup(self):
        """Initialize test environment."""
        print("🔧 Setting up conflict resolution test...")
        print(f"   ├─ Test duration: {self.config.duration_seconds}s")
        print(f"   ├─ Target entities: {self.config.num_targets}")
        print(f"   ├─ Operation rate: {self.config.operation_rate_per_second:.1f} ops/sec")
        print(f"   ├─ Grace period: {self.config.grace_period_seconds:.1f}s")
        print(f"   └─ Target completion rate: {self.config.target_completion_rate:.1%}")
        
        # Create target entities
        for i in range(self.config.num_targets):
            target = TestTarget(name=f"test_target_{i}", capacity=1)
            target.promote_to_root()
            self.target_entities.append(target)
            
        print(f"✅ Created {len(self.target_entities)} target entities")
        
    async def submit_operation(self, target: TestTarget, priority: OperationPriority) -> Optional[OperationEntity]:
        """Submit a real operation to the system."""
        self.operation_counter += 1
        
        op_class = {
            OperationPriority.CRITICAL: StructuralOperation,
            OperationPriority.LOW: LowPriorityOperation
        }.get(priority, NormalOperation)
        
        operation = op_class.create_and_register(
            op_type=f"test_op_{self.operation_counter}",
            priority=priority,
            target_entity_id=target.ecs_id,
            max_retries=1
        )
        
        self.submitted_operations.add(operation.ecs_id)
        self.metrics.record_operation_submitted(priority)
        
        return operation
        
    def _get_operation_status(self, op_id: UUID) -> Optional[OperationStatus]:
        """Get the current status of an operation."""
        for root_id in EntityRegistry.tree_registry.keys():
            tree = EntityRegistry.tree_registry.get(root_id)
            if tree and op_id in tree.nodes:
                op = tree.nodes[op_id]
                if isinstance(op, OperationEntity):
                    return op.status
        return None
        
    async def detect_and_resolve_conflicts(self, target_entity_id: UUID):
        """Detect and resolve conflicts using the system's conflict resolution."""
        start_time = time.time()
        
        try:
            conflicts = get_conflicting_operations(target_entity_id)
            
            if len(conflicts) > 1:
                self.metrics.record_conflict_detected(len(conflicts))
                
                # Emit conflict detection event
                await emit(OperationConflictEvent(
                    process_name="conflict_resolution_test",
                    op_id=conflicts[0].ecs_id,  # Primary conflicting operation
                    op_type=conflicts[0].op_type,
                    target_entity_id=target_entity_id,
                    priority=conflicts[0].priority,
                    conflict_details={
                        "total_conflicts": len(conflicts),
                        "conflict_priorities": [op.priority for op in conflicts]
                    },
                    conflicting_op_ids=[op.ecs_id for op in conflicts[1:]]
                ))
                
                # Track protection stats BEFORE resolution
                protected = self.grace_tracker.get_protected_operations()
                protected_count = len([op for op in conflicts if op.ecs_id in protected])
                
                if protected_count > 0:
                    self.metrics.record_operation_protected()
                
                # ACTUALLY CALL THE CONFLICT RESOLUTION SYSTEM
                winners = resolve_operation_conflicts(target_entity_id, conflicts)
                
                # Track what happened to the losing operations
                losers = [op for op in conflicts if op not in winners]
                for loser in losers:
                    if loser.status == OperationStatus.REJECTED:
                        self.metrics.record_operation_rejected(loser.priority)
                        # Remove rejected operations from our tracking
                        self.submitted_operations.discard(loser.ecs_id)
                        self.grace_tracker.end_grace_period(loser.ecs_id)
                        
                        # Emit rejection event
                        await emit(OperationRejectedEvent(
                            op_id=loser.ecs_id,
                            op_type=loser.op_type,
                            target_entity_id=loser.target_entity_id,
                            from_state="pending",
                            to_state="rejected",
                            rejection_reason="preempted_by_higher_priority",
                            retry_count=loser.retry_count
                        ))
                
                resolution_time = (time.time() - start_time) * 1000
                self.metrics.record_conflict_resolved(resolution_time)
                
        except Exception as e:
            print(f"⚠️  Error in conflict detection: {e}")
            
    async def operation_submission_worker(self):
        """Submit operations at the configured rate."""
        interval = 1.0 / self.config.operation_rate_per_second
        
        while not self.stop_flag:
            try:
                # Select priority based on distribution using round-robin
                priorities = list(self.config.priority_distribution.keys())
                weights = list(self.config.priority_distribution.values())
                
                # Use operation counter to select priority in a predictable pattern
                # This ensures we get the exact distribution specified
                cumulative_weights = []
                running_total = 0
                for weight in weights:
                    running_total += weight
                    cumulative_weights.append(running_total)
                
                # Normalize to operation counter
                normalized_counter = (self.operation_counter % 1000) / 1000.0
                
                selected_priority = priorities[0]
                for i, cumulative_weight in enumerate(cumulative_weights):
                    if normalized_counter <= cumulative_weight:
                        selected_priority = priorities[i]
                        break
                
                # Select target (round-robin to avoid random)
                target_index = self.operation_counter % len(self.target_entities)
                target = self.target_entities[target_index]
                
                await self.submit_operation(target, selected_priority)
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"⚠️  Error in operation submission: {e}")
                await asyncio.sleep(interval)
                
    async def conflict_monitoring_worker(self):
        """Monitor for conflicts and measure resolution."""
        while not self.stop_flag:
            try:
                for target in self.target_entities:
                    await self.detect_and_resolve_conflicts(target.ecs_id)
                    
                await asyncio.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                print(f"⚠️  Error in conflict monitoring: {e}")
                
    async def operation_lifecycle_driver(self):
        """Drive operation lifecycle - start and complete operations."""
        while not self.stop_flag:
            try:
                # Find pending operations and start some of them
                started_count = 0
                for op_id in list(self.submitted_operations):
                    if started_count >= 5:  # Limit concurrent starts per cycle
                        break
                        
                    for root_id in EntityRegistry.tree_registry.keys():
                        tree = EntityRegistry.tree_registry.get(root_id)
                        if tree and op_id in tree.nodes:
                            op = tree.nodes[op_id]
                            if isinstance(op, OperationEntity):
                                
                                # Start pending operations
                                if op.status == OperationStatus.PENDING:
                                    try:
                                        op.start_execution()
                                        self.grace_tracker.start_grace_period(op.ecs_id)
                                        self.metrics.record_operation_started(op.priority)
                                        started_count += 1
                                        
                                        await emit(OperationStartedEvent(
                                            process_name="conflict_resolution_test",
                                            op_id=op.ecs_id,
                                            op_type=op.op_type,
                                            priority=op.priority,
                                            target_entity_id=op.target_entity_id
                                        ))
                                    except Exception as e:
                                        # Operation couldn't start (maybe rejected by conflict resolution)
                                        # Try to retry if it failed
                                        if op.status == OperationStatus.FAILED:
                                            if op.increment_retry():
                                                self.metrics.record_operation_retried(op.priority)
                                            else:
                                                # Max retries exceeded - now rejected
                                                self.submitted_operations.discard(op_id)
                                                self.metrics.record_operation_rejected(op.priority)
                                
                                # Complete executing operations when they've had time to execute
                                elif op.status == OperationStatus.EXECUTING:
                                    execution_time = (datetime.now(timezone.utc) - op.started_at).total_seconds() if op.started_at else 0
                                    
                                    # Allow operations to complete after they've had some execution time
                                    min_execution_time = 0.05  # Minimum 50ms execution time
                                    if execution_time >= min_execution_time:
                                        try:
                                            # Real operation execution - no artificial failures
                                            # Let the system fail naturally through real ECS operations
                                            
                                            # Normal operation completion
                                            op.complete_operation(success=True)
                                            self.grace_tracker.end_grace_period(op.ecs_id)
                                            self.submitted_operations.discard(op_id)
                                            self.metrics.record_operation_completed(op.priority)
                                            
                                            await emit(OperationCompletedEvent(
                                                process_name="conflict_resolution_test",
                                                op_id=op.ecs_id,
                                                op_type=op.op_type,
                                                target_entity_id=op.target_entity_id,
                                                execution_duration_ms=execution_time * 1000
                                            ))
                                            
                                        except Exception as e:
                                            # REAL failure occurred during ECS operations - this should trigger retries!
                                            op.complete_operation(success=False, error_message=str(e))
                                            self.grace_tracker.end_grace_period(op.ecs_id)
                                            
                                            # Try to retry the failed operation using built-in retry system
                                            if op.increment_retry():
                                                self.metrics.record_operation_retried(op.priority)
                                                
                                                # Calculate backoff delay and emit retry event
                                                backoff_delay = op.get_backoff_delay() * 1000  # Convert to ms
                                                await emit(OperationRetryEvent(
                                                    op_id=op.ecs_id,
                                                    op_type=op.op_type,
                                                    target_entity_id=op.target_entity_id,
                                                    retry_count=op.retry_count,
                                                    max_retries=op.max_retries,
                                                    backoff_delay_ms=backoff_delay,
                                                    retry_reason=str(e)
                                                ))
                                                # Operation goes back to PENDING for retry
                                            else:
                                                # Max retries exceeded - now rejected
                                                self.submitted_operations.discard(op_id)
                                                self.metrics.record_operation_rejected(op.priority)
                                
                                # Clean up rejected operations
                                elif op.status == OperationStatus.REJECTED:
                                    self.grace_tracker.end_grace_period(op.ecs_id)
                                    self.submitted_operations.discard(op_id)
                                    self.metrics.record_operation_rejected(op.priority)
                                
                                # Handle failed operations - try to retry them
                                elif op.status == OperationStatus.FAILED:
                                    self.grace_tracker.end_grace_period(op.ecs_id)
                                    
                                    # Try to retry the failed operation
                                    if op.increment_retry():
                                        self.metrics.record_operation_retried(op.priority)
                                        # Operation goes back to PENDING for retry
                                    else:
                                        # Max retries exceeded - now rejected
                                        self.submitted_operations.discard(op_id)
                                        self.metrics.record_operation_rejected(op.priority)
                            break
                
                # Update in-progress count
                in_progress_count = len([op_id for op_id in self.submitted_operations 
                                       if self._get_operation_status(op_id) == OperationStatus.EXECUTING])
                self.metrics.update_operations_in_progress(in_progress_count)
                
                await asyncio.sleep(0.05)  # Check every 50ms
                
            except Exception as e:
                print(f"⚠️  Error in lifecycle driver: {e}")
                
    async def operation_lifecycle_observer(self):
        """Observe operation lifecycle for additional metrics."""
        while not self.stop_flag:
            try:
                # Just observe for additional metrics/validation
                for op_id in list(self.submitted_operations):
                    for root_id in EntityRegistry.tree_registry.keys():
                        tree = EntityRegistry.tree_registry.get(root_id)
                        if tree and op_id in tree.nodes:
                            op = tree.nodes[op_id]
                            if isinstance(op, OperationEntity):
                                # Additional validation could go here
                                pass
                            break
                            
                await asyncio.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                print(f"⚠️  Error in lifecycle observer: {e}")
                
    async def metrics_collector(self):
        """Collect system metrics."""
        while not self.stop_flag:
            try:
                self.metrics.record_system_stats()
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"⚠️  Error in metrics collection: {e}")
                
    async def progress_reporter(self):
        """Report progress during test."""
        last_report = time.time()
        
        while not self.stop_flag:
            await asyncio.sleep(10)
            
            current_time = time.time()
            elapsed = current_time - self.metrics.start_time
            remaining = self.config.duration_seconds - elapsed
            
            if current_time - last_report >= 10:
                completion_rate = self.metrics.get_current_completion_rate()
                throughput = self.metrics.get_current_throughput()
                
                print(f"\n📊 Test Progress ({elapsed:.1f}s elapsed, {remaining:.1f}s remaining):")
                print(f"   ├─ Operations submitted: {self.metrics.operations_submitted}")
                print(f"   ├─ Operations started: {self.metrics.operations_started}")
                print(f"   ├─ Operations completed: {self.metrics.operations_completed}")
                print(f"   ├─ Operations rejected: {self.metrics.operations_rejected}")
                print(f"   ├─ Completion rate: {completion_rate:.1%}")
                print(f"   ├─ Throughput: {throughput:.1f} ops/sec")
                print(f"   ├─ Conflicts detected: {self.metrics.conflicts_detected}")
                print(f"   └─ Grace period saves: {self.metrics.grace_period_saves}")
                
                last_report = current_time
                
    async def run_test(self):
        """Run the complete conflict resolution test."""
        print(f"\n🚀 Starting Conflict Resolution Test...")
        
        # Start all workers
        tasks = [
            asyncio.create_task(self.operation_submission_worker()),
            asyncio.create_task(self.conflict_monitoring_worker()),
            asyncio.create_task(self.operation_lifecycle_driver()),
            asyncio.create_task(self.operation_lifecycle_observer()),
            asyncio.create_task(self.metrics_collector()),
            asyncio.create_task(self.progress_reporter())
        ]
        
        try:
            await asyncio.sleep(self.config.duration_seconds)
        finally:
            self.stop_flag = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def analyze_results(self):
        """Analyze test results."""
        print("\n🧹 Analyzing results...")
        
        elapsed = time.time() - self.metrics.start_time
        throughput = self.metrics.operations_submitted / elapsed
        completion_rate = self.metrics.get_current_completion_rate()
        rejection_rate = self.metrics.operations_rejected / self.metrics.operations_submitted if self.metrics.operations_submitted > 0 else 0
        
        print("\n" + "=" * 80)
        print("🚀 CONFLICT RESOLUTION TEST RESULTS")
        print("=" * 80)
        
        print(f"\n📋 Test Configuration:")
        print(f"   ├─ Duration: {self.config.duration_seconds}s")
        print(f"   ├─ Targets: {self.config.num_targets}")
        print(f"   ├─ Operation rate: {self.config.operation_rate_per_second:.1f} ops/sec")
        print(f"   ├─ Grace period: {self.config.grace_period_seconds:.1f}s")
        print(f"   └─ Priority distribution: {dict(self.config.priority_distribution)}")
        
        print(f"\n⏱️  Performance Results:")
        print(f"   ├─ Actual duration: {elapsed:.1f}s")
        print(f"   ├─ Actual throughput: {throughput:.1f} ops/sec")
        print(f"   ├─ Operations submitted: {self.metrics.operations_submitted}")
        print(f"   ├─ Operations started: {self.metrics.operations_started}")
        print(f"   ├─ Operations completed: {self.metrics.operations_completed}")
        print(f"   ├─ Operations rejected: {self.metrics.operations_rejected}")
        print(f"   ├─ Operations failed: {self.metrics.operations_failed}")
        print(f"   ├─ Operations retried: {self.metrics.operations_retried}")
        print(f"   ├─ Operations in progress: {self.metrics.operations_in_progress}")
        print(f"   ├─ Completion rate: {completion_rate:.1%}")
        print(f"   └─ Rejection rate: {rejection_rate:.1%}")
        
        # Operation accounting verification
        total_accounted = (self.metrics.operations_completed + 
                          self.metrics.operations_rejected + 
                          self.metrics.operations_failed + 
                          self.metrics.operations_in_progress)
        unaccounted = self.metrics.operations_submitted - total_accounted
        
        print(f"\n📊 Operation Accounting:")
        print(f"   ├─ Total submitted: {self.metrics.operations_submitted}")
        print(f"   ├─ Total accounted: {total_accounted}")
        print(f"   ├─ Unaccounted: {unaccounted}")
        if unaccounted > 0:
            print(f"   └─ ⚠️  {unaccounted} operations may be stuck in PENDING state")
        else:
            print(f"   └─ ✅ All operations accounted for")
        
        print(f"\n🛡️  Grace Period Results:")
        print(f"   ├─ Grace period saves: {self.metrics.grace_period_saves}")
        print(f"   └─ Operations protected: {self.metrics.operations_protected}")
        
        print(f"\n⚔️  Conflict Resolution Results:")
        print(f"   ├─ Conflicts detected: {self.metrics.conflicts_detected}")
        print(f"   ├─ Conflicts resolved: {self.metrics.conflicts_resolved}")
        if self.metrics.resolution_times:
            avg_resolution_time = statistics.mean(self.metrics.resolution_times)
            print(f"   └─ Avg resolution time: {avg_resolution_time:.1f}ms")
        
        # System performance
        if self.metrics.memory_samples:
            avg_memory = statistics.mean(self.metrics.memory_samples)
            max_memory = max(self.metrics.memory_samples)
            print(f"\n💾 System Resources:")
            print(f"   ├─ Avg memory: {avg_memory:.1f} MB")
            print(f"   ├─ Max memory: {max_memory:.1f} MB")
            print(f"   └─ Memory status: {'✅ Good' if max_memory < self.config.max_memory_mb else '⚠️ High'}")
        
        # Assessment
        print(f"\n🎯 SYSTEM ASSESSMENT:")
        if completion_rate >= self.config.target_completion_rate:
            print(f"   ✅ PASSED: Completion rate {completion_rate:.1%} meets target {self.config.target_completion_rate:.1%}")
        else:
            print(f"   ❌ FAILED: Completion rate {completion_rate:.1%} below target {self.config.target_completion_rate:.1%}")
            
        if self.metrics.grace_period_saves > 0:
            print("   ✅ GRACE PERIODS: Successfully protected executing operations")
        
        if self.metrics.conflicts_resolved > 0:
            print("   ✅ CONFLICT RESOLUTION: System handled conflicts")
        
        return {
            'completion_rate': completion_rate,
            'rejection_rate': rejection_rate,
            'throughput': throughput,
            'conflicts_resolved': self.metrics.conflicts_resolved,
            'grace_period_saves': self.metrics.grace_period_saves,
            'passed': completion_rate >= self.config.target_completion_rate
        }


async def run_conflict_resolution_test(config: TestConfig) -> Dict[str, Any]:
    """Run a conflict resolution test with the given configuration."""
    bus = get_event_bus()
    await bus.start()
    
    try:
        test = ConflictResolutionTest(config)
        
        await test.setup()
        await test.run_test()
        results = await test.analyze_results()
        
        print("\n🎉 Conflict resolution test completed!")
        return results
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'passed': False, 'error': str(e)}
    finally:
        await bus.stop()


async def main():
    """Run the conflict resolution test with example configuration."""
    
    # Example test configuration - all parameters explicit
    config = TestConfig(
        duration_seconds=120,  # 2 minutes
        num_targets=5,
        operation_rate_per_second=10000.0,  # Increased mega stress test 10000 ops/sec
        priority_distribution={
            OperationPriority.LOW: 0.5,
            OperationPriority.NORMAL: 0.2,
            OperationPriority.HIGH: 0.2,
            OperationPriority.CRITICAL: 0.1
        },
        target_completion_rate=0.02,  # 2% minimum completion rate expected
        max_memory_mb=200,
        grace_period_seconds=1.5
    )
    
    print("🚀 CONFLICT RESOLUTION ALGORITHM TEST")
    print("=" * 60)
    print("Production test - submits real operations and measures results")
    print("=" * 60)
    
    results = await run_conflict_resolution_test(config)
    
    if results.get('passed', False):
        print(f"\n✅ TEST PASSED - System meets performance requirements")
    else:
        print(f"\n❌ TEST FAILED - System does not meet requirements")


if __name__ == "__main__":
    asyncio.run(main()) 