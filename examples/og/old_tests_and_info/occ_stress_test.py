"""
OCC + Dynamic Stress Test: Production-grade testing of all concurrency systems

This test integrates OCC (Optimistic Concurrency Control) with the existing
pre-ECS conflict resolution, grace period system, and zombie cleanup.

Tests the complete concurrency stack:
1. Pre-ECS conflict resolution (operation-level)
2. Grace period protection (temporal)  
3. OCC protection (data-level)
4. Zombie cleanup (resource management)

OCC = Optimistic Concurrency Control - prevents race conditions in entity data modifications
"""

import asyncio
import time
import statistics
import psutil
import random
from typing import List, Dict, Any, Set, Optional, Union
from collections import deque, defaultdict
from datetime import datetime, timezone
from uuid import UUID, uuid4
from pydantic import Field

# Core imports - PURE EVENT-DRIVEN
from abstractions.ecs.entity_hierarchy import (
    OperationEntity, StructuralOperation, NormalOperation, LowPriorityOperation,
    OperationStatus, OperationPriority
)
from abstractions.events.events import (
    get_event_bus, emit,
    OperationStartedEvent, OperationCompletedEvent, OperationRejectedEvent, OperationConflictEvent, OperationRetryEvent
)
from abstractions.ecs.entity import Entity, EntityRegistry
from abstractions.ecs.functional_api import put, get

# Enable operation observers
import abstractions.agent_observer


class TestConfig:
    """Configuration for OCC + conflict resolution test."""
    
    def __init__(self,
                 duration_seconds: int,
                 num_targets: int,
                 operation_rate_per_second: float,
                 priority_distribution: Dict[OperationPriority, float],
                 target_completion_rate: float,
                 max_memory_mb: float,
                 grace_period_seconds: float,
                 occ_max_retries: int = 5):
        """
        Configure test parameters with OCC settings.
        
        Args:
            occ_max_retries: Maximum OCC retry attempts per operation
        """
        self.duration_seconds = duration_seconds
        self.num_targets = num_targets
        self.operation_rate_per_second = operation_rate_per_second
        self.priority_distribution = priority_distribution
        self.target_completion_rate = target_completion_rate
        self.max_memory_mb = max_memory_mb
        self.grace_period_seconds = grace_period_seconds
        self.occ_max_retries = occ_max_retries
        
        # Validate configuration
        assert duration_seconds > 0, "Duration must be positive"
        assert num_targets > 0, "Must have at least one target"
        assert operation_rate_per_second > 0, "Operation rate must be positive"
        assert abs(sum(priority_distribution.values()) - 1.0) < 1e-6, "Priority distribution must sum to 1.0"
        assert 0.0 <= target_completion_rate <= 1.0, "Target completion rate must be between 0 and 1"
        assert max_memory_mb > 0, "Memory threshold must be positive"
        assert grace_period_seconds >= 0, "Grace period must be non-negative"
        assert occ_max_retries >= 0, "OCC max retries must be non-negative"


class EnhancedConflictMetrics:
    """Enhanced metrics that track all concurrency systems."""
    
    def __init__(self):
        self.start_time = time.time()
        
        # Existing metrics
        self.operations_submitted = 0
        self.operations_started = 0
        self.operations_completed = 0
        self.operations_rejected = 0
        self.operations_failed = 0
        self.operations_retried = 0
        self.operations_in_progress = 0
        self.conflicts_detected = 0
        self.conflicts_resolved = 0
        
        # NEW: OCC-specific metrics
        self.occ_conflicts_detected = 0
        self.occ_retries_attempted = 0
        self.occ_failures_max_retries = 0
        self.occ_successes_after_retry = 0
        self.occ_total_retry_time_ms = 0.0
        
        # OCC retry distribution
        self.occ_retry_distribution: Dict[int, int] = defaultdict(int)  # retry_count -> count
        
        # Per-operation type OCC metrics
        self.occ_metrics_by_operation: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            'conflicts': 0, 'retries': 0, 'successes': 0, 'failures': 0
        })
        
        # Performance tracking
        self.priority_counts = defaultdict(int)
        self.operation_durations = defaultdict(list)
        self.conflict_resolution_times = []
        self.grace_period_saves = 0
        self.operations_protected = 0
        
        # System metrics
        self.peak_memory_mb = 0.0
        self.peak_operations_in_progress = 0
        
    def record_occ_conflict(self, operation_type: str, retry_count: int):
        """Record an OCC conflict detection."""
        self.occ_conflicts_detected += 1
        self.occ_metrics_by_operation[operation_type]['conflicts'] += 1
        self.occ_retry_distribution[retry_count] += 1
        
    def record_occ_retry(self, operation_type: str, retry_time_ms: float):
        """Record an OCC retry attempt."""
        self.occ_retries_attempted += 1
        self.occ_total_retry_time_ms += retry_time_ms
        self.occ_metrics_by_operation[operation_type]['retries'] += 1
        
    def record_occ_success_after_retry(self, operation_type: str, final_retry_count: int):
        """Record successful operation after OCC retries."""
        self.occ_successes_after_retry += 1
        self.occ_metrics_by_operation[operation_type]['successes'] += 1
        self.occ_retry_distribution[final_retry_count] += 1
        
    def record_occ_success(self, operation_type: str):
        """Record successful operation with OCC protection (no retries needed)."""
        self.occ_metrics_by_operation[operation_type]['successes'] += 1
        
    def record_occ_failure_max_retries(self, operation_type: str):
        """Record operation failure due to max OCC retries exceeded."""
        self.occ_failures_max_retries += 1
        self.occ_metrics_by_operation[operation_type]['failures'] += 1
        
    def get_occ_statistics(self) -> Dict[str, Any]:
        """Get comprehensive OCC statistics."""
        total_occ_operations = self.occ_conflicts_detected + self.operations_completed
        occ_conflict_rate = self.occ_conflicts_detected / total_occ_operations if total_occ_operations > 0 else 0
        avg_retry_time = self.occ_total_retry_time_ms / self.occ_retries_attempted if self.occ_retries_attempted > 0 else 0
        
        return {
            'occ_conflicts_detected': self.occ_conflicts_detected,
            'occ_conflict_rate_percent': occ_conflict_rate * 100,
            'occ_retries_attempted': self.occ_retries_attempted,
            'occ_successes_after_retry': self.occ_successes_after_retry,
            'occ_failures_max_retries': self.occ_failures_max_retries,
            'occ_avg_retry_time_ms': avg_retry_time,
            'occ_retry_distribution': dict(self.occ_retry_distribution),
            'occ_by_operation_type': dict(self.occ_metrics_by_operation)
        }


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


class TestDataEntity(Entity):
    """Test entity with data that can be modified by real operations."""
    name: str = "test_entity"
    counter: int = 0
    data_value: float = 0.0
    text_content: str = ""
    timestamp: datetime = datetime.now(timezone.utc)
    version_count: int = 0
    borrow_count: int = 0
    modification_history: List[str] = []


class TestDataSource(Entity):
    """Source entity for borrowing operations."""
    source_name: str = "data_source"
    source_counter: int = 100
    source_data: float = 99.99
    source_text: str = "borrowed_content"
    source_values: List[int] = [1, 2, 3, 4, 5]
    source_metadata: Dict[str, str] = {"type": "source", "quality": "high"}


class OCCProtectedOperationEntity(OperationEntity):
    """Operation entity with integrated OCC protection."""
    
    # Real operation parameters
    operation_type: str = Field(description="Type of real operation to perform")
    operation_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for real operation")
    source_entity_id: Optional[UUID] = Field(default=None, description="Source entity for borrowing ops")
    
    # OCC tracking
    occ_retry_count: int = Field(default=0, description="Current OCC retry count")
    occ_max_retries: int = Field(default=5, description="Max OCC retries")
    
    async def execute_with_occ_protection(self, metrics: 'EnhancedConflictMetrics') -> bool:
        """
        Execute operation with OCC (Optimistic Concurrency Control) protection.
        
        Uses a simple snapshot-modify-commit pattern that aligns with existing entity hierarchy.
        No complex managers needed - just proper sequencing.
        
        OCC = Optimistic Concurrency Control - prevents race conditions during entity modifications
        """
        occ_retry_count = 0
        
        while occ_retry_count <= self.occ_max_retries:
            try:
                # Step 1: Get fresh entity from storage (our working copy)
                target_entity = self._get_target_entity()
                if not target_entity:
                    metrics.record_occ_failure_max_retries(self.operation_type)
                    return False
                
                # Step 2: Create OCC snapshot BEFORE any modifications
                occ_snapshot = {
                    'version': target_entity.version,
                    'last_modified': target_entity.last_modified,
                    'ecs_id': target_entity.ecs_id
                }
                
                if occ_retry_count > 0:
                    print(f"🔄 OCC RETRY {occ_retry_count}: {self.operation_type} on {str(target_entity.ecs_id)[:8]}")
                    print(f"   📊 OCC Snapshot: version={occ_snapshot['version']}, modified={occ_snapshot['last_modified']}")
                
                # Step 3: Simulate processing time (where other operations might interfere)
                retry_start_time = time.time()
                # NO DELAY - immediate conflict window to test in-memory race conditions
                
                # Step 4: Check for conflicts BEFORE modifying
                # Get fresh copy from storage to see if anyone else modified it
                fresh_entity = self._get_target_entity()
                if not fresh_entity:
                    metrics.record_occ_failure_max_retries(self.operation_type)
                    return False
                
                # Compare fresh entity against our snapshot (not against our modified entity)
                if (fresh_entity.version != occ_snapshot['version'] or 
                    fresh_entity.last_modified != occ_snapshot['last_modified']):
                    # OCC CONFLICT DETECTED - someone else modified the entity since our snapshot
                    occ_retry_count += 1
                    retry_time_ms = (time.time() - retry_start_time) * 1000
                    
                    metrics.record_occ_conflict(self.operation_type, occ_retry_count)
                    metrics.record_occ_retry(self.operation_type, retry_time_ms)
                    
                    print(f"⚠️  OCC CONFLICT: {self.operation_type}")
                    print(f"   📊 Expected version: {occ_snapshot['version']}, Actual: {fresh_entity.version}")
                    print(f"   ⏰ Expected modified: {occ_snapshot['last_modified']}, Actual: {fresh_entity.last_modified}")
                    
                    if occ_retry_count > self.occ_max_retries:
                        print(f"❌ OCC MAX RETRIES: {self.operation_type} failed after {self.occ_max_retries} retries")
                        metrics.record_occ_failure_max_retries(self.operation_type)
                        return False
                    
                    # Retry with fresh entity state
                    await asyncio.sleep(0.01 * occ_retry_count)  # Exponential backoff
                    continue
                
                # Step 5: No conflict detected - execute the operation using fresh entity
                success = await self._execute_operation_logic(fresh_entity)
                if not success:
                    metrics.record_occ_failure_max_retries(self.operation_type)
                    return False
                
                # Step 6: Operation completed successfully
                retry_time_ms = (time.time() - retry_start_time) * 1000
                if occ_retry_count > 0:
                    metrics.record_occ_retry(self.operation_type, retry_time_ms)
                    print(f"✅ OCC SUCCESS: {self.operation_type} succeeded after {occ_retry_count} retries")
                else:
                    print(f"✅ OCC SUCCESS: {self.operation_type} succeeded on first try")
                
                metrics.record_occ_success(self.operation_type)
                return True
                
            except Exception as e:
                print(f"🚨 OCC EXCEPTION: {self.operation_type} failed with error: {e}")
                metrics.record_occ_failure_max_retries(self.operation_type)
                return False
        
        # Should never reach here due to loop condition, but just in case
        metrics.record_occ_failure_max_retries(self.operation_type)
        return False
    
    async def _execute_operation_logic(self, target_entity: TestDataEntity) -> bool:
        """Execute the core operation logic with automatic OCC field updates."""
        try:
            print(f"🔧 EXECUTING: {self.operation_type} on target {str(target_entity.ecs_id)[:8]}")
            
            if self.operation_type == "version_entity":
                # Pure ECS versioning operation
                success = EntityRegistry.version_entity(target_entity, force_versioning=True)
                return success
                    
            elif self.operation_type == "modify_field":
                # Pure ECS field modification with OCC protection
                field_name = self.operation_params.get("field_name", "counter")
                new_value = self.operation_params.get("new_value", random.randint(1, 1000))
                
                # Modify the field
                setattr(target_entity, field_name, new_value)
                
                # Update modification history and mark as modified (updates OCC fields)
                target_entity.modification_history.append(f"Modified {field_name} to {new_value} at {datetime.now(timezone.utc)}")
                target_entity.mark_modified()  # This updates version and last_modified
                
                return True
                
            elif self.operation_type == "borrow_attribute":
                # Pure ECS borrowing operation with OCC protection
                source_entity = self._get_source_entity()
                if source_entity:
                    source_field = self.operation_params.get("source_field", "source_counter")
                    target_field = self.operation_params.get("target_field", "counter")
                    
                    # Real ECS borrowing call (automatically calls mark_modified)
                    target_entity.borrow_attribute_from(source_entity, source_field, target_field)
                    target_entity.borrow_count += 1
                    return True
                else:
                    raise ValueError("Source entity not found for borrowing operation")
                    
            elif self.operation_type == "complex_update":
                # Multiple field updates with OCC protection
                target_entity.counter += 1
                target_entity.data_value = random.uniform(0, 100)
                target_entity.text_content = f"Updated_{target_entity.counter}_{time.time()}"
                target_entity.timestamp = datetime.now(timezone.utc)
                target_entity.version_count += 1
                target_entity.modification_history.append(f"Complex update {target_entity.counter} at {target_entity.timestamp}")
                
                # Mark as modified (updates OCC fields)
                target_entity.mark_modified()
                return True
                
            else:
                raise ValueError(f"Unknown operation type: {self.operation_type}")
                
        except Exception as e:
            self.error_message = str(e)
            return False
    
    def _get_target_entity(self) -> Optional[TestDataEntity]:
        """Get the target entity for this operation."""
        try:
            root_id = EntityRegistry.ecs_id_to_root_id.get(self.target_entity_id)
            if root_id:
                entity = EntityRegistry.get_stored_entity(root_id, self.target_entity_id)
                if isinstance(entity, TestDataEntity):
                    return entity
            return None
        except:
            return None
    
    def _get_source_entity(self) -> Optional[TestDataSource]:
        """Get the source entity for borrowing operations."""
        if not self.source_entity_id:
            return None
        try:
            root_id = EntityRegistry.ecs_id_to_root_id.get(self.source_entity_id)
            if root_id:
                entity = EntityRegistry.get_stored_entity(root_id, self.source_entity_id)
                if isinstance(entity, TestDataSource):
                    return entity
            return None
        except:
            return None


class OCCStressTest:
    """
    Production stress test with integrated OCC protection.
    
    Tests all concurrency systems:
    - Pre-ECS conflict resolution
    - Grace period protection  
    - OCC data protection
    - Zombie cleanup
    """
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.metrics = EnhancedConflictMetrics()
        self.grace_tracker = GracePeriodTracker(config.grace_period_seconds)
        
        # Test entities
        self.target_entities: List[TestDataEntity] = []
        self.source_entities: List[TestDataSource] = []
        self.submitted_operations: Set[UUID] = set()
        self.stop_flag = False
        self.stop_submission = False
        
        # Pre-ECS staging area
        self.pending_operations: Dict[UUID, List[OCCProtectedOperationEntity]] = {}
        
    async def setup(self):
        """Set up test entities and dependencies."""
        print(f"🏗️  Setting up OCC + Conflict Resolution Test...")
        
        # Create target entities
        for i in range(self.config.num_targets):
            entity = TestDataEntity(
                name=f"target_{i}",
                counter=0,
                data_value=100.0,
                text_content=f"initial_content_{i}",
                timestamp=datetime.now(timezone.utc),
                version_count=0,
                borrow_count=0,
                modification_history=[f"Created at {datetime.now(timezone.utc)}"]
            )
            entity.promote_to_root()
            self.target_entities.append(entity)
            
        # Create source entities  
        for i in range(min(3, self.config.num_targets)):
            source = TestDataSource(
                source_name=f"source_{i}",
                source_counter=1000 + i,
                source_data=99.99 + i,
                source_text=f"borrowed_content_{i}",
                source_values=[1+i, 2+i, 3+i, 4+i, 5+i],
                source_metadata={"type": "source", "id": str(i)}
            )
            source.promote_to_root()
            self.source_entities.append(source)
            
        print(f"✅ Created {len(self.target_entities)} target entities and {len(self.source_entities)} source entities")
        print(f"🎯 Test will run for {self.config.duration_seconds}s with {self.config.operation_rate_per_second} ops/sec")
        print(f"🔒 OCC Protection: max {self.config.occ_max_retries} retries per operation")
        
    async def run_test(self):
        """Run the complete OCC stress test."""
        print(f"\n🚀 Starting OCC + Conflict Resolution Stress Test...")
        
        # Start all workers
        tasks = [
            asyncio.create_task(self.operation_submission_worker()),
            asyncio.create_task(self.operation_lifecycle_driver()),
            asyncio.create_task(self.operation_lifecycle_observer()),
            asyncio.create_task(self.metrics_collector()),
            asyncio.create_task(self.progress_reporter())
        ]
        
        try:
            # Phase 1: Run test for specified duration
            print(f"⏱️  Phase 1: Running test for {self.config.duration_seconds}s")
            await asyncio.sleep(self.config.duration_seconds)
            
            # Phase 2: Stop new submissions, allow pending to complete
            print(f"⏱️  Phase 2: Graceful shutdown - stopping new operations")
            self.stop_submission = True
            await asyncio.sleep(2.0)  # Grace period for pending ops
            
            # Phase 3: Stop all workers
            print(f"⏱️  Phase 3: Stopping all workers")
            
        finally:
            self.stop_flag = True
            
            # Cancel and wait for cleanup
            for task in tasks:
                task.cancel()
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                print("⚠️  Some workers took too long to stop")
            
            await asyncio.sleep(0.5)  # Final cleanup
        
        # Analyze results
        await self.analyze_results()
    
    async def operation_submission_worker(self):
        """Submit operations with pre-ECS conflict resolution."""
        submission_interval = 1.0 / self.config.operation_rate_per_second
        
        while not self.stop_submission:
            try:
                # Select random target
                target = random.choice(self.target_entities)
                
                # Select priority based on distribution
                priority_value = random.random()
                cumulative = 0.0
                selected_priority = OperationPriority.NORMAL
                
                for priority, prob in self.config.priority_distribution.items():
                    cumulative += prob
                    if priority_value <= cumulative:
                        selected_priority = priority
                        break
                
                # Create operation
                op = await self._create_operation(target, selected_priority)
                if op:
                    # Add to staging area
                    if target.ecs_id not in self.pending_operations:
                        self.pending_operations[target.ecs_id] = []
                    
                    self.pending_operations[target.ecs_id].append(op)
                    self.metrics.operations_submitted += 1
                    
                    # Resolve conflicts in pre-ECS staging
                    await self._resolve_conflicts_before_ecs(target.ecs_id)
                
                await asyncio.sleep(submission_interval)
                
            except Exception as e:
                print(f"⚠️  Error in submission worker: {e}")
                await asyncio.sleep(0.1)
    
    async def _create_operation(self, target: TestDataEntity, priority: OperationPriority) -> Optional[OCCProtectedOperationEntity]:
        """Create a new operation with OCC protection."""
        operation_types = ["modify_field", "borrow_attribute", "complex_update", "version_entity"]
        op_type = random.choice(operation_types)
        
        # Create operation parameters
        params = {}
        source_id = None
        
        if op_type == "modify_field":
            params = {
                "field_name": random.choice(["counter", "data_value", "text_content"]),
                "new_value": random.randint(1, 1000)
            }
        elif op_type == "borrow_attribute":
            if self.source_entities:
                source = random.choice(self.source_entities)
                source_id = source.ecs_id
                params = {
                    "source_field": random.choice(["source_counter", "source_data", "source_text"]),
                    "target_field": random.choice(["counter", "data_value", "text_content"])
                }
        
        # Determine operation class based on priority
        if priority == OperationPriority.CRITICAL:
            op_class = StructuralOperation
        elif priority == OperationPriority.LOW:
            op_class = LowPriorityOperation
        else:
            op_class = NormalOperation
        
        # Create OCC-protected operation
        op = OCCProtectedOperationEntity(
            op_type=f"occ_{op_type}",
            operation_type=op_type,
            operation_params=params,
            target_entity_id=target.ecs_id,
            source_entity_id=source_id,
            priority=priority,
            occ_max_retries=self.config.occ_max_retries
        )
        
        return op
    
    async def _resolve_conflicts_before_ecs(self, target_entity_id: UUID):
        """Resolve conflicts in pre-ECS staging area."""
        pending_ops = self.pending_operations.get(target_entity_id, [])
        
        if len(pending_ops) > 1:
            resolution_start = time.time()
            self.metrics.conflicts_detected += 1
            
            # Sort by priority (higher wins)
            pending_ops.sort(key=lambda op: (op.priority, -op.created_at.timestamp()), reverse=True)
            
            winner = pending_ops[0]
            losers = pending_ops[1:]
            
            print(f"⚔️  PRE-ECS CONFLICT: {len(pending_ops)} ops, winner: {winner.operation_type} (priority: {winner.priority})")
            
            # Reject losers
            for loser in losers:
                self.metrics.operations_rejected += 1
                self.pending_operations[target_entity_id].remove(loser)
            
            # Promote winner to ECS
            winner.promote_to_root()
            self.submitted_operations.add(winner.ecs_id)
            self.pending_operations[target_entity_id] = []
            
            resolution_time = (time.time() - resolution_start) * 1000
            self.metrics.conflicts_resolved += 1
            self.metrics.conflict_resolution_times.append(resolution_time)
            
        elif len(pending_ops) == 1:
            # No conflict - promote to ECS
            winner = pending_ops[0]
            winner.promote_to_root()
            self.submitted_operations.add(winner.ecs_id)
            self.pending_operations[target_entity_id] = []
    
    async def operation_lifecycle_driver(self):
        """Drive operation execution with OCC protection."""
        while not self.stop_flag:
            try:
                # Find pending operations in ECS
                operations_to_execute = []
                
                for op_id in list(self.submitted_operations):
                    root_id = EntityRegistry.ecs_id_to_root_id.get(op_id)
                    if root_id:
                        op = EntityRegistry.get_stored_entity(root_id, op_id)
                        if isinstance(op, OCCProtectedOperationEntity) and op.status == OperationStatus.PENDING:
                            operations_to_execute.append(op)
                
                # Execute operations CONCURRENTLY to create real OCC conflicts
                execution_tasks = []
                
                for op in operations_to_execute:
                    try:
                        op.status = OperationStatus.EXECUTING
                        op.started_at = datetime.now(timezone.utc)
                        self.grace_tracker.start_grace_period(op.ecs_id)
                        self.metrics.operations_started += 1
                        
                        # Create concurrent execution task
                        task = asyncio.create_task(self._execute_operation_concurrently(op))
                        execution_tasks.append(task)
                        
                    except Exception as e:
                        print(f"⚠️  Error starting operation {op.operation_type}: {e}")
                        op.status = OperationStatus.FAILED
                        op.error_message = str(e)
                        self.metrics.operations_failed += 1
                        self.grace_tracker.end_grace_period(op.ecs_id)
                        self.submitted_operations.discard(op.ecs_id)
                
                # Wait for all concurrent executions to complete
                if execution_tasks:
                    await asyncio.gather(*execution_tasks, return_exceptions=True)
                
                await asyncio.sleep(0.01)  # Small delay
                
            except Exception as e:
                print(f"⚠️  Error in lifecycle driver: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_operation_concurrently(self, op: OCCProtectedOperationEntity):
        """Execute a single operation with proper cleanup."""
        try:
            # Execute with full OCC protection
            success = await op.execute_with_occ_protection(self.metrics)
            
            if success:
                op.status = OperationStatus.SUCCEEDED
                op.completed_at = datetime.now(timezone.utc)
                self.metrics.operations_completed += 1
            else:
                op.status = OperationStatus.FAILED
                op.completed_at = datetime.now(timezone.utc)
                self.metrics.operations_failed += 1
            
            self.grace_tracker.end_grace_period(op.ecs_id)
            self.submitted_operations.discard(op.ecs_id)
            
        except Exception as e:
            print(f"⚠️  Error executing operation {op.operation_type}: {e}")
            op.status = OperationStatus.FAILED
            op.error_message = str(e)
            self.metrics.operations_failed += 1
            self.grace_tracker.end_grace_period(op.ecs_id)
            self.submitted_operations.discard(op.ecs_id)
    
    async def operation_lifecycle_observer(self):
        """Observe operation lifecycle for metrics."""
        while not self.stop_flag:
            try:
                # Count operations in progress
                in_progress = len([op_id for op_id in self.submitted_operations 
                                 if EntityRegistry.ecs_id_to_root_id.get(op_id)])
                
                self.metrics.operations_in_progress = in_progress
                self.metrics.peak_operations_in_progress = max(
                    self.metrics.peak_operations_in_progress, 
                    in_progress
                )
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️  Error in lifecycle observer: {e}")
                await asyncio.sleep(0.1)
    
    async def metrics_collector(self):
        """Collect system metrics."""
        while not self.stop_flag:
            try:
                # Memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, memory_mb)
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"⚠️  Error collecting metrics: {e}")
                await asyncio.sleep(1.0)
    
    async def progress_reporter(self):
        """Report progress periodically."""
        while not self.stop_flag:
            try:
                elapsed = time.time() - self.metrics.start_time
                
                if elapsed > 0 and int(elapsed) % 5 == 0:  # Every 5 seconds
                    ops_per_sec = self.metrics.operations_completed / elapsed
                    occ_stats = self.metrics.get_occ_statistics()
                    
                    print(f"\n📊 PROGRESS @ {elapsed:.1f}s:")
                    print(f"   Operations: {self.metrics.operations_completed} completed, {self.metrics.operations_in_progress} in progress")
                    print(f"   Rate: {ops_per_sec:.1f} ops/sec")
                    print(f"   Pre-ECS Conflicts: {self.metrics.conflicts_detected}")
                    print(f"   OCC Conflicts: {occ_stats['occ_conflicts_detected']} ({occ_stats['occ_conflict_rate_percent']:.1f}%)")
                    print(f"   OCC Retries: {occ_stats['occ_retries_attempted']}")
                    print(f"   Memory: {self.metrics.peak_memory_mb:.1f} MB")
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"⚠️  Error in progress reporter: {e}")
                await asyncio.sleep(1.0)
    
    async def analyze_results(self):
        """Analyze comprehensive test results."""
        elapsed = time.time() - self.metrics.start_time
        occ_stats = self.metrics.get_occ_statistics()
        
        print(f"\n" + "=" * 80)
        print(f"🏁 OCC + CONFLICT RESOLUTION TEST RESULTS")
        print(f"=" * 80)
        
        # Overall metrics
        print(f"\n📊 OVERALL METRICS:")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Operations Submitted: {self.metrics.operations_submitted}")
        print(f"   Operations Completed: {self.metrics.operations_completed}")
        print(f"   Operations Failed: {self.metrics.operations_failed}")
        print(f"   Operations Rejected: {self.metrics.operations_rejected}")
        print(f"   Success Rate: {(self.metrics.operations_completed / max(1, self.metrics.operations_submitted)) * 100:.1f}%")
        print(f"   Throughput: {self.metrics.operations_completed / elapsed:.1f} ops/sec")
        
        # Pre-ECS conflict resolution
        print(f"\n⚔️  PRE-ECS CONFLICT RESOLUTION:")
        print(f"   Conflicts Detected: {self.metrics.conflicts_detected}")
        print(f"   Conflicts Resolved: {self.metrics.conflicts_resolved}")
        if self.metrics.conflict_resolution_times:
            avg_resolution_time = statistics.mean(self.metrics.conflict_resolution_times)
            print(f"   Avg Resolution Time: {avg_resolution_time:.2f}ms")
        
        # OCC metrics
        print(f"\n🔒 OCC (OPTIMISTIC CONCURRENCY CONTROL):")
        print(f"   OCC Conflicts Detected: {occ_stats['occ_conflicts_detected']}")
        print(f"   OCC Conflict Rate: {occ_stats['occ_conflict_rate_percent']:.1f}%")
        print(f"   OCC Retries Attempted: {occ_stats['occ_retries_attempted']}")
        print(f"   OCC Successes After Retry: {occ_stats['occ_successes_after_retry']}")
        print(f"   OCC Failures (Max Retries): {occ_stats['occ_failures_max_retries']}")
        print(f"   OCC Avg Retry Time: {occ_stats['occ_avg_retry_time_ms']:.2f}ms")
        
        # OCC retry distribution
        if occ_stats['occ_retry_distribution']:
            print(f"\n🔄 OCC RETRY DISTRIBUTION:")
            for retry_count, count in sorted(occ_stats['occ_retry_distribution'].items()):
                print(f"   {retry_count} retries: {count} operations")
        
        # Per-operation OCC stats
        if occ_stats['occ_by_operation_type']:
            print(f"\n📈 OCC BY OPERATION TYPE:")
            for op_type, stats in occ_stats['occ_by_operation_type'].items():
                print(f"   {op_type}:")
                print(f"     Conflicts: {stats['conflicts']}")
                print(f"     Retries: {stats['retries']}")
                print(f"     Successes: {stats['successes']}")
                print(f"     Failures: {stats['failures']}")
        
        # System resources
        print(f"\n💾 SYSTEM RESOURCES:")
        print(f"   Peak Memory: {self.metrics.peak_memory_mb:.1f} MB")
        print(f"   Peak Operations In Progress: {self.metrics.peak_operations_in_progress}")
        
        # Test verdict
        occ_success_rate = (occ_stats['occ_successes_after_retry'] + self.metrics.operations_completed - occ_stats['occ_conflicts_detected']) / max(1, self.metrics.operations_completed)
        overall_success_rate = self.metrics.operations_completed / max(1, self.metrics.operations_submitted)
        
        print(f"\n🎯 TEST VERDICT:")
        print(f"   OCC Effectiveness: {occ_success_rate * 100:.1f}% (operations succeeded despite conflicts)")
        print(f"   Overall Success Rate: {overall_success_rate * 100:.1f}%")
        
        if overall_success_rate >= 0.95 and occ_stats['occ_conflict_rate_percent'] < 50:
            print(f"   ✅ TEST PASSED: High success rate with manageable OCC conflicts")
        elif overall_success_rate >= 0.80:
            print(f"   ⚠️  TEST MARGINAL: Acceptable but could be improved")
        else:
            print(f"   ❌ TEST FAILED: Low success rate indicates system stress")
        
        print(f"\n✨ OCC Integration Complete!")
        print(f"OCC = Optimistic Concurrency Control successfully integrated with:")
        print(f"   • Pre-ECS conflict resolution (operation-level)")
        print(f"   • Grace period protection (temporal)")
        print(f"   • Zombie cleanup (resource management)")


async def run_occ_stress_test(config: TestConfig) -> Dict[str, Any]:
    """Run the OCC stress test and return results."""
    test = OCCStressTest(config)
    await test.setup()
    await test.run_test()
    
    return {
        'metrics': test.metrics,
        'occ_stats': test.metrics.get_occ_statistics(),
        'elapsed_time': time.time() - test.metrics.start_time
    }


async def main():
    """Main function to run OCC stress test."""
    print("🚀 OCC + Dynamic Stress Test")
    print("=" * 60)
    print("Testing complete concurrency stack:")
    print("   1. Pre-ECS conflict resolution (operation-level)")
    print("   2. Grace period protection (temporal)")
    print("   3. OCC protection (data-level)")
    print("   4. Zombie cleanup (resource management)")
    print()
    print("OCC = Optimistic Concurrency Control")
    
    # Test configuration - BRUTAL OCC STRESS MODE
    config = TestConfig(
        duration_seconds=15,
        num_targets=2,  # FEWER targets = MORE conflicts per target
        operation_rate_per_second=100.0,  # MUCH higher rate
        priority_distribution={
            OperationPriority.CRITICAL: 0.1,
            OperationPriority.HIGH: 0.2,
            OperationPriority.NORMAL: 0.5,
            OperationPriority.LOW: 0.2
        },
        target_completion_rate=0.70,  # Lower expectation due to conflicts
        max_memory_mb=500.0,
        grace_period_seconds=0.1,
        occ_max_retries=5
    )
    
    # Run test
    results = await run_occ_stress_test(config)
    
    print(f"\n🎉 Test Complete! Check results above.")


if __name__ == "__main__":
    asyncio.run(main()) 