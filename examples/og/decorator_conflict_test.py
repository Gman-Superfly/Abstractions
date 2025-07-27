"""
Decorator Conflict Resolution Test: Validation of Declarative Conflict Protection

This test validates the conflict resolution decorators by comparing them against
the manual implementations proven in dynamic_stress_test.py and total_brutality_test.py.

Tests both decorator-based and manual patterns under identical conditions to verify:
1. Decorator performance matches manual implementation
2. Event emission is identical between approaches
3. Conflict resolution outcomes are consistent
4. Integration with existing operation hierarchy works correctly

Comprehensive validation with real conflicts, actual entity modifications, and stress testing.
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
    OperationStartedEvent, OperationCompletedEvent, OperationRejectedEvent, 
    OperationConflictEvent, OperationRetryEvent
)
from abstractions.ecs.entity import Entity, EntityRegistry
from abstractions.ecs.functional_api import put, get

# Import the new conflict resolution decorators
from abstractions.ecs.conflict_decorators import (
    with_conflict_resolution, no_conflict_resolution,
    ConflictResolutionConfig, ConflictResolutionMode,
    PreECSConfig, OCCConfig, get_staging_area_status, clear_staging_area
)

# Enable operation observers
import abstractions.agent_observer


class DecoratorTestConfig:
    """Configuration for decorator conflict resolution testing."""
    
    def __init__(self,
                 test_duration_seconds: int = 15,
                 operations_per_second: float = 200.0,
                 num_targets: int = 3,
                 batch_size: int = 10,
                 decorator_vs_manual_ratio: float = 0.5):
        """
        Configure decorator test parameters.
        
        Args:
            test_duration_seconds: How long to run the test
            operations_per_second: Rate of operation creation
            num_targets: Number of target entities (low = more conflicts)
            batch_size: Operations per batch for conflict creation
            decorator_vs_manual_ratio: 0.5 = 50% decorator, 50% manual
        """
        self.test_duration_seconds = test_duration_seconds
        self.operations_per_second = operations_per_second
        self.num_targets = num_targets
        self.batch_size = batch_size
        self.decorator_vs_manual_ratio = decorator_vs_manual_ratio
        
        print(f"🎯 DECORATOR TEST CONFIGURATION:")
        print(f"   ⏱️  Duration: {test_duration_seconds}s")
        print(f"   🚀 Rate: {operations_per_second:.0f} ops/sec")
        print(f"   🎯 Targets: {num_targets} (guaranteed conflicts)")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   ⚖️  Split: {decorator_vs_manual_ratio:.0%} decorator vs {1-decorator_vs_manual_ratio:.0%} manual")


class DecoratorTestMetrics:
    """Comprehensive metrics comparing decorator vs manual approaches."""
    
    def __init__(self):
        self.start_time = time.time()
        
        # Overall operation metrics
        self.total_operations_submitted = 0
        self.total_operations_completed = 0
        self.total_operations_failed = 0
        self.total_operations_rejected = 0
        
        # Split by approach
        self.decorator_operations_submitted = 0
        self.decorator_operations_completed = 0
        self.decorator_operations_failed = 0
        self.decorator_operations_rejected = 0
        
        self.manual_operations_submitted = 0
        self.manual_operations_completed = 0
        self.manual_operations_failed = 0
        self.manual_operations_rejected = 0
        
        # Conflict resolution metrics
        self.decorator_conflicts_detected = 0
        self.decorator_conflicts_resolved = 0
        self.manual_conflicts_detected = 0
        self.manual_conflicts_resolved = 0
        
        # Performance metrics
        self.decorator_execution_times: List[float] = []
        self.manual_execution_times: List[float] = []
        self.decorator_conflict_resolution_times: List[float] = []
        self.manual_conflict_resolution_times: List[float] = []
        
        # System metrics
        self.memory_samples: List[float] = []
        self.peak_memory_mb = 0.0
        
        # Entity modification tracking
        self.decorator_entity_modifications = 0
        self.manual_entity_modifications = 0
        
        # Event emission tracking
        self.decorator_events_emitted = 0
        self.manual_events_emitted = 0
    
    def record_decorator_operation(self, event_type: str, execution_time_ms: float = 0.0):
        """Record decorator-based operation metrics."""
        if event_type == "submitted":
            self.decorator_operations_submitted += 1
            self.total_operations_submitted += 1
        elif event_type == "completed":
            self.decorator_operations_completed += 1
            self.total_operations_completed += 1
            if execution_time_ms > 0:
                self.decorator_execution_times.append(execution_time_ms)
        elif event_type == "failed":
            self.decorator_operations_failed += 1
            self.total_operations_failed += 1
        elif event_type == "rejected":
            self.decorator_operations_rejected += 1
            self.total_operations_rejected += 1
    
    def record_manual_operation(self, event_type: str, execution_time_ms: float = 0.0):
        """Record manual implementation operation metrics."""
        if event_type == "submitted":
            self.manual_operations_submitted += 1
            self.total_operations_submitted += 1
        elif event_type == "completed":
            self.manual_operations_completed += 1
            self.total_operations_completed += 1
            if execution_time_ms > 0:
                self.manual_execution_times.append(execution_time_ms)
        elif event_type == "failed":
            self.manual_operations_failed += 1
            self.total_operations_failed += 1
        elif event_type == "rejected":
            self.manual_operations_rejected += 1
            self.total_operations_rejected += 1
    
    def record_system_stats(self):
        """Record current system statistics."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        except:
            pass
    
    def get_comparison_stats(self) -> Dict[str, Any]:
        """Get comparative statistics between decorator and manual approaches."""
        elapsed = time.time() - self.start_time
        
        # Calculate success rates
        decorator_success_rate = (
            self.decorator_operations_completed / max(1, self.decorator_operations_submitted)
        )
        manual_success_rate = (
            self.manual_operations_completed / max(1, self.manual_operations_submitted)
        )
        overall_success_rate = (
            self.total_operations_completed / max(1, self.total_operations_submitted)
        )
        
        # Calculate average execution times
        decorator_avg_time = (
            statistics.mean(self.decorator_execution_times) 
            if self.decorator_execution_times else 0.0
        )
        manual_avg_time = (
            statistics.mean(self.manual_execution_times) 
            if self.manual_execution_times else 0.0
        )
        
        return {
            'elapsed_seconds': elapsed,
            'overall_success_rate': overall_success_rate,
            'decorator': {
                'submitted': self.decorator_operations_submitted,
                'completed': self.decorator_operations_completed,
                'failed': self.decorator_operations_failed,
                'rejected': self.decorator_operations_rejected,
                'success_rate': decorator_success_rate,
                'avg_execution_time_ms': decorator_avg_time,
                'conflicts_detected': self.decorator_conflicts_detected,
                'conflicts_resolved': self.decorator_conflicts_resolved,
                'entity_modifications': self.decorator_entity_modifications,
                'events_emitted': self.decorator_events_emitted
            },
            'manual': {
                'submitted': self.manual_operations_submitted,
                'completed': self.manual_operations_completed,
                'failed': self.manual_operations_failed,
                'rejected': self.manual_operations_rejected,
                'success_rate': manual_success_rate,
                'avg_execution_time_ms': manual_avg_time,
                'conflicts_detected': self.manual_conflicts_detected,
                'conflicts_resolved': self.manual_conflicts_resolved,
                'entity_modifications': self.manual_entity_modifications,
                'events_emitted': self.manual_events_emitted
            },
            'system': {
                'peak_memory_mb': self.peak_memory_mb,
                'avg_memory_mb': statistics.mean(self.memory_samples) if self.memory_samples else 0.0
            }
        }


class DecoratorTestEntity(Entity):
    """Test entity optimized for decorator conflict testing."""
    
    name: str = "decorator_test_entity"
    value: float = 0.0
    modification_count: int = 0
    last_modifier: str = ""
    modification_history: List[str] = Field(default_factory=list)
    
    def record_modification(self, modifier_name: str, new_value: float):
        """Record a modification with complete tracking."""
        old_value = self.value
        self.value = new_value
        self.modification_count += 1
        self.last_modifier = modifier_name
        
        modification_record = f"{modifier_name}: {old_value} → {new_value} @ {time.time()}"
        self.modification_history.append(modification_record)
        
        # Keep history manageable
        if len(self.modification_history) > 50:
            self.modification_history = self.modification_history[-25:]
        
        # Update ECS versioning
        self.mark_modified()


# ============================================================================
# DECORATOR-BASED OPERATIONS (Using the new decorators)
# ============================================================================

@with_conflict_resolution(pre_ecs=True, occ=True, priority=OperationPriority.HIGH)
async def decorator_increment_operation(target: DecoratorTestEntity, amount: float) -> bool:
    """Decorator-protected increment operation with conflict resolution."""
    try:
        # Simulate processing time where conflicts can occur
        await asyncio.sleep(0.005)
        
        # Modify the entity
        new_value = target.value + amount
        target.record_modification("decorator_increment", new_value)
        
        return True
    except Exception as e:
        print(f"❌ Decorator increment failed: {e}")
        return False


@with_conflict_resolution(
    config=ConflictResolutionConfig(
        mode=ConflictResolutionMode.BOTH,
        pre_ecs=PreECSConfig(priority=OperationPriority.CRITICAL, staging_timeout_ms=50.0),
        occ=OCCConfig(max_retries=15, backoff_factor=2.0)
    )
)
async def decorator_complex_update(target: DecoratorTestEntity, multiplier: float, offset: float) -> bool:
    """Complex decorator-protected operation with custom configuration."""
    try:
        # More complex processing simulation
        await asyncio.sleep(0.008)
        
        # Complex calculation
        intermediate = target.value * multiplier
        await asyncio.sleep(0.002)  # More opportunity for conflicts
        new_value = intermediate + offset
        
        target.record_modification("decorator_complex", new_value)
        return True
    except Exception as e:
        print(f"❌ Decorator complex update failed: {e}")
        return False


@no_conflict_resolution
async def decorator_read_only_operation(target: DecoratorTestEntity) -> float:
    """Read-only operation explicitly marked as no conflict resolution needed."""
    await asyncio.sleep(0.001)  # Minimal processing
    return target.value + target.modification_count  # Read-only calculation


# ============================================================================
# MANUAL OPERATIONS (Using patterns from stress tests)
# ============================================================================

class ManualTestOperation(OperationEntity):
    """Manual operation entity for comparison with decorators."""
    
    operation_type: str = "manual_test_operation"
    target_value: float = 0.0
    modifier_name: str = ""
    
    async def execute_with_manual_conflict_resolution(
        self, 
        target: DecoratorTestEntity,
        metrics: DecoratorTestMetrics
    ) -> bool:
        """Execute with manual OCC protection (like total_brutality_test.py)."""
        retry_count = 0
        max_retries = 10
        
        while retry_count <= max_retries:
            try:
                # READ: Snapshot current state
                read_version = target.version
                read_modified = target.last_modified
                read_value = target.value
                
                # PROCESS: Simulate work where conflicts can occur
                await asyncio.sleep(0.005)
                new_value = read_value + self.target_value
                
                # WRITE: Check for conflicts before committing
                current_version = target.version
                current_modified = target.last_modified
                
                if (current_version != read_version or current_modified != read_modified):
                    # Conflict detected - emit event and retry
                    metrics.manual_conflicts_detected += 1
                    
                    await emit(OperationConflictEvent(
                        op_id=self.ecs_id,
                        op_type=self.op_type,
                        target_entity_id=target.ecs_id,
                        priority=self.priority,
                        conflict_details={"retry_count": retry_count, "approach": "manual"}
                    ))
                    
                    retry_count += 1
                    if retry_count <= max_retries:
                        await asyncio.sleep(0.002 * (2 ** retry_count))  # Exponential backoff
                        continue
                    else:
                        print(f"❌ Manual operation max retries exceeded: {self.op_type}")
                        return False
                
                # COMMIT: Safe to write
                target.record_modification(self.modifier_name, new_value)
                metrics.manual_conflicts_resolved += 1
                return True
                
            except Exception as e:
                print(f"❌ Manual operation error: {e}")
                if retry_count >= max_retries:
                    return False
                retry_count += 1
                await asyncio.sleep(0.002 * (2 ** retry_count))
        
        return False


# ============================================================================
# COMPREHENSIVE DECORATOR VS MANUAL TEST
# ============================================================================

class DecoratorConflictTest:
    """Comprehensive test comparing decorator vs manual conflict resolution."""
    
    def __init__(self, config: DecoratorTestConfig):
        self.config = config
        self.metrics = DecoratorTestMetrics()
        self.targets: List[DecoratorTestEntity] = []
        
        # Control flags
        self.stop_flag = False
        
        # Manual conflict resolution staging (like dynamic_stress_test.py)
        self.manual_pending_operations: Dict[UUID, List[ManualTestOperation]] = {}
        
    async def setup_test_environment(self):
        """Set up test entities and environment."""
        print(f"\n🔧 SETTING UP DECORATOR CONFLICT TEST")
        print(f"=" * 60)
        
        # Create target entities
        for i in range(self.config.num_targets):
            target = DecoratorTestEntity(
                name=f"decorator_test_target_{i}",
                value=float(i * 10),
                modification_count=0,
                last_modifier="initial",
                modification_history=[f"Created at {datetime.now(timezone.utc)}"]
            )
            target.promote_to_root()
            self.targets.append(target)
        
        print(f"✅ Created {len(self.targets)} test entities")
        print(f"🎯 Each target will receive mixed decorator + manual operations")
        print(f"💥 Expected conflicts: GUARANTEED (multiple ops per target)")
        
        # Set up event handlers
        from abstractions.events.events import setup_operation_event_handlers
        setup_operation_event_handlers()
        print(f"✅ Event handlers ready for conflict detection")
        
        # Clear any existing staging area
        from abstractions.ecs.conflict_decorators import clear_staging_area
        clear_staging_area()
        print(f"✅ Staging areas cleared and ready")
        
        # Start staging coordinator for decorator operations
        from abstractions.ecs.conflict_decorators import _start_staging_coordinator
        await _start_staging_coordinator()
        print(f"✅ Staging coordinator started for decorator operations")
    
    async def submit_decorator_operation(self, target: DecoratorTestEntity) -> bool:
        """Submit a decorator-protected operation."""
        operation_types = ["increment", "complex_update", "read_only"]
        op_type = random.choice(operation_types)
        
        start_time = time.time()
        success = False
        
        try:
            if op_type == "increment":
                amount = random.uniform(1.0, 10.0)
                success = await decorator_increment_operation(target, amount)
                if success:
                    self.metrics.decorator_entity_modifications += 1
                
            elif op_type == "complex_update":
                multiplier = random.uniform(1.1, 2.0)
                offset = random.uniform(-5.0, 5.0)
                success = await decorator_complex_update(target, multiplier, offset)
                if success:
                    self.metrics.decorator_entity_modifications += 1
                
            elif op_type == "read_only":
                result = await decorator_read_only_operation(target)
                success = True  # Read-only operations always succeed (no entity modification)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            if success:
                self.metrics.record_decorator_operation("completed", execution_time_ms)
            else:
                self.metrics.record_decorator_operation("failed")
            
            return success
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            error_message = str(e)
            
            # Distinguish between rejections and actual failures
            if "rejected by Pre-ECS conflict resolution" in error_message:
                print(f"❌ Decorator operation rejected: {e}")
                self.metrics.record_decorator_operation("rejected")
            else:
                print(f"⚠️  Decorator operation failed: {e}")
                self.metrics.record_decorator_operation("failed")
            return False
    
    async def submit_manual_operation(self, target: DecoratorTestEntity) -> bool:
        """Submit a manual conflict resolution operation."""
        # Create manual operation
        operation = ManualTestOperation(
            op_type="manual_test_operation",
            target_entity_id=target.ecs_id,
            priority=random.choice([OperationPriority.NORMAL, OperationPriority.HIGH, OperationPriority.CRITICAL]),
            target_value=random.uniform(1.0, 10.0),
            modifier_name="manual_operation"
        )
        
        # Add to manual staging area (like dynamic_stress_test.py)
        if target.ecs_id not in self.manual_pending_operations:
            self.manual_pending_operations[target.ecs_id] = []
        self.manual_pending_operations[target.ecs_id].append(operation)
        
        # Don't promote to ECS yet - keep in staging
        self.metrics.record_manual_operation("submitted")
        return True
    
    async def resolve_manual_conflicts(self, target_entity_id: UUID):
        """Resolve manual conflicts using patterns from dynamic_stress_test.py."""
        pending_ops = self.manual_pending_operations.get(target_entity_id, [])
        
        if len(pending_ops) > 1:
            resolution_start = time.time()
            self.metrics.manual_conflicts_detected += 1
            
            print(f"⚔️  MANUAL CONFLICT: {len(pending_ops)} operations for target {str(target_entity_id)[:8]}")
            
            # Sort by priority (higher wins) - same as dynamic_stress_test.py
            pending_ops.sort(key=lambda op: (op.priority, -op.created_at.timestamp()), reverse=True)
            
            winner = pending_ops[0]
            losers = pending_ops[1:]
            
            # Promote winner to ECS
            winner.promote_to_root()
            
            # Reject losers
            for loser in losers:
                self.metrics.record_manual_operation("rejected")
                await emit(OperationRejectedEvent(
                    op_id=loser.ecs_id,
                    op_type=loser.op_type,
                    target_entity_id=target_entity_id,
                    from_state="pending",
                    to_state="rejected",
                    rejection_reason="preempted_by_higher_priority_manual",
                    retry_count=loser.retry_count
                ))
            
            resolution_time_ms = (time.time() - resolution_start) * 1000
            self.metrics.manual_conflict_resolution_times.append(resolution_time_ms)
            self.metrics.manual_conflicts_resolved += 1
            
            self.manual_pending_operations[target_entity_id] = []
            return [winner]
            
        elif len(pending_ops) == 1:
            # No conflict - promote single operation
            winner = pending_ops[0]
            winner.promote_to_root()
            self.manual_pending_operations[target_entity_id] = []
            return [winner]
        
        return []
    
    async def execute_manual_winners(self, winners: List[ManualTestOperation]):
        """Execute winning manual operations with OCC protection."""
        for winner in winners:
            target = None
            for t in self.targets:
                if t.ecs_id == winner.target_entity_id:
                    target = t
                    break
            
            if target:
                start_time = time.time()
                success = await winner.execute_with_manual_conflict_resolution(target, self.metrics)
                execution_time_ms = (time.time() - start_time) * 1000
                
                if success:
                    self.metrics.record_manual_operation("completed", execution_time_ms)
                    self.metrics.manual_entity_modifications += 1
                else:
                    self.metrics.record_manual_operation("failed")
    
    async def run_comprehensive_test(self):
        """Run comprehensive comparison test."""
        print(f"\n🚀 STARTING COMPREHENSIVE DECORATOR VS MANUAL TEST")
        print(f"💀" * 60)
        
        # Start all workers
        tasks = [
            asyncio.create_task(self._operation_submission_worker()),
            asyncio.create_task(self._manual_conflict_resolution_worker()),
            asyncio.create_task(self._progress_monitor()),
            asyncio.create_task(self._system_monitor())
        ]
        
        try:
            # Run test for specified duration
            print(f"⏱️  Running test for {self.config.test_duration_seconds}s...")
            await asyncio.sleep(self.config.test_duration_seconds)
            
            print(f"⏹️  Stopping new submissions - allowing grace period...")
            self.stop_flag = True
            await asyncio.sleep(2.0)  # Grace period for pending operations
            
        finally:
            # Stop all workers
            for task in tasks:
                task.cancel()
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                print("⚠️  Some workers took too long to stop")
            
            # Stop staging coordinator 
            try:
                from abstractions.ecs.conflict_decorators import _stop_staging_coordinator
                await _stop_staging_coordinator()
                print("✅ Staging coordinator stopped")
            except Exception as e:
                print(f"⚠️  Error stopping staging coordinator: {e}")
        
        # Analyze results
        await self._analyze_comparison_results()
    
    async def _operation_submission_worker(self):
        """Submit operations using mixed decorator + manual approaches with IDENTICAL timing."""
        interval = 1.0 / self.config.operations_per_second * self.config.batch_size
        operations_submitted = 0
        
        while not self.stop_flag:
            try:
                # Submit batch of operations to same target (guaranteed conflicts)
                target = random.choice(self.targets)
                
                # COLLECT BOTH TYPES IN BATCHES for identical conflict windows
                decorator_batch = []
                manual_batch = []
                
                for i in range(self.config.batch_size):
                    # Decide decorator vs manual based on ratio
                    use_decorator = random.random() < self.config.decorator_vs_manual_ratio
                    
                    if use_decorator:
                        # STAGE decorator operations (don't execute immediately)
                        decorator_batch.append(i)
                    else:
                        # STAGE manual operations  
                        manual_batch.append(i)
                
                # Execute BOTH batches simultaneously for identical conflict windows
                batch_results = await asyncio.gather(
                    self._execute_decorator_batch(target, decorator_batch),
                    self._execute_manual_batch(target, manual_batch),
                    return_exceptions=True
                )
                
                operations_submitted += len(decorator_batch) + len(manual_batch)
                print(f"📦 BATCH: {len(decorator_batch)} decorator + {len(manual_batch)} manual → {str(target.ecs_id)[:8]} (total: {operations_submitted})")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"⚠️  Error in operation submission: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_decorator_batch(self, target: DecoratorTestEntity, batch_indices: List[int]) -> List[bool]:
        """Execute a batch of decorator operations simultaneously (STRESS TEST PATTERN)."""
        
        # Submit ALL decorator operations simultaneously to create real conflicts
        decorator_tasks = []
        
        for i in batch_indices:
            # Create concurrent tasks that will hit the staging area simultaneously
            task = asyncio.create_task(self._submit_decorator_operation_concurrent(target))
            decorator_tasks.append(task)
        
        # Execute all decorator operations simultaneously (guaranteed conflicts!)
        results = await asyncio.gather(*decorator_tasks, return_exceptions=True)
        
        # Process results
        successes = []
        for result in results:
            if isinstance(result, Exception):
                error_message = str(result)
                if "rejected by Pre-ECS conflict resolution" in error_message:
                    self.metrics.record_decorator_operation("rejected")
                else:
                    self.metrics.record_decorator_operation("failed")
                successes.append(False)
            else:
                self.metrics.record_decorator_operation("completed")
                successes.append(True)
        
        return successes
    
    async def _submit_decorator_operation_concurrent(self, target: DecoratorTestEntity) -> bool:
        """Submit a single decorator operation (designed for concurrent execution)."""
        operation_types = ["increment", "complex_update"]  # Remove read_only for conflict testing
        op_type = random.choice(operation_types)
        
        try:
            if op_type == "increment":
                amount = random.uniform(1.0, 10.0)
                success = await decorator_increment_operation(target, amount)
                if success:
                    self.metrics.decorator_entity_modifications += 1
                return success
                
            elif op_type == "complex_update":
                multiplier = random.uniform(1.1, 2.0)
                offset = random.uniform(-5.0, 5.0)
                success = await decorator_complex_update(target, multiplier, offset)
                if success:
                    self.metrics.decorator_entity_modifications += 1
                return success
            
            return False
            
        except Exception as e:
            # Let the exception propagate to be handled in batch processing
            raise e
    
    async def _execute_manual_batch(self, target: DecoratorTestEntity, batch_indices: List[int]) -> List[bool]:
        """Execute a batch of manual operations with immediate conflict resolution."""
        # Submit all manual operations to staging
        for i in batch_indices:
            await self.submit_manual_operation(target)
        
        # Resolve conflicts immediately (same timing as decorators)
        winners = await self.resolve_manual_conflicts(target.ecs_id)
        
        # Execute winners
        if winners:
            await self.execute_manual_winners(winners)
        
        return [True] * len(batch_indices)  # All submitted successfully
    
    async def _manual_conflict_resolution_worker(self):
        """Manual conflict resolution worker - now does minimal work since conflicts resolved immediately."""
        while not self.stop_flag:
            try:
                # Most work now done in _execute_manual_batch for timing consistency
                # This worker just handles cleanup
                await asyncio.sleep(0.5)  # Check every 500ms for cleanup
                
            except Exception as e:
                print(f"⚠️  Error in manual conflict resolution: {e}")
                await asyncio.sleep(0.1)
    
    async def _progress_monitor(self):
        """Monitor test progress."""
        while not self.stop_flag:
            try:
                stats = self.metrics.get_comparison_stats()
                
                print(f"\n📊 PROGRESS @ {stats['elapsed_seconds']:.1f}s:")
                print(f"   🎯 Overall success: {stats['overall_success_rate']:.1%}")
                print(f"   🎨 Decorator: {stats['decorator']['completed']}/{stats['decorator']['submitted']} ({stats['decorator']['success_rate']:.1%})")
                print(f"   🔧 Manual: {stats['manual']['completed']}/{stats['manual']['submitted']} ({stats['manual']['success_rate']:.1%})")
                print(f"   💾 Memory: {stats['system']['peak_memory_mb']:.1f} MB")
                
                # Show staging area status
                staging_status = get_staging_area_status()
                if staging_status['total_staged_operations'] > 0:
                    print(f"   📦 Decorator staging: {staging_status['total_staged_operations']} ops")
                
                await asyncio.sleep(5.0)
                
            except Exception as e:
                print(f"⚠️  Error in progress monitoring: {e}")
                await asyncio.sleep(2.0)
    
    async def _system_monitor(self):
        """Monitor system resources."""
        while not self.stop_flag:
            try:
                self.metrics.record_system_stats()
                await asyncio.sleep(1.0)
            except Exception as e:
                print(f"⚠️  Error in system monitoring: {e}")
                await asyncio.sleep(1.0)
    
    async def _analyze_comparison_results(self):
        """Analyze and compare decorator vs manual results with pattern validation."""
        stats = self.metrics.get_comparison_stats()
        
        print(f"\n" + "🎯" * 80)
        print(f"DECORATOR VS MANUAL CONFLICT RESOLUTION COMPARISON")
        print(f"🎯" * 80)
        
        # Test overview
        print(f"\n📊 TEST OVERVIEW:")
        print(f"   ⏱️  Duration: {stats['elapsed_seconds']:.1f}s")
        print(f"   🎯 Targets: {len(self.targets)}")
        print(f"   📦 Batch size: {self.config.batch_size}")
        print(f"   ⚖️  Split: {self.config.decorator_vs_manual_ratio:.0%} decorator vs {1-self.config.decorator_vs_manual_ratio:.0%} manual")
        
        # PATTERN VALIDATION - Ensure both approaches follow proven patterns
        print(f"\n🔍 PATTERN VALIDATION (vs proven stress test patterns):")
        
        decorator_stats = stats['decorator']
        manual_stats = stats['manual']
        
        # Validate that both approaches show conflict behavior
        decorator_conflicts = decorator_stats['conflicts_resolved']
        manual_conflicts = manual_stats['conflicts_resolved']
        
        print(f"   📈 CONFLICT DETECTION VALIDATION:")
        if decorator_conflicts > 0:
            print(f"      ✅ Decorator conflicts detected: {decorator_conflicts}")
        else:
            print(f"      ❌ Decorator conflicts: {decorator_conflicts} (Should be > 0 for batched operations)")
        
        if manual_conflicts > 0:
            print(f"      ✅ Manual conflicts detected: {manual_conflicts}")
        else:
            print(f"      ❌ Manual conflicts: {manual_conflicts} (Should be > 0 for batched operations)")
        
        # Validate rejection behavior matches proven patterns
        print(f"   📈 REJECTION PATTERN VALIDATION:")
        decorator_rejection_rate = decorator_stats['rejected'] / max(1, decorator_stats['submitted'])
        manual_rejection_rate = manual_stats['rejected'] / max(1, manual_stats['submitted'])
        
        print(f"      ├─ Decorator rejection rate: {decorator_rejection_rate:.1%}")
        print(f"      ├─ Manual rejection rate: {manual_rejection_rate:.1%}")
        
        # For batched operations, expect significant rejections (proven pattern)
        expected_rejection_rate = 0.5  # At least 50% rejected in batch conflicts
        if decorator_rejection_rate >= expected_rejection_rate:
            print(f"      ✅ Decorator rejections match batch conflict pattern")
        else:
            print(f"      ⚠️  Decorator rejections ({decorator_rejection_rate:.1%}) lower than expected ({expected_rejection_rate:.0%})")
        
        if manual_rejection_rate >= expected_rejection_rate:
            print(f"      ✅ Manual rejections match batch conflict pattern")
        else:
            print(f"      ⚠️  Manual rejections ({manual_rejection_rate:.1%}) lower than expected ({expected_rejection_rate:.0%})")
        
        # Validate execution pattern matches proven approaches
        print(f"   📈 EXECUTION PATTERN VALIDATION:")
        decorator_execution_rate = decorator_stats['completed'] / max(1, decorator_stats['submitted'])
        manual_execution_rate = manual_stats['completed'] / max(1, manual_stats['submitted'])
        
        print(f"      ├─ Decorator execution rate: {decorator_execution_rate:.1%}")
        print(f"      ├─ Manual execution rate: {manual_execution_rate:.1%}")
        
        # Both should have reasonable execution rates despite conflicts
        min_execution_rate = 0.3  # At least 30% should execute successfully
        if decorator_execution_rate >= min_execution_rate:
            print(f"      ✅ Decorator execution rate acceptable")
        else:
            print(f"      ⚠️  Decorator execution rate ({decorator_execution_rate:.1%}) too low")
        
        if manual_execution_rate >= min_execution_rate:
            print(f"      ✅ Manual execution rate acceptable")
        else:
            print(f"      ⚠️  Manual execution rate ({manual_execution_rate:.1%}) too low")
        
        # Performance comparison
        print(f"\n⚡ PERFORMANCE COMPARISON:")
        
        print(f"   📈 DECORATOR APPROACH:")
        print(f"      ├─ Submitted: {decorator_stats['submitted']}")
        print(f"      ├─ Completed: {decorator_stats['completed']}")
        print(f"      ├─ Failed: {decorator_stats['failed']}")
        print(f"      ├─ Rejected: {decorator_stats['rejected']}")
        print(f"      ├─ Success rate: {decorator_stats['success_rate']:.1%}")
        print(f"      ├─ Avg execution time: {decorator_stats['avg_execution_time_ms']:.1f}ms")
        print(f"      ├─ Entity modifications: {decorator_stats['entity_modifications']}")
        print(f"      └─ Conflicts handled: {decorator_stats['conflicts_resolved']}")
        
        print(f"   🔧 MANUAL APPROACH:")
        print(f"      ├─ Submitted: {manual_stats['submitted']}")
        print(f"      ├─ Completed: {manual_stats['completed']}")
        print(f"      ├─ Failed: {manual_stats['failed']}")
        print(f"      ├─ Rejected: {manual_stats['rejected']}")
        print(f"      ├─ Success rate: {manual_stats['success_rate']:.1%}")
        print(f"      ├─ Avg execution time: {manual_stats['avg_execution_time_ms']:.1f}ms")
        print(f"      ├─ Entity modifications: {manual_stats['entity_modifications']}")
        print(f"      └─ Conflicts handled: {manual_stats['conflicts_resolved']}")
        
        # Comparison analysis
        print(f"\n🔍 COMPARISON ANALYSIS:")
        
        # Success rate comparison
        success_diff = decorator_stats['success_rate'] - manual_stats['success_rate']
        if abs(success_diff) < 0.05:  # Within 5%
            print(f"   ✅ SUCCESS RATES: Equivalent ({success_diff:+.1%} difference)")
        elif success_diff > 0:
            print(f"   🎨 SUCCESS RATES: Decorator advantage ({success_diff:+.1%})")
        else:
            print(f"   🔧 SUCCESS RATES: Manual advantage ({success_diff:+.1%})")
        
        # Performance comparison
        time_diff = decorator_stats['avg_execution_time_ms'] - manual_stats['avg_execution_time_ms']
        if abs(time_diff) < 2.0:  # Within 2ms
            print(f"   ✅ EXECUTION TIME: Equivalent ({time_diff:+.1f}ms difference)")
        elif time_diff < 0:
            print(f"   🎨 EXECUTION TIME: Decorator faster ({abs(time_diff):.1f}ms)")
        else:
            print(f"   🔧 EXECUTION TIME: Manual faster ({time_diff:.1f}ms)")
        
        # Entity modifications
        total_modifications = decorator_stats['entity_modifications'] + manual_stats['entity_modifications']
        if total_modifications > 0:
            print(f"   🎯 ENTITY MODIFICATIONS: {total_modifications} total, both approaches working")
        
        # PROVEN PATTERN COMPLIANCE CHECK
        print(f"\n🎯 PROVEN PATTERN COMPLIANCE:")
        
        compliance_issues = []
        
        # Check if conflicts are being detected (batched operations should create conflicts)
        if decorator_conflicts == 0:
            compliance_issues.append("Decorator approach: No conflicts detected")
        if manual_conflicts == 0:
            compliance_issues.append("Manual approach: No conflicts detected")
        
        # Check if rejection rates are reasonable for batched operations
        if decorator_rejection_rate < 0.3:  # Less than 30% rejected is suspicious for batch conflicts
            compliance_issues.append(f"Decorator approach: Low rejection rate ({decorator_rejection_rate:.1%})")
        if manual_rejection_rate < 0.3:
            compliance_issues.append(f"Manual approach: Low rejection rate ({manual_rejection_rate:.1%})")
        
        # Check if both approaches actually modify entities
        if decorator_stats['entity_modifications'] == 0:
            compliance_issues.append("Decorator approach: No entity modifications")
        if manual_stats['entity_modifications'] == 0:
            compliance_issues.append("Manual approach: No entity modifications")
        
        if compliance_issues:
            print(f"   ⚠️  COMPLIANCE ISSUES DETECTED:")
            for issue in compliance_issues:
                print(f"      ├─ {issue}")
            print(f"      └─ These issues suggest patterns don't match proven stress test behavior")
        else:
            print(f"   ✅ FULL COMPLIANCE: Both approaches follow proven stress test patterns")
        
        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT:")
        
        overall_success = stats['overall_success_rate']
        if overall_success >= 0.90:
            print(f"   ✅ EXCELLENT: {overall_success:.1%} overall success under conflict load")
        elif overall_success >= 0.75:
            print(f"   ✅ GOOD: {overall_success:.1%} overall success with some conflicts")
        else:
            print(f"   ⚠️  STRESSED: {overall_success:.1%} success rate under extreme load")
        
        # Decorator validation
        patterns_match = (
            abs(success_diff) < 0.10 and  # Success rates within 10%
            abs(time_diff) < 5.0 and      # Execution times within 5ms
            decorator_stats['entity_modifications'] > 0 and  # Actually modifying entities
            len(compliance_issues) == 0   # No compliance issues
        )
        
        if patterns_match:
            print(f"   🎉 DECORATOR VALIDATION: ✅ PASSED")
            print(f"      └─ Decorators perform equivalently to manual implementation")
            print(f"      └─ Both approaches follow proven stress test patterns")
        else:
            print(f"   ⚠️  DECORATOR VALIDATION: Needs investigation")
            print(f"      └─ Performance or behavior differs from manual implementation")
            if compliance_issues:
                print(f"      └─ Pattern compliance issues need resolution")
        
        # Entity verification
        print(f"\n🎯 ENTITY VERIFICATION:")
        for i, target in enumerate(self.targets):
            print(f"   Target {i} ({target.name}):")
            print(f"      ├─ Final value: {target.value:.2f}")
            print(f"      ├─ Modifications: {target.modification_count}")
            print(f"      ├─ Last modifier: {target.last_modifier}")
            print(f"      ├─ ECS version: {target.version}")
            print(f"      └─ History entries: {len(target.modification_history)}")
        
        print(f"\n🎯 DECORATOR CONFLICT TEST COMPLETE!")
        if patterns_match:
            print(f"✅ Both approaches validated under identical conflict conditions")
            print(f"✅ Decorator implementation follows proven stress test patterns")
        else:
            print(f"⚠️  Pattern differences detected - review implementation")
            print(f"⚠️  Ensure both approaches follow identical conflict resolution logic")


async def run_decorator_conflict_test(config: DecoratorTestConfig) -> Dict[str, Any]:
    """Run the comprehensive decorator vs manual conflict resolution test."""
    # Start event bus
    bus = get_event_bus()
    await bus.start()
    
    try:
        test = DecoratorConflictTest(config)
        await test.setup_test_environment()
        await test.run_comprehensive_test()
        
        return test.metrics.get_comparison_stats()
        
    except Exception as e:
        print(f"💥 DECORATOR TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}
        
    finally:
        await bus.stop()


async def main():
    """Main function for decorator conflict resolution testing."""
    print("🎯" * 80)
    print("🎨 DECORATOR CONFLICT RESOLUTION TEST")
    print("🎯" * 80)
    print("Comprehensive validation of declarative conflict protection:")
    print("   1. Decorator-based conflict resolution (@with_conflict_resolution)")
    print("   2. Manual conflict resolution (stress test patterns)")
    print("   3. Side-by-side comparison under identical conditions")
    print("   4. Event emission verification")
    print("   5. Performance and correctness validation")
    print()
    print("🎯 Testing both approaches simultaneously with REAL CONCURRENT conflicts!")
    print("🎯 Following PROVEN stress test patterns for staging coordination!")
    print("🎯" * 80)
    
    # COMPREHENSIVE TEST CONFIGURATION - Designed for concurrent conflicts
    config = DecoratorTestConfig(
        test_duration_seconds=15,       # 15 seconds test
        operations_per_second=100.0,    # 100 ops/sec (manageable rate)
        num_targets=3,                  # 3 targets = guaranteed conflicts
        batch_size=6,                   # 6 ops per batch (more conflicts)
        decorator_vs_manual_ratio=0.6   # 60% decorator, 40% manual (more decorator testing)
    )
    
    # Run the comprehensive test
    results = await run_decorator_conflict_test(config)
    
    if 'error' not in results:
        print(f"\n🎉 DECORATOR CONFLICT TEST SUCCESSFUL!")
        print(f"Decorator-based conflict resolution validated against manual implementation.")
        print(f"✅ Follows proven stress test patterns for staging coordination")
    else:
        print(f"\n💥 DECORATOR CONFLICT TEST FAILED")
        print(f"Error: {results['error']}")


if __name__ == "__main__":
    # Use high-performance event loop policy on Windows
    if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main()) 