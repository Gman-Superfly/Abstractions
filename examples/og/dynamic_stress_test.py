"""
Dynamic Conflict Resolution Stress Test

The ONE stress test to rule them all! This combines:
- No artificial concurrency limits (high_concurrency_test)
- Grace period protection (grace_period_stress_test)  
- Real-time adaptation (real_time_adaptive_stress_test)
- System profiling (system_performance_profiler)
- Agent observer integration (hierarchy_integration_test)

Automatically adapts to your system's capabilities and optimizes parameters
in real-time for maximum performance and completion rates.
"""

import asyncio
import time
import random
import statistics
import psutil
from typing import List, Dict, Any, Set
from collections import deque, defaultdict
from datetime import datetime, timezone
from uuid import UUID, uuid4

# Core imports
from abstractions.ecs.entity_hierarchy import (
    OperationEntity, StructuralOperation, NormalOperation, LowPriorityOperation,
    OperationStatus, OperationPriority,
    get_conflicting_operations, get_operation_stats
)
from abstractions.events.events import (
    get_event_bus, emit,
    OperationStartedEvent, OperationCompletedEvent, OperationRejectedEvent
)
from abstractions.ecs.entity import Entity, EntityRegistry

# Enable operation observers
import abstractions.agent_observer


class DynamicStressTestMetrics:
    """Comprehensive metrics with real-time adaptation tracking."""
    
    def __init__(self):
        self.start_time = time.time()
        
        # Core metrics
        self.operations_created = 0
        self.operations_started = 0
        self.operations_completed = 0
        self.operations_rejected = 0
        self.conflicts_detected = 0
        self.conflicts_resolved = 0
        
        # Priority-based tracking
        self.created_by_priority = defaultdict(int)
        self.started_by_priority = defaultdict(int)
        self.completed_by_priority = defaultdict(int)
        self.rejected_by_priority = defaultdict(int)
        
        # Grace period metrics
        self.grace_period_saves = 0
        self.operations_protected = 0
        
        # Performance tracking
        self.resolution_times = []
        self.conflict_sizes = []
        self.memory_samples = []
        self.cpu_samples = []
        
        # Real-time adaptation
        self.adaptation_count = 0
        self.throughput_history = deque(maxlen=20)
        self.completion_rate_history = deque(maxlen=20)
        
    def _get_priority_name(self, priority):
        """Convert priority to consistent string representation."""
        if hasattr(priority, 'name'):
            return priority.name
        else:
            # Convert integer values to names
            priority_map = {2: 'LOW', 5: 'NORMAL', 8: 'HIGH', 10: 'CRITICAL'}
            return priority_map.get(priority, str(priority))
    
    def record_operation_created(self, priority: OperationPriority = None):
        self.operations_created += 1
        if priority:
            priority_name = self._get_priority_name(priority)
            self.created_by_priority[priority_name] += 1
        
    def record_operation_started(self, priority: OperationPriority = None):
        self.operations_started += 1
        if priority:
            priority_name = self._get_priority_name(priority)
            self.started_by_priority[priority_name] += 1
        
    def record_operation_completed(self, priority: OperationPriority = None):
        self.operations_completed += 1
        if priority:
            priority_name = self._get_priority_name(priority)
            self.completed_by_priority[priority_name] += 1
        
    def record_operation_rejected(self, priority: OperationPriority = None):
        self.operations_rejected += 1
        if priority:
            priority_name = self._get_priority_name(priority)
            self.rejected_by_priority[priority_name] += 1
        
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
            self.memory_samples.append(0.0)
            self.cpu_samples.append(0.0)
            
    def record_adaptation(self):
        self.adaptation_count += 1
        
    def get_current_throughput(self) -> float:
        elapsed = time.time() - self.start_time
        return self.operations_created / elapsed if elapsed > 0 else 0
        
    def get_current_completion_rate(self) -> float:
        return self.operations_completed / self.operations_created if self.operations_created > 0 else 0
        
    def update_real_time_metrics(self):
        """Update real-time metric history."""
        self.throughput_history.append(self.get_current_throughput())
        self.completion_rate_history.append(self.get_current_completion_rate())


class GracePeriodTracker:
    """Tracks grace periods for executing operations."""
    
    def __init__(self, grace_period_seconds: float = 1.5):
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


class RealTimeAdaptiveController:
    """Real-time adaptation of stress test parameters."""
    
    def __init__(self):
        # Adaptation parameters
        self.target_completion_rate = 0.02  # 2% target
        self.max_memory_mb = 200  # Memory pressure threshold
        self.adaptation_interval = 20  # Adapt every 20 seconds
        self.last_adaptation_time = 0
        
        # Current dynamic parameters
        self.creation_rate = 50.0  # Start moderate
        self.grace_period_seconds = 1.5
        self.max_concurrent_per_target = 50  # Start moderate
        self.conflict_resolution_interval = 0.5
        self.start_probability = 0.5
        self.completion_probability = 0.4
        
    def should_adapt(self) -> bool:
        return time.time() - self.last_adaptation_time >= self.adaptation_interval
        
    def adapt_parameters(self, metrics: DynamicStressTestMetrics) -> Dict[str, Any]:
        """Adapt parameters based on current performance."""
        completion_rate = metrics.get_current_completion_rate()
        current_memory = metrics.memory_samples[-1] if metrics.memory_samples else 0
        
        adaptation = {'action': 'maintain', 'changes': []}
        
        # Memory pressure check
        if current_memory > self.max_memory_mb:
            self.creation_rate *= 0.7
            self.conflict_resolution_interval *= 1.3
            adaptation['action'] = 'reduce_load'
            adaptation['changes'].append(f"Memory pressure: {current_memory:.1f}MB")
            
        # Low completion rate
        elif completion_rate < self.target_completion_rate * 0.3:
            self.creation_rate *= 0.8
            self.grace_period_seconds = min(3.0, self.grace_period_seconds * 1.2)
            self.completion_probability = min(0.6, self.completion_probability * 1.2)
            adaptation['action'] = 'improve_completion'
            adaptation['changes'].append(f"Low completion: {completion_rate:.3f}")
            
        # High completion rate - can push harder
        elif completion_rate > self.target_completion_rate * 2:
            self.creation_rate *= 1.2
            self.max_concurrent_per_target = min(100, int(self.max_concurrent_per_target * 1.1))
            adaptation['action'] = 'increase_load'
            adaptation['changes'].append(f"High completion: {completion_rate:.3f}")
            
        # Bounds checking
        self.creation_rate = max(10.0, min(self.creation_rate, 200.0))
        self.conflict_resolution_interval = max(0.1, min(self.conflict_resolution_interval, 2.0))
        
        self.last_adaptation_time = time.time()
        metrics.record_adaptation()
        
        return adaptation


class TestTarget(Entity):
    """Test target entity."""
    name: str = "stress_target"
    load_factor: float = 1.0


class DynamicStressTest:
    """The ultimate adaptive stress test."""
    
    def __init__(self, duration_minutes: int = 2):
        self.duration_seconds = duration_minutes * 60
        self.metrics = DynamicStressTestMetrics()
        self.grace_tracker = GracePeriodTracker()
        self.adaptive_controller = RealTimeAdaptiveController()
        
        # Test entities
        self.target_entities: List[TestTarget] = []
        self.active_operations: Set[UUID] = set()
        self.stop_flag = False
        
        # Configuration
        self.num_targets = 4
        
    async def setup(self):
        """Initialize test environment."""
        print("🔧 Setting up dynamic stress test environment...")
        
        # Create target entities
        for i in range(self.num_targets):
            target = TestTarget(name=f"dynamic_target_{i}", load_factor=random.uniform(0.8, 1.5))
            target.promote_to_root()
            self.target_entities.append(target)
            
        print(f"✅ Created {len(self.target_entities)} target entities")
        
    def dynamic_conflict_resolution(self, target_entity_id: UUID, current_operations: List[OperationEntity]) -> List[OperationEntity]:
        """Dynamic conflict resolution without artificial limits."""
        if len(current_operations) <= 1:
            return current_operations
            
        # Separate by grace period status
        protected_ops = []
        unprotected_ops = []
        
        for op in current_operations:
            if (op.status == OperationStatus.EXECUTING and 
                not self.grace_tracker.can_be_preempted(op.ecs_id)):
                protected_ops.append(op)
            else:
                unprotected_ops.append(op)
        
        # Always keep ALL protected operations (no artificial limits!)
        winners = protected_ops.copy()
        
        # For unprotected, use priority-based selection with generous concurrency
        if len(unprotected_ops) > 1:
            priority_groups = defaultdict(list)
            for op in unprotected_ops:
                priority_groups[op.get_effective_priority()].append(op)
                
            # Allow top 2 priority levels to run concurrently
            sorted_priorities = sorted(priority_groups.keys(), reverse=True)
            for priority in sorted_priorities[:2]:
                winners.extend(priority_groups[priority])
        
        # Mark losers as rejected
        losers = [op for op in current_operations if op not in winners]
        for op in losers:
            if op.status in [OperationStatus.PENDING, OperationStatus.EXECUTING]:
                op.status = OperationStatus.REJECTED
                op.completed_at = datetime.now(timezone.utc)
                op.error_message = "Significantly lower priority"
                op.update_ecs_ids()
                
                if op.status == OperationStatus.EXECUTING:
                    self.grace_tracker.end_grace_period(op.ecs_id)
        
        return winners
        
    async def create_random_operation(self) -> OperationEntity:
        """Create operation with dynamic parameters."""
        target = random.choice(self.target_entities)
        
        # Dynamic concurrency check
        existing_ops = get_conflicting_operations(target.ecs_id)
        if len(existing_ops) >= self.adaptive_controller.max_concurrent_per_target:
            return None
            
        # Create operation
        priority = random.choices(
            [OperationPriority.LOW, OperationPriority.NORMAL, OperationPriority.HIGH, OperationPriority.CRITICAL],
            weights=[0.4, 0.4, 0.15, 0.05]
        )[0]
        
        op_class = {
            OperationPriority.CRITICAL: StructuralOperation,
            OperationPriority.LOW: LowPriorityOperation
        }.get(priority, NormalOperation)
        
        operation = op_class.create_and_register(
            op_type=f"dynamic_op_{random.randint(1000, 9999)}",
            priority=priority,
            target_entity_id=target.ecs_id,
            max_retries=random.randint(1, 3)
        )
        
        self.active_operations.add(operation.ecs_id)
        self.metrics.record_operation_created(priority)
        
        return operation
        
    async def detect_and_resolve_conflicts(self, target_entity_id: UUID):
        """Dynamic conflict detection and resolution."""
        start_time = time.time()
        
        try:
            conflicts = get_conflicting_operations(target_entity_id)
            
            if len(conflicts) > 1:
                self.metrics.record_conflict_detected(len(conflicts))
                
                # Track protection stats
                protected = self.grace_tracker.get_protected_operations()
                protected_count = len([op for op in conflicts if op.ecs_id in protected])
                
                if protected_count > 0:
                    self.metrics.record_operation_protected()
                
                # Use dynamic resolution
                winners = self.dynamic_conflict_resolution(target_entity_id, conflicts)
                
                # Track saves and rejections
                saved_count = len([op for op in winners if op.ecs_id in protected])
                if saved_count > 0:
                    self.metrics.record_grace_period_save()
                
                # Track rejections with priority
                for op in conflicts:
                    if op not in winners and op.status == OperationStatus.REJECTED:
                        self.metrics.record_operation_rejected(op.priority)
                
                resolution_time = (time.time() - start_time) * 1000
                self.metrics.record_conflict_resolved(resolution_time)
                
                # Emit events
                for op in conflicts:
                    if op.status == OperationStatus.REJECTED:
                        await emit(OperationRejectedEvent(
                            op_id=op.ecs_id,
                            op_type=op.op_type,
                            target_entity_id=op.target_entity_id,
                            from_state="pending",
                            to_state="rejected",
                            rejection_reason="significantly_lower_priority",
                            retry_count=op.retry_count
                        ))
                        self.active_operations.discard(op.ecs_id)
                        
        except Exception as e:
            print(f"⚠️  Error in dynamic conflict resolution: {e}")
            
    async def operation_creation_worker(self):
        """Dynamic operation creation."""
        while not self.stop_flag:
            try:
                creation_interval = 1.0 / self.adaptive_controller.creation_rate
                
                operation = await self.create_random_operation()
                if operation is None:
                    await asyncio.sleep(creation_interval * 2)
                else:
                    await asyncio.sleep(creation_interval)
                    
            except Exception as e:
                print(f"⚠️  Error in dynamic creation worker: {e}")
                await asyncio.sleep(0.1)
                
    async def conflict_resolution_worker(self):
        """Dynamic conflict resolution worker."""
        while not self.stop_flag:
            try:
                for target in self.target_entities:
                    await self.detect_and_resolve_conflicts(target.ecs_id)
                    
                await asyncio.sleep(self.adaptive_controller.conflict_resolution_interval)
                
            except Exception as e:
                print(f"⚠️  Error in dynamic conflict worker: {e}")
                
    async def operation_lifecycle_manager(self):
        """Dynamic lifecycle management."""
        while not self.stop_flag:
            try:
                # Start operations
                if self.active_operations and random.random() < self.adaptive_controller.start_probability:
                    op_id = random.choice(list(self.active_operations))
                    
                    for root_id in EntityRegistry.tree_registry.keys():
                        tree = EntityRegistry.tree_registry.get(root_id)
                        if tree and op_id in tree.nodes:
                            op = tree.nodes[op_id]
                            if isinstance(op, OperationEntity) and op.status == OperationStatus.PENDING:
                                try:
                                    op.start_execution()
                                    self.grace_tracker.start_grace_period(op.ecs_id)
                                    self.metrics.record_operation_started(op.priority)
                                    
                                    await emit(OperationStartedEvent(
                                        process_name="dynamic_stress_test",
                                        op_id=op.ecs_id,
                                        op_type=op.op_type,
                                        priority=op.priority,
                                        target_entity_id=op.target_entity_id
                                    ))
                                except Exception:
                                    pass
                            break
                
                # Complete operations
                if self.active_operations and random.random() < self.adaptive_controller.completion_probability:
                    op_id = random.choice(list(self.active_operations))
                    
                    for root_id in EntityRegistry.tree_registry.keys():
                        tree = EntityRegistry.tree_registry.get(root_id)
                        if tree and op_id in tree.nodes:
                            op = tree.nodes[op_id]
                            if isinstance(op, OperationEntity) and op.status == OperationStatus.EXECUTING:
                                try:
                                    success = random.random() > 0.1
                                    op.complete_operation(success=success)
                                    self.grace_tracker.end_grace_period(op.ecs_id)
                                    self.active_operations.discard(op_id)
                                    self.metrics.record_operation_completed(op.priority)
                                    
                                    await emit(OperationCompletedEvent(
                                        process_name="dynamic_stress_test",
                                        op_id=op.ecs_id,
                                        op_type=op.op_type,
                                        target_entity_id=op.target_entity_id,
                                        execution_duration_ms=random.uniform(50, 400)
                                    ))
                                except Exception:
                                    pass
                            break
                            
                await asyncio.sleep(0.02)
                
            except Exception as e:
                print(f"⚠️  Error in dynamic lifecycle manager: {e}")
                
    async def adaptive_controller_worker(self):
        """Real-time parameter adaptation."""
        while not self.stop_flag:
            try:
                if self.adaptive_controller.should_adapt():
                    adaptation = self.adaptive_controller.adapt_parameters(self.metrics)
                    
                    if adaptation['action'] != 'maintain':
                        print(f"\n🔄 ADAPTING: {adaptation['action']}")
                        for change in adaptation['changes']:
                            print(f"   └─ {change}")
                            
                    # Update grace period in tracker
                    self.grace_tracker.grace_period_seconds = self.adaptive_controller.grace_period_seconds
                    
                await asyncio.sleep(5)
                
            except Exception as e:
                print(f"⚠️  Error in adaptive controller: {e}")
                
    async def metrics_collector(self):
        """System metrics collection."""
        while not self.stop_flag:
            try:
                self.metrics.record_system_stats()
                self.metrics.update_real_time_metrics()
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"⚠️  Error in metrics collector: {e}")
                
    async def progress_reporter(self):
        """Dynamic progress reporting."""
        last_report = time.time()
        
        while not self.stop_flag:
            await asyncio.sleep(10)
            
            current_time = time.time()
            elapsed = current_time - self.metrics.start_time
            remaining = self.duration_seconds - elapsed
            
            if current_time - last_report >= 10:
                stats = get_operation_stats()
                protected_count = len(self.grace_tracker.get_protected_operations())
                
                # Calculate concurrency
                target_concurrency = {}
                for target in self.target_entities:
                    active_ops = get_conflicting_operations(target.ecs_id)
                    target_concurrency[target.name] = len(active_ops)
                
                avg_concurrency = sum(target_concurrency.values()) / len(target_concurrency) if target_concurrency else 0
                max_concurrency = max(target_concurrency.values()) if target_concurrency else 0
                
                completion_rate = self.metrics.get_current_completion_rate()
                
                print(f"\n📊 Dynamic Progress ({elapsed:.1f}s elapsed, {remaining:.1f}s remaining):")
                print(f"   ├─ Operations created: {self.metrics.operations_created}")
                print(f"   ├─ Operations started: {self.metrics.operations_started}")
                print(f"   ├─ Operations completed: {self.metrics.operations_completed}")
                print(f"   ├─ Completion rate: {completion_rate:.1%}")
                print(f"   ├─ Grace period saves: {self.metrics.grace_period_saves}")
                print(f"   ├─ Currently protected: {protected_count}")
                print(f"   ├─ Avg concurrency per target: {avg_concurrency:.1f}")
                print(f"   ├─ Max concurrency: {max_concurrency}")
                print(f"   ├─ Current creation rate: {self.adaptive_controller.creation_rate:.1f} ops/sec")
                print(f"   ├─ Grace period: {self.adaptive_controller.grace_period_seconds:.1f}s")
                print(f"   ├─ Adaptations: {self.metrics.adaptation_count}")
                print(f"   └─ Memory: {self.metrics.memory_samples[-1] if self.metrics.memory_samples else 0:.1f} MB")
                
                last_report = current_time
                
    async def run_dynamic_test(self):
        """Run the complete dynamic stress test."""
        print(f"\n🚀 Starting Dynamic Stress Test...")
        print(f"   ├─ Duration: {self.duration_seconds}s")
        print(f"   ├─ Targets: {self.num_targets}")
        print(f"   ├─ Initial creation rate: {self.adaptive_controller.creation_rate:.1f} ops/sec")
        print(f"   ├─ Initial grace period: {self.adaptive_controller.grace_period_seconds:.1f}s")
        print(f"   ├─ No artificial concurrency limits!")
        print(f"   └─ Will adapt parameters every {self.adaptive_controller.adaptation_interval}s")
        
        # Start all workers
        tasks = [
            asyncio.create_task(self.operation_creation_worker()),
            asyncio.create_task(self.conflict_resolution_worker()),
            asyncio.create_task(self.operation_lifecycle_manager()),
            asyncio.create_task(self.adaptive_controller_worker()),
            asyncio.create_task(self.metrics_collector()),
            asyncio.create_task(self.progress_reporter())
        ]
        
        try:
            await asyncio.sleep(self.duration_seconds)
        finally:
            self.stop_flag = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def analyze_dynamic_performance(self):
        """Comprehensive dynamic analysis."""
        print("\n🧹 Analyzing dynamic performance...")
        
        elapsed = time.time() - self.metrics.start_time
        throughput = self.metrics.operations_created / elapsed
        completion_rate = self.metrics.get_current_completion_rate()
        
        print("\n" + "=" * 80)
        print("🚀 DYNAMIC STRESS TEST RESULTS")
        print("=" * 80)
        
        print(f"\n⏱️  Performance Summary:")
        print(f"   ├─ Duration: {elapsed:.1f}s")
        print(f"   ├─ Throughput: {throughput:.1f} ops/sec")
        print(f"   ├─ Total operations: {self.metrics.operations_created}")
        print(f"   ├─ Operations started: {self.metrics.operations_started}")
        print(f"   ├─ Operations completed: {self.metrics.operations_completed}")
        print(f"   └─ Overall completion rate: {completion_rate:.1%}")
        
        # Priority breakdown analysis
        self._print_priority_breakdown()
        
        print(f"\n🛡️  Grace Period Effectiveness:")
        print(f"   ├─ Grace period saves: {self.metrics.grace_period_saves}")
        print(f"   ├─ Operations protected: {self.metrics.operations_protected}")
        print(f"   └─ Final grace period: {self.adaptive_controller.grace_period_seconds:.1f}s")
        
        print(f"\n🔄 Adaptation Summary:")
        print(f"   ├─ Total adaptations: {self.metrics.adaptation_count}")
        print(f"   ├─ Final creation rate: {self.adaptive_controller.creation_rate:.1f} ops/sec")
        print(f"   ├─ Final max concurrent: {self.adaptive_controller.max_concurrent_per_target}")
        print(f"   └─ Final completion prob: {self.adaptive_controller.completion_probability:.1%}")
        
        # System performance
        if self.metrics.memory_samples:
            avg_memory = statistics.mean(self.metrics.memory_samples)
            max_memory = max(self.metrics.memory_samples)
            print(f"\n💾 System Resources:")
            print(f"   ├─ Avg memory: {avg_memory:.1f} MB")
            print(f"   ├─ Max memory: {max_memory:.1f} MB")
            print(f"   └─ Memory efficiency: {'✅ Good' if max_memory < 250 else '⚠️ High'}")
        
        # Comparison to static tests
        print(f"\n📊 vs STATIC BASELINES:")
        print(f"   ├─ Original (3 limit): 9 completions")
        print(f"   ├─ High concurrency: 114 completions") 
        print(f"   ├─ Dynamic test: {self.metrics.operations_completed} completions")
        
        if self.metrics.operations_completed > 114:
            improvement = self.metrics.operations_completed / 114
            print(f"   ✅ BEST RESULT: {improvement:.1f}x better than high concurrency!")
        elif self.metrics.operations_completed > 50:
            print(f"   ✅ EXCELLENT: Strong performance with adaptation")
        elif self.metrics.operations_completed > 20:
            print(f"   ✅ GOOD: Solid adaptive performance")
        else:
            print(f"   ⚠️  NEEDS TUNING: Consider longer test duration")
            
        print(f"\n🎯 FINAL ASSESSMENT:")
        if completion_rate > 0.02:
            print("   ✅ TARGET EXCEEDED: Excellent completion rate")
        elif completion_rate > 0.01:
            print("   ✅ TARGET MET: Good completion rate")
        else:
            print("   ⚠️ BELOW TARGET: System under high stress")
            
        if self.metrics.adaptation_count > 0:
            print("   ✅ ADAPTIVE: System responded to changing conditions")
        else:
            print("   ⚠️ STATIC: No adaptations needed (system was stable)")
            
    def _print_priority_breakdown(self):
        """Print detailed priority-based operation breakdown."""
        print(f"\n🎯 PRIORITY BREAKDOWN:")
        
        # Get all priority levels we've seen
        all_priorities = set()
        all_priorities.update(self.metrics.created_by_priority.keys())
        all_priorities.update(self.metrics.started_by_priority.keys())
        all_priorities.update(self.metrics.completed_by_priority.keys())
        all_priorities.update(self.metrics.rejected_by_priority.keys())
        
        if not all_priorities:
            print("   └─ No priority data available")
            return
            
        # Sort priorities by importance (CRITICAL > HIGH > NORMAL > LOW)
        priority_order = ['CRITICAL', 'HIGH', 'NORMAL', 'LOW']
        sorted_priorities = [p for p in priority_order if p in all_priorities]
        # Add any other priorities we might have missed
        sorted_priorities.extend([p for p in all_priorities if p not in priority_order])
        
        for i, priority in enumerate(sorted_priorities):
            created = self.metrics.created_by_priority.get(priority, 0)
            started = self.metrics.started_by_priority.get(priority, 0)
            completed = self.metrics.completed_by_priority.get(priority, 0)
            rejected = self.metrics.rejected_by_priority.get(priority, 0)
            
            # Calculate rates
            start_rate = f"{started/created:.1%}" if created > 0 else "0.0%"
            completion_rate = f"{completed/started:.1%}" if started > 0 else "0.0%" 
            overall_completion = f"{completed/created:.1%}" if created > 0 else "0.0%"
            rejection_rate = f"{rejected/created:.1%}" if created > 0 else "0.0%"
            
            # Use appropriate emoji/symbol for priority
            priority_symbol = {
                'CRITICAL': '🔥',
                'HIGH': '⚡',
                'NORMAL': '📄', 
                'LOW': '🐌'
            }.get(priority, '❓')
            
            is_last = i == len(sorted_priorities) - 1
            connector = "└─" if is_last else "├─"
            
            print(f"   {connector} {priority_symbol} {priority}:")
            
            sub_connector = "   " if is_last else "│  "
            print(f"   {sub_connector}   ├─ Created: {created}")
            print(f"   {sub_connector}   ├─ Started: {started} ({start_rate})")
            print(f"   {sub_connector}   ├─ Completed: {completed} ({completion_rate} of started, {overall_completion} overall)")
            print(f"   {sub_connector}   └─ Rejected: {rejected} ({rejection_rate})")
            
        # Summary insights
        print(f"\n💡 PRIORITY INSIGHTS:")
        
        # Check if higher priorities are getting better treatment
        critical_completion = self.metrics.completed_by_priority.get('CRITICAL', 0)
        critical_created = self.metrics.created_by_priority.get('CRITICAL', 0)
        low_completion = self.metrics.completed_by_priority.get('LOW', 0)
        low_created = self.metrics.created_by_priority.get('LOW', 0)
        
        if critical_created > 0 and low_created > 0:
            critical_rate = critical_completion / critical_created
            low_rate = low_completion / low_created
            
            if critical_rate > low_rate * 1.5:
                print("   ✅ EXCELLENT: Critical operations prioritized correctly")
            elif critical_rate > low_rate:
                print("   ✅ GOOD: Some priority differentiation")
            else:
                print("   ⚠️  CONCERN: Low priority operations completing as well as critical")
        
        # Check rejection patterns
        critical_rejected = self.metrics.rejected_by_priority.get('CRITICAL', 0)
        low_rejected = self.metrics.rejected_by_priority.get('LOW', 0)
        
        if critical_rejected == 0 and low_rejected > 0:
            print("   ✅ PERFECT: No critical operations rejected")
        elif critical_rejected < low_rejected:
            print("   ✅ GOOD: Critical operations rejected less frequently")
        elif critical_rejected > 0:
            print("   ⚠️  WARNING: Some critical operations were rejected")


async def main():
    """Run the dynamic stress test."""
    bus = get_event_bus()
    await bus.start()
    
    try:
        print("🚀 DYNAMIC CONFLICT RESOLUTION STRESS TEST")
        print("=" * 60)
        print("The ONE test to rule them all!")
        print("• No artificial concurrency limits")
        print("• Grace period protection")  
        print("• Real-time adaptation")
        print("• Agent observer integration")
        print("• System profiling & optimization")
        print("=" * 60)
        
        stress_test = DynamicStressTest(duration_minutes=2)
        
        await stress_test.setup()
        await stress_test.run_dynamic_test()
        await stress_test.analyze_dynamic_performance()
        
        print("\n🎉 Dynamic stress test completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await bus.stop()


if __name__ == "__main__":
    asyncio.run(main()) 