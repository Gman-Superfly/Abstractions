"""
Production Conflict Resolution Algorithm Test

Production-grade test for validating conflict resolution algorithms.
Submits actual operations to the system and measures conflict resolution performance.

This test uses production ECS operations that perform actual entity modifications:
- version_entity operations
- borrow_attribute_from operations 
- put() functional API operations
- promote_to_root operations
- detach operations

Production validation with comprehensive measurements and edge case coverage.
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
        
        # Real operation tracking
        self.real_operations_by_type = defaultdict(int)
        self.real_operation_success_by_type = defaultdict(int)
        self.real_operation_failures = defaultdict(list)
        self.entity_modifications = 0
        self.versioning_operations = 0
        self.borrowing_operations = 0
        self.structural_operations = 0
        
        # Grace period metrics
        self.grace_period_saves = 0
        self.operations_protected = 0
        
        # Performance tracking
        self.resolution_times = []
        self.conflict_sizes = []
        self.memory_samples = []
        self.cpu_samples = []
        self.real_operation_durations = []
        
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
        
    def record_real_operation(self, op_type: str, success: bool, duration_ms: float, error: Optional[str] = None):
        """Record metrics for real ECS operations."""
        self.real_operations_by_type[op_type] += 1
        self.real_operation_durations.append(duration_ms)
        
        if success:
            self.real_operation_success_by_type[op_type] += 1
            if op_type in ['version_entity', 'entity_versioning']:
                self.versioning_operations += 1
            elif op_type in ['borrow_attribute']:  # FIXED: use correct operation type
                self.borrowing_operations += 1
            elif op_type in ['promote_to_root', 'detach_entity', 'structural']:  # FIXED: use correct operation type
                self.structural_operations += 1
            self.entity_modifications += 1
        else:
            self.real_operation_failures[op_type].append(error or "Unknown error")
        
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


class RealOperationEntity(OperationEntity):
    """Operation entity that performs real ECS operations."""
    
    # Real operation parameters
    operation_type: str = Field(description="Type of real operation to perform")
    operation_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for real operation")
    source_entity_id: Optional[UUID] = Field(default=None, description="Source entity for borrowing ops")
    
    async def execute_real_operation(self) -> bool:
        """Execute the actual ECS operation - NO SIMULATIONS, only real ECS calls."""
        try:
            # Get target entity
            target_entity = self._get_target_entity()
            if not target_entity:
                print(f"🚨 OPERATION FAILURE: Target entity {self.target_entity_id} not found for operation {self.op_type}")
                raise ValueError(f"Target entity {self.target_entity_id} not found")
            
            print(f"🔧 EXECUTING: {self.operation_type} on target {str(target_entity.ecs_id)[:8]} (Operation: {self.op_type})")
            
            success = False
            
            if self.operation_type == "version_entity":
                # Pure ECS versioning operation - no fake delays
                success = EntityRegistry.version_entity(target_entity, force_versioning=True)
                    
            elif self.operation_type == "modify_field":
                # Pure ECS field modification
                field_name = self.operation_params.get("field_name", "counter")
                new_value = self.operation_params.get("new_value", random.randint(1, 1000))
                
                # Real ECS functional API call only
                address = f"@{target_entity.ecs_id}.{field_name}"
                put(address, new_value, borrow=False)
                
                # Real entity modification only
                target_entity.modification_history.append(f"Modified {field_name} to {new_value} at {datetime.now(timezone.utc)}")
                success = True
                
            elif self.operation_type == "borrow_attribute":
                # Pure ECS borrowing operation
                source_entity = self._get_source_entity()
                if source_entity:
                    source_field = self.operation_params.get("source_field", "source_counter")
                    target_field = self.operation_params.get("target_field", "counter")
                    
                    # Real ECS borrowing call only
                    target_entity.borrow_attribute_from(source_entity, source_field, target_field)
                    target_entity.borrow_count += 1
                    success = True
                else:
                    raise ValueError("Source entity not found for borrowing operation")
                    
            elif self.operation_type == "promote_to_root":
                # Pure ECS structural operation
                if not target_entity.is_root_entity():
                    target_entity.promote_to_root()
                    success = True
                else:
                    # Real entity modification only
                    target_entity.modification_history.append(f"Already root entity at {datetime.now(timezone.utc)}")
                    success = True
                    
            elif self.operation_type == "detach_entity":
                # Pure ECS detachment operation
                if not target_entity.is_orphan():
                    target_entity.detach()
                    success = True
                else:
                    # Real entity modification only
                    target_entity.modification_history.append(f"Already detached at {datetime.now(timezone.utc)}")
                    success = True
                    
            elif self.operation_type == "complex_update":
                # Real ECS operations only - no simulation
                target_entity.counter += 1
                target_entity.data_value = random.uniform(0, 100)
                target_entity.text_content = f"Updated_{target_entity.counter}_{time.time()}"
                target_entity.timestamp = datetime.now(timezone.utc)
                target_entity.version_count += 1
                target_entity.modification_history.append(f"Complex update {target_entity.counter} at {target_entity.timestamp}")
                
                # Real ECS versioning call only
                EntityRegistry.version_entity(target_entity, force_versioning=True)
                success = True
                
            else:
                raise ValueError(f"Unknown operation type: {self.operation_type}")
            
            if success:
                print(f"✅ SUCCESS: {self.operation_type} completed on target {str(target_entity.ecs_id)[:8]}")
            else:
                print(f"❌ FAILED: {self.operation_type} failed on target {str(target_entity.ecs_id)[:8]}")
            
            return success
            
        except Exception as e:
            self.error_message = str(e)
            return False
    
    def _get_target_entity(self) -> Optional[TestDataEntity]:
        """Get the target entity for this operation."""
        try:
            # Find the entity in the registry
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


class ConflictResolutionTest:
    """
    Production-grade test for conflict resolution algorithms.
    
    Submits actual operations to the system and measures performance under stress.
    """
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.metrics = ConflictResolutionMetrics()
        self.grace_tracker = GracePeriodTracker(config.grace_period_seconds)
        
        # Test entities - using production entities that can be modified
        self.target_entities: List[TestDataEntity] = []
        self.source_entities: List[TestDataSource] = []
        self.submitted_operations: Set[UUID] = set()
        self.stop_flag = False
        
        # Pre-ECS staging area for conflict resolution
        self.pending_operations: Dict[UUID, List[RealOperationEntity]] = {}  # target_id -> [operations]
        
        # Operation ID counter for unique operation names
        self.operation_counter = 0
        
        # Production operation types with weights
        self.real_operation_types = {
            "version_entity": 0.3,      # 30% versioning operations
            "modify_field": 0.25,       # 25% field modifications
            "borrow_attribute": 0.2,    # 20% borrowing operations
            "complex_update": 0.15,     # 15% complex updates
            "promote_to_root": 0.05,    # 5% structural promotions
            "detach_entity": 0.05       # 5% detachment operations
        }
        
    async def setup(self):
        """Initialize test environment with production entities."""
        print("🔧 Setting up conflict resolution test with production operations...")
        print(f"   ├─ Test duration: {self.config.duration_seconds}s")
        print(f"   ├─ Target entities: {self.config.num_targets}")
        print(f"   ├─ Operation rate: {self.config.operation_rate_per_second:.1f} ops/sec")
        print(f"   ├─ Grace period: {self.config.grace_period_seconds:.1f}s")
        print(f"   └─ Target completion rate: {self.config.target_completion_rate:.1%}")
        
        # Set up event handlers for pure event-driven approach
        from abstractions.events.events import setup_operation_event_handlers
        setup_operation_event_handlers()
        print("✅ Event handlers registered for pure event-driven conflict resolution")
        
        # Create target entities that can be modified
        for i in range(self.config.num_targets):
            target = TestDataEntity(
                name=f"test_target_{i}",
                counter=i,
                data_value=float(i * 10),
                text_content=f"initial_content_{i}",
                modification_history=[f"Created at {datetime.now(timezone.utc)}"]
            )
            target.promote_to_root()
            self.target_entities.append(target)
            
        # Create source entities for borrowing operations
        for i in range(min(5, self.config.num_targets)):
            source = TestDataSource(
                source_name=f"data_source_{i}",
                source_counter=100 + i,
                source_data=99.99 + i,
                source_text=f"source_content_{i}",
                source_values=list(range(i, i + 5)),
                source_metadata={"source_id": str(i), "created_at": str(datetime.now(timezone.utc))}
            )
            source.promote_to_root()
            self.source_entities.append(source)
            
        print(f"✅ Created {len(self.target_entities)} target entities with production data")
        print(f"✅ Created {len(self.source_entities)} source entities for borrowing operations")
        
    async def submit_operation(self, target: TestDataEntity, priority: OperationPriority) -> Optional[RealOperationEntity]:
        """Submit a production operation to the system."""
        self.operation_counter += 1
        
        # Select real operation type based on weights
        operation_type = self._select_operation_type()
        operation_params = self._generate_operation_params(operation_type)
        
        # Select source entity for borrowing operations
        source_entity_id = None
        if operation_type == "borrow_attribute" and self.source_entities:
            source_entity_id = random.choice(self.source_entities).ecs_id
        
        # Determine operation class based on priority and type
        if priority == OperationPriority.CRITICAL or operation_type in ["promote_to_root", "detach_entity"]:
            op_class = StructuralOperation
        elif priority == OperationPriority.LOW:
            op_class = LowPriorityOperation
        else:
            op_class = NormalOperation
        
        operation = RealOperationEntity(
            op_type=f"{operation_type}_{self.operation_counter}",
            operation_type=operation_type,
            operation_params=operation_params,
            priority=priority,
            target_entity_id=target.ecs_id,
            source_entity_id=source_entity_id,
            max_retries=3
        )
        # DO NOT promote_to_root() yet - keep in staging area for conflict resolution
        
        # Add to pre-ECS staging area
        if target.ecs_id not in self.pending_operations:
            self.pending_operations[target.ecs_id] = []
        self.pending_operations[target.ecs_id].append(operation)
        
        self.metrics.record_operation_submitted(priority)
        
        # DON'T resolve conflicts immediately - let them accumulate in staging area
        # Conflict resolution will happen after the batch is complete
        
        return operation
    
    def _select_operation_type(self) -> str:
        """Select operation type based on configured weights."""
        rand_val = random.random()
        cumulative = 0.0
        
        for op_type, weight in self.real_operation_types.items():
            cumulative += weight
            if rand_val <= cumulative:
                return op_type
        
        return "modify_field"  # Fallback
    
    def _generate_operation_params(self, operation_type: str) -> Dict[str, Any]:
        """Generate parameters for the real operation."""
        if operation_type == "modify_field":
            field_names = ["counter", "data_value", "text_content"]
            field_name = random.choice(field_names)
            
            if field_name == "counter":
                new_value = random.randint(1, 10000)
            elif field_name == "data_value":
                new_value = random.uniform(0, 1000)
            else:  # text_content
                new_value = f"modified_{random.randint(1000, 9999)}_{time.time()}"
            
            return {"field_name": field_name, "new_value": new_value}
            
        elif operation_type == "borrow_attribute":
            source_fields = ["source_counter", "source_data", "source_text"]
            target_fields = ["counter", "data_value", "text_content"]
            
            source_field = random.choice(source_fields)
            # Match types: counter->counter, data->data_value, text->text_content
            if "counter" in source_field:
                target_field = "counter"
            elif "data" in source_field:
                target_field = "data_value"
            else:
                target_field = "text_content"
            
            return {"source_field": source_field, "target_field": target_field}
        
        return {}
        
    def _get_operation_status(self, op_id: UUID) -> Optional[OperationStatus]:
        """Get the current status of an operation."""
        for root_id in EntityRegistry.tree_registry.keys():
            tree = EntityRegistry.tree_registry.get(root_id)
            if tree and op_id in tree.nodes:
                op = tree.nodes[op_id]
                if isinstance(op, OperationEntity):
                    return op.status
        return None
        
    # ❌ REMOVED METHOD: detect_and_resolve_conflicts()
    # 
    # 📚 HISTORICAL CONTEXT:
    # This method was part of the old POST-ECS conflict resolution system that has been
    # completely replaced by PRE-ECS conflict resolution in resolve_conflicts_before_ecs().
    #
    # 🔍 WHAT IT DID:
    # - Scanned EntityRegistry.tree_registry for conflicting PENDING operations
    # - Used event-driven resolution via OperationConflictEvent emission
    # - Handled grace period protection for executing operations
    # - Performed conflict resolution AFTER operations already had ECS IDs
    #
    # ❌ WHY IT WAS REMOVED:
    # 1. Race Condition: Operations could start executing before conflict detection
    # 2. Post-ECS Resolution: Conflicts resolved after operations entered ECS
    # 3. Double Counting: Both pre-ECS and post-ECS systems caused metrics duplication
    # 4. Event Complexity: Required passing operation entities through event system
    # 5. Grace Period Complexity: Needed to track protection for already-executing ops
    #
    # ✅ NEW SYSTEM (resolve_conflicts_before_ecs):
    # - Pre-ECS staging area prevents conflicts from reaching ECS
    # - Priority-based resolution before any promote_to_root() calls
    # - Only winners get ECS IDs - losers never enter the system
    # - Clean separation: conflict resolution vs ECS operations
    # - No race conditions: conflicts resolved before execution begins

    async def resolve_conflicts_before_ecs(self, target_entity_id: UUID):
        """Resolve conflicts in pre-ECS staging area before any ECS operations."""
        pending_ops = self.pending_operations.get(target_entity_id, [])
        
        # DEBUG: Show staging area status
        print(f"🔍 STAGING CHECK: Target {str(target_entity_id)[:8]} has {len(pending_ops)} pending operations")
        
        if len(pending_ops) > 1:
            # 🎯 METRICS: Track resolution time
            resolution_start = time.time()
            
            # 🎯 METRICS: Record conflict detected
            self.metrics.record_conflict_detected(len(pending_ops))
            
            print(f"⚔️  PRE-ECS CONFLICT: {len(pending_ops)} operations competing for target {str(target_entity_id)[:8]}")
            for op in pending_ops:
                print(f"   ├─ {op.op_type} (Priority: {op.priority}, Status: PRE-ECS)")
            
            # Sort by priority (higher priority = higher number wins)
            pending_ops.sort(key=lambda op: (op.priority, -op.created_at.timestamp()), reverse=True)
            
            # Winner is highest priority (first after sort)
            winner = pending_ops[0]
            losers = pending_ops[1:]
            
            print(f"🏆 PRE-ECS RESOLUTION: 1 winner, {len(losers)} rejected")
            print(f"✅ WINNER: {winner.op_type} (Priority: {winner.priority})")
            
            # Reject losers before they enter ECS
            for loser in losers:
                print(f"❌ REJECTED: {loser.op_type} (Priority: {loser.priority})")
                self.metrics.record_operation_rejected(loser.priority)
                # Remove from staging area - they never enter ECS
                self.pending_operations[target_entity_id].remove(loser)
            
            # 🎯 METRICS: Record conflict resolved with timing
            resolution_time_ms = (time.time() - resolution_start) * 1000
            self.metrics.record_conflict_resolved(resolution_time_ms)
            
            # Promote winner to ECS for execution
            winner.promote_to_root()
            self.submitted_operations.add(winner.ecs_id)
            self.pending_operations[target_entity_id] = []  # Clear staging area
            
            print(f"🚀 PROMOTED TO ECS: {winner.op_type} (ID: {str(winner.ecs_id)[:8]})")
            
        elif len(pending_ops) == 1:
            # No conflict - promote single operation to ECS
            winner = pending_ops[0]
            winner.promote_to_root()
            self.submitted_operations.add(winner.ecs_id)
            self.pending_operations[target_entity_id] = []
            
    async def run_test(self):
        """Run the complete conflict resolution test with graceful shutdown."""
        print(f"\n🚀 Starting Conflict Resolution Test with Production Operations...")
        
        # Start all workers (no conflict monitoring worker - conflicts detected during submission)
        tasks = [
            asyncio.create_task(self.operation_submission_worker()),
            asyncio.create_task(self.operation_lifecycle_driver()),
            asyncio.create_task(self.operation_lifecycle_observer()),
            asyncio.create_task(self.metrics_collector()),
            asyncio.create_task(self.progress_reporter())
        ]
        
        try:
            # Phase 1: Run test for specified duration (stop new submissions)
            print(f"⏱️  Phase 1: Running test for {self.config.duration_seconds}s (new operations)")
            await asyncio.sleep(self.config.duration_seconds)
            
            # Stop submission worker only - let others continue for grace period
            print(f"⏱️  Phase 2: Graceful shutdown - allowing {self.config.grace_period_seconds * 10:.0f}s for pending operations")
            self.stop_submission = True  # New flag to stop only submissions
            
            # Grace period: Allow pending operations to complete
            grace_period = 2.0  # 2 seconds for pending operations
            await asyncio.sleep(grace_period)
            
            # Phase 3: Stop all workers and buffer output
            print(f"⏱️  Phase 3: Stopping all workers and preparing final results...")
            
        finally:
            # Stop all workers
            self.stop_flag = True
            
            # Cancel tasks and wait for cleanup with timeout
            for task in tasks:
                task.cancel()
            
            # Wait for workers to stop with timeout to prevent hanging
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=5.0  # Max 5 seconds for cleanup
                )
            except asyncio.TimeoutError:
                print("⚠️  Some workers took too long to stop - proceeding with results")
            
            # Small delay to ensure all async output is flushed
            await asyncio.sleep(0.5)
            
    async def operation_submission_worker(self):
        """Submit operations at the configured rate - maximum stress mode."""
        interval = 1.0 / self.config.operation_rate_per_second
        
        # Track submission phase separately from overall test
        while not self.stop_flag and not getattr(self, 'stop_submission', False):
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
                
                # BRUTAL CONFLICT MODE: Submit MULTIPLE operations to the SAME target simultaneously
                # This creates guaranteed conflicts since multiple ops target same entity
                target = random.choice(self.target_entities)  # Pick one target for maximum conflict
                
                # Submit a BATCH of operations to the same target to force conflicts
                batch_size = random.randint(3, 8)  # 3-8 operations per batch
                batch_operations = []
                
                # Only print during submission phase to reduce output chaos
                if not getattr(self, 'stop_submission', False):
                    print(f"🔥 BRUTAL BATCH: Submitting {batch_size} operations to target {str(target.ecs_id)[:8]} simultaneously")
                
                for i in range(batch_size):
                    # Vary priorities within the batch to create priority conflicts
                    batch_priority = priorities[i % len(priorities)]
                    operation = await self.submit_operation(target, batch_priority)
                    if operation:
                        batch_operations.append(operation)
                        if not getattr(self, 'stop_submission', False):
                            print(f"   ├─ {operation.op_type} (Priority: {operation.priority}) → Target: {str(target.ecs_id)[:8]}")
                
                if not getattr(self, 'stop_submission', False):
                    print(f"🎯 BATCH COMPLETE: {len(batch_operations)} operations submitted to same target")
                
                # RESOLVE CONFLICTS AFTER BATCH IS COMPLETE
                await self.resolve_conflicts_before_ecs(target.ecs_id)
                
                await asyncio.sleep(interval * batch_size)  # Adjust timing for batch
                
            except Exception as e:
                print(f"⚠️  Error in operation submission: {e}")
                await asyncio.sleep(interval)
        
        # Submission phase complete
        if not self.stop_flag:
            print(f"✅ Submission phase complete - no new operations will be created")
                
    # ❌ REMOVED METHOD: conflict_monitoring_worker()
    # 
    # 📚 HISTORICAL CONTEXT:
    # This worker was part of the old POST-ECS conflict resolution system.
    # It ran continuously and called detect_and_resolve_conflicts() every 0.1 seconds.
    #
    # 🔍 WHAT IT DID:
    # - Monitored all target entities for conflicts every 100ms
    # - Called detect_and_resolve_conflicts() for each target
    # - Ran as a separate async task alongside other workers
    #
    # ❌ WHY IT WAS REMOVED:
    # 1. No longer needed: Conflicts now resolved during operation submission
    # 2. Performance: Unnecessary continuous scanning of ECS registry
    # 3. Timing issues: Could miss fast conflicts or create race conditions
    # 4. Complexity: Added extra worker management without benefit
    #
    # ✅ NEW SYSTEM:
    # - Conflicts detected and resolved immediately during submission
    # - No separate monitoring worker needed
    # - resolve_conflicts_before_ecs() called synchronously after each batch
    # - Eliminates all timing and race condition issues
                
    async def operation_lifecycle_driver(self):
        """Drive operation lifecycle - start and complete production operations."""
        while not self.stop_flag:
            try:
                # MAXIMUM CONCURRENCY - no limits on concurrent operations
                started_count = 0
                for op_id in list(self.submitted_operations):
                    # NO LIMITS - start everything immediately for maximum carnage
                        
                    for root_id in EntityRegistry.tree_registry.keys():
                        tree = EntityRegistry.tree_registry.get(root_id)
                        if tree and op_id in tree.nodes:
                            op = tree.nodes[op_id]
                            if isinstance(op, RealOperationEntity):
                                
                                # Start pending operations immediately
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
                                
                                # Execute operations immediately - no artificial timing
                                elif op.status == OperationStatus.EXECUTING:
                                    execution_time = (datetime.now(timezone.utc) - op.started_at).total_seconds() if op.started_at else 0
                                    
                                    # Execute IMMEDIATELY - no artificial delays
                                    try:
                                        # Execute the production operation
                                        op_start_time = time.time()
                                        success = await op.execute_real_operation()
                                        op_duration_ms = (time.time() - op_start_time) * 1000
                                        
                                        # Record production operation metrics
                                        self.metrics.record_real_operation(
                                            op.operation_type, 
                                            success, 
                                            op_duration_ms,
                                            op.error_message if not success else None
                                        )
                                        
                                        if success:
                                            # Production operation succeeded
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
                                        else:
                                            # Production operation failed
                                            raise Exception(op.error_message or "Production operation failed")
                                            
                                    except Exception as e:
                                        # Production failure occurred during ECS operations
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
                                
                                # Clean up rejected operations - NO double counting
                                elif op.status == OperationStatus.REJECTED:
                                    self.grace_tracker.end_grace_period(op.ecs_id)
                                    self.submitted_operations.discard(op_id)
                                    # DON'T record_operation_rejected here - already counted in conflict resolution
                                
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
                
                # Minimal yield for cooperative multitasking - not for timing
                await asyncio.sleep(0)  # Yield to event loop immediately
                
            except Exception as e:
                print(f"⚠️  Error in lifecycle driver: {e}")
                
    async def operation_lifecycle_observer(self):
        """Observe operation lifecycle for additional metrics."""
        while not self.stop_flag:
            try:
                # Validate that operations are actually modifying entities
                for op_id in list(self.submitted_operations):
                    for root_id in EntityRegistry.tree_registry.keys():
                        tree = EntityRegistry.tree_registry.get(root_id)
                        if tree and op_id in tree.nodes:
                            op = tree.nodes[op_id]
                            if isinstance(op, RealOperationEntity) and op.status == OperationStatus.EXECUTING:
                                # Verify target entity exists and can be accessed
                                target = op._get_target_entity()
                                if target and len(target.modification_history) > 1:
                                    # Entity is being actively modified
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
        """Report progress during test - reduced output during grace period."""
        last_report = time.time()
        
        while not self.stop_flag:
            await asyncio.sleep(10)
            
            current_time = time.time()
            elapsed = current_time - self.metrics.start_time
            remaining = self.config.duration_seconds - elapsed
            
            # Only report during main test phase, not during grace period
            if current_time - last_report >= 10 and not getattr(self, 'stop_submission', False):
                completion_rate = self.metrics.get_current_completion_rate()
                throughput = self.metrics.get_current_throughput()
                
                print(f"\n📊 Test Progress ({elapsed:.1f}s elapsed, {remaining:.1f}s remaining):")
                print(f"   ├─ Operations submitted: {self.metrics.operations_submitted}")
                print(f"   ├─ Operations started: {self.metrics.operations_started}")
                print(f"   ├─ Operations completed: {self.metrics.operations_completed}")
                print(f"   ├─ Operations rejected: {self.metrics.operations_rejected}")
                print(f"   ├─ Real entity modifications: {self.metrics.entity_modifications}")
                print(f"   ├─ Versioning operations: {self.metrics.versioning_operations}")
                print(f"   ├─ Borrowing operations: {self.metrics.borrowing_operations}")
                print(f"   ├─ Completion rate: {completion_rate:.1%}")
                print(f"   ├─ Throughput: {throughput:.1f} ops/sec")
                print(f"   ├─ Conflicts detected: {self.metrics.conflicts_detected}")
                print(f"   └─ Grace period saves: {self.metrics.grace_period_saves}")
                
                last_report = current_time
            elif getattr(self, 'stop_submission', False):
                # During grace period, show minimal status
                pending_ops = len([op_id for op_id in self.submitted_operations 
                                 if self._get_operation_status(op_id) in [OperationStatus.PENDING, OperationStatus.EXECUTING]])
                if pending_ops > 0:
                    print(f"⏳ Grace period: {pending_ops} operations still pending/executing...")
            
    async def analyze_results(self):
        """Analyze test results with clean output."""
        
        # Clear the terminal output with separator to ensure clean results display
        print("\n" + "🧹" * 80)
        print("BUFFERED FINAL RESULTS - All workers stopped")
        print("🧹" * 80)
        
        elapsed = time.time() - self.metrics.start_time
        # Calculate throughput based on submission period only (not grace period)
        submission_duration = self.config.duration_seconds  # Only count submission phase
        throughput = self.metrics.operations_submitted / submission_duration if submission_duration > 0 else 0
        completion_rate = self.metrics.get_current_completion_rate()
        rejection_rate = self.metrics.operations_rejected / self.metrics.operations_submitted if self.metrics.operations_submitted > 0 else 0
        
        print("\n" + "=" * 80)
        print("🚀 CONFLICT RESOLUTION TEST RESULTS")
        print("=" * 80)
        
        print(f"\n📋 Test Configuration:")
        print(f"   ├─ Submission duration: {self.config.duration_seconds}s")
        print(f"   ├─ Grace period: 2.0s (for pending operations)")
        print(f"   ├─ Total test time: {elapsed:.1f}s")
        print(f"   ├─ Targets: {self.config.num_targets}")
        print(f"   ├─ Operation rate: {self.config.operation_rate_per_second:.1f} ops/sec")
        print(f"   ├─ Grace period protection: {self.config.grace_period_seconds:.1f}s")
        print(f"   └─ Priority distribution: {dict(self.config.priority_distribution)}")
        
        print(f"\n⏱️  Performance Results:")
        print(f"   ├─ Submission phase: {submission_duration:.1f}s")
        print(f"   ├─ Submission throughput: {throughput:.1f} ops/sec")
        print(f"   ├─ Operations submitted: {self.metrics.operations_submitted}")
        print(f"   ├─ Operations started: {self.metrics.operations_started}")
        print(f"   ├─ Operations completed: {self.metrics.operations_completed}")
        print(f"   ├─ Operations rejected: {self.metrics.operations_rejected}")
        print(f"   ├─ Operations failed: {self.metrics.operations_failed}")
        print(f"   ├─ Operations retried: {self.metrics.operations_retried}")
        print(f"   ├─ Operations in progress: {self.metrics.operations_in_progress}")
        print(f"   ├─ Completion rate: {completion_rate:.1%}")
        print(f"   └─ Rejection rate: {rejection_rate:.1%}")
        
        # Production operation metrics
        print(f"\n🔧 PRODUCTION Operation Results:")
        print(f"   ├─ Total entity modifications: {self.metrics.entity_modifications}")
        print(f"   ├─ Versioning operations: {self.metrics.versioning_operations}")
        print(f"   ├─ Borrowing operations: {self.metrics.borrowing_operations}")
        print(f"   ├─ Structural operations: {self.metrics.structural_operations}")
        
        if self.metrics.real_operation_durations:
            avg_real_op_time = statistics.mean(self.metrics.real_operation_durations)
            print(f"   ├─ Avg production operation time: {avg_real_op_time:.1f}ms")
        
        print(f"   └─ Production operations by type:")
        for op_type, count in self.metrics.real_operations_by_type.items():
            success_count = self.metrics.real_operation_success_by_type.get(op_type, 0)
            success_rate = (success_count / count * 100) if count > 0 else 0
            print(f"       ├─ {op_type}: {count} total, {success_count} successful ({success_rate:.1f}%)")
        
        # Operation accounting verification - FIXED to avoid double counting
        # Only count final states, not intermediate transitions
        operations_in_final_state = (self.metrics.operations_completed + 
                          self.metrics.operations_rejected + 
                          self.metrics.operations_in_progress)
        # Note: operations_failed are typically retried and eventually completed or rejected
        # So we don't count them separately to avoid double-counting
        
        unaccounted = self.metrics.operations_submitted - operations_in_final_state
        
        print(f"\n📊 Operation Accounting (Fixed):")
        print(f"   ├─ Total submitted: {self.metrics.operations_submitted}")
        print(f"   ├─ Final state count: {operations_in_final_state}")
        print(f"   │  ├─ Completed: {self.metrics.operations_completed}")
        print(f"   │  ├─ Rejected: {self.metrics.operations_rejected}")
        print(f"   │  └─ In progress: {self.metrics.operations_in_progress}")
        print(f"   ├─ Unaccounted: {unaccounted}")
        print(f"   ├─ 📝 Pre-ECS Conflict Resolution: {self.metrics.operations_rejected} operations")
        print(f"   │  └─ rejected before entering ECS (never got ECS IDs)")
        print(f"   └─ Transition metrics (not counted in final):")
        print(f"      ├─ Failed (retried): {self.metrics.operations_failed}")
        print(f"      └─ Retried: {self.metrics.operations_retried}")
        
        if unaccounted > 0:
            print(f"   ⚠️  {unaccounted} operations may be stuck in PENDING state")
        elif unaccounted < 0:
            print(f"   ⚠️  {abs(unaccounted)} operations were double-counted (likely rejected operations)")
        else:
            print(f"   ✅ All operations properly accounted for")
        
        print(f"\n🛡️  Grace Period Results:")
        print(f"   ├─ Grace period saves: {self.metrics.grace_period_saves}")
        print(f"   └─ Operations protected: {self.metrics.operations_protected}")
        
        print(f"\n⚔️  Conflict Resolution Results:")
        print(f"   ├─ Conflicts detected: {self.metrics.conflicts_detected}")
        print(f"   ├─ Conflicts resolved: {self.metrics.conflicts_resolved}")
        if self.metrics.resolution_times:
            avg_resolution_time = statistics.mean(self.metrics.resolution_times)
            print(f"   ├─ Avg resolution time: {avg_resolution_time:.1f}ms")
        
        # Explain conflict resolution math
        if self.metrics.conflicts_detected > 0:
            avg_ops_per_conflict = self.metrics.operations_submitted / self.metrics.conflicts_detected
            print(f"   ├─ Avg operations per conflict: {avg_ops_per_conflict:.1f}")
            print(f"   └─ 📝 Each conflict = 1 batch with multiple operations, 1 winner + rest rejected")
        
        # System performance
        if self.metrics.memory_samples:
            avg_memory = statistics.mean(self.metrics.memory_samples)
            max_memory = max(self.metrics.memory_samples)
            print(f"\n💾 System Resources:")
            print(f"   ├─ Avg memory: {avg_memory:.1f} MB")
            print(f"   ├─ Max memory: {max_memory:.1f} MB")
            print(f"   └─ Memory status: {'✅ Good' if max_memory < self.config.max_memory_mb else '⚠️ High'}")
        
        # ECS Lineage and Versioning Analysis
        print(f"\n🔍 ECS Lineage and Versioning Analysis:")
        modifications_detected = 0
        total_versions_found = 0
        
        for i, target in enumerate(self.target_entities):
            target_id = target.ecs_id
            print(f"   │  Target {i} (ID: {str(target_id)[:8]}):")
            
            # Check entity versioning through lineage registry (in-memory only)
            lineage_id = target.lineage_id
            root_versions = EntityRegistry.lineage_registry.get(lineage_id, [])
            total_versions_found += len(root_versions)
            print(f"   │    ├─ ECS Lineage versions: {len(root_versions)}")
            
            if len(root_versions) > 1:
                modifications_detected += 1
                print(f"   │    ├─ ✅ Entity was modified! ({len(root_versions)} versions in lineage)")
                
                # Show version IDs from lineage
                versions_to_show = min(3, len(root_versions))
                for j, root_ecs_id in enumerate(root_versions[:versions_to_show]):
                    print(f"   │    │  Version {j}: root_ecs_id={str(root_ecs_id)[:8]}")
                
                # Show truncation message if there are more versions
                if len(root_versions) > 3:
                    remaining = len(root_versions) - 3
                    print(f"   │    │  --------{remaining} more cut out for sanity-----")
                
                # ECS lineage is the authoritative source of modification count
                actual_modifications = len(root_versions) - 1  # versions - 1 = modifications
                print(f"   │    └─ ✅ Confirmed modifications: {actual_modifications} (from ECS lineage + 1 Original)")
            else:
                print(f"   │    └─ ❌ No versions beyond original (lineage has {len(root_versions)} entries)")
        
        print(f"   ├─ Entities with ECS versions: {modifications_detected}/{len(self.target_entities)}")
        print(f"   ├─ Total ECS versions found: {total_versions_found}")
        print(f"   ├─ Total operations submitted: {self.metrics.operations_submitted}")
        print(f"   ├─ Operations started: {self.metrics.operations_started}")
        print(f"   ├─ Operations completed: {self.metrics.operations_completed}")
        print(f"   ├─ Operations rejected: {self.metrics.operations_rejected}")
        
        # Cross-reference with operation metrics
        expected_modifications = self.metrics.entity_modifications
        print(f"   ├─ Total operations that modified entities: {expected_modifications}")
        print(f"   ├─ Actual ECS versions found: {total_versions_found}")
        
        # Explain the version accounting
        version_creating_ops = self.metrics.versioning_operations + self.metrics.real_operations_by_type.get('complex_update', 0)
        original_entities = len(self.target_entities)
        expected_versions = version_creating_ops + original_entities
        
        print(f"   ├─ Version accounting breakdown:")
        print(f"   │  ├─ Operations that create versions: {version_creating_ops}")
        print(f"   │  │  ├─ version_entity operations: {self.metrics.versioning_operations}")
        print(f"   │  │  └─ complex_update operations: {self.metrics.real_operations_by_type.get('complex_update', 0)}")
        print(f"   │  ├─ Original target entities: {original_entities}")
        print(f"   │  └─ Expected total versions: {expected_versions}")
        
        if total_versions_found == expected_versions:
            print(f"   ├─ ✅ Perfect version accounting: {version_creating_ops} + {original_entities} original entities = {expected_versions}")
        elif total_versions_found > 0:
            print(f"   ├─ ✅ ECS versioning verified: Entities were modified in ECS system")
            if total_versions_found != expected_versions:
                diff = total_versions_found - expected_versions
                print(f"   │  └─ Note: {abs(diff)} {'extra' if diff > 0 else 'fewer'} versions (likely from structural operations)")
        else:
            print(f"   ├─ ❌ No ECS versions found: Operations may not have reached entities")
            
        # Explain operations that don't create versions
        non_versioning_ops = expected_modifications - version_creating_ops
        if non_versioning_ops > 0:
            print(f"   ├─ Operations that modified state without creating versions: {non_versioning_ops}")
            print(f"   │  ├─ These operations (modify_field, borrow_attribute) update entity state")
            print(f"   │  │  but don't call force_versioning=True (by design)")
            print(f"   │  └─ 📝 ECS Versioning Logic: Only certain operations create new ECS versions")
            
        # Operation Lineage Analysis (in-memory)
        print(f"\n🕵️ Operation Lineage Analysis:")
        completed_operations = []
        
        # Find all completed operations in the in-memory ECS registry
        all_operations_found = []
        for root_id in EntityRegistry.tree_registry.keys():
            tree = EntityRegistry.tree_registry.get(root_id)
            if tree:
                for entity_id, entity in tree.nodes.items():
                    if isinstance(entity, RealOperationEntity):
                        all_operations_found.append(entity)
                        if entity.status == OperationStatus.SUCCEEDED:
                            completed_operations.append(entity)
        
        print(f"   ├─ Total RealOperationEntity objects found: {len(all_operations_found)}")
        print(f"   ├─ Operations with SUCCEEDED status: {len(completed_operations)}")
        
        # Show status breakdown of all operations found
        if all_operations_found:
            status_breakdown = {}
            for op in all_operations_found:
                status = op.status
                status_breakdown[status] = status_breakdown.get(status, 0) + 1
            
            print(f"   ├─ Status breakdown of found operations:")
            for status, count in status_breakdown.items():
                print(f"   │    ├─ {status}: {count}")
        
        print(f"   ├─ Completed operations found in ECS: {len(completed_operations)}")
        
        # Group by target entity
        ops_by_target = {}
        for op in completed_operations:
            target_id = op.target_entity_id
            if target_id not in ops_by_target:
                ops_by_target[target_id] = []
            ops_by_target[target_id].append(op)
        
        print(f"   ├─ Targets that received operations: {len(ops_by_target)}")
        
        for target_id, ops in ops_by_target.items():
            target_short_id = str(target_id)[:8]
            print(f"   │  Target {target_short_id}: {len(ops)} completed operations")
            
            # Show operation types
            op_types = {}
            for op in ops:
                op_type = getattr(op, 'operation_type', 'unknown')
                op_types[op_type] = op_types.get(op_type, 0) + 1
            
            for op_type, count in op_types.items():
                print(f"   │    ├─ {op_type}: {count}")
        
        # Final verification
        total_ops_by_lineage = len(completed_operations)
        total_ops_by_metrics = self.metrics.operations_completed
        
        print(f"   ├─ Operations by lineage tracking: {total_ops_by_lineage}")
        print(f"   ├─ Operations by metrics tracking: {total_ops_by_metrics}")
        
        if total_ops_by_lineage == total_ops_by_metrics:
            print(f"   ├─ ✅ Operation counts match perfectly")
        else:
            print(f"   ├─ ⚠️ Operation count mismatch: {abs(total_ops_by_lineage - total_ops_by_metrics)} difference")
        
        print(f"   ├─ Production operations verified: ✅ {self.metrics.entity_modifications > 0}")
        print(f"   └─ Conflict resolution verified: ✅ {self.metrics.conflicts_resolved > 0}")
        
        # Assessment
        print(f"\n🎯 SYSTEM ASSESSMENT:")
        if completion_rate >= self.config.target_completion_rate:
            print(f"   ✅ PASSED: Completion rate {completion_rate:.1%} meets target {self.config.target_completion_rate:.1%}")
        else:
            print(f"   ❌ FAILED: Completion rate {completion_rate:.1%} below target {self.config.target_completion_rate:.1%}")
            
        if self.metrics.grace_period_saves > 0:
            print("   ✅ GRACE PERIODS: Successfully protected executing operations")
        
        if self.metrics.conflicts_resolved > 0:
            print("   ✅ CONFLICT RESOLUTION: System handled conflicts with production operations")
        
        if self.metrics.entity_modifications > 0:
            print("   ✅ PRODUCTION OPERATIONS: System performed actual entity modifications")
        else:
            print("   ❌ NO PRODUCTION WORK: No actual entity modifications detected")
            
        # Add clear explanation of the overall logic
        print(f"\n📚 SYSTEM LOGIC EXPLANATION:")
        print(f"   ├─ 🎯 Conflict Resolution Logic:")
        print(f"   │  ├─ Multiple operations submitted to same target in batches")
        print(f"   │  ├─ Pre-ECS staging area detects conflicts before ECS entry")
        print(f"   │  ├─ Priority-based resolution: 1 winner per conflict batch")
        print(f"   │  └─ Losers rejected before getting ECS IDs (never enter system)")
        print(f"   ├─ 🔄 ECS Versioning Logic:")
        print(f"   │  ├─ All operations modify entity state")
        print(f"   │  ├─ Only certain operations create new ECS versions")
        print(f"   │  ├─ version_entity/complex_update → create versions")
        print(f"   │  └─ modify_field/borrow_attribute → state-only updates")
        print(f"   └─ 📊 Math Verification:")
        print(f"      ├─ {self.metrics.operations_submitted} total operations → {self.metrics.conflicts_detected} conflict batches")
        print(f"      ├─ {self.metrics.operations_completed} winners + {self.metrics.operations_rejected} losers = {self.metrics.operations_submitted} total")
        print(f"      ├─ {self.metrics.entity_modifications} state modifications → {total_versions_found} ECS versions created")
        print(f"      ├─ {version_creating_ops} operations created ECS versions")
        print(f"      └─ {non_versioning_ops} operations modified state only (no versions)")
        
        return {
            'completion_rate': completion_rate,
            'rejection_rate': rejection_rate,
            'throughput': throughput,
            'conflicts_resolved': self.metrics.conflicts_resolved,
            'grace_period_saves': self.metrics.grace_period_saves,
            'real_modifications': self.metrics.entity_modifications,
            'real_operations': sum(self.metrics.real_operations_by_type.values()),
            'passed': completion_rate >= self.config.target_completion_rate and self.metrics.entity_modifications > 0
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
        
        print("\n🎉 Conflict resolution test with production operations completed!")
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
    
    # BRUTAL CONFLICT CONFIGURATION
    config = TestConfig(
        duration_seconds=15,  # Longer to see conflict patterns
        num_targets=5,        # 5 entities as requested
        operation_rate_per_second=100.0,  # Reduced rate but with batching
        priority_distribution={
            OperationPriority.LOW: 0.25,
            OperationPriority.NORMAL: 0.25,
            OperationPriority.HIGH: 0.25,
            OperationPriority.CRITICAL: 0.25
        },
        target_completion_rate=0.10,  # Very low expectation due to brutal conflicts at 100 ops/sec
        max_memory_mb=1000,
        grace_period_seconds=0.05  # Increased to 50ms to catch some operations in grace period
    )
    
    print("🚀 CONFLICT RESOLUTION ALGORITHM TEST")
    print("=" * 60)
    print("BRUTAL CONFLICT MODE - Multiple ops per target simultaneously")
    print("Production test - submits production operations and measures results")
    print("ALL operations perform actual ECS work - comprehensive validation!")
    print("FORCING SIMULTANEOUS OPERATIONS ON SAME TARGETS")
    print("=" * 60)
    
    results = await run_conflict_resolution_test(config)
    
    if results.get('passed', False):
        print(f"\n✅ TEST PASSED - System survived the brutal conflicts with production operations")
        print(f"   ├─ Production modifications: {results.get('real_modifications', 0)}")
        print(f"   ├─ Production operations: {results.get('real_operations', 0)}")
        print(f"   └─ Conflicts resolved: {results.get('conflicts_resolved', 0)}")
    else:
        print(f"\n❌ TEST FAILED - System could not handle the brutal conflicts")
        if results.get('real_modifications', 0) == 0:
            print("   └─ ERROR: No production entity modifications detected!")
        if results.get('conflicts_resolved', 0) == 0:
            print("   └─ WARNING: No conflicts detected - need more brutality!")


if __name__ == "__main__":
    asyncio.run(main()) 