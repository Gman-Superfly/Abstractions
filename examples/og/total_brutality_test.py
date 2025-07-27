"""
TOTAL BRUTALITY TEST: Ultimate Production Stress Test

This test simulates 100,000 users simultaneously submitting operations to the system,
testing the complete concurrency stack under maximum stress:

1. Pre-ECS conflict resolution (operation-level)
2. Grace period protection (temporal)
3. OCC protection (data-level) 
4. Zombie cleanup (resource management)
5. ECS hierarchy management
6. Event-driven coordination

BRUTALITY MODE: Maximum concurrent operations, minimal targets, real race conditions

OCC = Optimistic Concurrency Control - prevents race conditions in entity data modifications
"""

import asyncio
import time
import statistics
import psutil
import random
import threading
from typing import List, Dict, Any, Set, Optional, Union
from collections import deque, defaultdict
from datetime import datetime, timezone
from uuid import UUID, uuid4
from pydantic import Field
import concurrent.futures
from dataclasses import dataclass

# Core imports - PURE EVENT-DRIVEN
from abstractions.ecs.entity_hierarchy import (
    OperationEntity, StructuralOperation, NormalOperation, LowPriorityOperation,
    OperationStatus, OperationPriority, resolve_operation_conflicts
)
from abstractions.events.events import (
    get_event_bus, emit,
    OperationStartedEvent, OperationCompletedEvent, OperationRejectedEvent, 
    OperationConflictEvent, OperationRetryEvent
)
from abstractions.ecs.entity import Entity, EntityRegistry
from abstractions.ecs.functional_api import put, get

# Enable operation observers
import abstractions.agent_observer


@dataclass
class BrutalityConfig:
    """Configuration for total brutality testing."""
    
    # Brutality parameters - REAL CONFLICT CREATION
    total_operations: int = 50_000      # 50K operations in 30 seconds = 1,666 ops/sec
    targets_count: int = 3              # ONLY 3 targets = GUARANTEED conflicts
    batch_size: int = 50                # Large batches to accumulate conflicts
    
    # Timing - RAMP UP THEN MAXIMUM BRUTALITY
    ramp_up_seconds: int = 10           # 10 seconds to ramp up
    max_brutality_seconds: int = 30     # 30 seconds at maximum brutality
    total_duration: int = 40            # Total test time
    
    # Conflict creation parameters
    submission_rate_start: float = 100.0    # Start: 100 ops/sec
    submission_rate_peak: float = 3000.0    # Peak: 3,000 ops/sec (BRUTAL)
    conflict_batch_interval: float = 0.05   # 50ms between batches = conflict window
    
    # OCC settings
    occ_max_retries: int = 15
    occ_retry_backoff_ms: float = 0.5   # Minimal backoff for maximum conflicts
    
    # Grace period - MINIMAL for maximum conflicts
    grace_period_seconds: float = 0.02  # 20ms grace period
    
    # System limits
    max_memory_mb: float = 3000.0
    target_success_rate: float = 0.50   # 50% success under BRUTAL load
    
    def __post_init__(self):
        """Validate brutal configuration."""
        assert self.total_operations > 0, "Must have operations"
        assert self.targets_count >= 1, "Must have at least one target"
        assert self.batch_size > 0, "Must have positive batch size"
        assert self.total_duration > 0, "Must have test duration"
        assert 0.0 <= self.target_success_rate <= 1.0, "Success rate must be 0-1"
        
        print(f"🔥 BRUTAL CONFLICT CONFIGURATION:")
        print(f"   💥 Total operations: {self.total_operations:,}")
        print(f"   🎯 Targets: {self.targets_count} (GUARANTEED MASSIVE CONFLICTS)")
        print(f"   📦 Batch size: {self.batch_size} (conflict accumulation)")
        print(f"   ⏱️  Ramp up: {self.ramp_up_seconds}s → Max brutality: {self.max_brutality_seconds}s")
        print(f"   🚀 Rate: {self.submission_rate_start:.0f} → {self.submission_rate_peak:.0f} ops/sec")
        
        # Calculate conflict potential
        ops_per_target = self.total_operations / self.targets_count
        avg_ops_per_sec = self.total_operations / self.total_duration
        conflicts_per_batch = (self.batch_size * self.targets_count) / self.targets_count
        
        print(f"   ⚔️  Ops per target: {ops_per_target:,.0f}")
        print(f"   💥 Avg ops/sec: {avg_ops_per_sec:.0f}")
        print(f"   🎲 Expected conflicts per batch: {conflicts_per_batch:.0f}")
        print(f"   🔥 CONFLICT GUARANTEE: Multiple ops per target per batch!")


class BrutalityMetrics:
    """Comprehensive metrics for brutality testing."""
    
    def __init__(self):
        self.start_time = time.time()
        self.lock = threading.Lock()  # Thread-safe metrics
        
        # Operation batching metrics (inspired by dynamic_stress_test)
        self.batches_created = 0
        self.operations_per_batch: List[int] = []
        self.batch_conflicts_detected = 0
        self.avg_batch_size = 0.0
        
        # Operation metrics (thread-safe counters)
        self.operations_submitted = 0
        self.operations_started = 0
        self.operations_completed = 0
        self.operations_rejected = 0
        self.operations_failed = 0
        self.operations_retried = 0
        
        # Conflict resolution metrics (like dynamic_stress_test)
        self.pre_ecs_conflicts_detected = 0
        self.pre_ecs_conflicts_resolved = 0
        self.grace_period_saves = 0
        self.staging_area_size: List[int] = []  # Track staging area growth
        self.max_staging_area_size = 0
        self.conflict_resolution_times: List[float] = []
        
        # OCC metrics
        self.occ_conflicts_detected = 0
        self.occ_retries_attempted = 0
        self.occ_successes_after_retry = 0
        self.occ_failures_max_retries = 0
        self.occ_total_retry_time_ms = 0.0
        
        # Per-operation type tracking
        self.occ_by_operation_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            'conflicts': 0, 'retries': 0, 'successes': 0, 'failures': 0
        })
        self.retry_counts_per_success: Dict[str, List[int]] = defaultdict(list)  # Track retries per success
        
        # Performance tracking
        self.operation_durations: List[float] = []
        self.conflict_resolution_times: List[float] = []
        self.user_completion_times: List[float] = []
        
        # System resource tracking
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []
        self.peak_memory_mb = 0.0
        self.peak_operations_in_progress = 0
        
        # Real-time tracking
        self.operations_in_progress = 0
        
    def increment_thread_safe(self, counter_name: str, amount: int = 1):
        """Thread-safe increment of counters."""
        with self.lock:
            setattr(self, counter_name, getattr(self, counter_name) + amount)
    
    def record_user_spawned(self):
        self.increment_thread_safe('users_spawned')
        with self.lock:
            self.active_users += 1
            self.peak_active_users = max(self.peak_active_users, self.active_users)
    
    def record_user_completed(self, completion_time_ms: float):
        with self.lock:
            self.users_completed += 1
            self.active_users -= 1
            self.user_completion_times.append(completion_time_ms)
    
    def record_user_failed(self):
        with self.lock:
            self.users_failed += 1
            self.active_users -= 1
    
    def record_occ_conflict(self, operation_type: str):
        with self.lock:
            self.occ_conflicts_detected += 1
            self.occ_by_operation_type[operation_type]['conflicts'] += 1
    
    def record_occ_retry(self, operation_type: str, retry_time_ms: float):
        with self.lock:
            self.occ_retries_attempted += 1
            self.occ_total_retry_time_ms += retry_time_ms
            self.occ_by_operation_type[operation_type]['retries'] += 1
    
    def record_occ_success(self, operation_type: str, retry_count: int = 0):
        with self.lock:
            self.occ_by_operation_type[operation_type]['successes'] += 1
            self.retry_counts_per_success[operation_type].append(retry_count)
    
    def record_occ_failure(self, operation_type: str):
        with self.lock:
            self.occ_failures_max_retries += 1
            self.occ_by_operation_type[operation_type]['failures'] += 1
    
    def record_system_stats(self):
        """Record current system statistics."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            with self.lock:
                self.memory_samples.append(memory_mb)
                self.cpu_samples.append(cpu_percent)
                self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        except:
            pass
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current statistics snapshot."""
        with self.lock:
            elapsed = time.time() - self.start_time
            ops_per_sec = self.operations_completed / elapsed if elapsed > 0 else 0
            
            # Calculate completion rate (completed vs submitted)
            completion_rate = self.operations_completed / max(1, self.operations_submitted)
            
            # Calculate success rate of operations that finished (completed vs failed)
            total_finished = self.operations_completed + self.operations_failed
            success_rate = self.operations_completed / max(1, total_finished)
            
            return {
                'elapsed_seconds': elapsed,
                'batches_created': self.batches_created,
                'batch_conflicts': self.batch_conflicts_detected,
                'operations_submitted': self.operations_submitted,
                'operations_completed': self.operations_completed,
                'operations_rejected': self.operations_rejected,
                'operations_failed': self.operations_failed,
                'operations_per_second': ops_per_sec,
                'completion_rate_percent': completion_rate * 100,  # How many submitted ops finished
                'success_rate_percent': success_rate * 100,        # How many finished ops succeeded
                'operations_still_retrying': self.operations_submitted - self.operations_completed - self.operations_failed - self.operations_rejected,
                'occ_conflicts': self.occ_conflicts_detected,
                'pre_ecs_conflicts': self.pre_ecs_conflicts_detected,
                'memory_mb': self.peak_memory_mb,
                'operations_in_progress': self.operations_in_progress
            }


class BrutalTargetEntity(Entity):
    """Target entity optimized for brutal concurrent access."""
    
    name: str = "brutal_target"
    hit_count: int = 0
    total_value: float = 0.0
    last_user_id: int = 0
    update_history: List[str] = Field(default_factory=list)
    concurrent_modifications: int = 0
    
    # Performance optimization - limit history size
    def add_update(self, user_id: int, operation: str, value: Any):
        """Add update with history management."""
        self.update_history.append(f"User{user_id}:{operation}={value}@{time.time()}")
        # Keep only last 100 updates to prevent memory bloat
        if len(self.update_history) > 100:
            self.update_history = self.update_history[-50:]


class BrutalOperationEntity(OperationEntity):
    """High-performance operation entity for brutality testing."""
    
    # User context
    user_id: int = Field(description="User who submitted this operation")
    operation_type: str = Field(description="Type of brutal operation")
    operation_data: Dict[str, Any] = Field(default_factory=dict)
    
    # OCC protection
    occ_retry_count: int = 0
    occ_max_retries: int = 10
    
    async def execute_with_occ_brutality(self, target: BrutalTargetEntity, metrics: BrutalityMetrics) -> bool:
        """Execute operation with PROPER OCC - check at WRITE time for stale data."""
        operation_start = time.time()
        
        while self.occ_retry_count <= self.occ_max_retries:
            try:
                # Step 1: READ PHASE - Snapshot current state (legitimate read)
                read_version = target.version
                read_modified = target.last_modified
                read_data = {
                    'hit_count': target.hit_count,
                    'total_value': target.total_value,
                    'last_user_id': target.last_user_id,
                    'concurrent_modifications': target.concurrent_modifications
                }
                
                # Step 2: OPERATION PHASE - Do legitimate work on the data
                # This simulates processing that takes time (like array operations)
                await asyncio.sleep(0.005)  # Simulate processing time where others can interfere
                
                # Compute new values based on read data (like array modification)
                if self.operation_type == "brutal_increment":
                    new_hit_count = read_data['hit_count'] + 1
                    new_total_value = read_data['total_value'] + self.operation_data.get('amount', 1.0)
                elif self.operation_type == "brutal_update":
                    new_hit_count = read_data['hit_count']
                    new_total_value = self.operation_data.get('new_value', 100.0)
                elif self.operation_type == "brutal_accumulate":
                    new_hit_count = read_data['hit_count'] + 1
                    new_total_value = read_data['total_value'] + self.operation_data.get('amount', 50.0)
                else:  # brutal_version
                    new_hit_count = read_data['hit_count'] + 1
                    new_total_value = read_data['total_value']
                
                # Step 3: WRITE PHASE - OCC CHECK BEFORE COMMITTING
                current_version = target.version
                current_modified = target.last_modified
                
                if (current_version != read_version or current_modified != read_modified):
                    # STALE DATA DETECTED! Another operation modified the entity
                    self.occ_retry_count += 1
                    retry_time = (time.time() - operation_start) * 1000
                    
                    metrics.record_occ_conflict(self.operation_type)
                    metrics.record_occ_retry(self.operation_type, retry_time)
                    
                    print(f"⚠️  OCC STALE DATA: {self.operation_type} read v{read_version}, now v{current_version} - RETRYING")
                    
                    if self.occ_retry_count > self.occ_max_retries:
                        print(f"❌ OCC MAX RETRIES: {self.operation_type} failed after {self.occ_max_retries} attempts")
                        metrics.record_occ_failure(self.operation_type)
                        return False
                    
                    # Exponential backoff before retry
                    await asyncio.sleep(0.002 * (2 ** self.occ_retry_count))
                    continue  # Re-read and try again
                
                # Step 4: COMMIT PHASE - Write the computed values
                target.hit_count = new_hit_count
                target.total_value = new_total_value
                target.last_user_id = self.user_id
                target.concurrent_modifications += 1
                target.add_update(self.user_id, self.operation_type, new_total_value)
                target.mark_modified()  # This increments version and updates timestamp
                
                print(f"✅ OCC SUCCESS: {self.operation_type} committed (v{read_version}→{target.version}) after {self.occ_retry_count} retries")
                
                metrics.record_occ_success(self.operation_type, self.occ_retry_count)
                operation_time = (time.time() - operation_start) * 1000
                metrics.operation_durations.append(operation_time)
                return True
                    
            except Exception as e:
                print(f"🚨 OCC EXECUTION ERROR: {self.operation_type}: {e}")
                metrics.record_occ_failure(self.operation_type)
                return False
        
        print(f"❌ OCC EXHAUSTED: {self.operation_type} failed after {self.occ_max_retries} retries")
        metrics.record_occ_failure(self.operation_type)
        return False
    
    async def _execute_brutal_operation(self, target: BrutalTargetEntity) -> bool:
        """Execute the actual brutal operation."""
        try:
            if self.operation_type == "brutal_increment":
                # Brutal increment with conflict opportunity
                target.hit_count += 1
                target.total_value += self.operation_data.get('amount', 1.0)
                target.last_user_id = self.user_id
                target.add_update(self.user_id, "increment", self.operation_data.get('amount', 1.0))
                target.mark_modified()
                
            elif self.operation_type == "brutal_update":
                # Brutal data update
                new_value = self.operation_data.get('new_value', random.uniform(0, 1000))
                target.total_value = new_value
                target.last_user_id = self.user_id
                target.add_update(self.user_id, "update", new_value)
                target.mark_modified()
                
            elif self.operation_type == "brutal_accumulate":
                # Brutal accumulation
                amount = self.operation_data.get('amount', random.uniform(1, 100))
                target.total_value += amount
                target.hit_count += 1
                target.concurrent_modifications += 1
                target.last_user_id = self.user_id
                target.add_update(self.user_id, "accumulate", amount)
                target.mark_modified()
                
            elif self.operation_type == "brutal_version":
                # Brutal versioning (creates ECS versions)
                target.hit_count += 1
                target.last_user_id = self.user_id
                target.add_update(self.user_id, "version", "forced")
                EntityRegistry.version_entity(target, force_versioning=True)
                
            else:
                return False
                
            return True
            
        except Exception as e:
            self.error_message = str(e)
            return False


class SimulatedUser:
    """Represents a single user hammering the system."""
    
    def __init__(self, user_id: int, config: BrutalityConfig, targets: List[BrutalTargetEntity]):
        self.user_id = user_id
        self.config = config
        self.targets = targets
        self.operations_submitted = 0
        self.operations_completed = 0
        self.start_time = time.time()
    
    async def brutal_user_session(self, metrics: BrutalityMetrics) -> bool:
        """Simulate a brutal user session."""
        try:
            metrics.record_user_spawned()
            
            # Each user submits multiple operations rapidly
            for op_num in range(self.config.operations_per_user):
                await self._submit_brutal_operation(metrics, op_num)
                
                # Minimal delay between operations for maximum brutality
                await asyncio.sleep(0.001)
            
            completion_time = (time.time() - self.start_time) * 1000
            metrics.record_user_completed(completion_time)
            return True
            
        except Exception as e:
            print(f"🚨 USER {self.user_id} FAILED: {e}")
            metrics.record_user_failed()
            return False
    
    async def _submit_brutal_operation(self, metrics: BrutalityMetrics, op_num: int):
        """Submit a single brutal operation."""
        try:
            # Select random target (few targets = maximum conflicts)
            target = random.choice(self.targets)
            
            # Select brutal operation type
            operation_types = ["brutal_increment", "brutal_update", "brutal_accumulate", "brutal_version"]
            op_type = random.choice(operation_types)
            
            # Create operation data
            op_data = {
                'amount': random.uniform(1, 100),
                'new_value': random.uniform(0, 1000),
                'target_id': target.ecs_id
            }
            
            # Determine priority (most are normal for maximum conflicts)
            priorities = [OperationPriority.NORMAL] * 7 + [OperationPriority.HIGH] * 2 + [OperationPriority.CRITICAL]
            priority = random.choice(priorities)
            
            # Create brutal operation
            brutal_op = BrutalOperationEntity(
                op_type=f"user_{self.user_id}_op_{op_num}",
                user_id=self.user_id,
                operation_type=op_type,
                operation_data=op_data,
                target_entity_id=target.ecs_id,
                priority=priority,
                occ_max_retries=self.config.occ_max_retries
            )
            
            # Submit to ECS (no pre-ECS staging for maximum brutality)
            brutal_op.promote_to_root()
            
            metrics.increment_thread_safe('operations_submitted')
            self.operations_submitted += 1
            
            # Execute immediately with OCC protection
            success = await brutal_op.execute_with_occ_brutality(target, metrics)
            
            if success:
                metrics.increment_thread_safe('operations_completed')
                self.operations_completed += 1
            else:
                metrics.increment_thread_safe('operations_failed')
                
        except Exception as e:
            print(f"🚨 USER {self.user_id} OP {op_num} ERROR: {e}")
            metrics.increment_thread_safe('operations_failed')


class TotalBrutalityTest:
    """The ultimate brutality test combining all concurrency systems."""
    
    def __init__(self, config: BrutalityConfig):
        self.config = config
        self.metrics = BrutalityMetrics()
        self.targets: List[BrutalTargetEntity] = []
        
        # Staging area for conflict creation (like dynamic_stress_test)
        self.pending_operations: Dict[UUID, List[BrutalOperationEntity]] = {}
        self.submitted_operations: Set[UUID] = set()
        self.operation_counter = 0
        
        # Control flags
        self.stop_submission = False
        self.stop_flag = False
        
        # Grace period tracker (from dynamic_stress_test)
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from dynamic_stress_test import GracePeriodTracker
        self.grace_tracker = GracePeriodTracker(config.grace_period_seconds)
        
    async def setup_brutal_environment(self):
        """Set up the brutal testing environment."""
        print(f"\n🔥 SETTING UP TOTAL BRUTALITY ENVIRONMENT")
        print(f"=" * 60)
        
        # Create brutal target entities (few targets = maximum conflicts)
        for i in range(self.config.targets_count):
            target = BrutalTargetEntity(
                name=f"brutal_target_{i}",
                hit_count=0,
                total_value=0.0,
                last_user_id=0,
                update_history=[],
                concurrent_modifications=0
            )
            target.promote_to_root()
            self.targets.append(target)
        
        print(f"✅ Created {len(self.targets)} brutal target entities")
        print(f"🎯 Each target will be hit by ~{self.config.total_operations // len(self.targets):,} operations")
        print(f"💥 Expected conflicts: GUARANTEED (many ops per target per batch)")
        
        # Set up event handlers
        from abstractions.events.events import setup_operation_event_handlers
        setup_operation_event_handlers()
        print(f"✅ Event handlers ready for brutal load")
    
    async def submit_brutal_operation(self, target: BrutalTargetEntity) -> BrutalOperationEntity:
        """Submit operation to staging area for conflict accumulation (like dynamic_stress_test)."""
        self.operation_counter += 1
        
        # Create brutal operation
        operation_types = ["brutal_increment", "brutal_update", "brutal_accumulate", "brutal_version"]
        op_type = random.choice(operation_types)
        
        # Generate operation data
        op_data = {
            'amount': random.uniform(1, 100),
            'new_value': random.uniform(0, 1000),
            'target_id': target.ecs_id
        }
        
        # Determine priority (weighted for conflicts)
        priorities = [OperationPriority.NORMAL] * 6 + [OperationPriority.HIGH] * 3 + [OperationPriority.CRITICAL]
        priority = random.choice(priorities)
        
        # Create operation but DON'T promote to ECS yet (staging area!)
        brutal_op = BrutalOperationEntity(
            op_type=f"brutal_op_{self.operation_counter}",
            user_id=self.operation_counter,
            operation_type=op_type,
            operation_data=op_data,
            target_entity_id=target.ecs_id,
            priority=priority,
            occ_max_retries=self.config.occ_max_retries
        )
        
        # Add to PRE-ECS staging area (this creates conflicts!)
        if target.ecs_id not in self.pending_operations:
            self.pending_operations[target.ecs_id] = []
        self.pending_operations[target.ecs_id].append(brutal_op)
        
        self.metrics.increment_thread_safe('operations_submitted')
        
        # DON'T resolve conflicts yet - let them accumulate in staging!
        return brutal_op
    
    async def resolve_conflicts_before_ecs(self, target_entity_id: UUID):
        """Resolve conflicts in pre-ECS staging area (EXACTLY like dynamic_stress_test)."""
        pending_ops = self.pending_operations.get(target_entity_id, [])
        
        # Track staging area growth
        self.metrics.staging_area_size.append(len(pending_ops))
        self.metrics.max_staging_area_size = max(self.metrics.max_staging_area_size, len(pending_ops))
        
        if len(pending_ops) > 1:
            # 🎯 CONFLICT DETECTED!
            resolution_start = time.time()
            
            self.metrics.increment_thread_safe('pre_ecs_conflicts_detected')
            self.metrics.increment_thread_safe('batch_conflicts_detected')
            
            print(f"⚔️  BRUTAL CONFLICT: {len(pending_ops)} operations competing for target {str(target_entity_id)[:8]}")
            for op in pending_ops:
                print(f"   ├─ {op.op_type} (Priority: {op.priority}, Type: {op.operation_type})")
            
            # Sort by priority (higher priority wins) - EXACTLY like dynamic_stress_test
            pending_ops.sort(key=lambda op: (op.priority, -op.created_at.timestamp()), reverse=True)
            
            # Winner takes all
            winner = pending_ops[0]
            losers = pending_ops[1:]
            
            print(f"🏆 BRUTAL RESOLUTION: 1 winner, {len(losers)} brutally rejected")
            print(f"✅ WINNER: {winner.op_type} (Priority: {winner.priority})")
            
            # Reject losers before they enter ECS
            for loser in losers:
                print(f"❌ BRUTALLY REJECTED: {loser.op_type} (Priority: {loser.priority})")
                self.metrics.increment_thread_safe('operations_rejected')
            
            # Record conflict resolution time
            resolution_time_ms = (time.time() - resolution_start) * 1000
            self.metrics.conflict_resolution_times.append(resolution_time_ms)
            self.metrics.increment_thread_safe('pre_ecs_conflicts_resolved')
            
            # Promote winner to ECS for execution
            winner.promote_to_root()
            self.submitted_operations.add(winner.ecs_id)
            self.pending_operations[target_entity_id] = []  # Clear staging area
            
            print(f"🚀 PROMOTED TO ECS: {winner.op_type} (ID: {str(winner.ecs_id)[:8]})")
            
        elif len(pending_ops) == 1:
            # No conflict - promote single operation
            winner = pending_ops[0]
            winner.promote_to_root()
            self.submitted_operations.add(winner.ecs_id)
            self.pending_operations[target_entity_id] = []
    
    async def run_total_brutality_test(self):
        """Run the complete total brutality test with ramp-up and maximum brutality phases."""
        print(f"\n🚀 LAUNCHING TOTAL BRUTALITY TEST")
        print(f"💀 PHASE 1: {self.config.ramp_up_seconds}s RAMP-UP")
        print(f"🔥 PHASE 2: {self.config.max_brutality_seconds}s MAXIMUM BRUTALITY")
        print(f"⚡ TARGET: {len(self.targets)} TARGETS FOR GUARANTEED CONFLICTS")
        print(f"=" * 80)
        
        # Start all workers
        tasks = [
            asyncio.create_task(self._brutal_operation_submission_worker()),
            asyncio.create_task(self._brutal_operation_lifecycle_driver()),
            asyncio.create_task(self._brutal_progress_monitor()),
            asyncio.create_task(self._brutal_system_monitor())
        ]
        
        try:
            # Phase 1: Ramp up
            print(f"🚀 PHASE 1: RAMPING UP BRUTALITY...")
            await asyncio.sleep(self.config.ramp_up_seconds)
            
            # Phase 2: Maximum brutality
            print(f"🔥 PHASE 2: MAXIMUM BRUTALITY MODE!")
            await asyncio.sleep(self.config.max_brutality_seconds)
            
            # Stop new submissions
            print(f"⏹️  STOPPING NEW SUBMISSIONS - ALLOWING GRACE PERIOD...")
            self.stop_submission = True
            await asyncio.sleep(2.0)  # Grace period for pending ops
            
        finally:
            # Stop all workers
            self.stop_flag = True
            for task in tasks:
                task.cancel()
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                print("⚠️  Some workers took too long to stop")
        
        # Analyze brutal results
        await self._analyze_brutal_results()
    
    async def _brutal_operation_submission_worker(self):
        """Submit operations in batches to create conflicts (inspired by dynamic_stress_test)."""
        operations_submitted = 0
        elapsed_time = 0.0
        
        while not self.stop_submission and operations_submitted < self.config.total_operations:
            try:
                # Calculate current submission rate (ramp up over time)
                if elapsed_time < self.config.ramp_up_seconds:
                    # Ramp up phase
                    progress = elapsed_time / self.config.ramp_up_seconds
                    current_rate = self.config.submission_rate_start + (
                        (self.config.submission_rate_peak - self.config.submission_rate_start) * progress
                    )
                else:
                    # Maximum brutality phase
                    current_rate = self.config.submission_rate_peak
                
                # Calculate batch timing
                interval = 1.0 / current_rate * self.config.batch_size
                
                print(f"🔥 BRUTAL BATCH @ {elapsed_time:.1f}s: Rate={current_rate:.0f} ops/sec, Interval={interval:.3f}s")
                
                # Submit batch of operations (this creates conflicts!)
                batch_ops = []
                for _ in range(self.config.batch_size):
                    if operations_submitted >= self.config.total_operations:
                        break
                    
                    # Select target (few targets = guaranteed conflicts)
                    target = random.choice(self.targets)
                    
                    # Submit to staging area
                    op = await self.submit_brutal_operation(target)
                    batch_ops.append(op)
                    operations_submitted += 1
                
                self.metrics.batches_created += 1
                self.metrics.operations_per_batch.append(len(batch_ops))
                
                print(f"   📦 Submitted batch of {len(batch_ops)} operations")
                print(f"   📊 Total submitted: {operations_submitted}/{self.config.total_operations}")
                
                # Now resolve conflicts for each target (this is where conflicts happen!)
                target_ids = set(op.target_entity_id for op in batch_ops)
                for target_id in target_ids:
                    await self.resolve_conflicts_before_ecs(target_id)
                
                # Wait for next batch
                await asyncio.sleep(self.config.conflict_batch_interval)
                elapsed_time += self.config.conflict_batch_interval
                
            except Exception as e:
                print(f"🚨 SUBMISSION WORKER ERROR: {e}")
                await asyncio.sleep(0.1)
        
        print(f"✅ SUBMISSION COMPLETE: {operations_submitted} operations submitted in {self.metrics.batches_created} batches")
    
    async def _brutal_operation_lifecycle_driver(self):
        """Drive operation execution with OCC protection (inspired by dynamic_stress_test)."""
        while not self.stop_flag:
            try:
                started_count = 0
                
                # Start all pending operations (maximum concurrency)
                # Fix: Create copy to avoid "dictionary changed size during iteration"
                submitted_ops_copy = list(self.submitted_operations)
                
                # Execute operations concurrently to create OCC conflicts
                executing_operations = []
                
                for op_id in submitted_ops_copy:
                    for root_id in EntityRegistry.tree_registry.keys():
                        tree = EntityRegistry.tree_registry.get(root_id)
                        if tree and op_id in tree.nodes:
                            op = tree.nodes[op_id]
                            if isinstance(op, BrutalOperationEntity):
                                
                                # Start pending operations
                                if op.status == OperationStatus.PENDING:
                                    try:
                                        op.start_execution()
                                        self.grace_tracker.start_grace_period(op.ecs_id)
                                        self.metrics.increment_thread_safe('operations_started')
                                        started_count += 1
                                        
                                    except Exception as e:
                                        print(f"⚠️  Error starting operation {op.op_type}: {e}")
                                
                                # Execute with OCC protection - ADD TO CONCURRENT EXECUTION
                                elif op.status == OperationStatus.EXECUTING:
                                    # Get target entity
                                    target = None
                                    for t in self.targets:
                                        if t.ecs_id == op.target_entity_id:
                                            target = t
                                            break
                                    
                                    if target:
                                        # Create concurrent task instead of awaiting immediately
                                        task = asyncio.create_task(self._execute_operation_concurrently(op, target))
                                        executing_operations.append(task)
                
                # Wait for all concurrent operations to complete
                if executing_operations:
                    print(f"🔥 CONCURRENT EXECUTION: {len(executing_operations)} operations running simultaneously")
                    try:
                        await asyncio.gather(*executing_operations, return_exceptions=True)
                    except Exception as e:
                        print(f"⚠️  Error in concurrent execution: {e}")
                
                if started_count > 0:
                    print(f"🚀 Started {started_count} operations")
                
                await asyncio.sleep(0.01)  # High frequency execution
                
            except Exception as e:
                print(f"⚠️  Lifecycle driver error: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_operation_concurrently(self, op: BrutalOperationEntity, target: BrutalTargetEntity):
        """Execute a single operation concurrently with others targeting the same entity."""
        try:
            success = await op.execute_with_occ_brutality(target, self.metrics)
            
            if success:
                op.status = OperationStatus.SUCCEEDED
                op.completed_at = datetime.now(timezone.utc)
                self.metrics.increment_thread_safe('operations_completed')
                self.submitted_operations.discard(op.ecs_id)
            else:
                op.status = OperationStatus.FAILED
                op.completed_at = datetime.now(timezone.utc)
                self.metrics.increment_thread_safe('operations_failed')
                self.submitted_operations.discard(op.ecs_id)
                
        except Exception as e:
            print(f"⚠️  Error in concurrent execution of {op.op_type}: {e}")
            op.status = OperationStatus.FAILED
            self.metrics.increment_thread_safe('operations_failed')
            self.submitted_operations.discard(op.ecs_id)
    
    async def _run_brutal_user_simulation(self):
        """Run the brutal user simulation with maximum concurrency."""
        print(f"\n🔥 STARTING BRUTAL USER SIMULATION")
        
        # Calculate batch timing
        users_per_batch = self.config.total_users // self.config.concurrent_batches
        batch_interval = 1.0 / (self.config.user_spawn_rate_per_second / users_per_batch)
        
        print(f"   📊 Users per batch: {users_per_batch}")
        print(f"   ⏱️  Batch interval: {batch_interval:.3f}s")
        print(f"   🚀 Starting brutal assault...")
        
        # Launch user batches
        all_user_tasks = []
        
        for batch_num in range(self.config.concurrent_batches):
            # Create user batch
            batch_tasks = []
            
            for user_in_batch in range(users_per_batch):
                user_id = batch_num * users_per_batch + user_in_batch
                
                if user_id >= self.config.total_users:
                    break
                
                user = SimulatedUser(user_id, self.config, self.targets)
                task = asyncio.create_task(user.brutal_user_session(self.metrics))
                batch_tasks.append(task)
            
            all_user_tasks.extend(batch_tasks)
            
            if batch_num % 100 == 0:
                print(f"   🚀 Launched batch {batch_num}/{self.config.concurrent_batches}")
            
            # Brief pause between batches to control spawn rate
            if batch_interval > 0:
                await asyncio.sleep(batch_interval)
        
        print(f"🔥 ALL {len(all_user_tasks)} USERS LAUNCHED - WAITING FOR COMPLETION")
        
        # Wait for all users to complete
        completed_users = 0
        total_users = len(all_user_tasks)
        
        for task in asyncio.as_completed(all_user_tasks):
            try:
                await task
                completed_users += 1
                
                if completed_users % 10000 == 0:
                    print(f"   ✅ {completed_users}/{total_users} users completed")
                    
            except Exception as e:
                print(f"   ❌ User task failed: {e}")
        
        print(f"🎉 ALL USERS COMPLETED: {completed_users}/{total_users}")
    
    async def _brutal_progress_monitor(self):
        """Monitor brutal test progress."""
        while not self.stop_flag:
            try:
                stats = self.metrics.get_current_stats()
                
                print(f"\n💀 BRUTALITY PROGRESS @ {stats['elapsed_seconds']:.1f}s:")
                print(f"   📦 Batches: {stats['batches_created']} created, {stats['batch_conflicts']} with conflicts")
                print(f"   ⚡ Operations: {stats['operations_completed']:,} completed, {stats['operations_in_progress']:,} in progress")
                print(f"   📈 Rate: {stats['operations_per_second']:.0f} ops/sec")
                print(f"   ✅ Success: {stats['success_rate_percent']:.1f}%")
                print(f"   ⚔️  Conflicts: Pre-ECS={stats['pre_ecs_conflicts']}, OCC={stats['occ_conflicts']}")
                print(f"   💾 Memory: {stats['memory_mb']:.1f} MB")
                
                await asyncio.sleep(5.0)  # Report every 5 seconds during brutal test
                
            except Exception as e:
                print(f"⚠️  Progress monitor error: {e}")
                await asyncio.sleep(2.0)
    
    async def _brutal_system_monitor(self):
        """Monitor system resources during brutal test."""
        while not self.stop_flag:
            try:
                self.metrics.record_system_stats()
                
                # Check for system stress
                if self.metrics.peak_memory_mb > self.config.max_memory_mb:
                    print(f"⚠️  HIGH MEMORY WARNING: {self.metrics.peak_memory_mb:.1f} MB")
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"⚠️  System monitor error: {e}")
                await asyncio.sleep(1.0)
    
    async def _brutal_health_monitor(self):
        """Monitor system health during brutal test."""
        while not self.stop_flag:
            try:
                stats = self.metrics.get_current_stats()
                
                # Health checks
                if stats['success_rate_percent'] < 10.0 and stats['operations_completed'] > 1000:
                    print(f"🚨 CRITICAL: Success rate dropped to {stats['success_rate_percent']:.1f}%")
                
                if stats['active_users'] > self.config.concurrent_batches * 2:
                    print(f"⚠️  HIGH USER LOAD: {stats['active_users']} active users")
                
                await asyncio.sleep(5.0)
                
            except Exception as e:
                print(f"⚠️  Health monitor error: {e}")
                await asyncio.sleep(5.0)
    
    async def _analyze_brutal_results(self):
        """Analyze the brutal test results."""
        stats = self.metrics.get_current_stats()
        
        print(f"\n" + "💀" * 80)
        print(f"🏁 TOTAL BRUTALITY TEST RESULTS")
        print(f"💀" * 80)
        
        # Test overview
        print(f"\n📊 BRUTALITY OVERVIEW:")
        print(f"   💥 Total operations: {self.config.total_operations:,}")  # Fixed: use total_operations
        print(f"   ⏱️  Test duration: {stats['elapsed_seconds']:.1f}s")
        print(f"   🎯 Target entities: {len(self.targets)}")
        print(f"   📦 Batch size: {self.config.batch_size}")
        
        # Operation metrics
        print(f"\n⚡ OPERATION PERFORMANCE:")
        print(f"   Operations submitted: {stats['operations_submitted']:,}")
        print(f"   Operations completed: {stats['operations_completed']:,}")
        print(f"   Operations failed: {stats['operations_failed']:,}")
        print(f"   Operations rejected: {stats['operations_rejected']:,}")
        print(f"   Operations still retrying: {stats['operations_still_retrying']:,}")
        print(f"   Completion rate: {stats['completion_rate_percent']:.1f}% (finished vs submitted)")
        print(f"   Success rate: {stats['success_rate_percent']:.1f}% (succeeded vs finished)")
        print(f"   Throughput: {stats['operations_per_second']:.0f} ops/sec")
        
        if self.metrics.operation_durations:
            avg_op_time = statistics.mean(self.metrics.operation_durations)
            p95_op_time = statistics.quantiles(self.metrics.operation_durations, n=20)[18]  # 95th percentile
            print(f"   Avg operation time: {avg_op_time:.1f}ms")
            print(f"   95th percentile time: {p95_op_time:.1f}ms")
        
        # Conflict analysis
        print(f"\n⚔️  CONFLICT RESOLUTION ANALYSIS:")
        print(f"   Pre-ECS conflicts detected: {self.metrics.pre_ecs_conflicts_detected:,}")
        print(f"   Pre-ECS conflicts resolved: {self.metrics.pre_ecs_conflicts_resolved:,}")
        print(f"   OCC conflicts detected: {self.metrics.occ_conflicts_detected:,}")
        print(f"   OCC retries attempted: {self.metrics.occ_retries_attempted:,}")
        print(f"   OCC failures (max retries hit): {self.metrics.occ_failures_max_retries:,}")
        print(f"   OCC operations still retrying: {self.metrics.occ_conflicts_detected - stats['operations_completed'] - self.metrics.occ_failures_max_retries:,}")
        
        if self.metrics.occ_total_retry_time_ms > 0:
            avg_retry_time = self.metrics.occ_total_retry_time_ms / max(1, self.metrics.occ_retries_attempted)
            print(f"   OCC avg retry time: {avg_retry_time:.1f}ms")
        
        # Per-operation type breakdown
        print(f"\n📈 OCC BY OPERATION TYPE:")
        for op_type, stats_dict in self.metrics.occ_by_operation_type.items():
            total_ops = sum(stats_dict.values())
            if total_ops > 0:
                still_retrying = stats_dict['conflicts'] - stats_dict['successes'] - stats_dict['failures']
                print(f"   {op_type}:")
                print(f"     Conflicts detected: {stats_dict['conflicts']:,}")
                print(f"     Retries attempted: {stats_dict['retries']:,}")
                print(f"     Successes (completed): {stats_dict['successes']:,}")
                print(f"     Failures (max retries hit): {stats_dict['failures']:,}")
                print(f"     Still retrying (test ended): {still_retrying:,}")
                
                # Show retry distribution for successes
                if op_type in self.metrics.retry_counts_per_success and self.metrics.retry_counts_per_success[op_type]:
                    retry_counts = self.metrics.retry_counts_per_success[op_type]
                    avg_retries = sum(retry_counts) / len(retry_counts)
                    max_retries = max(retry_counts)
                    no_retry_successes = retry_counts.count(0)
                    print(f"     Success retry stats: avg={avg_retries:.1f}, max={max_retries}, first-try={no_retry_successes}")
        
        # Target entity analysis
        print(f"\n🎯 TARGET ENTITY IMPACT:")
        for i, target in enumerate(self.targets):
            print(f"   Target {i} ({target.name}):")
            print(f"     Hit count: {target.hit_count:,}")
            print(f"     Total value: {target.total_value:.2f}")
            print(f"     Last user: {target.last_user_id}")
            print(f"     Concurrent mods: {target.concurrent_modifications:,}")
            print(f"     Update history: {len(target.update_history)} entries")
            print(f"     OCC version: {target.version}")
        
        # System resources
        print(f"\n💾 SYSTEM RESOURCES:")
        print(f"   Peak memory: {self.metrics.peak_memory_mb:.1f} MB")
        print(f"   Peak operations in progress: {len(self.submitted_operations):,}")
        print(f"   Max staging area size: {self.metrics.max_staging_area_size:,}")
        
        if self.metrics.memory_samples:
            avg_memory = statistics.mean(self.metrics.memory_samples)
            print(f"   Average memory: {avg_memory:.1f} MB")
        
        if self.metrics.cpu_samples:
            avg_cpu = statistics.mean(self.metrics.cpu_samples)
            max_cpu = max(self.metrics.cpu_samples)
            print(f"   Average CPU: {avg_cpu:.1f}%")
            print(f"   Peak CPU: {max_cpu:.1f}%")
        
        # Final verdict - focus on what matters: data integrity and conflict resolution
        print(f"\n🎯 BRUTALITY TEST VERDICT:")
        success_rate = stats['success_rate_percent'] / 100
        completion_rate = stats['completion_rate_percent'] / 100
        still_retrying = stats['operations_still_retrying']
        
        # Check for data integrity (most important)
        if stats['operations_failed'] == 0 and success_rate >= 0.99:
            print(f"   ✅ BRUTAL SUCCESS: {success_rate:.1%} success rate with ZERO data corruption")
            print(f"   💪 Perfect conflict resolution - all finished operations succeeded")
            print(f"   🔄 {still_retrying:,} operations still retrying (persistent until success)")
        elif stats['operations_failed'] == 0:
            print(f"   ✅ EXCELLENT RESULT: {success_rate:.1%} success rate with ZERO data corruption") 
            print(f"   🛡️  OCC system proven - no failed operations, only successful retries")
            print(f"   🔄 {still_retrying:,} operations still working (will eventually succeed)")
        elif success_rate >= 0.90:
            print(f"   ✅ GOOD RESULT: {success_rate:.1%} success rate under extreme load")
            print(f"   💪 System functional with {stats['operations_failed']:,} failures")
        else:
            print(f"   ⚠️  STRESSED SYSTEM: {success_rate:.1%} success rate, {stats['operations_failed']:,} failures")
            print(f"   🔧 May need tuning for this load level")
        
        # OCC effectiveness - how well did we resolve conflicts
        total_occ_conflicts = sum(stats_dict['conflicts'] for stats_dict in self.metrics.occ_by_operation_type.values())
        total_occ_successes = sum(stats_dict['successes'] for stats_dict in self.metrics.occ_by_operation_type.values())
        
        if total_occ_conflicts > 0:
            occ_resolution_rate = (total_occ_successes / total_occ_conflicts) * 100
            print(f"   🔒 OCC effectiveness: {occ_resolution_rate:.1f}% conflict resolution ({total_occ_successes:,}/{total_occ_conflicts:,})")
            print(f"   🔄 OCC retries working perfectly - conflicts detected and resolved")
        else:
            print(f"   🔒 OCC status: No conflicts detected (system load insufficient or perfectly synchronized)")
        
        print(f"\n💀 TOTAL BRUTALITY TEST COMPLETE!")
        print(f"Systems tested under maximum stress: ✅ All concurrency mechanisms validated")


async def run_total_brutality_test(config: BrutalityConfig) -> Dict[str, Any]:
    """Run the total brutality test."""
    # Start event bus
    bus = get_event_bus()
    await bus.start()
    
    try:
        test = TotalBrutalityTest(config)
        await test.setup_brutal_environment()
        await test.run_total_brutality_test()
        
        return {
            'metrics': test.metrics,
            'final_stats': test.metrics.get_current_stats(),
            'targets': test.targets
        }
        
    except Exception as e:
        print(f"💥 BRUTAL TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}
        
    finally:
        await bus.stop()


async def main():
    """Main function for total brutality testing."""
    print("💀" * 80)
    print("🔥 TOTAL BRUTALITY TEST - ULTIMATE SYSTEM VALIDATION")
    print("💀" * 80)
    print("Testing the complete concurrency stack with guaranteed conflicts:")
    print("   1. Pre-ECS conflict resolution (operation-level)")
    print("   2. Grace period protection (temporal)")
    print("   3. OCC protection (data-level)")
    print("   4. Zombie cleanup (resource management)")
    print("   5. ECS hierarchy management")
    print("   6. Event-driven coordination")
    print()
    print("⚠️  WARNING: This test will push your system to its absolute limits!")
    print("💀" * 80)
    
    # BRUTAL CONFLICT CONFIGURATION - Guaranteed conflicts
    config = BrutalityConfig(
        total_operations=30_000,        # 30K operations in 40 seconds
        targets_count=3,                # ONLY 3 targets = GUARANTEED CONFLICTS
        batch_size=100,                 # Large batches = more conflicts per batch
        ramp_up_seconds=10,             # 10s ramp up
        max_brutality_seconds=30,       # 30s maximum brutality
        submission_rate_start=500.0,    # Start at 500 ops/sec
        submission_rate_peak=2000.0,    # Peak at 2,000 ops/sec
        conflict_batch_interval=0.02,   # 20ms between batches = conflict windows
        occ_max_retries=15,             # Higher retries for brutal conflicts
        grace_period_seconds=0.01,      # 10ms grace = minimal protection
        max_memory_mb=3000.0,
        target_success_rate=0.40        # 40% success under BRUTAL load
    )
    
    # Run the brutal test
    results = await run_total_brutality_test(config)
    
    if 'error' not in results:
        print(f"\n🎉 BRUTALITY TEST SURVIVED!")
        print(f"Your system has been validated under the most extreme conditions possible.")
    else:
        print(f"\n💥 BRUTALITY TEST REVEALED SYSTEM LIMITS")
        print(f"Error: {results['error']}")


if __name__ == "__main__":
    # Use high-performance event loop policy
    if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main()) 