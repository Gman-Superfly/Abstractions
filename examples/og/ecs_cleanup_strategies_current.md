# ECS Cleanup Strategies: Modern Conflict Resolution and Resource Management

## The ECS Pollution Problem

In production systems using the Abstractions framework, operations can create "zombie" ECS entities through various conflict resolution scenarios. This creates resource consumption and registry pollution that needs systematic cleanup.

### Core Problem Scenarios

#### **1. Pre-ECS Staging Rejections**
```python
# With @with_conflict_resolution decorators:
@with_conflict_resolution(pre_ecs=PreECSConfig(enabled=True))
async def update_account(account: BankAccount, amount: float) -> BankAccount:
    account.balance += amount
    return account

# Scenario:
# - 5 operations submitted simultaneously to same account
# - All 5 get ConflictResolutionOperation entities created
# - Pre-ECS staging resolves conflicts, picks 1 winner
# - 4 losers are rejected BUT already have ECS IDs
# - Result: 4 zombie ConflictResolutionOperation entities in registry
```

#### **2. OCC (Optimistic Concurrency Control) Failures**
```python
# Even winners can fail at OCC level:
@with_conflict_resolution(
    pre_ecs=PreECSConfig(enabled=True),
    occ=OCCConfig(enabled=True, max_retries=3)
)
async def critical_update(entity: CriticalEntity) -> CriticalEntity:
    # Complex business logic here
    return modified_entity

# Scenario:
# - Operation wins Pre-ECS conflict
# - Starts execution, gets ECS ID
# - During execution, another process modifies the entity
# - OCC detects version conflict, operation fails after max retries
# - Result: Failed ConflictResolutionOperation zombie in ECS registry
```

#### **3. Grace Period Conflicts (Manual Patterns)**
```python
# In manual conflict resolution (from stress tests):
# Operation A: Enters ECS, starts executing (protected by grace period)
# Operation B: Enters ECS, waits for grace period to expire
# Operation C: Enters ECS, higher priority than B
# Grace period expires: A can be preempted by C
# Result: A gets rejected but remains in ECS as zombie
```

### Production Impact

- **Registry Pollution**: Zombie operations consume ECS entity IDs permanently
- **Memory Waste**: Failed operations hold references to target entities
- **Query Performance**: ECS traversals include useless rejected operations
- **Audit Confusion**: Failed operations appear "successful" in ECS registry
- **Resource Leaks**: Long-running systems accumulate thousands of zombies

## Manual Implementation Approaches

### Pattern 1: Immediate Cleanup (Aggressive)

**Best for: Resource-constrained systems, high-throughput operations**

```python
class ImmediateCleanupStrategy:
    """Aggressively remove rejected operations immediately."""
    
    async def handle_operation_rejection(self, operation: ConflictResolutionOperation):
        """Clean up rejected operation immediately."""
        try:
            # Mark as rejected
            operation.status = OperationStatus.REJECTED
            operation.completed_at = datetime.now(timezone.utc)
            
            # Remove from ECS registry immediately
            await self._remove_from_ecs_registry(operation.ecs_id)
            
            # Signal any waiting processes
            if hasattr(operation, 'execution_event') and operation.execution_event:
                operation.execution_event.set()
            
            # Clean up partial work
            await self._rollback_partial_operations(operation)
            
            logger.info(f"Immediately cleaned zombie operation {operation.ecs_id}")
            
        except Exception as e:
            logger.error(f"Failed immediate cleanup of {operation.ecs_id}: {e}")
    
    async def _remove_from_ecs_registry(self, ecs_id: UUID):
        """Remove entity from ECS registry completely."""
        # Implementation depends on your ECS registry structure
        # This is the aggressive approach - immediate removal
        from abstractions.ecs.base_registry import EntityRegistry
        
        # Remove from tree registry
        root_id = EntityRegistry.ecs_id_to_root_id.get(ecs_id)
        if root_id and root_id in EntityRegistry.tree_registry:
            tree = EntityRegistry.tree_registry[root_id]
            if ecs_id in tree.nodes:
                del tree.nodes[ecs_id]
                tree.node_count = len(tree.nodes)
        
        # Remove from ID mappings
        EntityRegistry.ecs_id_to_root_id.pop(ecs_id, None)
        
        # Remove from lineage if present
        if hasattr(EntityRegistry, 'lineage_registry'):
            EntityRegistry.lineage_registry.pop(ecs_id, None)
    
    async def _rollback_partial_operations(self, operation: ConflictResolutionOperation):
        """Clean up any partial work done by the operation."""
        # Implementation depends on operation type
        # This is where you'd undo any partial state changes
        pass
```

**Pros:** No resource waste, immediate cleanup, predictable memory usage  
**Cons:** No debugging window, complex rollback logic, risk of cleaning operations that might be retried

### Pattern 2: Deferred Cleanup (Conservative)

**Best for: Development environments, systems needing debugging capability**

```python
class DeferredCleanupStrategy:
    """Retain rejected operations briefly for debugging, then clean up."""
    
    def __init__(self, retention_minutes: int = 30):
        self.retention_minutes = retention_minutes
        self.rejected_operations: Dict[UUID, datetime] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
    
    async def handle_operation_rejection(self, operation: ConflictResolutionOperation):
        """Mark for deferred cleanup."""
        # Mark as rejected but keep in ECS temporarily
        operation.status = OperationStatus.REJECTED
        operation.completed_at = datetime.now(timezone.utc)
        
        # Schedule for cleanup
        self.rejected_operations[operation.ecs_id] = operation.completed_at
        
        # Start cleanup worker if not running
        if not self.cleanup_task or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        logger.info(f"Scheduled deferred cleanup for {operation.ecs_id}")
    
    async def _cleanup_worker(self):
        """Background worker to clean expired rejections."""
        while self.rejected_operations:
            try:
                now = datetime.now(timezone.utc)
                cutoff = now - timedelta(minutes=self.retention_minutes)
                
                expired_ops = [
                    op_id for op_id, rejected_at in self.rejected_operations.items()
                    if rejected_at < cutoff
                ]
                
                for op_id in expired_ops:
                    await self._cleanup_expired_operation(op_id)
                    del self.rejected_operations[op_id]
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_operation(self, op_id: UUID):
        """Clean up specific expired operation."""
        try:
            await self._remove_from_ecs_registry(op_id)
            logger.info(f"Cleaned expired operation {op_id}")
        except Exception as e:
            logger.error(f"Failed to clean expired operation {op_id}: {e}")
    
    async def _remove_from_ecs_registry(self, op_id: UUID):
        """Remove entity from ECS registry completely."""
        from abstractions.ecs.base_registry import EntityRegistry
        
        # Same implementation as immediate cleanup
        root_id = EntityRegistry.ecs_id_to_root_id.get(op_id)
        if root_id and root_id in EntityRegistry.tree_registry:
            tree = EntityRegistry.tree_registry[root_id]
            if op_id in tree.nodes:
                del tree.nodes[op_id]
                tree.node_count = len(tree.nodes)
        
        EntityRegistry.ecs_id_to_root_id.pop(op_id, None)
        if hasattr(EntityRegistry, 'lineage_registry'):
            EntityRegistry.lineage_registry.pop(op_id, None)
```

**Pros:** Debugging window preserved, safe verification, background processing  
**Cons:** Temporary resource consumption, requires background worker, more complex

### Pattern 3: Lazy Cleanup (On-Demand)

**Best for: Low-traffic systems, systems with irregular cleanup needs**

```python
class LazyCleanupStrategy:
    """Clean up zombies when encountered during normal operations."""
    
    def __init__(self, max_zombie_age_hours: int = 1):
        self.max_zombie_age_hours = max_zombie_age_hours
    
    async def handle_operation_rejection(self, operation: ConflictResolutionOperation):
        """Simply mark as rejected - cleanup happens later."""
        operation.status = OperationStatus.REJECTED
        operation.completed_at = datetime.now(timezone.utc)
        
        # No immediate cleanup - relies on lazy cleanup during queries
        logger.info(f"Marked {operation.ecs_id} for lazy cleanup")
    
    def filter_active_operations(self, operations: List[ConflictResolutionOperation]) -> List[ConflictResolutionOperation]:
        """Filter out zombie operations during ECS queries."""
        active_ops = []
        
        for op in operations:
            if self.is_zombie_operation(op):
                # Clean up zombie during query
                asyncio.create_task(self._cleanup_zombie_operation(op))
                logger.info(f"Lazy cleanup triggered for {op.ecs_id}")
            else:
                active_ops.append(op)
        
        return active_ops
    
    def is_zombie_operation(self, operation: ConflictResolutionOperation) -> bool:
        """Check if operation is an expired zombie."""
        if operation.status != OperationStatus.REJECTED:
            return False
        
        if not operation.completed_at:
            return False
        
        age = datetime.now(timezone.utc) - operation.completed_at
        return age.total_seconds() > (self.max_zombie_age_hours * 3600)
    
    async def _cleanup_zombie_operation(self, operation: ConflictResolutionOperation):
        """Clean up zombie found during query."""
        try:
            await self._remove_from_ecs_registry(operation.ecs_id)
            logger.info(f"Lazy cleaned zombie {operation.ecs_id}")
        except Exception as e:
            logger.error(f"Failed lazy cleanup of {operation.ecs_id}: {e}")
    
    async def _remove_from_ecs_registry(self, op_id: UUID):
        """Remove entity from ECS registry completely."""
        from abstractions.ecs.base_registry import EntityRegistry
        
        # Same implementation as other strategies
        root_id = EntityRegistry.ecs_id_to_root_id.get(op_id)
        if root_id and root_id in EntityRegistry.tree_registry:
            tree = EntityRegistry.tree_registry[root_id]
            if op_id in tree.nodes:
                del tree.nodes[op_id]
                tree.node_count = len(tree.nodes)
        
        EntityRegistry.ecs_id_to_root_id.pop(op_id, None)
        if hasattr(EntityRegistry, 'lineage_registry'):
            EntityRegistry.lineage_registry.pop(op_id, None)
```

**Pros:** No background workers, cleanup happens naturally, minimal overhead  
**Cons:** Inconsistent timing, zombies persist in unused areas, query performance impact

## Current Decorator Behavior

The current `@with_conflict_resolution` decorators handle cleanup implicitly through the staging coordinator. When operations are rejected in Pre-ECS staging:

```python
# Current implementation (simplified):
# 1. Operations submitted to staging area
# 2. Staging coordinator resolves conflicts after 100ms timeout
# 3. Winner executes, losers are marked as rejected
# 4. Rejected operations remain in ECS registry (potential zombies)
```

**Note:** The current implementation does NOT automatically clean up rejected operations. This is a known limitation that may need manual cleanup strategies in production systems.

## Monitoring and Health Checks

### Essential Metrics for All Strategies

```python
class CleanupHealthMonitor:
    """Monitor cleanup effectiveness across all strategies."""
    
    def __init__(self):
        self.zombie_count = 0
        self.cleanup_failures = 0
        self.oldest_zombie_age_hours = 0.0
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform cleanup system health check."""
        
        # Count current zombies in ECS registry
        zombie_count = await self._count_zombie_operations()
        oldest_zombie_age = await self._get_oldest_zombie_age_hours()
        cleanup_failure_rate = self._get_cleanup_failure_rate()
        
        status = "healthy"
        issues = []
        
        if zombie_count > 1000:
            status = "degraded"
            issues.append(f"High zombie count: {zombie_count}")
        
        if oldest_zombie_age > 24:
            status = "degraded"
            issues.append(f"Old zombies detected: {oldest_zombie_age:.1f}h")
        
        if cleanup_failure_rate > 0.1:  # 10% failure rate
            status = "unhealthy"
            issues.append(f"High cleanup failure rate: {cleanup_failure_rate:.2%}")
        
        return {
            "status": status,
            "zombie_count": zombie_count,
            "oldest_zombie_age_hours": oldest_zombie_age,
            "cleanup_failure_rate": cleanup_failure_rate,
            "issues": issues
        }
    
    async def _count_zombie_operations(self) -> int:
        """Count rejected operations still in ECS registry."""
        from abstractions.ecs.base_registry import EntityRegistry
        
        zombie_count = 0
        
        # Iterate through all trees in registry
        for root_id, tree in EntityRegistry.tree_registry.items():
            for ecs_id, entity in tree.nodes.items():
                # Check if entity is a ConflictResolutionOperation with rejected status
                if (hasattr(entity, 'status') and 
                    hasattr(entity, 'op_type') and
                    entity.status == OperationStatus.REJECTED):
                    zombie_count += 1
        
        return zombie_count
    
    async def _get_oldest_zombie_age_hours(self) -> float:
        """Find age of oldest zombie operation."""
        from abstractions.ecs.base_registry import EntityRegistry
        
        oldest_age_hours = 0.0
        now = datetime.now(timezone.utc)
        
        # Iterate through all trees in registry
        for root_id, tree in EntityRegistry.tree_registry.items():
            for ecs_id, entity in tree.nodes.items():
                if (hasattr(entity, 'status') and 
                    hasattr(entity, 'completed_at') and
                    entity.status == OperationStatus.REJECTED and
                    entity.completed_at):
                    
                    age = now - entity.completed_at
                    age_hours = age.total_seconds() / 3600
                    oldest_age_hours = max(oldest_age_hours, age_hours)
        
        return oldest_age_hours
    
    def _get_cleanup_failure_rate(self) -> float:
        """Calculate cleanup failure rate from recent operations."""
        # This would track cleanup attempts vs successes
        # Implementation depends on your metrics collection
        return 0.0  # Placeholder
```

## Production Recommendations

### Strategy Selection Guide

| System Type | Recommended Strategy | Retention | Monitoring Priority |
|-------------|---------------------|-----------|-------------------|
| **High-throughput APIs** | Immediate | None | Memory usage, cleanup latency |
| **Financial/Critical** | Deferred | 1-2 hours | Failure patterns, audit trail |
| **Development/Testing** | Deferred | 4-8 hours | Debug effectiveness, zombie patterns |
| **Background/Batch** | Lazy | 6-24 hours | Cleanup trigger frequency |
| **Memory-constrained** | Immediate | None | Memory pressure, cleanup effectiveness |

### Essential Production Monitoring

**All systems should track:**
1. **Zombie Count**: Current number of rejected operations in ECS
2. **Zombie Age**: Maximum age of unclean zombies
3. **Cleanup Latency**: Time from rejection to cleanup completion
4. **Cleanup Success Rate**: Percentage of successful cleanup operations
5. **Memory Impact**: Resource consumption of zombie operations

**Alerts should trigger on:**
- Zombie count > 1000 operations
- Zombie age > 24 hours
- Cleanup failure rate > 10%
- Memory pressure > 80% with active zombies

## Testing Cleanup Strategies

```python
class CleanupStrategyTest:
    """Test suite for validating cleanup strategies."""
    
    async def test_zombie_prevention(self):
        """Test that zombies are properly cleaned up."""
        
        # Create operation that will be rejected
        operation = ConflictResolutionOperation(
            function_name="test_operation",
            target_entities=[uuid4()],
            priority=OperationPriority.NORMAL
        )
        operation.promote_to_root()  # Enters ECS
        
        # Simulate rejection
        cleanup_strategy = ImmediateCleanupStrategy()
        await cleanup_strategy.handle_operation_rejection(operation)
        
        # Verify cleanup
        assert not self._is_in_ecs_registry(operation.ecs_id)
    
    def _is_in_ecs_registry(self, ecs_id: UUID) -> bool:
        """Check if entity is still in ECS registry."""
        from abstractions.ecs.base_registry import EntityRegistry
        
        return ecs_id in EntityRegistry.ecs_id_to_root_id
    
    async def test_cleanup_performance(self):
        """Test cleanup performance under load."""
        
        # Create many operations for rejection
        operations = []
        for i in range(1000):
            op = ConflictResolutionOperation(
                function_name=f"test_operation_{i}",
                target_entities=[uuid4()],
                priority=OperationPriority.NORMAL
            )
            op.promote_to_root()
            operations.append(op)
        
        start_time = time.time()
        cleanup_strategy = ImmediateCleanupStrategy()
        
        # Reject all operations
        for op in operations:
            await cleanup_strategy.handle_operation_rejection(op)
        
        cleanup_time = time.time() - start_time
        
        # Verify performance
        assert cleanup_time < 5.0  # Should complete in 5 seconds
        
        # Verify all cleaned up
        for op in operations:
            assert not self._is_in_ecs_registry(op.ecs_id)
```

## Conclusion

ECS cleanup is a critical production concern that requires manual implementation since the current decorator system does not include automatic cleanup strategies. The manual patterns provided offer different trade-offs between debugging capability, resource efficiency, and implementation complexity.

**Key takeaways:**
- **Zombie operations** are a real resource management problem in production
- **Current decorators** do not automatically clean up rejected operations
- **Manual cleanup strategies** are required for production systems
- **Strategy selection** depends on system requirements (throughput vs debugging)
- **Monitoring** is essential for detecting cleanup issues early
- **Testing** cleanup behavior is crucial for production readiness

Choose the cleanup strategy that best matches your system's requirements and implement appropriate monitoring to ensure production health. 