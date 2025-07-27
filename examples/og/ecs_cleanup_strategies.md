# ECS Cleanup Strategies: Managing Rejected Operations in Production

## The ECS Pollution Problem

In production ECS systems, operations can enter the ECS registry and then be rejected by conflict resolution mechanisms (such as grace period protection). This creates **"zombie" ECS entities** that consume resources and pollute the registry.

### Problem Scenario

```python
GRACE PERIOD CONFLICT SCENARIO:
1. Operation A: Enters ECS, starts executing (protected by grace period)
2. Operation B: Enters ECS, higher priority but A is protected
3. Operation C: Enters ECS, even higher priority
4. Grace Period Expires: A can now be preempted by C
5. Result: A gets rejected but remains in ECS registry as "zombie"
```

### Production Impact

- **Registry Pollution**: Rejected operations consume ECS entity IDs permanently
- **Memory Waste**: Zombie entities take up memory until cleanup
- **Query Performance**: ECS queries include useless rejected operations  
- **Audit Confusion**: "Failed" operations appear successful in ECS registry
- **Resource Leaks**: Long-running systems accumulate zombie entities

## Cleanup Strategies

### Strategy 1: Immediate Cleanup (Aggressive)

**When**: Immediate cleanup upon operation rejection.

```python
class ImmediateCleanupStrategy:
    """Aggressively clean rejected operations immediately."""
    
    def handle_operation_rejection(self, operation: OperationEntity):
        """Immediately remove rejected operation from ECS."""
        try:
            # Remove from ECS registry immediately
            self.remove_from_ecs_registry(operation.ecs_id)
            
            # Clean up any partial work done
            self.rollback_partial_operations(operation)
            
            # Update metrics
            self.metrics.record_zombie_prevented()
            
        except Exception as e:
            # Log error but don't fail the rejection process
            logger.error(f"Failed to clean up rejected operation {operation.ecs_id}: {e}")
    
    def remove_from_ecs_registry(self, ecs_id: UUID):
        """Remove entity from all ECS mappings."""
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
        EntityRegistry.lineage_registry.pop(ecs_id, None)
```

**Pros:**
- No memory waste
- Clean registry at all times
- Immediate resource reclamation

**Cons:**
- Aggressive cleanup might interfere with debugging
- Risk of cleaning up operations that might be retried
- Complex rollback logic required

### Strategy 2: Deferred Cleanup (Conservative)

**When**: Short-term retention for debugging, then cleanup.

```python
class DeferredCleanupStrategy:
    """Conservative cleanup with short retention for debugging."""
    
    def __init__(self, retention_minutes: int = 30):
        self.retention_minutes = retention_minutes
        self.rejected_operations: Dict[UUID, datetime] = {}
        self.cleanup_task = None
    
    def handle_operation_rejection(self, operation: OperationEntity):
        """Mark operation for deferred cleanup."""
        # Mark operation as rejected but keep in ECS temporarily
        operation.status = OperationStatus.REJECTED
        operation.completed_at = datetime.now(timezone.utc)
        
        # Schedule for cleanup
        self.rejected_operations[operation.ecs_id] = operation.completed_at
        
        # Start cleanup task if not running
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self.cleanup_worker())
    
    async def cleanup_worker(self):
        """Background worker to clean up expired rejected operations."""
        while True:
            try:
                now = datetime.now(timezone.utc)
                cutoff = now - timedelta(minutes=self.retention_minutes)
                
                expired_ops = [
                    op_id for op_id, rejected_at in self.rejected_operations.items()
                    if rejected_at < cutoff
                ]
                
                for op_id in expired_ops:
                    await self.cleanup_expired_operation(op_id)
                    del self.rejected_operations[op_id]
                
                # Sleep before next cleanup cycle
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                await asyncio.sleep(60)
    
    async def cleanup_expired_operation(self, op_id: UUID):
        """Clean up a specific expired operation."""
        try:
            # Remove from ECS registry
            self.remove_from_ecs_registry(op_id)
            self.metrics.record_zombie_cleaned()
            
        except Exception as e:
            logger.error(f"Failed to clean expired operation {op_id}: {e}")
```

**Pros:**
- Debugging window preserved
- Safe cleanup with verification
- Background processing doesn't block main flow

**Cons:**
- Temporary resource consumption
- More complex lifecycle management
- Requires background worker

### Strategy 3: Lazy Cleanup (On-Demand)

**When**: Cleanup during ECS queries/operations when zombie entities are encountered.

```python
class LazyCleanupStrategy:
    """Clean up zombie entities when encountered during normal operations."""
    
    def __init__(self, max_zombie_age_hours: int = 1):
        self.max_zombie_age_hours = max_zombie_age_hours
    
    def filter_active_operations(self, operations: List[OperationEntity]) -> List[OperationEntity]:
        """Filter out zombie operations during queries."""
        active_ops = []
        
        for op in operations:
            if self.is_zombie_operation(op):
                # Clean up zombie during query
                asyncio.create_task(self.cleanup_zombie_operation(op))
            else:
                active_ops.append(op)
        
        return active_ops
    
    def is_zombie_operation(self, operation: OperationEntity) -> bool:
        """Check if operation is a zombie that should be cleaned."""
        if operation.status != OperationStatus.REJECTED:
            return False
        
        if not operation.completed_at:
            return False
        
        age = datetime.now(timezone.utc) - operation.completed_at
        return age.total_seconds() > (self.max_zombie_age_hours * 3600)
    
    async def cleanup_zombie_operation(self, operation: OperationEntity):
        """Clean up a zombie operation encountered during query."""
        try:
            logger.info(f"Lazy cleanup of zombie operation {operation.ecs_id}")
            self.remove_from_ecs_registry(operation.ecs_id)
            self.metrics.record_lazy_cleanup()
            
        except Exception as e:
            logger.error(f"Failed lazy cleanup of {operation.ecs_id}: {e}")
```

**Pros:**
- No background workers needed
- Cleanup happens naturally during system use
- Minimal performance impact

**Cons:**
- Inconsistent cleanup timing
- Zombies might persist in unused parts of registry
- Query performance can be affected

### Strategy 4: Hybrid Cleanup (Recommended)

**When**: Combines multiple strategies for optimal production behavior.

```python
class HybridCleanupStrategy:
    """Production-ready hybrid cleanup combining multiple strategies."""
    
    def __init__(self, config: CleanupConfig):
        self.config = config
        self.immediate_cleanup = ImmediateCleanupStrategy()
        self.deferred_cleanup = DeferredCleanupStrategy(config.retention_minutes)
        self.lazy_cleanup = LazyCleanupStrategy(config.max_zombie_age_hours)
        
    def handle_operation_rejection(self, operation: OperationEntity):
        """Handle rejection with appropriate cleanup strategy."""
        
        # Determine cleanup strategy based on operation characteristics
        if self.should_cleanup_immediately(operation):
            # Critical operations or resource-heavy operations
            self.immediate_cleanup.handle_operation_rejection(operation)
            
        elif self.should_retain_for_debugging(operation):
            # Operations that might need debugging
            self.deferred_cleanup.handle_operation_rejection(operation)
            
        else:
            # Standard operations - lazy cleanup
            operation.status = OperationStatus.REJECTED
            operation.completed_at = datetime.now(timezone.utc)
    
    def should_cleanup_immediately(self, operation: OperationEntity) -> bool:
        """Determine if operation should be cleaned immediately."""
        return (
            operation.priority == OperationPriority.CRITICAL or
            operation.operation_type in self.config.resource_heavy_operations or
            self.get_system_memory_pressure() > 0.8
        )
    
    def should_retain_for_debugging(self, operation: OperationEntity) -> bool:
        """Determine if operation should be retained for debugging."""
        return (
            operation.retry_count > 0 or  # Failed operations
            operation.operation_type in self.config.debug_important_operations or
            self.is_debug_mode_enabled()
        )
```

## Production Implementation

### Integration with Operation Lifecycle

```python
# In operation lifecycle driver
async def handle_operation_rejection(self, operation: OperationEntity, reason: str):
    """Handle operation rejection with appropriate cleanup."""
    
    # Update operation status
    operation.status = OperationStatus.REJECTED
    operation.error_message = reason
    operation.completed_at = datetime.now(timezone.utc)
    
    # Apply cleanup strategy
    await self.cleanup_strategy.handle_operation_rejection(operation)
    
    # Emit rejection event
    await emit(OperationRejectedEvent(
        op_id=operation.ecs_id,
        op_type=operation.op_type,
        target_entity_id=operation.target_entity_id,
        rejection_reason=reason,
        cleanup_strategy=self.cleanup_strategy.__class__.__name__
    ))
    
    # Update metrics
    self.metrics.record_operation_rejected(operation.priority)
```

### Configuration Management

```python
class CleanupConfig:
    """Configuration for ECS cleanup strategies."""
    
    def __init__(self):
        # Strategy selection
        self.strategy = "hybrid"  # immediate, deferred, lazy, hybrid
        
        # Timing configuration
        self.retention_minutes = 30  # For deferred cleanup
        self.max_zombie_age_hours = 1  # For lazy cleanup
        self.cleanup_check_interval_seconds = 60
        
        # Operation classification
        self.resource_heavy_operations = [
            "complex_update", "structural_operation", "batch_operation"
        ]
        
        self.debug_important_operations = [
            "critical_business_logic", "financial_transaction", "user_data_update"
        ]
        
        # Performance thresholds
        self.memory_pressure_threshold = 0.8
        self.max_zombie_count = 1000
        self.force_cleanup_threshold = 5000
```

### Monitoring and Alerting

```python
class CleanupMetrics:
    """Metrics for ECS cleanup operations."""
    
    def __init__(self):
        # Cleanup effectiveness
        self.zombies_prevented = Counter()
        self.zombies_cleaned = Counter()
        self.lazy_cleanups = Counter()
        self.cleanup_failures = Counter()
        
        # Performance impact
        self.cleanup_duration = Histogram()
        self.registry_size_after_cleanup = Gauge()
        self.zombie_age_at_cleanup = Histogram()
        
        # Health indicators
        self.current_zombie_count = Gauge()
        self.oldest_zombie_age_hours = Gauge()
        self.cleanup_backlog_size = Gauge()
    
    def record_cleanup_health_check(self):
        """Record health metrics for monitoring."""
        zombie_count = self.count_zombie_operations()
        oldest_zombie_age = self.get_oldest_zombie_age()
        
        self.current_zombie_count.set(zombie_count)
        self.oldest_zombie_age_hours.set(oldest_zombie_age)
        
        # Alert if thresholds exceeded
        if zombie_count > 1000:
            logger.warning(f"High zombie count: {zombie_count}")
        
        if oldest_zombie_age > 24:  # hours
            logger.warning(f"Old zombies detected: {oldest_zombie_age} hours")
```

### Health Checks

```python
async def cleanup_health_check() -> HealthStatus:
    """Health check for ECS cleanup system."""
    
    zombie_count = count_zombie_operations()
    oldest_zombie_age = get_oldest_zombie_age_hours()
    cleanup_worker_status = check_cleanup_worker_status()
    
    issues = []
    
    if zombie_count > 1000:
        issues.append(f"High zombie count: {zombie_count}")
    
    if oldest_zombie_age > 24:
        issues.append(f"Old zombies: {oldest_zombie_age}h")
    
    if not cleanup_worker_status:
        issues.append("Cleanup worker not running")
    
    if not issues:
        return HealthStatus.HEALTHY
    elif len(issues) <= 2:
        return HealthStatus.DEGRADED
    else:
        return HealthStatus.UNHEALTHY
```

## Production Recommendations

### For High-Throughput Systems

**Use Hybrid Strategy** with:
- **Immediate cleanup** for critical/resource-heavy operations
- **Deferred cleanup (15min)** for debuggable operations  
- **Lazy cleanup** for standard operations
- Background cleanup every 30 seconds
- Monitor zombie count, age, and cleanup latency

### For Debug-Heavy Environments

**Use Deferred Strategy** with:
- **2-hour retention window** for investigation
- Detailed logging of rejected operations
- Manual cleanup tools for investigation
- Alerting on unusual rejection patterns
- Monitor zombie count, age, and cleanup latency

### For Resource-Constrained Systems

**Use Immediate Strategy (aggressive)** with:
- Aggressive cleanup to preserve memory
- Minimal retention (5 minutes max)
- Priority-based cleanup ordering
- Memory pressure monitoring
- Monitor zombie count, age, and cleanup latency

### Essential Monitoring

**All production systems should monitor:**
- **Zombie count**: Current number of rejected operations in ECS
- **Zombie age**: How long zombies remain before cleanup
- **Cleanup latency**: Time taken to clean up zombies
- **Cleanup effectiveness**: Percentage of zombies successfully cleaned
- **Memory impact**: Resource consumption of zombie entities
- **Registry pollution**: Growth rate of ECS registry due to zombies

## Testing Cleanup Strategies

```python
class CleanupStrategyTest:
    """Test suite for validating cleanup strategies."""
    
    async def test_zombie_prevention(self):
        """Test that zombies are properly cleaned up."""
        
        # Create operation that will be rejected
        operation = create_test_operation()
        operation.promote_to_root()  # Enters ECS
        
        # Simulate rejection
        await self.cleanup_strategy.handle_operation_rejection(operation)
        
        # Verify cleanup based on strategy
        if isinstance(self.cleanup_strategy, ImmediateCleanupStrategy):
            assert not self.is_in_ecs_registry(operation.ecs_id)
        else:
            assert self.is_marked_for_cleanup(operation.ecs_id)
    
    async def test_cleanup_performance(self):
        """Test cleanup performance under load."""
        
        # Create many operations for rejection
        operations = [create_test_operation() for _ in range(1000)]
        
        start_time = time.time()
        
        # Reject all operations
        for op in operations:
            await self.cleanup_strategy.handle_operation_rejection(op)
        
        cleanup_time = time.time() - start_time
        
        # Verify performance
        assert cleanup_time < 5.0  # Should complete in 5 seconds
        assert self.get_memory_usage_increase() < 100  # MB
```

## Conclusion

The ECS pollution problem is a real production concern that requires careful strategy selection based on system requirements. The hybrid approach provides the best balance of performance, debuggability, and resource efficiency for most production systems.

**Key Takeaways:**
- Rejected operations create "zombie" ECS entities
- Different cleanup strategies have different trade-offs
- Monitoring and health checks are essential
- Strategy should match system requirements (throughput vs debugging)
- Testing cleanup behavior is crucial for production readiness 