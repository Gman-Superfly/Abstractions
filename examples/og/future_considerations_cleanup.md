# Future Considerations: Configurable Cleanup Strategies for Decorators

## Current State Analysis

The existing `@with_conflict_resolution` decorator system provides robust conflict resolution through the staging coordinator, but **cleanup of rejected operations is implicit and non-configurable**:

### Current Behavior
```python
# Current decorator usage - no cleanup configuration
@with_conflict_resolution(
    pre_ecs=PreECSConfig(enabled=True),
    occ=OCCConfig(enabled=True, max_retries=3)
)
async def update_account(account: BankAccount, amount: float) -> BankAccount:
    account.balance += amount
    return account

# What happens today:
# 1. Operations submitted to staging coordinator
# 2. Conflicts resolved by priority 
# 3. Winner executes, losers are rejected
# 4. Rejected operations remain in ECS registry as "zombies"
# 5. No automatic cleanup of zombies
```

### Current Limitations
- **No cleanup configuration**: All operations use the same implicit cleanup behavior
- **One-size-fits-all**: Cannot customize cleanup strategy per operation type
- **Resource accumulation**: Zombie operations accumulate indefinitely
- **Manual intervention required**: Production systems need custom cleanup solutions

## Proposed Enhancement: CleanupConfig

### Enhanced Decorator API

```python
# Future decorator API with cleanup configuration
@with_conflict_resolution(
    pre_ecs=PreECSConfig(enabled=True),
    occ=OCCConfig(enabled=True, max_retries=3),
    cleanup=CleanupConfig(
        strategy="deferred",
        retention_minutes=30,
        aggressive_on_memory_pressure=True,
        custom_cleanup_handler=None
    )
)
async def update_critical_data(entity: CriticalEntity, data: dict) -> CriticalEntity:
    entity.data.update(data)
    return entity
```

### CleanupConfig Design

```python
from enum import Enum
from typing import Optional, Callable, Any
from pydantic import BaseModel, Field

class CleanupStrategy(str, Enum):
    """Available cleanup strategies for rejected operations."""
    NONE = "none"              # No cleanup - zombies remain indefinitely
    IMMEDIATE = "immediate"    # Clean up rejected operations immediately
    DEFERRED = "deferred"      # Clean up after retention period
    LAZY = "lazy"             # Clean up when encountered during queries
    CUSTOM = "custom"         # Use custom cleanup handler

class CleanupConfig(BaseModel):
    """Configuration for cleanup of rejected operations."""
    
    # Basic strategy selection
    strategy: CleanupStrategy = CleanupStrategy.DEFERRED
    
    # Timing configuration
    retention_minutes: int = Field(default=30, ge=0, le=1440)  # 0-24 hours
    cleanup_check_interval_seconds: int = Field(default=60, ge=10, le=3600)  # 10s-1h
    
    # Adaptive behavior
    aggressive_on_memory_pressure: bool = True
    memory_pressure_threshold: float = Field(default=0.8, ge=0.1, le=1.0)
    
    # Operation classification
    immediate_cleanup_priorities: List[int] = Field(default_factory=lambda: [OperationPriority.CRITICAL])
    debug_retention_priorities: List[int] = Field(default_factory=lambda: [OperationPriority.LOW])
    
    # Custom handling
    custom_cleanup_handler: Optional[str] = None  # Reference to registered cleanup handler
    
    # Performance limits
    max_zombie_count_per_target: int = Field(default=100, ge=1, le=10000)
    force_cleanup_threshold: int = Field(default=1000, ge=100, le=100000)

    def validate_config(self) -> bool:
        """Validate cleanup configuration."""
        if self.strategy == CleanupStrategy.CUSTOM and not self.custom_cleanup_handler:
            raise ValueError("Custom strategy requires custom_cleanup_handler")
        
        if self.strategy == CleanupStrategy.DEFERRED and self.retention_minutes == 0:
            raise ValueError("Deferred strategy requires retention_minutes > 0")
        
        return True
```

### Enhanced ConflictResolutionConfig

```python
class ConflictResolutionConfig(BaseModel):
    """Enhanced configuration with cleanup support."""
    
    mode: ConflictResolutionMode = ConflictResolutionMode.PRE_ECS
    pre_ecs: PreECSConfig = Field(default_factory=PreECSConfig)
    occ: OCCConfig = Field(default_factory=OCCConfig)
    cleanup: CleanupConfig = Field(default_factory=CleanupConfig)  # New field
    
    def get_effective_cleanup_strategy(self, operation: 'ConflictResolutionOperation') -> CleanupStrategy:
        """Determine effective cleanup strategy for an operation."""
        
        # Check for immediate cleanup priorities
        if operation.priority in self.cleanup.immediate_cleanup_priorities:
            return CleanupStrategy.IMMEDIATE
        
        # Check memory pressure for adaptive behavior
        if (self.cleanup.aggressive_on_memory_pressure and 
            _get_system_memory_pressure() > self.cleanup.memory_pressure_threshold):
            return CleanupStrategy.IMMEDIATE
        
        # Check for force cleanup threshold
        current_zombie_count = _get_current_zombie_count()
        if current_zombie_count > self.cleanup.force_cleanup_threshold:
            return CleanupStrategy.IMMEDIATE
        
        return self.cleanup.strategy
```

## Implementation Architecture

### Enhanced Staging Coordinator

```python
# Enhanced staging coordinator with cleanup integration
async def _process_conflict_group(target_id: UUID, operations: List[ConflictResolutionOperation]):
    """Process conflicts with integrated cleanup handling."""
    
    if len(operations) == 1:
        asyncio.create_task(_execute_winning_operation(operations[0]))
    else:
        # Resolve conflict
        winner = resolve_operation_conflicts(operations)
        losers = [op for op in operations if op.ecs_id != winner.ecs_id]
        
        # Execute winner
        asyncio.create_task(_execute_winning_operation(winner))
        
        # Handle rejected operations with configured cleanup
        for loser in losers:
            await _handle_operation_rejection_with_cleanup(loser, winner)

async def _handle_operation_rejection_with_cleanup(
    loser: ConflictResolutionOperation, 
    winner: ConflictResolutionOperation
):
    """Handle operation rejection with configured cleanup strategy."""
    
    # Get cleanup configuration from operation
    cleanup_config = _get_cleanup_config_for_operation(loser)
    effective_strategy = cleanup_config.get_effective_cleanup_strategy(loser)
    
    # Mark as rejected
    loser.status = OperationStatus.REJECTED
    loser.error_message = f"Lost Pre-ECS priority conflict (priority {loser.priority} vs winner {winner.priority})"
    loser.completed_at = datetime.now(timezone.utc)
    
    # Apply cleanup strategy
    cleanup_handler = _get_cleanup_handler(effective_strategy, cleanup_config)
    await cleanup_handler.handle_operation_rejection(loser)
    
    # Signal completion
    if hasattr(loser, 'execution_event') and loser.execution_event:
        loser.execution_event.set()
    
    # Emit events
    await emit(OperationRejectedEvent(
        op_id=loser.ecs_id,
        target_entity_id=loser.target_entity_id,
        rejection_reason=loser.error_message,
        cleanup_strategy=effective_strategy.value
    ))

def _get_cleanup_handler(strategy: CleanupStrategy, config: CleanupConfig) -> 'CleanupHandler':
    """Get appropriate cleanup handler for strategy."""
    
    if strategy == CleanupStrategy.IMMEDIATE:
        return ImmediateCleanupHandler()
    elif strategy == CleanupStrategy.DEFERRED:
        return DeferredCleanupHandler(retention_minutes=config.retention_minutes)
    elif strategy == CleanupStrategy.LAZY:
        return LazyCleanupHandler()
    elif strategy == CleanupStrategy.CUSTOM:
        return CustomCleanupHandler(config.custom_cleanup_handler)
    else:  # NONE
        return NoOpCleanupHandler()
```

### Cleanup Handler Registry

```python
from typing import Dict, Type
from abstractions.ecs.callable_registry import CallableRegistry

class CleanupHandler:
    """Base class for cleanup handlers."""
    
    async def handle_operation_rejection(self, operation: ConflictResolutionOperation):
        """Handle rejection of an operation."""
        raise NotImplementedError

class CleanupHandlerRegistry:
    """Registry for cleanup handlers."""
    
    _handlers: Dict[str, Type[CleanupHandler]] = {}
    
    @classmethod
    def register(cls, name: str, handler_class: Type[CleanupHandler]):
        """Register a cleanup handler."""
        cls._handlers[name] = handler_class
    
    @classmethod
    def get(cls, name: str) -> Type[CleanupHandler]:
        """Get a cleanup handler by name."""
        if name not in cls._handlers:
            raise ValueError(f"Unknown cleanup handler: {name}")
        return cls._handlers[name]
    
    @classmethod
    def create(cls, name: str, **kwargs) -> CleanupHandler:
        """Create an instance of a cleanup handler."""
        handler_class = cls.get(name)
        return handler_class(**kwargs)

# Register standard handlers
CleanupHandlerRegistry.register("immediate", ImmediateCleanupHandler)
CleanupHandlerRegistry.register("deferred", DeferredCleanupHandler)
CleanupHandlerRegistry.register("lazy", LazyCleanupHandler)
CleanupHandlerRegistry.register("noop", NoOpCleanupHandler)

# Custom handler support
@CleanupHandlerRegistry.register("financial_audit")
class FinancialAuditCleanupHandler(CleanupHandler):
    """Custom cleanup handler for financial operations."""
    
    def __init__(self, audit_retention_hours: int = 72):
        self.retention_hours = audit_retention_hours
    
    async def handle_operation_rejection(self, operation: ConflictResolutionOperation):
        # Custom financial audit cleanup logic
        await self._log_to_audit_trail(operation)
        await self._schedule_extended_retention(operation)
```

### Enhanced Operation Entity

```python
class ConflictResolutionOperation(OperationEntity):
    """Enhanced operation entity with cleanup configuration."""
    
    # ... existing fields ...
    
    # Cleanup configuration
    cleanup_config: Optional[CleanupConfig] = None
    cleanup_scheduled_at: Optional[datetime] = None
    cleanup_handler_name: Optional[str] = None
    
    def get_cleanup_config(self) -> CleanupConfig:
        """Get cleanup configuration for this operation."""
        return self.cleanup_config or CleanupConfig()  # Default config
    
    def schedule_cleanup(self, strategy: CleanupStrategy, retention_minutes: int = None):
        """Schedule cleanup for this operation."""
        if strategy == CleanupStrategy.IMMEDIATE:
            # Immediate cleanup - no scheduling needed
            return
        
        if strategy == CleanupStrategy.DEFERRED:
            retention = retention_minutes or self.get_cleanup_config().retention_minutes
            self.cleanup_scheduled_at = datetime.now(timezone.utc) + timedelta(minutes=retention)
        
        self.cleanup_handler_name = strategy.value
    
    def is_cleanup_due(self) -> bool:
        """Check if cleanup is due for this operation."""
        if not self.cleanup_scheduled_at:
            return False
        
        return datetime.now(timezone.utc) >= self.cleanup_scheduled_at
```

## Usage Examples

### High-Throughput Financial System

```python
# Aggressive cleanup for financial operations
@with_conflict_resolution(
    pre_ecs=PreECSConfig(enabled=True),
    occ=OCCConfig(enabled=True, max_retries=5),
    cleanup=CleanupConfig(
        strategy=CleanupStrategy.IMMEDIATE,  # No zombies allowed
        aggressive_on_memory_pressure=True,
        immediate_cleanup_priorities=[OperationPriority.CRITICAL, OperationPriority.HIGH]
    )
)
async def process_payment(transaction: PaymentTransaction) -> PaymentTransaction:
    transaction.process()
    return transaction
```

### Development Environment

```python
# Debug-friendly cleanup
@with_conflict_resolution(
    pre_ecs=PreECSConfig(enabled=True),
    cleanup=CleanupConfig(
        strategy=CleanupStrategy.DEFERRED,
        retention_minutes=120,  # 2-hour debugging window
        debug_retention_priorities=[OperationPriority.LOW, OperationPriority.NORMAL]
    )
)
async def update_user_profile(user: UserProfile, updates: dict) -> UserProfile:
    user.apply_updates(updates)
    return user
```

### Custom Financial Audit System

```python
# Custom cleanup for regulatory compliance
@with_conflict_resolution(
    pre_ecs=PreECSConfig(enabled=True),
    cleanup=CleanupConfig(
        strategy=CleanupStrategy.CUSTOM,
        custom_cleanup_handler="financial_audit",
        retention_minutes=4320  # 72 hours for financial audit
    )
)
async def financial_transaction(account: Account, amount: Decimal) -> Account:
    account.process_transaction(amount)
    return account
```

### Adaptive Cleanup Based on System Load

```python
# Smart cleanup that adapts to system conditions
@with_conflict_resolution(
    pre_ecs=PreECSConfig(enabled=True),
    cleanup=CleanupConfig(
        strategy=CleanupStrategy.DEFERRED,  # Default strategy
        retention_minutes=30,
        aggressive_on_memory_pressure=True,  # Switch to immediate if memory > 80%
        memory_pressure_threshold=0.8,
        force_cleanup_threshold=5000,  # Immediate cleanup if >5000 zombies
        max_zombie_count_per_target=50  # Per-target limits
    )
)
async def batch_process_records(record: DataRecord) -> DataRecord:
    record.process_batch_update()
    return record
```

## Migration Strategy

### Phase 1: Add CleanupConfig (Non-Breaking)

```python
# Add CleanupConfig to existing configuration
class ConflictResolutionConfig(BaseModel):
    mode: ConflictResolutionMode = ConflictResolutionMode.PRE_ECS
    pre_ecs: PreECSConfig = Field(default_factory=PreECSConfig)
    occ: OCCConfig = Field(default_factory=OCCConfig)
    cleanup: CleanupConfig = Field(default_factory=lambda: CleanupConfig(strategy=CleanupStrategy.NONE))  # Default to current behavior

# Existing decorators continue to work unchanged
@with_conflict_resolution()  # No cleanup configuration - defaults to NONE
async def existing_operation(entity: Entity) -> Entity:
    return entity
```

### Phase 2: Implement Cleanup Handlers

```python
# Add cleanup handler infrastructure
# - CleanupHandlerRegistry
# - Standard handler implementations (immediate, deferred, lazy)
# - Integration with staging coordinator
```

### Phase 3: Enable Cleanup by Default

```python
# Change default cleanup strategy
cleanup: CleanupConfig = Field(default_factory=lambda: CleanupConfig(strategy=CleanupStrategy.DEFERRED))

# Provide migration path for systems that want old behavior
@with_conflict_resolution(cleanup=CleanupConfig(strategy=CleanupStrategy.NONE))
async def legacy_operation(entity: Entity) -> Entity:
    return entity
```

### Phase 4: Advanced Features

```python
# Add advanced cleanup features
# - Custom cleanup handlers
# - Adaptive cleanup strategies
# - Per-operation cleanup configuration
# - Cleanup metrics and monitoring integration
```

## Monitoring and Observability

### Enhanced Metrics

```python
class CleanupMetrics:
    """Metrics for cleanup system."""
    
    def __init__(self):
        # Cleanup effectiveness
        self.operations_cleaned_by_strategy = Counter(["strategy"])
        self.cleanup_latency_by_strategy = Histogram(["strategy"])
        self.cleanup_failures_by_strategy = Counter(["strategy"])
        
        # System health
        self.current_zombie_count_by_target = Gauge(["target_type"])
        self.memory_pressure_triggered_cleanups = Counter()
        self.force_cleanup_triggered = Counter()
        
        # Configuration usage
        self.decorators_by_cleanup_strategy = Gauge(["strategy"])
        self.custom_cleanup_handlers_used = Counter(["handler_name"])

    def record_cleanup_operation(self, strategy: CleanupStrategy, latency_ms: float, success: bool):
        """Record a cleanup operation."""
        self.cleanup_latency_by_strategy.observe(latency_ms, labels={"strategy": strategy.value})
        
        if success:
            self.operations_cleaned_by_strategy.inc(labels={"strategy": strategy.value})
        else:
            self.cleanup_failures_by_strategy.inc(labels={"strategy": strategy.value})
```

### Health Checks

```python
async def cleanup_system_health_check() -> Dict[str, Any]:
    """Comprehensive health check for cleanup system."""
    
    # Current zombie counts
    zombie_counts = await _get_zombie_counts_by_target()
    total_zombies = sum(zombie_counts.values())
    
    # Cleanup handler status
    active_cleanup_tasks = _get_active_cleanup_task_count()
    failed_cleanup_handlers = _get_failed_cleanup_handlers()
    
    # Memory pressure
    memory_pressure = _get_system_memory_pressure()
    
    # Determine overall health
    status = "healthy"
    issues = []
    
    if total_zombies > 10000:
        status = "degraded"
        issues.append(f"High zombie count: {total_zombies}")
    
    if memory_pressure > 0.9:
        status = "degraded"  
        issues.append(f"High memory pressure: {memory_pressure:.1%}")
    
    if failed_cleanup_handlers:
        status = "unhealthy"
        issues.append(f"Failed cleanup handlers: {failed_cleanup_handlers}")
    
    return {
        "status": status,
        "total_zombie_operations": total_zombies,
        "zombie_counts_by_target": zombie_counts,
        "memory_pressure": memory_pressure,
        "active_cleanup_tasks": active_cleanup_tasks,
        "failed_cleanup_handlers": failed_cleanup_handlers,
        "issues": issues
    }
```

## Testing Strategy

### Unit Tests

```python
class TestCleanupConfiguration:
    """Test cleanup configuration functionality."""
    
    def test_cleanup_config_validation(self):
        """Test CleanupConfig validation."""
        
        # Valid configuration
        config = CleanupConfig(
            strategy=CleanupStrategy.DEFERRED,
            retention_minutes=30
        )
        assert config.validate_config()
        
        # Invalid custom configuration
        with pytest.raises(ValueError):
            invalid_config = CleanupConfig(
                strategy=CleanupStrategy.CUSTOM,
                custom_cleanup_handler=None
            )
            invalid_config.validate_config()
    
    async def test_adaptive_cleanup_strategy(self):
        """Test adaptive cleanup based on system conditions."""
        
        config = CleanupConfig(
            strategy=CleanupStrategy.DEFERRED,
            aggressive_on_memory_pressure=True,
            memory_pressure_threshold=0.8
        )
        
        operation = ConflictResolutionOperation(priority=OperationPriority.NORMAL)
        
        # Normal conditions - should use deferred strategy
        with mock.patch('_get_system_memory_pressure', return_value=0.5):
            strategy = config.get_effective_cleanup_strategy(operation)
            assert strategy == CleanupStrategy.DEFERRED
        
        # High memory pressure - should switch to immediate
        with mock.patch('_get_system_memory_pressure', return_value=0.9):
            strategy = config.get_effective_cleanup_strategy(operation)
            assert strategy == CleanupStrategy.IMMEDIATE
```

### Integration Tests

```python
async def test_decorator_cleanup_integration():
    """Test decorator with cleanup configuration."""
    
    cleanup_events = []
    
    class TestCleanupHandler(CleanupHandler):
        async def handle_operation_rejection(self, operation):
            cleanup_events.append(operation.ecs_id)
    
    CleanupHandlerRegistry.register("test", TestCleanupHandler)
    
    @with_conflict_resolution(
        cleanup=CleanupConfig(
            strategy=CleanupStrategy.CUSTOM,
            custom_cleanup_handler="test"
        )
    )
    async def test_operation(entity: TestEntity) -> TestEntity:
        entity.value += 1
        return entity
    
    entity = TestEntity(value=10)
    
    # Submit conflicting operations
    tasks = [
        asyncio.create_task(test_operation(entity))
        for _ in range(5)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify cleanup was called for rejected operations
    successes = [r for r in results if isinstance(r, TestEntity)]
    failures = [r for r in results if isinstance(r, Exception)]
    
    assert len(successes) == 1
    assert len(failures) == 4
    assert len(cleanup_events) == 4  # All rejected operations cleaned up
```

## Benefits and Trade-offs

### Benefits

**✅ Configurable Resource Management**: Each operation type can have appropriate cleanup strategy  
**✅ Production Flexibility**: Adapt cleanup behavior to system requirements  
**✅ Memory Efficiency**: Prevent zombie accumulation through configurable cleanup  
**✅ Debugging Support**: Retain operations for debugging when needed  
**✅ Adaptive Behavior**: Respond to system conditions (memory pressure, load)  
**✅ Custom Integration**: Support for custom cleanup logic for specialized use cases  

### Trade-offs

**⚠️ Increased Complexity**: More configuration options and code paths  
**⚠️ Performance Overhead**: Cleanup operations consume CPU and memory  
**⚠️ Configuration Burden**: Users need to understand cleanup implications  
**⚠️ Testing Complexity**: More scenarios to test and validate  
**⚠️ Backward Compatibility**: Migration path needed for existing code  

### Recommended Defaults

```python
# Production-ready defaults
DEFAULT_CLEANUP_CONFIG = CleanupConfig(
    strategy=CleanupStrategy.DEFERRED,  # Safe default with debugging window
    retention_minutes=30,               # 30-minute debugging window
    aggressive_on_memory_pressure=True, # Adapt to system conditions
    memory_pressure_threshold=0.8,      # Switch to immediate at 80% memory
    force_cleanup_threshold=1000,       # Emergency cleanup at 1000 zombies
    immediate_cleanup_priorities=[OperationPriority.CRITICAL]  # Critical ops get immediate cleanup
)
```

## Conclusion

Implementing configurable cleanup strategies would transform the decorator system from a conflict resolution tool into a comprehensive operation lifecycle management system. The proposed design provides:

- **Backward compatibility** through careful migration strategy
- **Production flexibility** through multiple cleanup strategies  
- **Adaptive behavior** through system condition monitoring
- **Extensibility** through custom cleanup handlers
- **Observability** through comprehensive metrics and health checks

This enhancement would address the current limitation of zombie operation accumulation while maintaining the performance and simplicity benefits of the existing staging coordinator approach.

**Implementation Priority:**
1. **High**: Basic CleanupConfig and standard handlers (immediate, deferred, lazy)
2. **Medium**: Adaptive behavior and system condition monitoring  
3. **Low**: Custom cleanup handlers and advanced configuration options

The key is to implement this incrementally while maintaining the existing system's proven reliability and performance characteristics. 