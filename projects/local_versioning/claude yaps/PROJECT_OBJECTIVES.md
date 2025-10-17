# Local Versioning Optimization: Project Objectives

## Problem Statement

### Current System Behavior

When executing a function like `move_agent(gridmap, agent, target_position) -> gridmap`:

1. **Input Divergence Check** (EXPENSIVE):
   - Builds full tree from current `gridmap` state
   - Compares against stored tree
   - Checks ALL 10,000+ entities even though gridmap hasn't changed

2. **Function Execution**:
   - Function receives isolated copy of gridmap
   - Modifies 2 nodes (source and target)
   - Returns mutated gridmap

3. **Semantic Detection**:
   - Detects "mutation" (same Python object)
   - Triggers `version_entity(gridmap)`

4. **Full Versioning** (EXPENSIVE):
   - Builds NEW tree from mutated gridmap (10,000+ entities)
   - Compares against OLD tree (10,000+ entities)
   - Detects 2 nodes changed
   - Versions 4 entities (agent + 2 nodes + map)
   - Updates ALL tree mappings (10,000+ entities)

**Total Operations**: ~30,000+ for moving 1 agent!

---

## Optimization Goals

### Goal 1: Eliminate Redundant Divergence Checks

**Target**: Skip divergence check when we control execution flow.

**Implementation**:
```python
# Option A: Flag in execute methods
CallableRegistry.execute(
    "move_agent",
    gridmap=map,
    agent=agent,
    target=pos,
    skip_divergence_check=True  # ← NEW
)

# Option B: Global registry config
CallableRegistry.set_config(skip_divergence_checks=True)
CallableRegistry.execute("move_agent", ...)
```

**Safety**:
- ✅ Safe when: No external mutations between calls
- ❌ Unsafe when: Manual entity modifications outside CallableRegistry
- **Use case**: Controlled simulation loops

**Expected Speedup**: 2x (eliminates one full tree build + diff)

---

### Goal 2: Partial Versioning with Reattachment

**Target**: Version only modified subtrees, reattach to parent.

#### Current Behavior

Function signature: `node, entity, node -> node, node`

**Returns**: Two detached nodes (orphaned roots)

**Problem**: No automatic reattachment to gridmap

#### Desired Behavior

**Execution with reattachment**:
```python
@CallableRegistry.register(
    input_entity=Node,
    output_entity=Node,
    reattach_outputs={
        "output_0": {"parent_field": "nodes", "mode": "replace"},
        "output_1": {"parent_field": "nodes", "mode": "replace"}
    }
)
def move_agent_between_nodes(
    source_node: Node,
    agent: Agent,
    target_node: Node
) -> Tuple[Node, Node]:
    # Remove agent from source
    source_node.agents.remove(agent)
    
    # Add agent to target
    target_node.agents.append(agent)
    
    return source_node, target_node

# Execute with reattachment
result_map = CallableRegistry.execute(
    "move_agent_between_nodes",
    source_node=map.nodes[5],
    agent=agent,
    target_node=map.nodes[10],
    reattach_to=map,  # ← Parent entity
    skip_full_diff=True  # ← Use targeted versioning
)
```

**Flow**:
1. Execute function → get detached nodes
2. Find nodes in `map.nodes` by `ecs_id`
3. Replace with new versions
4. Version ONLY modified nodes + map (skip full diff)
5. Return versioned map

**Expected Speedup**: 100x+ (version 4 entities instead of 10,000)

---

### Goal 3: Greedy Versioning with Known Changes

**Target**: Skip diff computation when we know what changed.

**Implementation**:
```python
# In EntityRegistry
@classmethod
def version_entity_partial(
    cls,
    entity: Entity,
    modified_child_ids: Set[UUID],  # ← Known modified entities
    skip_full_diff: bool = True
):
    old_tree = cls.get_stored_tree(entity.root_ecs_id)
    new_tree = build_entity_tree(entity)
    
    if skip_full_diff:
        # Use provided set instead of computing diff
        modified_entities = modified_child_ids
    else:
        # Fallback to full diff
        modified_entities = find_modified_entities(new_tree, old_tree)
    
    # Continue with normal versioning...
```

**Usage**:
```python
# After reattachment, we know exactly what changed
modified_ids = {source_node.ecs_id, target_node.ecs_id, agent.ecs_id}

EntityRegistry.version_entity_partial(
    gridmap,
    modified_child_ids=modified_ids,
    skip_full_diff=True
)
```

**Expected Speedup**: 1000x+ for diff phase (O(4) instead of O(10,000))

---

## Test Scenario Design

### Hierarchical Structure

```python
class Agent(Entity):
    name: str
    speed: int

class Node(Entity):
    position: Tuple[int, int]
    agents: List[Agent] = Field(default_factory=list)

class GridMap(Entity):
    nodes: List[Node] = Field(default_factory=list)
```

### Parameterized Creation

```python
def create_test_scenario(num_nodes: int, agents_per_node: int) -> GridMap:
    """
    Create test scenario with configurable size.
    
    Args:
        num_nodes: Number of grid nodes (e.g., 100)
        agents_per_node: Agents per node (e.g., 100)
    
    Returns:
        GridMap with num_nodes * agents_per_node total entities
    """
    gridmap = GridMap()
    
    for i in range(num_nodes):
        node = Node(position=(i % 10, i // 10))
        
        for j in range(agents_per_node):
            agent = Agent(
                name=f"agent_{i}_{j}",
                speed=random.randint(1, 10)
            )
            node.agents.append(agent)
        
        gridmap.nodes.append(node)
    
    gridmap.promote_to_root()
    return gridmap
```

### Test Configurations

| Config | Nodes | Agents/Node | Total Entities | Use Case |
|--------|-------|-------------|----------------|----------|
| Small | 10 | 10 | 100 | Quick tests |
| Medium | 50 | 50 | 2,500 | Realistic game |
| Large | 100 | 100 | 10,000 | Stress test |
| XLarge | 200 | 100 | 20,000 | Extreme scale |

---

## Performance Metrics

### Metrics to Track

```python
@dataclass
class PerformanceMetrics:
    # Timing
    divergence_check_ms: float
    tree_build_ms: float
    diff_computation_ms: float
    versioning_ms: float
    total_execution_ms: float
    
    # Operations
    entities_checked: int
    entities_versioned: int
    tree_mappings_updated: int
    
    # Comparisons
    attribute_comparisons: int
    edge_comparisons: int
    
    # Memory
    tree_size_bytes: int
    peak_memory_mb: float
```

### Benchmark Operations

1. **Baseline (Current System)**:
   - Move 1 agent
   - Move 10 agents (sequential)
   - Move 100 agents (sequential)

2. **Optimized (Lazy Divergence)**:
   - Same operations with `skip_divergence_check=True`

3. **Optimized (Partial Versioning)**:
   - Same operations with reattachment + targeted diff

4. **Optimized (Full Stack)**:
   - Lazy + Partial + Greedy

### Expected Results

| Operation | Baseline (ms) | Lazy (ms) | Partial (ms) | Full (ms) | Speedup |
|-----------|---------------|-----------|--------------|-----------|---------|
| Move 1 agent | 300 | 150 | 5 | 2 | 150x |
| Move 10 agents | 3000 | 1500 | 50 | 20 | 150x |
| Move 100 agents | 30000 | 15000 | 500 | 200 | 150x |

---

## Implementation Plan

### Phase 1: Infrastructure Setup ✓

- [x] Study `entity.py` completely
- [x] Document entity graph structure
- [x] Create test scenario structure
- [x] Define objectives

### Phase 2: Lazy Divergence Checking

**Files to modify**:
- `callable_registry.py`: Add `skip_divergence_check` parameter
- `callable_registry.py`: Add global config option

**Implementation**:
```python
class CallableRegistry:
    _config = {
        "skip_divergence_checks": False,
        "enable_partial_versioning": False,
        "enable_greedy_diff": False
    }
    
    @classmethod
    def set_config(cls, **kwargs):
        cls._config.update(kwargs)
    
    @classmethod
    async def _check_entity_divergence(cls, entity: Entity):
        if cls._config["skip_divergence_checks"]:
            return  # Skip check
        
        # Existing divergence check logic...
```

**Tests**:
- Verify no divergence check when flag enabled
- Verify detection still works when flag disabled
- Measure performance improvement

### Phase 3: Reattachment Pattern

**Files to modify**:
- `callable_registry.py`: Add reattachment logic to `_finalize_multi_entity_result()`
- `callable_registry.py`: Add `reattach_outputs` metadata to `FunctionMetadata`

**Implementation**:
```python
async def _finalize_multi_entity_result_with_reattachment(
    cls,
    result: Any,
    metadata: FunctionMetadata,
    parent_entity: Optional[Entity],
    reattach_config: Dict[str, Any]
):
    # Unpack result
    entities = unpack_result(result)
    
    if parent_entity and reattach_config:
        # Find and replace entities in parent
        for i, entity in enumerate(entities):
            config = reattach_config.get(f"output_{i}")
            if config:
                parent_field = config["parent_field"]
                mode = config["mode"]  # "replace", "append", "insert"
                
                if mode == "replace":
                    # Find entity in parent's field by ecs_id
                    field_value = getattr(parent_entity, parent_field)
                    if isinstance(field_value, list):
                        for j, item in enumerate(field_value):
                            if isinstance(item, Entity) and item.ecs_id == entity.ecs_id:
                                field_value[j] = entity
                                break
        
        # Version parent with known changes
        modified_ids = {e.ecs_id for e in entities}
        EntityRegistry.version_entity_partial(
            parent_entity,
            modified_child_ids=modified_ids,
            skip_full_diff=True
        )
        
        return parent_entity
    else:
        # Standard finalization
        return entities
```

**Tests**:
- Verify reattachment replaces correct entities
- Verify parent versioning with targeted diff
- Measure performance improvement

### Phase 4: Greedy Versioning

**Files to modify**:
- `entity.py`: Add `version_entity_partial()` method
- `entity.py`: Add `skip_full_diff` parameter to `find_modified_entities()`

**Implementation**:
```python
@classmethod
def version_entity_partial(
    cls,
    entity: Entity,
    modified_child_ids: Set[UUID],
    skip_full_diff: bool = True
) -> bool:
    old_tree = cls.get_stored_tree(entity.root_ecs_id)
    new_tree = build_entity_tree(entity)
    
    if skip_full_diff:
        # Use provided set + ancestry propagation
        modified_entities = set()
        for child_id in modified_child_ids:
            path = new_tree.get_ancestry_path(child_id)
            modified_entities.update(path)
    else:
        modified_entities = find_modified_entities(new_tree, old_tree)
    
    # Continue with standard versioning flow...
```

**Tests**:
- Verify correct ancestry propagation
- Verify tree consistency after partial versioning
- Compare against full versioning (should match)
- Measure performance improvement

### Phase 5: Integration & Benchmarking

**Create benchmark suite**:
```python
def benchmark_move_operations(config_name: str, num_moves: int):
    # Setup
    gridmap = create_test_scenario(...)
    
    # Baseline
    metrics_baseline = run_moves_baseline(gridmap, num_moves)
    
    # Lazy
    CallableRegistry.set_config(skip_divergence_checks=True)
    metrics_lazy = run_moves_optimized(gridmap, num_moves)
    
    # Partial
    CallableRegistry.set_config(enable_partial_versioning=True)
    metrics_partial = run_moves_optimized(gridmap, num_moves)
    
    # Full
    CallableRegistry.set_config(
        skip_divergence_checks=True,
        enable_partial_versioning=True,
        enable_greedy_diff=True
    )
    metrics_full = run_moves_optimized(gridmap, num_moves)
    
    # Report
    print_comparison_table(metrics_baseline, metrics_lazy, metrics_partial, metrics_full)
```

---

## Success Criteria

### Functional Requirements

✅ **Correctness**:
- Partial versioning produces same result as full versioning
- Tree consistency maintained after all operations
- No entity lookup failures
- Lineage tracking preserved

✅ **Safety**:
- Clear documentation of when optimizations are safe
- Runtime checks for unsafe usage
- Fallback to full versioning on error

### Performance Requirements

✅ **Speedup Targets**:
- Lazy divergence: 2x faster
- Partial versioning: 50x faster
- Full optimization: 100x+ faster

✅ **Scalability**:
- Performance improvement scales with entity count
- Memory usage remains constant
- No performance degradation over time

### Code Quality

✅ **Maintainability**:
- Clear separation of optimized vs. standard paths
- Comprehensive tests for all code paths
- Detailed documentation of design decisions

✅ **Extensibility**:
- Easy to add new reattachment modes
- Pluggable diff strategies
- Configurable optimization levels

---

## Risk Mitigation

### Risk 1: Incorrect Partial Versioning

**Mitigation**:
- Comprehensive test suite comparing partial vs. full
- Tree consistency validation after every operation
- Fallback to full versioning on validation failure

### Risk 2: Unsafe Optimization Usage

**Mitigation**:
- Clear documentation of safety requirements
- Runtime warnings for potentially unsafe usage
- Config flags require explicit opt-in

### Risk 3: Performance Regression

**Mitigation**:
- Benchmark suite run on every change
- Performance regression tests in CI
- Profiling to identify bottlenecks

---

## Timeline

### Week 1: Foundation
- ✅ Study entity.py
- ✅ Create documentation
- ✅ Define objectives
- [ ] Create test scenario
- [ ] Implement baseline benchmarks

### Week 2: Lazy Divergence
- [ ] Implement skip_divergence_check flag
- [ ] Add global config
- [ ] Write tests
- [ ] Measure performance

### Week 3: Reattachment Pattern
- [ ] Implement reattachment logic
- [ ] Add metadata support
- [ ] Write tests
- [ ] Measure performance

### Week 4: Greedy Versioning
- [ ] Implement version_entity_partial
- [ ] Add targeted diff
- [ ] Write tests
- [ ] Measure performance

### Week 5: Integration & Optimization
- [ ] Integrate all optimizations
- [ ] Full benchmark suite
- [ ] Performance tuning
- [ ] Documentation

---

## Deliverables

1. **Code**:
   - Modified `callable_registry.py` with optimizations
   - Modified `entity.py` with partial versioning
   - Test scenario implementation
   - Benchmark suite

2. **Documentation**:
   - ✅ Entity graph understanding
   - ✅ Project objectives
   - Implementation guide
   - Performance analysis report
   - Usage examples

3. **Tests**:
   - Unit tests for all new features
   - Integration tests for optimization paths
   - Performance regression tests
   - Correctness validation tests

4. **Benchmarks**:
   - Baseline measurements
   - Optimization measurements
   - Comparison analysis
   - Scaling analysis
