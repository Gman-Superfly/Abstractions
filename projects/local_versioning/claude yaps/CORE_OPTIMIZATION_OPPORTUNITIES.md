# Core Performance Optimization Opportunities

**Date**: 2025-01-17  
**Context**: Before implementing lazy/local optimizations, we need to optimize the core operations themselves.

---

## 🚨 Current Performance Problem

### Real Baseline (with CallableRegistry):

| Config | Entities | Time/Op | Throughput | vs Direct Mutation |
|--------|----------|---------|------------|-------------------|
| 10×10 | 111 | **151 ms** | 6.6 ops/sec | 7,589x slower |
| 20×20 | 421 | **561 ms** | 1.8 ops/sec | 10,795x slower |
| 50×20 | 1,051 | **1,526 ms** | 0.7 ops/sec | ~75,000x slower |
| 50×50 | 2,551 | **~3,500 ms** | 0.3 ops/sec | ~175,000x slower |

**This is unacceptably slow even for "full validation" mode.**

---

## 🔍 Pipeline Breakdown

Let's isolate each step in the CallableRegistry execution:

### Step 1: Divergence Check (Pre-execution)
```python
async def _check_entity_divergence(cls, entity: Entity):
    stored_tree = EntityRegistry.get_stored_tree(entity.root_ecs_id)
    current_tree = build_entity_tree(entity)  # ← EXPENSIVE
    modified = find_modified_entities(current_tree, stored_tree)  # ← EXPENSIVE
    if modified:
        raise ValueError("Entity diverged")
```

**Operations**:
- Build tree from current state: O(N)
- Compare with stored tree: O(N)
- **Total: 2×O(N) for every execution**

### Step 2: Function Execution
```python
result = func(**kwargs)  # Fast (just Python mutation)
```

**Operations**: O(1) - just list operations

### Step 3: Semantic Detection
```python
async def _detect_execution_semantic(cls, ...):
    # Check object identity
    if result is input_entity:  # Mutation
    elif result not in inputs:  # Creation
    else:  # Detachment
```

**Operations**: O(1) - just identity checks

### Step 4: Versioning (Post-execution)
```python
async def _apply_semantic_actions(cls, ...):
    if semantic == "mutation":
        EntityRegistry.version_entity(entity)  # ← EXPENSIVE
```

**Operations**:
- Build new tree: O(N)
- Compare with old tree: O(N)
- Version modified entities: O(M)
- Update tree mappings: O(N)
- **Total: 3×O(N) + O(M)**

### **Total Per Operation: 5×O(N) + O(M)**

For 1,051 entities: **~5,000+ operations per move!**

---

## 🎯 Optimization Opportunities

### Category 1: Structural Optimizations

#### 1.1 Use Named Fields Instead of Lists

**Current Problem**:
```python
class GridMap(Entity):
    nodes: List[Node]  # ← Linear scan to find node
```

**Issues**:
- Finding node by position: O(N) linear scan
- `isinstance()` checks on every iteration
- No index/cache for lookups

**Proposed Solution**:
```python
class GridMap(Entity):
    # Option A: Dict by position
    nodes_by_position: Dict[Tuple[int, int], Node]
    
    # Option B: Named fields (for small grids)
    node_0_0: Node
    node_0_1: Node
    # ...
    
    # Option C: Hybrid
    nodes: List[Node]
    _position_index: Dict[Tuple[int, int], Node] = Field(exclude=True)
```

**Expected Speedup**: 10-100x for lookups

#### 1.2 Cache Pydantic Field Metadata

**Current Problem**:
```python
for field_name in entity.model_fields:  # ← Repeated for every entity
    field_type = get_pydantic_field_type_entities(entity, field_name)
    # Complex type introspection every time
```

**Issues**:
- Field metadata computed repeatedly
- Type checking on every tree build
- No caching of field structure

**Proposed Solution**:
```python
# Class-level cache
_FIELD_CACHE: Dict[Type[Entity], Dict[str, FieldInfo]] = {}

def get_cached_field_info(entity_class: Type[Entity]):
    if entity_class not in _FIELD_CACHE:
        _FIELD_CACHE[entity_class] = {
            field_name: {
                'type': field.annotation,
                'is_entity': is_entity_field(field),
                'is_container': is_container_field(field),
                'container_type': get_container_type(field)
            }
            for field_name, field in entity_class.model_fields.items()
        }
    return _FIELD_CACHE[entity_class]
```

**Expected Speedup**: 2-5x for tree building

---

### Category 2: Algorithm Optimizations

#### 2.1 Early Exit on First Difference

**Current Problem**:
```python
def find_modified_entities(new_tree, old_tree, greedy=True):
    # Computes ALL differences even if we just need to know "changed or not"
    modified_entities = set()
    
    # Phase 1: Compare all nodes
    added_entities = new_entity_ids - old_entity_ids
    
    # Phase 2: Compare all edges
    added_edges = new_edges - old_edges
    
    # Phase 3: Compare all attributes
    for entity_id in common_entities:
        has_changes = compare_non_entity_attributes(...)
```

**Issues**:
- For divergence check, we only need to know IF there are changes
- Currently computes WHICH entities changed (expensive)
- No early exit option

**Proposed Solution**:
```python
def has_any_differences(new_tree, old_tree) -> bool:
    """Fast check: returns True on first difference found."""
    
    # Quick checks first
    if new_tree.node_count != old_tree.node_count:
        return True
    
    if new_tree.edge_count != old_tree.edge_count:
        return True
    
    # Check root entity attributes (most likely to change)
    root_new = new_tree.get_entity(new_tree.root_ecs_id)
    root_old = old_tree.get_entity(old_tree.root_ecs_id)
    if compare_non_entity_attributes(root_new, root_old):
        return True  # ← Early exit!
    
    # Only if needed, check deeper
    # ...
    
    return False
```

**Expected Speedup**: 10-50x for divergence checks

#### 2.2 Incremental Tree Updates

**Current Problem**:
```python
def version_entity(entity):
    new_tree = build_entity_tree(entity)  # ← Rebuilds entire tree
    modified = find_modified_entities(new_tree, old_tree)
```

**Issues**:
- Rebuilds entire tree even if only 2 nodes changed
- No incremental update mechanism
- Wastes time traversing unchanged subtrees

**Proposed Solution**:
```python
def update_tree_incrementally(old_tree, modified_entities):
    """Update only the modified parts of the tree."""
    
    new_tree = old_tree.model_copy(deep=False)  # Shallow copy
    
    # Only rebuild subtrees for modified entities
    for entity_id in modified_entities:
        entity = get_live_entity(entity_id)
        update_entity_in_tree(new_tree, entity)
    
    return new_tree
```

**Expected Speedup**: 100x+ for small modifications

#### 2.3 Skip Ancestry Propagation When Possible

**Current Problem**:
```python
# If child changes, mark ENTIRE path to root
path = new_tree.get_ancestry_path(entity_id)
modified_entities.update(path)  # ← Marks all ancestors
```

**Issues**:
- Always propagates to root
- No flag to skip propagation
- Forces versioning of unchanged ancestors

**Proposed Solution**:
```python
def find_modified_entities(new_tree, old_tree, propagate_ancestry=True):
    modified_entities = set()
    
    # Find actual changes
    for entity_id in common_entities:
        if has_changes(entity_id):
            modified_entities.add(entity_id)
            
            if propagate_ancestry:
                path = new_tree.get_ancestry_path(entity_id)
                modified_entities.update(path)
    
    return modified_entities
```

**Expected Speedup**: Enables partial versioning

---

### Category 3: Data Structure Optimizations

#### 3.1 Use Frozen Sets for Edge Comparison

**Current Problem**:
```python
new_edges = set()
for (source_id, target_id), edge in new_tree.edges.items():
    new_edges.add((source_id, target_id))  # ← Creates new set every time

added_edges = new_edges - old_edges  # ← Set difference
```

**Issues**:
- Creates temporary sets on every comparison
- No caching of edge sets
- Repeated set operations

**Proposed Solution**:
```python
class EntityTree(BaseModel):
    edges: Dict[Tuple[UUID, UUID], EntityEdge]
    _edge_set: Optional[frozenset] = None  # ← Cached
    
    def get_edge_set(self) -> frozenset:
        if self._edge_set is None:
            self._edge_set = frozenset(self.edges.keys())
        return self._edge_set
```

**Expected Speedup**: 2-3x for edge comparisons

#### 3.2 Lazy Ancestry Path Computation

**Current Problem**:
```python
# Computed during tree building for ALL entities
ancestry_paths: Dict[UUID, List[UUID]] = {}
for entity_id in tree.nodes:
    path = compute_path_to_root(entity_id)
    ancestry_paths[entity_id] = path
```

**Issues**:
- Computed for all entities even if never used
- Stored in memory for entire tree
- Recomputed on every tree build

**Proposed Solution**:
```python
class EntityTree(BaseModel):
    _ancestry_cache: Dict[UUID, List[UUID]] = Field(default_factory=dict, exclude=True)
    
    def get_ancestry_path(self, entity_id: UUID) -> List[UUID]:
        if entity_id not in self._ancestry_cache:
            # Compute on demand
            path = []
            current = entity_id
            while current:
                path.append(current)
                parent = self.get_parent(current)
                current = parent
            self._ancestry_cache[entity_id] = path
        return self._ancestry_cache[entity_id]
```

**Expected Speedup**: 2-5x for tree building

---

### Category 4: Pydantic-Specific Optimizations

#### 4.1 Use model_copy(deep=False) Where Possible

**Current Problem**:
```python
stored_tree = cls.tree_registry.get(root_ecs_id)
new_tree = stored_tree.model_copy(deep=True)  # ← Expensive deep copy
```

**Issues**:
- Deep copies entire tree structure
- Copies all entities recursively
- Unnecessary for read-only operations

**Proposed Solution**:
```python
# For read-only: shallow copy
tree_view = stored_tree.model_copy(deep=False)

# For modification: selective deep copy
tree_copy = stored_tree.model_copy(deep=False)
tree_copy.nodes = {k: v.model_copy() for k, v in stored_tree.nodes.items()}
```

**Expected Speedup**: 5-10x for tree retrieval

#### 4.2 Use Field(exclude=True) for Computed Fields

**Current Problem**:
```python
class EntityTree(BaseModel):
    nodes: Dict[UUID, Entity]
    edges: Dict[Tuple[UUID, UUID], EntityEdge]
    outgoing_edges: Dict[UUID, List[UUID]]  # ← Redundant with edges
    incoming_edges: Dict[UUID, List[UUID]]  # ← Redundant with edges
```

**Issues**:
- Redundant data stored and copied
- Increases memory usage
- Slows down serialization

**Proposed Solution**:
```python
class EntityTree(BaseModel):
    nodes: Dict[UUID, Entity]
    edges: Dict[Tuple[UUID, UUID], EntityEdge]
    
    # Computed on demand, not stored
    _outgoing_edges: Dict[UUID, List[UUID]] = Field(default_factory=dict, exclude=True)
    _incoming_edges: Dict[UUID, List[UUID]] = Field(default_factory=dict, exclude=True)
    
    def get_outgoing_edges(self, entity_id: UUID) -> List[UUID]:
        if entity_id not in self._outgoing_edges:
            self._outgoing_edges[entity_id] = [
                target for (source, target) in self.edges.keys()
                if source == entity_id
            ]
        return self._outgoing_edges[entity_id]
```

**Expected Speedup**: 2-3x for tree operations

#### 4.3 Use __slots__ for Entity Classes

**Current Problem**:
```python
class Entity(BaseModel):
    ecs_id: UUID
    live_id: UUID
    # ... many fields
    # Uses __dict__ for attribute storage
```

**Issues**:
- Higher memory usage
- Slower attribute access
- No compile-time field checking

**Proposed Solution**:
```python
class Entity(BaseModel):
    __slots__ = ('ecs_id', 'live_id', 'lineage_id', ...)  # ← Define slots
    
    ecs_id: UUID
    live_id: UUID
    # ...
```

**Expected Speedup**: 10-20% memory reduction, 5-10% speed improvement

---

### Category 5: List vs Dict Performance

#### 5.1 Benchmark: List vs Dict for Node Storage

**Hypothesis**: Lists require O(N) linear scan, dicts provide O(1) lookup.

**Test**:
```python
# Current (List)
class GridMap(Entity):
    nodes: List[Node]
    
    def get_node_at(self, position):
        for node in self.nodes:  # O(N)
            if node.position == position:
                return node

# Alternative (Dict)
class GridMap(Entity):
    nodes: Dict[Tuple[int, int], Node]
    
    def get_node_at(self, position):
        return self.nodes.get(position)  # O(1)
```

**Expected Impact**:
- 10×10 grid: 10x speedup for lookups
- 100×100 grid: 100x speedup for lookups

#### 5.2 Polymorphic Type Checking Overhead

**Current Problem**:
```python
for entity in node.entities:
    if isinstance(entity, Agent):  # ← Type check on every iteration
        # process agent
```

**Issues**:
- `isinstance()` is relatively slow
- Repeated for every entity in every node
- No type segregation

**Proposed Solution**:
```python
class Node(Entity):
    # Segregate by type
    agents: List[Agent]
    apples: List[Apple]
    paths: List[Path]
    
    # Or use type index
    entities: List[GameEntity]
    _entities_by_type: Dict[Type, List[GameEntity]] = Field(exclude=True)
```

**Expected Speedup**: 2-5x for type-specific operations

---

## 🧪 Proposed Investigation Plan

### Phase 1: Profiling (Immediate)

1. **Instrument each pipeline step**:
   ```python
   with Timer("divergence_check"):
       await _check_entity_divergence(entity)
   
   with Timer("function_execution"):
       result = func(**kwargs)
   
   with Timer("versioning"):
       EntityRegistry.version_entity(entity)
   ```

2. **Measure breakdown**:
   - How much time in tree building?
   - How much time in diff computation?
   - How much time in tree copying?
   - How much time in field introspection?

3. **Identify hotspots**:
   - Which functions are called most?
   - Which operations are slowest?
   - Where are the memory allocations?

### Phase 2: Quick Wins (1-2 days)

1. **Implement early exit for divergence check**
   - Add `has_any_differences()` function
   - Skip full diff if not needed
   - **Expected: 10x speedup for divergence checks**

2. **Cache field metadata**
   - Class-level cache for field info
   - Avoid repeated type introspection
   - **Expected: 2-3x speedup for tree building**

3. **Use shallow copies where possible**
   - Identify read-only operations
   - Replace deep copies with shallow
   - **Expected: 5x speedup for tree retrieval**

### Phase 3: Structural Changes (3-5 days)

1. **Test Dict vs List for nodes**
   - Create benchmark comparing both
   - Measure tree building time
   - Measure lookup time
   - **Expected: 10-100x speedup for lookups**

2. **Implement incremental tree updates**
   - Add `update_tree_incrementally()` function
   - Only rebuild modified subtrees
   - **Expected: 100x+ speedup for small changes**

3. **Add lazy ancestry computation**
   - Compute paths on demand
   - Cache results
   - **Expected: 2-5x speedup for tree building**

### Phase 4: Algorithm Improvements (5-7 days)

1. **Optimize diff algorithm**
   - Add early exit options
   - Skip unnecessary comparisons
   - Use cached edge sets
   - **Expected: 10-50x speedup for diffs**

2. **Implement partial versioning**
   - Skip ancestry propagation when safe
   - Version only modified entities
   - **Expected: Enables local optimizations**

---

## 📊 Expected Overall Speedup

### Conservative Estimates

| Optimization | Speedup | Cumulative |
|--------------|---------|------------|
| Early exit divergence | 10x | 10x |
| Field metadata cache | 2x | 20x |
| Shallow copies | 5x | 100x |
| Dict-based nodes | 10x | 1,000x |
| Incremental updates | 100x | 100,000x |

### Realistic Target

**From**: 1,526 ms/op (1,051 entities)  
**To**: 1-10 ms/op (1,051 entities)  
**Speedup**: 150-1,500x

**This would make the system usable even WITHOUT lazy/local optimizations.**

---

## 🎯 Priority Ranking

### P0 (Critical - Do First)
1. ✅ Profile the pipeline to identify actual bottlenecks
2. ✅ Implement early exit for divergence checks
3. ✅ Cache field metadata

### P1 (High Impact)
4. Test Dict vs List for node storage
5. Implement shallow copies for read-only ops
6. Add lazy ancestry computation

### P2 (Medium Impact)
7. Optimize diff algorithm with early exits
8. Implement incremental tree updates
9. Use frozen sets for edge comparisons

### P3 (Nice to Have)
10. Add __slots__ to Entity classes
11. Segregate entities by type
12. Optimize Pydantic field handling

---

## 🚀 Next Steps

1. **Create profiling script** to measure each pipeline step
2. **Identify the #1 bottleneck** (likely tree building or diff)
3. **Implement quick wins** (early exit, caching)
4. **Benchmark improvements** after each change
5. **Iterate** until we reach acceptable performance

**Goal**: Get to <10ms per operation BEFORE implementing lazy/local optimizations.

Then the lazy/local optimizations will provide an additional 10-100x on top of that!
