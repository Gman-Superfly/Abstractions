# Diff Algorithm Deep Dive & Optimization Strategy

**Current Performance**: 0.17 ms per entity (18.8ms for 111 entities)  
**Target**: <0.05 ms per entity (<5ms for 111 entities)  
**Required Speedup**: 3-4x

---

## 🔍 Current Algorithm Analysis

### The `find_modified_entities()` Function (Lines 1022-1160)

```python
def find_modified_entities(new_tree, old_tree, greedy=True, debug=False):
    """
    Three-phase algorithm:
    1. Compare node sets (added/removed entities)
    2. Compare edge sets (moved entities)
    3. Compare attributes (changed entities)
    """
```

### Phase 1: Node Set Comparison (Lines 1056-1067)

```python
# Step 1: Compare node sets
new_entity_ids = set(new_tree.nodes.keys())  # ← Create set from dict keys
old_entity_ids = set(old_tree.nodes.keys())  # ← Create set from dict keys

added_entities = new_entity_ids - old_entity_ids  # ← Set difference
removed_entities = old_entity_ids - new_entity_ids
common_entities = new_entity_ids & old_entity_ids  # ← Set intersection

# Mark all added entities and their ancestry paths
for entity_id in added_entities:
    path = new_tree.get_ancestry_path(entity_id)  # ← Lookup ancestry
    modified_entities.update(path)  # ← Add all ancestors
```

**Cost Analysis**:
- Create 2 sets: O(N) each
- Set operations: O(N)
- For each added entity: lookup ancestry path
- **Total: O(N) + O(added × depth)**

**Optimization Opportunities**:
1. **Cache entity ID sets** - don't recreate on every diff
2. **Early exit if sets are identical** - no changes at all
3. **Lazy ancestry lookup** - only when needed

---

### Phase 2: Edge Set Comparison (Lines 1069-1106)

```python
# Step 2: Compare edge sets
new_edges = set()
old_edges = set()

for (source_id, target_id), edge in new_tree.edges.items():
    new_edges.add((source_id, target_id))  # ← Build edge set
    
for (source_id, target_id), edge in old_tree.edges.items():
    old_edges.add((source_id, target_id))  # ← Build edge set

added_edges = new_edges - old_edges  # ← Set difference
removed_edges = old_edges - new_edges

# Identify moved entities
for source_id, target_id in added_edges:
    if target_id in common_entities:
        # Check if entity has different parent
        old_parents = set()
        for old_source_id, old_target_id in old_edges:  # ← Linear scan!
            if old_target_id == target_id:
                old_parents.add(old_source_id)
        
        new_parents = set()
        for new_source_id, new_target_id in new_edges:  # ← Linear scan!
            if new_target_id == target_id:
                new_parents.add(new_source_id)
        
        if old_parents != new_parents:
            moved_entities.add(target_id)
            path = new_tree.get_ancestry_path(target_id)
            modified_entities.update(path)
```

**Cost Analysis**:
- Build 2 edge sets: O(E) each (E = number of edges)
- Set operations: O(E)
- For each added edge: **O(E) linear scans** to find parents!
- **Total: O(E) + O(added_edges × E) = O(E²) in worst case!**

**This is TERRIBLE! Nested loops over edges!**

**Optimization Opportunities**:
1. **Use tree.incoming_edges** instead of scanning all edges
2. **Cache edge sets as frozenset** in EntityTree
3. **Early exit if edge sets are identical**

---

### Phase 3: Attribute Comparison (Lines 1108-1153)

```python
# Step 3: Check attribute changes
remaining_entities = []

for entity_id in common_entities:
    if entity_id not in modified_entities and entity_id not in moved_entities:
        path_length = len(new_tree.get_ancestry_path(entity_id))  # ← Lookup
        remaining_entities.append((path_length, entity_id))

remaining_entities.sort(reverse=True)  # ← Sort by depth

for _, entity_id in remaining_entities:
    if entity_id in modified_entities or entity_id in unchanged_entities:
        continue  # ← Skip check
    
    new_entity = new_tree.get_entity(entity_id)
    old_entity = old_tree.get_entity(entity_id)
    
    comparison_count += 1
    has_changes = compare_non_entity_attributes(new_entity, old_entity)  # ← EXPENSIVE
    
    if has_changes:
        path = new_tree.get_ancestry_path(entity_id)
        modified_entities.update(path)
        
        if greedy:
            continue  # ← Early exit
    else:
        unchanged_entities.add(entity_id)
```

**Cost Analysis**:
- Build remaining list: O(N)
- Sort: O(N log N)
- For each entity: compare attributes
- `compare_non_entity_attributes()`: calls `get_non_entity_attributes()` twice
- **Total: O(N log N) + O(N × attr_comparison_cost)**

**Optimization Opportunities**:
1. **Cache non-entity attributes** in EntityTree
2. **Hash-based comparison** instead of field-by-field
3. **Early exit on first difference**
4. **Skip sorting** - not necessary for correctness

---

## 🎯 Major Bottlenecks Identified

### Bottleneck #1: Edge Parent Lookup (O(E²))

**Current Code** (Lines 1090-1098):
```python
old_parents = set()
for old_source_id, old_target_id in old_edges:  # ← Scans ALL edges
    if old_target_id == target_id:
        old_parents.add(old_source_id)
```

**Problem**: For each added edge, we scan ALL edges to find parents.

**Solution**: Use `tree.incoming_edges` which is already indexed!
```python
# Already available in EntityTree!
old_parents = set(old_tree.incoming_edges.get(target_id, []))
new_parents = set(new_tree.incoming_edges.get(target_id, []))
```

**Expected Speedup**: 10-50x for this phase (O(E²) → O(1))

---

### Bottleneck #2: Repeated Edge Set Creation

**Current Code** (Lines 1071-1078):
```python
new_edges = set()
for (source_id, target_id), edge in new_tree.edges.items():
    new_edges.add((source_id, target_id))
```

**Problem**: Creates edge sets from scratch on every diff.

**Solution**: Cache edge sets in EntityTree as frozenset:
```python
class EntityTree(BaseModel):
    edges: Dict[Tuple[UUID, UUID], EntityEdge]
    _edge_set: Optional[frozenset] = Field(default=None, exclude=True)
    
    def get_edge_set(self) -> frozenset:
        if self._edge_set is None:
            self._edge_set = frozenset(self.edges.keys())
        return self._edge_set
```

**Expected Speedup**: 2-3x for edge comparison

---

### Bottleneck #3: Attribute Comparison

**Current Code** (Lines 987-1019):
```python
def compare_non_entity_attributes(entity1, entity2):
    attrs1 = get_non_entity_attributes(entity1)  # ← Expensive
    attrs2 = get_non_entity_attributes(entity2)  # ← Expensive
    
    if set(attrs1.keys()) != set(attrs2.keys()):
        return True
    
    for field_name, value1 in attrs1.items():
        value2 = attrs2[field_name]
        if value1 != value2:
            return True
    
    return False
```

**Problem**: `get_non_entity_attributes()` calls `get_pydantic_field_type_entities()` for EVERY field!

**Solution**: Use cached field metadata:
```python
def compare_non_entity_attributes_fast(entity1, entity2):
    # Use cached metadata
    metadata = get_cached_field_metadata(type(entity1))
    
    for field_name, field_meta in metadata.items():
        # Skip entity fields and identity fields
        if field_meta.contains_entities or field_meta.is_identity_field:
            continue
        
        value1 = getattr(entity1, field_name)
        value2 = getattr(entity2, field_name)
        
        if value1 != value2:
            return True  # ← Early exit!
    
    return False
```

**Expected Speedup**: 5-10x for attribute comparison

---

### Bottleneck #4: Repeated Ancestry Path Lookups

**Current Code**:
```python
path = new_tree.get_ancestry_path(entity_id)  # ← Called many times
modified_entities.update(path)
```

**Problem**: Ancestry paths are looked up multiple times for the same entity.

**Solution**: Cache paths during diff:
```python
# Local cache for this diff operation
path_cache = {}

def get_path_cached(entity_id):
    if entity_id not in path_cache:
        path_cache[entity_id] = new_tree.get_ancestry_path(entity_id)
    return path_cache[entity_id]
```

**Expected Speedup**: 1.5-2x (reduces redundant lookups)

---

## 🚀 Optimization Strategy

### Phase 1: Quick Wins (2-3x speedup)

#### 1.1: Use incoming_edges for Parent Lookup
```python
# Replace O(E²) loops with O(1) lookups
old_parents = set(old_tree.incoming_edges.get(target_id, []))
new_parents = set(new_tree.incoming_edges.get(target_id, []))
```

#### 1.2: Cache Field Metadata in Attribute Comparison
```python
def compare_non_entity_attributes_fast(entity1, entity2):
    metadata = get_cached_field_metadata(type(entity1))
    
    for field_name, field_meta in metadata.items():
        if field_meta.contains_entities or field_meta.is_identity_field:
            continue
        
        if getattr(entity1, field_name) != getattr(entity2, field_name):
            return True
    
    return False
```

#### 1.3: Early Exit on Identical Trees
```python
def find_modified_entities(new_tree, old_tree, greedy=True):
    # Quick check: if trees are identical, return empty set
    if new_tree.node_count != old_tree.node_count:
        # Trees differ, continue with full diff
        pass
    elif new_tree.edge_count != old_tree.edge_count:
        # Trees differ, continue with full diff
        pass
    else:
        # Trees might be identical, check root
        new_root = new_tree.get_entity(new_tree.root_ecs_id)
        old_root = old_tree.get_entity(old_tree.root_ecs_id)
        
        if not compare_non_entity_attributes_fast(new_root, old_root):
            # Root unchanged, likely no changes
            # Could do deeper check here
            pass
```

**Expected Speedup**: 2-3x

---

### Phase 2: Caching Optimizations (2-3x additional)

#### 2.1: Cache Edge Sets in EntityTree
```python
class EntityTree(BaseModel):
    edges: Dict[Tuple[UUID, UUID], EntityEdge]
    _edge_set: Optional[frozenset] = Field(default=None, exclude=True)
    _entity_id_set: Optional[frozenset] = Field(default=None, exclude=True)
    
    def get_edge_set(self) -> frozenset:
        if self._edge_set is None:
            self._edge_set = frozenset(self.edges.keys())
        return self._edge_set
    
    def get_entity_id_set(self) -> frozenset:
        if self._entity_id_set is None:
            self._entity_id_set = frozenset(self.nodes.keys())
        return self._entity_id_set
```

#### 2.2: Cache Non-Entity Attributes in EntityTree
```python
class EntityTree(BaseModel):
    nodes: Dict[UUID, Entity]
    _non_entity_attrs: Dict[UUID, Dict[str, Any]] = Field(default_factory=dict, exclude=True)
    
    def get_non_entity_attrs(self, entity_id: UUID) -> Dict[str, Any]:
        if entity_id not in self._non_entity_attrs:
            entity = self.nodes[entity_id]
            metadata = get_cached_field_metadata(type(entity))
            
            attrs = {}
            for field_name, field_meta in metadata.items():
                if not field_meta.contains_entities and not field_meta.is_identity_field:
                    attrs[field_name] = getattr(entity, field_name)
            
            self._non_entity_attrs[entity_id] = attrs
        
        return self._non_entity_attrs[entity_id]
```

**Expected Speedup**: 2-3x additional (cumulative 4-9x total)

---

### Phase 3: Algorithm Improvements (1.5-2x additional)

#### 3.1: Skip Sorting (Not Needed)
```python
# Current: Sort by path length
remaining_entities.sort(reverse=True)

# Optimized: Just iterate (order doesn't matter for correctness)
for entity_id in remaining_entities:
    # Process...
```

#### 3.2: Hash-Based Comparison
```python
def get_entity_hash(entity: Entity) -> int:
    """Fast hash of non-entity attributes."""
    metadata = get_cached_field_metadata(type(entity))
    
    values = []
    for field_name, field_meta in metadata.items():
        if not field_meta.contains_entities and not field_meta.is_identity_field:
            values.append(getattr(entity, field_name))
    
    return hash(tuple(values))

# In diff:
if get_entity_hash(new_entity) != get_entity_hash(old_entity):
    # Entities differ
```

**Expected Speedup**: 1.5-2x additional (cumulative 6-18x total)

---

## 📊 Expected Results

### Conservative Estimate

| Optimization | Current | After | Speedup |
|--------------|---------|-------|---------|
| **Baseline** | 18.8 ms | - | 1x |
| Phase 1 (Quick wins) | 18.8 ms | 6.3 ms | 3x |
| Phase 2 (Caching) | 6.3 ms | 2.1 ms | 3x (9x total) |
| Phase 3 (Algorithm) | 2.1 ms | 1.4 ms | 1.5x (13x total) |

**Final**: 18.8ms → **1.4ms** (13x speedup)

### Per-Entity Cost

**Current**: 0.169 ms/entity  
**Target**: 0.013 ms/entity  
**Speedup**: 13x

---

## 🎯 Implementation Priority

### P0: Fix O(E²) Parent Lookup (Immediate)
**Impact**: 10-50x for edge comparison phase  
**Effort**: 10 minutes  
**Risk**: Very low

### P1: Cache Field Metadata in Attribute Comparison
**Impact**: 5-10x for attribute comparison  
**Effort**: 30 minutes  
**Risk**: Low

### P2: Early Exit on Identical Trees
**Impact**: Infinite speedup for "no changes" case  
**Effort**: 20 minutes  
**Risk**: Low

### P3: Cache Edge Sets in EntityTree
**Impact**: 2-3x for edge comparison  
**Effort**: 1 hour  
**Risk**: Medium (changes EntityTree structure)

### P4: Cache Non-Entity Attributes
**Impact**: 2-3x for attribute comparison  
**Effort**: 1-2 hours  
**Risk**: Medium (memory usage)

---

## 🔬 Tree Caching Strategy

### What Can Be Cached in EntityTree?

**Already Cached**:
- ✅ `nodes: Dict[UUID, Entity]`
- ✅ `edges: Dict[Tuple[UUID, UUID], EntityEdge]`
- ✅ `outgoing_edges: Dict[UUID, List[UUID]]`
- ✅ `incoming_edges: Dict[UUID, List[UUID]]`
- ✅ `ancestry_paths: Dict[UUID, List[UUID]]`

**Should Be Cached** (computed once, reused):
- ❌ Edge set: `frozenset(edges.keys())`
- ❌ Entity ID set: `frozenset(nodes.keys())`
- ❌ Non-entity attributes per entity
- ❌ Entity hashes for fast comparison

**Trade-off**: Memory vs Speed
- Caching increases memory usage
- But diff is called frequently (2x per operation)
- **Worth it for 10x+ speedup**

---

## 🎓 Key Insights

1. **O(E²) parent lookup is the worst offender** - must fix first
2. **Edge set creation is wasteful** - cache as frozenset
3. **Attribute comparison is expensive** - use cached metadata
4. **Sorting is unnecessary** - just iterate
5. **Early exit opportunities** - check for "no changes" case

**Next**: Implement P0-P2 optimizations and benchmark!
