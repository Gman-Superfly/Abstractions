# Entity Graph & Versioning System: Complete Understanding

## Core Data Structures

### 1. Entity Identity System

Every entity has **three identity layers**:

```python
class Entity(BaseModel):
    # Layer 1: Persistent Content Identity
    ecs_id: UUID              # Unique identifier for this version
    lineage_id: UUID          # Shared across all versions of same entity
    
    # Layer 2: Runtime Session Identity  
    live_id: UUID             # Changes on every retrieval from storage
    
    # Layer 3: Tree Position Identity
    root_ecs_id: UUID         # Points to root entity's ecs_id
    root_live_id: UUID        # Points to root entity's live_id
    
    # Versioning Metadata
    previous_ecs_id: UUID     # Previous version's ecs_id
    old_ids: List[UUID]       # Complete version history
    forked_at: datetime       # When this version was created
    created_at: datetime      # Original creation time
```

**Key Insight**: 
- `ecs_id` changes on every modification → **immutable versioning**
- `live_id` changes on every storage retrieval → **prevents accidental mutation**
- `lineage_id` stays constant → **tracks entity across all versions**

### 2. EntityTree Structure

The `EntityTree` is a **directed acyclic graph (DAG)** with optimized indexes:

```python
class EntityTree(BaseModel):
    # Core Identity
    root_ecs_id: UUID                              # Root entity's ecs_id
    lineage_id: UUID                               # Lineage of root entity
    
    # Primary Storage
    nodes: Dict[UUID, Entity]                      # ecs_id → Entity object
    edges: Dict[Tuple[UUID, UUID], EntityEdge]     # (source, target) → Edge details
    
    # Adjacency Lists (for graph traversal)
    outgoing_edges: Dict[UUID, List[UUID]]         # entity → [children]
    incoming_edges: Dict[UUID, List[UUID]]         # entity → [parents]
    
    # Path Optimization
    ancestry_paths: Dict[UUID, List[UUID]]         # entity → [path to root]
    
    # Live ID Mapping (for session tracking)
    live_id_to_ecs_id: Dict[UUID, UUID]            # live_id → ecs_id
    
    # Metadata
    node_count: int
    edge_count: int
    max_depth: int
```

**Graph Properties**:
- **Hierarchical**: Each entity has ONE primary parent (hierarchical edge)
- **DAG**: No circular references allowed (enforced in `build_entity_tree`)
- **Multi-indexed**: Fast lookup by ecs_id, live_id, or ancestry path

### 3. EntityEdge: Relationship Metadata

```python
class EntityEdge(BaseModel):
    source_id: UUID                    # Parent entity
    target_id: UUID                    # Child entity
    edge_type: EdgeType                # DIRECT, LIST, DICT, SET, TUPLE
    field_name: str                    # Field name in parent
    container_index: Optional[int]     # For LIST/TUPLE
    container_key: Optional[Any]       # For DICT
    ownership: bool                    # Always True (simplified)
    is_hierarchical: bool              # Primary ownership edge
```

**Edge Types**:
- `DIRECT`: `parent.child = entity`
- `LIST`: `parent.children[i] = entity`
- `DICT`: `parent.entities[key] = entity`
- `TUPLE`: `parent.items[i] = entity`
- `SET`: `parent.items = {entity, ...}`

**Critical**: `container_index` and `container_key` track **exact position** in containers.

---

## Tree Building Algorithm

### `build_entity_tree(root_entity)` - Single-Pass BFS

**Algorithm** (lines 636-840):

```python
def build_entity_tree(root_entity: Entity) -> EntityTree:
    tree = EntityTree(root_ecs_id=root_entity.ecs_id, lineage_id=root_entity.lineage_id)
    to_process = deque([(root_entity, None)])  # (entity, parent_id)
    processed = set()
    ancestry_paths = {root_entity.ecs_id: [root_entity.ecs_id]}
    
    while to_process:
        entity, parent_id = to_process.popleft()
        
        # Circular reference check
        if entity.ecs_id in processed and parent_id is not None:
            raise ValueError("Circular entity reference detected")
        
        processed.add(entity.ecs_id)
        
        # Update ancestry path
        if parent_id:
            parent_path = ancestry_paths[parent_id]
            entity_path = parent_path + [entity.ecs_id]
            ancestry_paths[entity.ecs_id] = entity_path
            tree.set_ancestry_path(entity.ecs_id, entity_path)
        
        # Process all fields
        for field_name in entity.model_fields:
            value = getattr(entity, field_name)
            if value is None:
                continue
            
            field_type = get_pydantic_field_type_entities(entity, field_name)
            
            # Direct entity
            if isinstance(value, Entity):
                tree.add_entity(value)
                tree.add_direct_edge(entity, value, field_name)
                to_process.append((value, entity.ecs_id))
            
            # List of entities
            elif isinstance(value, list) and field_type:
                for i, item in enumerate(value):
                    if isinstance(item, Entity):
                        tree.add_entity(item)
                        tree.add_list_edge(entity, item, field_name, i)
                        to_process.append((item, entity.ecs_id))
            
            # Dict of entities
            elif isinstance(value, dict) and field_type:
                for k, v in value.items():
                    if isinstance(v, Entity):
                        tree.add_entity(v)
                        tree.add_dict_edge(entity, v, field_name, k)
                        to_process.append((v, entity.ecs_id))
            
            # Tuple/Set similar...
    
    return tree
```

**Performance**:
- **Time**: O(N + E) where N = entities, E = edges
- **Space**: O(N) for nodes + O(E) for edges
- **Single pass**: Builds all indexes simultaneously

**Key Features**:
1. **BFS traversal**: Ensures shortest paths to root
2. **Ancestry tracking**: Computed on-the-fly during traversal
3. **Edge classification**: Marks hierarchical edges immediately
4. **Container awareness**: Tracks exact position in lists/dicts

---

## Diff Computation: `find_modified_entities()`

### Three-Phase Comparison (lines 901-1043)

```python
def find_modified_entities(new_tree, old_tree, greedy=True) -> Set[UUID]:
    modified_entities = set()
    
    # PHASE 1: Node Set Comparison
    new_entity_ids = set(new_tree.nodes.keys())
    old_entity_ids = set(old_tree.nodes.keys())
    
    added_entities = new_entity_ids - old_entity_ids
    removed_entities = old_entity_ids - new_entity_ids
    common_entities = new_entity_ids & old_entity_ids
    
    # Mark all added entities and their ancestry
    for entity_id in added_entities:
        path = new_tree.get_ancestry_path(entity_id)
        modified_entities.update(path)  # ENTIRE PATH marked
    
    # PHASE 2: Edge Set Comparison (detect moves)
    new_edges = {(src, tgt) for (src, tgt) in new_tree.edges.keys()}
    old_edges = {(src, tgt) for (src, tgt) in old_tree.edges.keys()}
    
    added_edges = new_edges - old_edges
    removed_edges = old_edges - new_edges
    
    # Detect moved entities (same entity, different parent)
    for source_id, target_id in added_edges:
        if target_id in common_entities:
            old_parents = {src for (src, tgt) in old_edges if tgt == target_id}
            new_parents = {src for (src, tgt) in new_edges if tgt == target_id}
            
            if old_parents != new_parents:
                path = new_tree.get_ancestry_path(target_id)
                modified_entities.update(path)  # ENTIRE PATH marked
    
    # PHASE 3: Attribute Comparison (for remaining entities)
    remaining_entities = []
    for entity_id in common_entities:
        if entity_id not in modified_entities:
            path_length = len(new_tree.get_ancestry_path(entity_id))
            remaining_entities.append((path_length, entity_id))
    
    # Sort by path length (descending) - process leaves first
    remaining_entities.sort(reverse=True)
    
    for _, entity_id in remaining_entities:
        if entity_id in modified_entities:
            continue  # Already marked by child
        
        new_entity = new_tree.get_entity(entity_id)
        old_entity = old_tree.get_entity(entity_id)
        
        # Compare non-entity attributes
        has_changes = compare_non_entity_attributes(new_entity, old_entity)
        
        if has_changes:
            path = new_tree.get_ancestry_path(entity_id)
            modified_entities.update(path)  # ENTIRE PATH marked
            
            if greedy:
                continue  # Skip parent checks (optimization)
    
    return modified_entities
```

**Critical Insights**:

1. **Set-based comparison**: O(N) instead of O(N²)
2. **Ancestry propagation**: If child changes, ALL ancestors marked
3. **Leaf-first processing**: Prevents redundant comparisons
4. **Greedy optimization**: Stops checking parents once child marked

**Why Ancestry Propagation?**
- Immutable versioning requires ALL ancestors to get new `ecs_id`s
- Parent's `ecs_id` must change if child's `ecs_id` changes
- Ensures tree consistency

---

## Versioning Flow: `version_entity()`

### Complete Versioning Process (lines 1438-1511)

```python
def version_entity(cls, entity: Entity, force_versioning=False) -> bool:
    # Step 1: Check if entity is registered
    old_tree = cls.get_stored_tree(entity.root_ecs_id)
    if old_tree is None:
        cls.register_entity(entity)  # First registration
        return True
    
    # Step 2: Build new tree from current state
    new_tree = build_entity_tree(entity)
    
    # Step 3: Compute diff
    if force_versioning:
        modified_entities = new_tree.nodes.keys()  # All entities
    else:
        modified_entities = find_modified_entities(new_tree, old_tree)
    
    if len(modified_entities) == 0:
        return False  # No changes
    
    # Step 4: Version root entity first
    current_root_ecs_id = new_tree.root_ecs_id
    root_entity = new_tree.get_entity(current_root_ecs_id)
    
    root_entity.update_ecs_ids()  # Generates new ecs_id
    new_root_ecs_id = root_entity.ecs_id
    
    id_mapping = {current_root_ecs_id: new_root_ecs_id}
    
    # Update nodes dict with new root ID
    new_tree.nodes.pop(current_root_ecs_id)
    new_tree.nodes[new_root_ecs_id] = root_entity
    
    # Step 5: Version all modified entities
    modified_entities.remove(current_root_ecs_id)  # Already versioned
    
    for modified_entity_id in modified_entities:
        modified_entity = new_tree.get_entity(modified_entity_id)
        if modified_entity:
            old_ecs_id = modified_entity.ecs_id
            modified_entity.update_ecs_ids(new_root_ecs_id, root_entity.live_id)
            new_ecs_id = modified_entity.ecs_id
            id_mapping[old_ecs_id] = new_ecs_id
    
    # Step 6: Update all tree mappings
    update_tree_mappings_after_versioning(new_tree, id_mapping)
    
    # Step 7: Register new tree version
    new_tree.lineage_id = root_entity.lineage_id
    cls.register_entity_tree(new_tree)
    
    return True
```

### `update_tree_mappings_after_versioning()` (lines 1047-1110)

**Critical Fix**: Updates ALL tree indexes after `ecs_id` changes:

```python
def update_tree_mappings_after_versioning(tree, id_mapping):
    # 1. Update nodes mapping
    updated_nodes = {}
    for old_ecs_id, entity in tree.nodes.items():
        new_ecs_id = id_mapping.get(old_ecs_id, old_ecs_id)
        updated_nodes[new_ecs_id] = entity
    tree.nodes = updated_nodes
    
    # 2. Update edges mapping and edge objects
    updated_edges = {}
    for (old_src, old_tgt), edge in tree.edges.items():
        new_src = id_mapping.get(old_src, old_src)
        new_tgt = id_mapping.get(old_tgt, old_tgt)
        edge.source_id = new_src  # Update edge object
        edge.target_id = new_tgt
        updated_edges[(new_src, new_tgt)] = edge
    tree.edges = updated_edges
    
    # 3. Update outgoing_edges
    updated_outgoing = defaultdict(list)
    for old_src, targets in tree.outgoing_edges.items():
        new_src = id_mapping.get(old_src, old_src)
        new_targets = [id_mapping.get(t, t) for t in targets]
        updated_outgoing[new_src] = new_targets
    tree.outgoing_edges = updated_outgoing
    
    # 4. Update incoming_edges
    updated_incoming = defaultdict(list)
    for old_tgt, sources in tree.incoming_edges.items():
        new_tgt = id_mapping.get(old_tgt, old_tgt)
        new_sources = [id_mapping.get(s, s) for s in sources]
        updated_incoming[new_tgt] = new_sources
    tree.incoming_edges = updated_incoming
    
    # 5. Update ancestry_paths
    updated_ancestry = {}
    for old_id, path in tree.ancestry_paths.items():
        new_id = id_mapping.get(old_id, old_id)
        new_path = [id_mapping.get(p, p) for p in path]
        updated_ancestry[new_id] = new_path
    tree.ancestry_paths = updated_ancestry
    
    # 6. Update live_id_to_ecs_id
    for live_id, old_ecs_id in tree.live_id_to_ecs_id.items():
        if old_ecs_id in id_mapping:
            tree.live_id_to_ecs_id[live_id] = id_mapping[old_ecs_id]
    
    # 7. Update root_ecs_id
    if tree.root_ecs_id in id_mapping:
        tree.root_ecs_id = id_mapping[tree.root_ecs_id]
```

**Why This Matters**:
- After `update_ecs_ids()`, entity objects have new `ecs_id`s
- But tree mappings still reference OLD `ecs_id`s
- This function synchronizes ALL mappings
- **Prevents lookup failures** in subsequent operations

---

## EntityRegistry: Global State Manager

### Five Registries (lines 1276-1289)

```python
class EntityRegistry:
    # 1. Tree Storage (primary)
    tree_registry: Dict[UUID, EntityTree] = {}
    # root_ecs_id → EntityTree
    
    # 2. Lineage Tracking
    lineage_registry: Dict[UUID, List[UUID]] = {}
    # lineage_id → [root_ecs_id_v1, root_ecs_id_v2, ...]
    
    # 3. Live Session Tracking
    live_id_registry: Dict[UUID, Entity] = {}
    # live_id → Entity (current Python objects)
    
    # 4. ECS ID to Root Mapping
    ecs_id_to_root_id: Dict[UUID, UUID] = {}
    # any_ecs_id → root_ecs_id (for navigation)
    
    # 5. Type Index
    type_registry: Dict[Type[Entity], List[UUID]] = {}
    # EntityClass → [lineage_id1, lineage_id2, ...]
```

### Key Operations

#### `register_entity_tree(tree)` (lines 1292-1316)

```python
def register_entity_tree(cls, tree: EntityTree):
    # Store tree
    cls.tree_registry[tree.root_ecs_id] = tree
    
    # Index all entities by live_id
    for entity in tree.nodes.values():
        cls.live_id_registry[entity.live_id] = entity
        cls.ecs_id_to_root_id[entity.ecs_id] = tree.root_ecs_id
    
    # Update lineage history
    if tree.lineage_id not in cls.lineage_registry:
        cls.lineage_registry[tree.lineage_id] = [tree.root_ecs_id]
    else:
        cls.lineage_registry[tree.lineage_id].append(tree.root_ecs_id)
    
    # Update type index
    root_entity = tree.get_entity(tree.root_ecs_id)
    if root_entity.__class__ not in cls.type_registry:
        cls.type_registry[root_entity.__class__] = [tree.lineage_id]
    else:
        cls.type_registry[root_entity.__class__].append(tree.lineage_id)
```

#### `get_stored_tree(root_ecs_id)` (lines 1364-1372)

```python
def get_stored_tree(cls, root_ecs_id: UUID) -> Optional[EntityTree]:
    stored_tree = cls.tree_registry.get(root_ecs_id)
    if stored_tree is None:
        return None
    
    # CRITICAL: Deep copy with new live_ids
    new_tree = stored_tree.model_copy(deep=True)
    new_tree.update_live_ids()  # Assigns new live_ids to all entities
    return new_tree
```

**Why Deep Copy + New Live IDs?**
- **Immutability enforcement**: Prevents accidental mutation of stored tree
- **Session isolation**: Each retrieval gets unique `live_id`s
- **Divergence detection**: Can compare `live_id` to detect if entity was modified

---

## List Handling: Critical Details

### How Lists Are Indexed

When you have `Map.nodes: List[Node]`, the tree stores:

```python
# In EntityTree.edges:
{
    (map.ecs_id, node0.ecs_id): EntityEdge(
        source_id=map.ecs_id,
        target_id=node0.ecs_id,
        edge_type=EdgeType.LIST,
        field_name="nodes",
        container_index=0  # ← Position in list
    ),
    (map.ecs_id, node1.ecs_id): EntityEdge(
        source_id=map.ecs_id,
        target_id=node1.ecs_id,
        edge_type=EdgeType.LIST,
        field_name="nodes",
        container_index=1  # ← Position in list
    ),
    # ...
}
```

### List Modification Detection

**Scenario**: Move agent from `node[5]` to `node[10]`

**What Changes**:
1. **Edge removal**: `(node5.ecs_id, agent.ecs_id)` removed
2. **Edge addition**: `(node10.ecs_id, agent.ecs_id)` added
3. **Diff detects**: Agent moved (same entity, different parent)
4. **Ancestry propagation**: 
   - `agent` marked modified
   - `node10` marked modified (new child)
   - `node5` marked modified (child removed)
   - `map` marked modified (children changed)

**Problem**: Even though only 2 nodes changed, we check ALL nodes.

---

## Performance Characteristics

### Current System

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| `build_entity_tree(N entities)` | O(N + E) | O(N + E) |
| `find_modified_entities(N entities)` | O(N) | O(N) |
| `version_entity(M modified)` | O(M) | O(M) |
| `update_tree_mappings(M modified)` | O(N + E) | O(N + E) |

**Bottlenecks**:
1. **Full tree build**: O(N) even if only 2 nodes changed
2. **Full diff**: O(N) comparisons even if we know what changed
3. **Full mapping update**: O(N + E) even for local changes

### Optimization Opportunities

**For your scenario** (100 nodes × 100 agents = 10,000 entities):

**Current**:
- Move 1 agent: Build 10,000-node tree, compare 10,000 entities
- **~10,000 operations per move**

**Optimized (with local versioning)**:
- Move 1 agent: Update 2 nodes, version 3 entities (agent + 2 nodes + map)
- **~4 operations per move**

**Speedup**: ~2,500x for single-agent moves!

---

## Key Takeaways

### 1. Immutability is Enforced at Multiple Levels
- `ecs_id` changes on modification
- `live_id` changes on retrieval
- Deep copy on `get_stored_tree()`

### 2. Ancestry Propagation is Fundamental
- Child change → parent must change
- Ensures tree consistency
- But causes cascading updates

### 3. Tree is Fully Indexed
- 7 different indexes maintained
- Fast lookup by any identifier
- But expensive to update

### 4. Lists Are First-Class
- `container_index` tracks position
- Edge changes detect moves
- But no optimization for local changes

### 5. Diff is Greedy by Default
- Stops checking parents once child marked
- Leaf-first processing
- But still checks all common entities

---

## Next Steps for Optimization

1. **Lazy divergence checking**: Skip when we control execution
2. **Partial versioning**: Version only known-modified subtrees
3. **Reattachment pattern**: Update parent without full rebuild
4. **Targeted diff**: Skip comparison for known-unchanged entities

These optimizations are **safe** when:
- We control the execution flow (CallableRegistry)
- We know exactly what changed (function signature tells us)
- No external mutations between calls
