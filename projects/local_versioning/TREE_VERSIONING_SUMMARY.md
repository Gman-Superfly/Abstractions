# Tree Structure Preservation - Implementation Summary

**Date**: October 19, 2025  
**Feature**: `preserve_tree_structure` runtime flag for intelligent tree-aware execution

---

## Overview

Implemented a runtime flag that enables tree-aware execution in the CallableRegistry, solving the problem of tree structure preservation during transactional execution.

---

## The Problem

### Before Implementation

When executing functions with entities from the same tree:

```python
gridmap = GridMap(nodes=[node1, node2])
result = execute("move_agent", gridmap=gridmap, node1=node1, node2=node2, agent=agent)
```

**Issue**: Each entity was copied independently, breaking tree relationships:
- `gridmap` copy had ORIGINAL nodes (unmodified)
- `node1` and `node2` were separate copies
- Modifications to nodes were lost in the gridmap result

### Root Cause

The transactional execution model creates isolated copies of each input entity to prevent side effects. However, this breaks parent-child relationships when multiple entities from the same tree are passed as arguments.

---

## The Solution

### Runtime Flag: `preserve_tree_structure`

```python
result = CallableRegistry.execute(
    "function_name",
    preserve_tree_structure=True,  # Enable tree-aware execution
    **kwargs
)
```

When enabled:
1. **Input Phase**: Groups entities by `root_ecs_id`, copies each tree ONCE
2. **Execution Phase**: Function operates on tree-connected entities
3. **Output Phase**: Versions affected trees, updates entity IDs

---

## Implementation Details

### Phase 1: Input Preparation (`_prepare_transactional_inputs`)

**Key Logic**:
```python
# Group entities by root_ecs_id
entities_by_root: Dict[UUID, List[Tuple[str, Entity]]] = {}

for param_name, entity in kwargs.items():
    root_id = entity.root_ecs_id or entity.ecs_id
    entities_by_root[root_id].append((param_name, entity))

# For each tree, copy ONCE and map all entities
for root_id, entities_list in entities_by_root.items():
    # Fetch root (from inputs or storage)
    root_entity = find_or_fetch_root(root_id)
    
    # Copy entire tree
    root_copy = root_entity.model_copy(deep=True)
    root_copies_map[root_id] = root_copy
    
    # Build tree and map input entities to their copies
    tree = build_entity_tree(root_copy)
    for param_name, original_entity in entities_list:
        copy_entity = tree.nodes[original_entity.ecs_id]
        execution_kwargs[param_name] = copy_entity
```

**Key Features**:
- Always copies full tree (even for single entities)
- Auto-fetches missing parent trees from storage
- Maintains tree structure during copying
- Tracks all root copies in `root_copies_map`

### Phase 2: Output Versioning (`_version_affected_trees`)

**Key Logic**:
```python
# Track which trees need versioning
trees_to_version = set()

# 1. Add trees that output entities belong to (after execution)
for output_entity in result:
    current_tree_id = output_entity.root_ecs_id or output_entity.ecs_id
    trees_to_version.add(current_tree_id)

# 2. Add trees that output entities came from (before execution)
#    Use lineage_id to track entities across copies
for output_entity in result:
    if output_entity.lineage_id in input_tree_membership:
        original_tree_id = input_tree_membership[output_entity.lineage_id]
        if original_tree_id != current_tree_id:
            # Entity moved trees! Version both
            trees_to_version.add(original_tree_id)

# 3. Version each affected tree
for tree_id in trees_to_version:
    root_copy = root_copies_map[tree_id]
    EntityRegistry.version_entity(root_copy, force_versioning=True)
```

**Key Features**:
- Detects cross-tree movement (entity moved from Tree A to Tree B)
- Uses `lineage_id` to track entities across copies
- Versions both source and target trees in cross-tree scenarios
- `version_entity()` modifies entities in-place (updates `ecs_id`, `root_ecs_id`)

---

## Test Coverage

### ✅ SCENARIO 1: Global Functions with Tree Preservation

**Setup**: Pass entire tree (gridmap + nodes)

```python
result = execute("move_agent_global", 
                 preserve_tree_structure=True,
                 gridmap=gridmap, node1=node1, node2=node2, agent=agent)
```

**Expected**: Tree copied once, all entities mapped, modifications preserved

**Result**: ✅ PASS

---

### ✅ SCENARIO 2: Borrowing Pattern

**Setup**: Independent root entities → new root entity

```python
result = execute("create_report", student=student, course=course)
# Returns: Report (new root)
```

**Expected**: Works without tree preservation (independent entities)

**Result**: ✅ PASS

---

### ✅ SCENARIO 3: Local Functions with Auto-Reattachment

**Setup**: Pass only child entities (nodes), system auto-fetches parent

```python
result = execute("move_agent_local",
                 preserve_tree_structure=True,
                 source_node=node1, target_node=node2, agent=agent)
# Returns: [node1, node2] (both modified)
```

**Expected**: 
- Parent gridmap auto-fetched from storage
- Parent gridmap versioned
- Output nodes point to new versioned tree

**Result**: ✅ PASS

---

### ✅ SCENARIO 3B: Cross-Tree Movement

**Setup**: Move entity from Tree A to Tree B

```python
# node_A from gridmap_A, node_B from gridmap_B
result = execute("move_agent_local",
                 preserve_tree_structure=True,
                 source_node=node_A, target_node=node_B, agent=agent)
# Returns: [node_A, node_B] (from different trees)
```

**Expected**:
- BOTH gridmap_A and gridmap_B versioned
- node_A belongs to new gridmap_A version
- node_B belongs to new gridmap_B version

**Result**: ✅ PASS

---

### ✅ SCENARIO 4: Pure Local Processing

**Setup**: Orphan entities (no parent tree)

```python
result = execute("process_nodes_orphan",
                 node1=orphan_node1, node2=orphan_node2)
```

**Expected**: Works as independent roots

**Result**: ✅ PASS

---

## Key Insights

### 1. Cross-Tree Movement Detection

The system correctly handles entities moving between trees by:
- Tracking input tree membership by `lineage_id`
- Comparing output entity's current tree vs original tree
- Versioning BOTH trees when movement is detected

### 2. In-Place ID Updates

`EntityRegistry.version_entity()` modifies entities in-place:
- Updates `ecs_id` to new value
- Updates `root_ecs_id` to new root
- Updates `live_id` to new version
- Preserves `lineage_id` for tracking

This means output entities automatically have correct IDs after versioning - no need to fetch from storage!

### 3. Always Copy Full Tree

When `preserve_tree_structure=True`, the system ALWAYS copies the full tree, even for single entities. This ensures:
- All trees are available for versioning
- Cross-tree scenarios work correctly
- Consistent behavior regardless of input count

---

## Next Steps: Optimization Opportunities

### Problem: Unnecessary Full Tree Versioning

Currently, when a tree is modified, we version the ENTIRE tree using `force_versioning=True`. This means:
- All entities in the tree get new `ecs_id` values
- All entities are re-serialized and stored
- Performance degrades with large trees

### Opportunity: Selective Entity Versioning

**Key Insight**: We can infer which specific sub-entities were modified from:
1. **Function signature**: Which entities were passed as inputs
2. **Function outputs**: Which entities were returned
3. **Tree structure**: Which entities are ancestors/descendants

**Example**:
```python
def move_agent(source_node: Node, target_node: Node, agent: Agent) -> Tuple[Node, Node]:
    # Only source_node and target_node are modified
    # Parent gridmap structure unchanged (just references updated)
    pass
```

**Optimization Strategy**:
- Only version entities that were:
  - Passed as inputs (modified directly)
  - Returned as outputs (modified and returned)
  - Ancestors of modified entities (path to root)
- Skip versioning unmodified siblings

### Test Scenarios Needed

1. **Output Filtering**: Only return modified tree
   - Move agent from Tree A to Tree B
   - Only return node from Tree B
   - Verify Tree A is NOT versioned (no outputs from it)

2. **Within-Tree Modification**: Partial tree output
   - Modify node within tree
   - Only return modified node
   - Verify modifications visible in non-returned siblings (full tree versioned)

3. **Ancestor Path Versioning**: Selective versioning
   - Modify deep child entity
   - Only version: child + ancestors to root
   - Skip versioning: unmodified siblings

---

## Performance Implications

### Current Approach (Full Tree Versioning)

**Pros**:
- Simple and correct
- No risk of missing modifications
- Consistent behavior

**Cons**:
- Scales poorly with tree size
- Unnecessary work for localized changes
- High storage overhead

### Proposed Approach (Selective Versioning)

**Pros**:
- Scales with modification size, not tree size
- Significant performance gains for large trees
- Reduced storage overhead

**Cons**:
- More complex logic
- Risk of missing edge cases
- Requires careful testing

**Estimated Impact**:
- Large trees (1000+ entities): 10-100x speedup
- Small modifications: 5-10x speedup
- Full tree modifications: No change (same work)
