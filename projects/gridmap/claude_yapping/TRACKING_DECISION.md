# Derivation Tracking Decision for GridMap

## Framework Limitation Discovered

**Single-entity returns from transactional path do NOT get automatic tracking.**

### Test Results (P10)
- ✅ Multi-entity returns: Full tracking (`derived_from_function`, `derived_from_execution_id`, siblings)
- ⚠️ Single-entity creation (no inputs): Partial tracking (`derived_from_function` only)
- ❌ Single-entity mutation (has inputs): No tracking

### Our Use Case
```python
@CallableRegistry.register("compute_navigation_graph")
def compute_navigation_graph(grid_map: GridMap) -> NavigationGraph:
    # Has Entity input → Transactional path
    # Returns single entity → _finalize_single_entity_result
    # Gets NO automatic tracking ❌
```

## Decision: Manual Tracking

We will use **manual causal tracking** via explicit fields:

```python
class NavigationGraph(Entity):
    source_grid_id: UUID  # Manual tracking: which GridMap this came from
    
    # derived_from_function will be None (framework limitation)
    # derived_from_execution_id will be None (framework limitation)
```

### Checking Staleness
```python
# Check if graph is stale
if nav_graph.source_grid_id != current_grid.ecs_id:
    # Graph is stale, recompute
    nav_graph = CallableRegistry.execute("compute_navigation_graph", grid_map=current_grid)
```

### Causal Chain
```
GridMap (ecs_id: abc-123)
    ↓ [compute_navigation_graph]
NavigationGraph (source_grid_id: abc-123)  ← Manual link
    ↓ [find_path_astar]
Path (source_graph_id: def-456)  ← Manual link
```

## Alternative: Fix Framework

To get automatic tracking, we would need to:
1. Add `execution_id` parameter to `_finalize_single_entity_result`
2. Call `_apply_semantic_actions` in that function
3. Update call site in `_execute_transactional`

This is a framework-level change that's out of scope for GridMap project.

## Conclusion

**Use manual tracking fields** (`source_grid_id`, etc.) for now.
- ✅ Simple and explicit
- ✅ Works with current framework
- ✅ Easy to understand causal relationships
- ❌ Not automatic (must remember to set the field)

If framework gets fixed later, we can migrate to automatic tracking.
