# Bug Investigation: List[Entity] Parent Not Versioned

## Summary
When an entity is added to a `List[Entity]` field, the framework detects the new edge but does NOT version the parent entity that owns the list.

## Evidence from Test

### Tree v0 (Before)
```
Container(7ccbd95f) → Item(6c6a3da7) [field: items]      ← apple
Container(7ccbd95f) → Item(2280fb21) [field: items]      ← banana  
Container(7ccbd95f) → Collector(7eb91303) [field: collector]
```

### Tree v1 (After transfer)
```
Container(9eb56ffe) → Item(2280fb21) [field: items]      ← banana only
Container(9eb56ffe) → Collector(7eb91303) [field: collector]  ← SAME ecs_id!
Collector(7eb91303) → Item(6c6a3da7) [field: inventory]  ← NEW EDGE!
```

### What Changed
1. ✅ Container versioned: `7ccbd95f → 9eb56ffe` (apple removed from items)
2. ❌ Collector NOT versioned: `7eb91303 → 7eb91303` (apple added to inventory)
3. ✅ Apple (Item) versioned: `6c6a3da7 → 6c6a3da7` (wait, it's the same!)

Actually, looking closer at the item IDs:
- Tree v0: apple=`6c6a3da7`, banana=`2280fb21`
- Tree v1: banana=`2280fb21`, apple=`6c6a3da7`

**The item ecs_ids are also the SAME!** Only the Container got a new ecs_id!

## Revised Understanding

Looking at the test output again:
```
Item[0]: 6c6a3da7... → 2280fb21...
```

This is comparing `container_v0.items[0]` to `container_v1.items[0]`:
- v0.items[0] = apple (6c6a3da7)
- v1.items[0] = banana (2280fb21)

They're DIFFERENT items, not the same item versioned!

Let me trace what actually happened...

## Actual State

### v0 Tree Entities
- Container: 7ccbd95f
- Collector: 7eb91303
- Apple: 6c6a3da7
- Banana: 2280fb21

### v1 Tree Entities  
- Container: 9eb56ffe (NEW)
- Collector: 7eb91303 (SAME)
- Apple: 6c6a3da7 (SAME)
- Banana: 2280fb21 (SAME)

**Only the Container got a new ecs_id!**

## The Bug

When we:
1. Remove apple from `container.items`
2. Add apple to `collector.inventory`

The framework should detect:
- Edge removed: `Container → Apple [items]`
- Edge added: `Collector → Apple [inventory]`

And version:
- ✅ Container (edge removed) → NEW ecs_id
- ❌ Collector (edge added) → SAME ecs_id (BUG!)
- ❌ Apple (moved) → SAME ecs_id (BUG!)

## Root Cause Hypothesis

In `find_modified_entities()` (entity.py lines 960-980):

```python
# Identify moved entities
for source_id, target_id in added_edges:
    if target_id in common_entities:
        # Check if this entity has a different parent
        if old_parents != new_parents:
            moved_entities.add(target_id)
            
            # Mark the entire path for the moved entity for versioning
            path = new_tree.get_ancestry_path(target_id)
            modified_entities.update(path)  # ← Only marks TARGET's path!
```

When `Collector → Apple` edge is added:
- `target_id` = Apple
- Apple's path = `[Container, Collector, Apple]`
- Marks: Container, Collector, Apple

Wait, that SHOULD mark the Collector! Let me check if Apple is in `common_entities`...

## Key Question

Is Apple in `common_entities`?

```python
common_entities = new_entity_ids & old_entity_ids
```

If Apple exists in BOTH trees with the SAME ecs_id, then yes.

But the edge logic only triggers if:
```python
if target_id in common_entities:
```

So it only checks moved entities that already existed. But what about NEW edges to EXISTING entities?

## Next Steps

1. Add debug output to see what `find_modified_entities` returns
2. Check if Apple is in `common_entities`
3. Check if the edge change is detected as `added_edges`
4. Trace through the exact logic flow
5. Identify the specific line where the bug occurs
