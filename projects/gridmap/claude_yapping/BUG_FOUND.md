# BUG FOUND: List[Entity] Mutations Not Detected

## The Problem

When a function mutates a `List[Entity]` field (like `collector.inventory.append(apple)`), the framework does NOT detect this as a change and does NOT version the entity.

## Root Cause

In `entity.py`, `compare_non_entity_attributes()` line 889:

```python
if value1 != value2:
    return True
```

This compares list objects directly. When a list is mutated in place:
- `old_collector.inventory` points to the SAME list object
- `new_collector.inventory` points to the SAME list object  
- They compare as equal!

## Evidence from Test

```
Collector: 92b6b5d2... → 92b6b5d2...  # ecs_id UNCHANGED!
Are they the same object? Collector: False  # Different Python objects
```

The collector is a different Python object but has the SAME ecs_id because the framework didn't detect the inventory change.

## Why This Happens

1. Function executes on stored copy from registry
2. Function mutates: `collector.inventory.append(apple)`
3. This mutates the list IN PLACE
4. Framework compares old vs new tree
5. Both trees have collectors pointing to SAME mutated list
6. Comparison: `[apple] == [apple]` → True (no change detected!)
7. Collector not versioned

## The Fix Needed

`compare_non_entity_attributes()` needs to:
1. Detect `List[Entity]` fields specially
2. Compare the CONTENTS (entity ecs_ids), not the list object
3. Or compare list lengths at minimum

## Correct Behavior Should Be

```
Collector: 92b6b5d2... → NEW_ID...  # Should get new ecs_id!
```

Because the inventory changed from `[]` to `[apple]`.

## Impact

This affects ANY function that:
- Appends to `List[Entity]`
- Removes from `List[Entity]`  
- Modifies `List[Entity]` in any way

The parent entity won't be versioned unless OTHER fields change!

## Workaround

Reassign the entire list:
```python
# Instead of:
collector.inventory.append(apple)

# Do:
collector.inventory = collector.inventory + [apple]
```

This creates a NEW list object, which will be detected as different.
