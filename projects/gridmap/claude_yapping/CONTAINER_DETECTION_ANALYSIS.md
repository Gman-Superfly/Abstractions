# Container Entity Detection Analysis

## The Question
How does the framework detect changes to `List[Entity]` fields?

## Code Flow

### 1. get_non_entity_attributes() - Line 855
```python
if get_pydantic_field_type_entities(entity, field_name, detect_non_entities=True) is True:
    non_entity_attrs[field_name] = getattr(entity, field_name)
```

Only includes fields where `get_pydantic_field_type_entities(..., detect_non_entities=True)` returns `True`.

### 2. get_pydantic_field_type_entities() - Lines 326-367

For `inventory: List[Item]` where `Item` is an `Entity`:

```python
# Line 326-328: Check if list contains entities
if isinstance(field_value, list) and field_value:
    if any(isinstance(item, Entity) for item in field_value):
        is_entity_container = True  # ✓ This is True!

# Line 366-367: Return value for entity containers
if is_entity_container:
    return None if detect_non_entities else None  # Returns None!
```

So for `List[Entity]` fields:
- `detect_non_entities=True` → returns `None`
- NOT included in `non_entity_attrs`!

### 3. What About Empty Lists?

```python
inventory: List[Item] = Field(default_factory=list)  # Empty initially
```

When the list is EMPTY:
- Line 326: `if isinstance(field_value, list) and field_value:` → **False** (empty list)
- `is_entity_container` stays `False`
- Falls through to line 371-372:
  ```python
  if detect_non_entities:
      return True  # Returns True for empty list!
  ```

So:
- **Empty** `List[Entity]` → treated as non-entity field (compared)
- **Non-empty** `List[Entity]` → treated as entity field (NOT compared)

## The Real Detection Mechanism

`List[Entity]` changes are detected through **edge changes**, not attribute comparison!

### find_modified_entities() - Lines 943-980

```python
# Step 2: Compare edge sets to identify moved entities
new_edges = set()  # All parent-child relationships
old_edges = set()

# Find edges that exist in new tree but not in old tree
added_edges = new_edges - old_edges

# Find edges that exist in old tree but not in new tree  
removed_edges = old_edges - new_edges
```

When we do `inventory.append(apple)`:
1. Apple becomes a child of Collector
2. New edge: `(collector_id, apple_id)` 
3. This edge didn't exist in old tree
4. Detected as `added_edge`
5. Collector's path marked for versioning

## So Why Didn't It Work?

Let me check if the edges are being created correctly for List fields...

Actually, looking at our test output:
```
Item[0]: 94a226fb... → a0f03637...  # Item DID get new ecs_id!
```

The item changed! So edges ARE being detected. But why didn't the Collector change?

## Hypothesis

The framework detected:
1. ✓ Item moved (new edge to collector.inventory)
2. ✓ Item versioned (new ecs_id)
3. ✓ Container versioned (new ecs_id)
4. ❌ Collector NOT versioned (same ecs_id)

Maybe the Collector wasn't marked for versioning because:
- The edge change was from Container → Item (item removed from container.items)
- NOT from Collector → Item (item added to collector.inventory)?

Let me check if List[Entity] fields create edges...
