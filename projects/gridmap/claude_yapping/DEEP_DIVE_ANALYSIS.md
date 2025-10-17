# Deep Dive: Why Collector Not Versioned

## The Bug Evidence

```
find_modified_entities returned: 0 modified entities
Container DID get versioned: a02feb0f → 453221c5
Collector did NOT get versioned: e240dd7a → e240dd7a (SAME!)

Edge changes:
  Added: Collector → Item [inventory]
  Removed: Container → Item [items] (one apple)
```

## Critical Questions

1. **Why did Container get versioned if find_modified_entities returned 0?**
2. **Why didn't find_modified_entities detect the edge changes?**
3. **Where in the code does versioning actually happen?**

## Let's Trace the Execution Flow

### Step 1: Function Call
```python
container_v1 = CallableRegistry.execute("transfer_item", container=container_v0, item_name="apple")
```

### Step 2: CallableRegistry.execute
Location: `callable_registry.py`

Need to find:
- Which execution path is taken?
- Where does it call versioning?
- Does it use find_modified_entities?

### Step 3: Transactional Execution Path
From our earlier investigation, we know it uses `_prepare_transactional_inputs`.

Let me trace the COMPLETE flow...

## Questions to Answer

1. Where is `EntityRegistry.version_entity()` called?
2. Does it use `find_modified_entities()`?
3. If yes, why did it return 0?
4. If no, what DOES it use to determine what to version?
5. Why did Container get versioned but not Collector?

## Investigation Plan

1. Find where `version_entity` is called in callable_registry
2. Read the complete `version_entity` implementation
3. Understand what it uses to detect changes
4. Find out why it only versioned Container
5. Identify the exact bug location
