# Entity Transfer Problem Analysis

## The Problem

When we execute a function that mutates an entity tree (e.g., `collect_apple`), the framework creates a **new version** of the tree. However, our test code still holds references to the **old versions** of entities.

## Example Scenario

```python
# Initial state
grid = GridMap(...)  # ecs_id: AAA, lineage_id: XXX
agent = Agent(inventory=[])  # Part of grid tree

# Execute function
grid_v1 = CallableRegistry.execute("collect_apple", grid_map=grid, agent=agent, ...)

# Problem:
# - grid_v1 is the NEW version (ecs_id: BBB, lineage_id: XXX)
# - agent still references the OLD version from grid (ecs_id: AAA)
# - agent.inventory is EMPTY because it's the old version!
```

## What Happens During Execution

1. **Input Preparation**: Framework may create copies or borrow entities
2. **Function Execution**: Function mutates entities directly
3. **Versioning**: Framework detects changes and creates new tree version
4. **Return**: Returns new root entity

## The Question

**Where do the mutations go?**

When we do:
```python
agent.inventory.append(apple)
```

Does this mutate:
- A) The original agent (old version)?
- B) A copy of the agent (new version)?
- C) Something else?

## Investigation Needed

1. **Execution Path**: Which path does `collect_apple` take?
   - PATH 1: single_entity_with_config
   - PATH 2: no_inputs
   - PATH 3: transactional (Entity inputs)
   - PATH 4/5: borrowing

2. **Input Handling**: What happens to `agent` parameter?
   - Is it copied?
   - Is it borrowed?
   - Is it used directly?

3. **Mutation Semantics**: When we mutate, what gets mutated?
   - Original entity?
   - Deep copy?
   - Shallow copy?

4. **Versioning Trigger**: When does versioning happen?
   - Before execution?
   - After execution?
   - During execution?

5. **How to Get Latest**: How do we get the latest version?
   - EntityRegistry.lineage_registry[lineage_id][-1]?
   - Some other method?
   - Do child entities (like Agent) have their own lineage?

## Test Case to Understand

```python
# Simple test
container = Container(items=[item1, item2])
container.promote_to_root()

print(f"Before: {len(container.items)} items")
print(f"Container ecs_id: {container.ecs_id}")

# Execute mutation
container_v1 = CallableRegistry.execute("remove_item", container=container, item_name="item1")

print(f"After: {len(container.items)} items")  # OLD version
print(f"After v1: {len(container_v1.items)} items")  # NEW version
print(f"Container ecs_id: {container.ecs_id}")  # OLD
print(f"Container_v1 ecs_id: {container_v1.ecs_id}")  # NEW
```

## Key Questions

1. Does the function mutate the input directly, or a copy?
2. When is the new version created?
3. How do we access child entities in the new version?
4. Do child entities have separate lineages or share parent's lineage?

## Next Steps

1. Read the transactional execution path code
2. Understand `_prepare_transactional_inputs`
3. Understand when versioning happens
4. Understand how to navigate from old to new versions
5. Test with simple case first
