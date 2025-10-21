# Framework Primitives Testing Plan

## Overview

Before implementing GridMap, we need to verify that the Abstractions framework can handle all the operations we need. This document enumerates each primitive capability and provides a test for it.

## Primitive Categories

1. **Entity Creation & Lifecycle**
2. **Entity Hierarchies & Trees**
3. **Direct Mutation & Versioning**
4. **Function Registration & Execution**
5. **Collection Manipulation**
6. **Distributed Addressing**
7. **Entity Tree Operations**

---

## 1. Entity Creation & Lifecycle

### P1.1: Create Simple Entity
**What we need**: Create an entity with basic fields.

**Test**:
```python
from abstractions.ecs.entity import Entity
from pydantic import Field

class SimpleEntity(Entity):
    name: str
    value: int = Field(default=0)

entity = SimpleEntity(name="test", value=42)
assert entity.name == "test"
assert entity.value == 42
assert entity.ecs_id is not None
```

**Status**: ⏳ To test

---

### P1.2: Promote Entity to Root
**What we need**: Make an entity a root entity for distributed addressing.

**Test**:
```python
entity = SimpleEntity(name="test", value=42)
entity.promote_to_root()

assert entity.is_root_entity()
assert entity.root_ecs_id == entity.ecs_id
assert entity.root_live_id == entity.live_id
```

**Status**: ⏳ To test

---

### P1.3: Entity with Collection Fields
**What we need**: Entity containing lists of other entities.

**Test**:
```python
class Container(Entity):
    items: List[SimpleEntity] = Field(default_factory=list)

container = Container(items=[])
item1 = SimpleEntity(name="item1", value=1)
item2 = SimpleEntity(name="item2", value=2)

container.items.append(item1)
container.items.append(item2)

assert len(container.items) == 2
assert container.items[0].name == "item1"
```

**Status**: ⏳ To test

---

## 2. Entity Hierarchies & Trees

### P2.1: Nested Entity Structure
**What we need**: Entity containing other entities (hierarchical tree).

**Test**:
```python
class Parent(Entity):
    name: str
    children: List[SimpleEntity] = Field(default_factory=list)

parent = Parent(name="parent", children=[])
child1 = SimpleEntity(name="child1", value=1)
child2 = SimpleEntity(name="child2", value=2)

parent.children.append(child1)
parent.children.append(child2)
parent.promote_to_root()

# Verify tree structure
assert parent.is_root_entity()
assert len(parent.children) == 2
```

**Status**: ⏳ To test

---

### P2.2: Build Entity Tree
**What we need**: Framework can build tree from nested entities.

**Test**:
```python
from abstractions.ecs.entity_tree import build_entity_tree

parent = Parent(name="parent", children=[])
child = SimpleEntity(name="child", value=1)
parent.children.append(child)
parent.promote_to_root()

tree = build_entity_tree(parent)

assert tree is not None
assert parent.ecs_id in tree.nodes
assert child.ecs_id in tree.nodes
```

**Status**: ⏳ To test

---

### P2.3: Three-Level Hierarchy
**What we need**: GridMap → GridNode → GameEntity (3 levels).

**Test**:
```python
class Level3(Entity):
    value: int

class Level2(Entity):
    items: List[Level3] = Field(default_factory=list)

class Level1(Entity):
    containers: List[Level2] = Field(default_factory=list)

root = Level1(containers=[])
mid = Level2(items=[])
leaf = Level3(value=42)

mid.items.append(leaf)
root.containers.append(mid)
root.promote_to_root()

tree = build_entity_tree(root)
assert root.ecs_id in tree.nodes
assert mid.ecs_id in tree.nodes
assert leaf.ecs_id in tree.nodes
```

**Status**: ⏳ To test

---

## 3. Direct Mutation & Versioning

### P3.1: Direct Field Mutation
**What we need**: Mutate entity field directly, framework handles versioning.

**Test**:
```python
entity = SimpleEntity(name="original", value=10)
entity.promote_to_root()

original_ecs_id = entity.ecs_id

# Direct mutation
entity.name = "modified"
entity.value = 20

# Entity mutated in place
assert entity.name == "modified"
assert entity.value == 20
assert entity.ecs_id == original_ecs_id  # Same object
```

**Status**: ⏳ To test

---

### P3.2: Mutation in Function Creates Version
**What we need**: Function that mutates entity creates new version.

**Test**:
```python
from abstractions.ecs.callable_registry import CallableRegistry

@CallableRegistry.register("mutate_entity")
def mutate_entity(entity: SimpleEntity, new_value: int) -> SimpleEntity:
    entity.value = new_value  # Direct mutation
    return entity

entity = SimpleEntity(name="test", value=10)
entity.promote_to_root()
original_ecs_id = entity.ecs_id

result = CallableRegistry.execute("mutate_entity", entity=entity, new_value=20)
updated = result if not isinstance(result, list) else result[0]

# New version created
assert updated.value == 20
assert updated.ecs_id != original_ecs_id
assert updated.lineage_id == entity.lineage_id
assert entity.value == 10  # Original unchanged
```

**Status**: ⏳ To test

---

### P3.3: List Mutation in Tree
**What we need**: Mutate list within entity tree, framework tracks changes.

**Test**:
```python
@CallableRegistry.register("add_to_container")
def add_to_container(container: Container, item: SimpleEntity) -> Container:
    container.items.append(item)  # Direct list mutation
    return container

container = Container(items=[])
container.promote_to_root()
original_ecs_id = container.ecs_id

item = SimpleEntity(name="new_item", value=99)
result = CallableRegistry.execute("add_to_container", container=container, item=item)
updated = result if not isinstance(result, list) else result[0]

assert len(updated.items) == 1
assert updated.items[0].name == "new_item"
assert updated.ecs_id != original_ecs_id  # New version
```

**Status**: ⏳ To test

---

## 4. Function Registration & Execution

### P4.1: Register Simple Function
**What we need**: Register and execute function with entity parameter.

**Test**:
```python
@CallableRegistry.register("simple_function")
def simple_function(entity: SimpleEntity) -> SimpleEntity:
    entity.value = entity.value * 2
    return entity

entity = SimpleEntity(name="test", value=5)
entity.promote_to_root()

result = CallableRegistry.execute("simple_function", entity=entity)
updated = result if not isinstance(result, list) else result[0]

assert updated.value == 10
```

**Status**: ⏳ To test

---

### P4.2: Function with Multiple Parameters
**What we need**: Function with entity + primitive parameters.

**Test**:
```python
@CallableRegistry.register("multi_param")
def multi_param(entity: SimpleEntity, multiplier: int, name: str) -> SimpleEntity:
    entity.value = entity.value * multiplier
    entity.name = name
    return entity

entity = SimpleEntity(name="old", value=3)
entity.promote_to_root()

result = CallableRegistry.execute("multi_param", 
                                  entity=entity, 
                                  multiplier=4, 
                                  name="new")
updated = result if not isinstance(result, list) else result[0]

assert updated.value == 12
assert updated.name == "new"
```

**Status**: ⏳ To test

---

### P4.3: Function Returning New Entity
**What we need**: Function creates and returns new entity.

**Test**:
```python
@CallableRegistry.register("create_entity")
def create_entity(name: str, value: int) -> SimpleEntity:
    return SimpleEntity(name=name, value=value)

result = CallableRegistry.execute("create_entity", name="created", value=100)
entity = result if not isinstance(result, list) else result[0]

assert entity.name == "created"
assert entity.value == 100
assert entity.ecs_id is not None
```

**Status**: ⏳ To test

---

### P4.4: Function Returning Tuple
**What we need**: Function returns multiple entities as tuple.

**Test**:
```python
from typing import Tuple

@CallableRegistry.register("create_pair")
def create_pair(value1: int, value2: int) -> Tuple[SimpleEntity, SimpleEntity]:
    e1 = SimpleEntity(name="first", value=value1)
    e2 = SimpleEntity(name="second", value=value2)
    return e1, e2

result = CallableRegistry.execute("create_pair", value1=10, value2=20)

# Tuple returns as list
assert isinstance(result, list)
assert len(result) == 2
assert result[0].name == "first"
assert result[1].name == "second"
```

**Status**: ⏳ To test

---

## 5. Collection Manipulation

### P5.1: Append to List in Tree
**What we need**: Add entity to list within tree structure.

**Test**:
```python
@CallableRegistry.register("append_item")
def append_item(parent: Parent, item: SimpleEntity) -> Parent:
    parent.children.append(item)
    return parent

parent = Parent(name="parent", children=[])
parent.promote_to_root()

item = SimpleEntity(name="child", value=42)
result = CallableRegistry.execute("append_item", parent=parent, item=item)
updated = result if not isinstance(result, list) else result[0]

assert len(updated.children) == 1
assert updated.children[0].name == "child"
```

**Status**: ⏳ To test

---

### P5.2: Remove from List in Tree
**What we need**: Remove entity from list within tree structure.

**Test**:
```python
@CallableRegistry.register("remove_item")
def remove_item(parent: Parent, item_name: str) -> Parent:
    parent.children = [c for c in parent.children if c.name != item_name]
    return parent

parent = Parent(name="parent", children=[])
child1 = SimpleEntity(name="keep", value=1)
child2 = SimpleEntity(name="remove", value=2)
parent.children.append(child1)
parent.children.append(child2)
parent.promote_to_root()

result = CallableRegistry.execute("remove_item", parent=parent, item_name="remove")
updated = result if not isinstance(result, list) else result[0]

assert len(updated.children) == 1
assert updated.children[0].name == "keep"
```

**Status**: ⏳ To test

---

### P5.3: Move Item Between Lists
**What we need**: Remove from one list, add to another (within same tree).

**Test**:
```python
class TwoContainers(Entity):
    list_a: List[SimpleEntity] = Field(default_factory=list)
    list_b: List[SimpleEntity] = Field(default_factory=list)

@CallableRegistry.register("move_item")
def move_item(container: TwoContainers, item_name: str) -> TwoContainers:
    # Find and remove from list_a
    item = None
    for i, entity in enumerate(container.list_a):
        if entity.name == item_name:
            item = container.list_a.pop(i)
            break
    
    # Add to list_b
    if item:
        container.list_b.append(item)
    
    return container

container = TwoContainers(list_a=[], list_b=[])
item = SimpleEntity(name="movable", value=42)
container.list_a.append(item)
container.promote_to_root()

result = CallableRegistry.execute("move_item", container=container, item_name="movable")
updated = result if not isinstance(result, list) else result[0]

assert len(updated.list_a) == 0
assert len(updated.list_b) == 1
assert updated.list_b[0].name == "movable"
```

**Status**: ⏳ To test

---

## 6. Distributed Addressing

### P6.1: Access Field via Address
**What we need**: Use @uuid.field to access entity data.

**Test**:
```python
from abstractions.ecs.functional_api import get

entity = SimpleEntity(name="test", value=42)
entity.promote_to_root()

# Access via address
name = get(f"@{entity.ecs_id}.name")
value = get(f"@{entity.ecs_id}.value")

assert name == "test"
assert value == 42
```

**Status**: ⏳ To test

---

### P6.2: Function with Address Parameter
**What we need**: Pass address string to function, framework resolves it.

**Test**:
```python
@CallableRegistry.register("process_name")
def process_name(name: str, suffix: str) -> str:
    return name + suffix

entity = SimpleEntity(name="hello", value=0)
entity.promote_to_root()

# Pass address as parameter
result = CallableRegistry.execute("process_name",
                                  name=f"@{entity.ecs_id}.name",
                                  suffix="_world")

# Result should be string, not entity
assert result == "hello_world" or (isinstance(result, list) and result[0] == "hello_world")
```

**Status**: ⏳ To test

---

### P6.3: Mixed Addresses and Values
**What we need**: Function with some address parameters, some direct values.

**Test**:
```python
@CallableRegistry.register("create_from_address")
def create_from_address(name: str, value: int, multiplier: int) -> SimpleEntity:
    return SimpleEntity(name=name, value=value * multiplier)

source = SimpleEntity(name="source", value=10)
source.promote_to_root()

result = CallableRegistry.execute("create_from_address",
                                  name=f"@{source.ecs_id}.name",  # Address
                                  value=f"@{source.ecs_id}.value",  # Address
                                  multiplier=3)  # Direct value

entity = result if not isinstance(result, list) else result[0]
assert entity.name == "source"
assert entity.value == 30
```

**Status**: ⏳ To test

---

## 7. Entity Tree Operations

### P7.1: Detach Entity from Tree
**What we need**: Remove entity from tree, call detach().

**Test**:
```python
parent = Parent(name="parent", children=[])
child = SimpleEntity(name="child", value=42)
parent.children.append(child)
parent.promote_to_root()

# Child is part of tree
assert not child.is_root_entity()

# Remove and detach
parent.children.remove(child)
child.detach()

# Child is now root
assert child.is_root_entity()
```

**Status**: ⏳ To test

---

### P7.2: Attach Root Entity to Tree
**What we need**: Add external root entity to tree, call attach().

**Test**:
```python
parent = Parent(name="parent", children=[])
parent.promote_to_root()

# Create separate root entity
external = SimpleEntity(name="external", value=99)
external.promote_to_root()
assert external.is_root_entity()

# Add to parent
parent.children.append(external)
external.attach(parent)

# External is now part of parent's tree
assert not external.is_root_entity()
assert external.root_ecs_id == parent.ecs_id
```

**Status**: ⏳ To test

---

### P7.3: Version Detection After Mutation
**What we need**: Framework detects which entities changed in tree.

**Test**:
```python
from abstractions.ecs.entity_registry import EntityRegistry

parent = Parent(name="parent", children=[])
child = SimpleEntity(name="child", value=10)
parent.children.append(child)
parent.promote_to_root()

original_parent_id = parent.ecs_id
original_child_id = child.ecs_id

# Mutate child
child.value = 20

# Version the tree
EntityRegistry.version_entity(parent)

# Parent gets new version
new_parent = EntityRegistry.get_entity(parent.ecs_id)
assert new_parent.ecs_id != original_parent_id

# Child should also be versioned
# (This depends on framework behavior - need to verify)
```

**Status**: ⏳ To test

---

## Testing Strategy

### Phase 1: Basic Primitives (P1.x, P4.1-P4.3)
Test fundamental entity operations and simple functions.

### Phase 2: Hierarchies (P2.x, P3.x)
Test nested structures and mutation tracking.

### Phase 3: Collections (P5.x)
Test list manipulation within trees.

### Phase 4: Advanced (P6.x, P7.x)
Test distributed addressing and tree operations.

---

## Test File Structure

Create `test_primitives.py` with:
```python
def test_p1_1_create_simple_entity():
    # Test code here
    pass

def test_p1_2_promote_to_root():
    # Test code here
    pass

# ... etc for all primitives

if __name__ == "__main__":
    # Run all tests
    pass
```

---

## Success Criteria

✅ All primitives pass their tests
✅ No unexpected framework errors
✅ Behavior matches documented patterns
✅ Edge cases handled gracefully

Once all primitives are verified, we can confidently implement GridMap.
