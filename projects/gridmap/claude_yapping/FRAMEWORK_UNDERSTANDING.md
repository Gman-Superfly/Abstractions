# Complete Framework Understanding

## Core Philosophy

**Entities are data. Functions are transformations. Framework handles everything else.**

The Abstractions framework unifies:
- **Functional programming** (pure functions, immutability)
- **Entity-component systems** (composition, identity)
- **Distributed systems** (location transparency, addressing)
- **Event sourcing** (provenance, audit trails)

## The Critical Pattern: Direct Mutation

### ❌ WRONG (what I initially thought)
```python
def move_entity(entity: GameEntity, position: Tuple[int, int]) -> GameEntity:
    return entity.model_copy(update={"position": position})
```

### ✅ CORRECT (from examples + README)
```python
@CallableRegistry.register("move_entity")
def move_entity(entity: GameEntity, position: Tuple[int, int]) -> GameEntity:
    entity.position = position  # MUTATE DIRECTLY
    return entity
```

**Why**: The framework intercepts the mutation and handles versioning/immutability automatically. You write simple imperative code, get functional guarantees.

## Entity Lifecycle

### 1. Define Entity Classes
```python
from abstractions.ecs.entity import Entity

class GameEntity(Entity):
    position: Tuple[int, int]
    walkable: bool
    transparent: bool
```

### 2. Create & Promote
```python
entity = GameEntity(position=(0, 0), walkable=True, transparent=True)
entity.promote_to_root()  # Enter distributed entity space
```

**After promotion**:
- `entity.ecs_id` = persistent UUID
- `entity.lineage_id` = lineage UUID (same across versions)
- `entity.root_ecs_id` = root entity UUID
- Entity is now addressable via `@{ecs_id}.field`

### 3. Transform via Registry
```python
result = CallableRegistry.execute("move_entity", entity=entity, position=(5, 5))

# Handle Union[Entity, List[Entity]] return
updated = result if not isinstance(result, list) else result[0]
```

**After transformation**:
- `entity.ecs_id` unchanged (original version)
- `updated.ecs_id` is new UUID (new version)
- `entity.lineage_id == updated.lineage_id` (same lineage)
- Original entity unchanged (immutability)

## Function Registration

### Basic Pattern
```python
@CallableRegistry.register("function_name")
def function_name(entity: Entity, param: type) -> Entity:
    entity.field = new_value  # Direct mutation
    return entity
```

### Framework Automatically:
1. **Extracts signature** → creates input/output entity models
2. **Validates types** → ensures type safety
3. **Resolves addresses** → `@uuid.field` strings become values
4. **Tracks provenance** → records function execution
5. **Versions entities** → creates new `ecs_id` for modified entities
6. **Emits events** → observable transformation pipeline

## Distributed Addressing

### The `@uuid.field` Pattern
```python
from abstractions.ecs.functional_api import get

# Direct access
name = entity.name

# Distributed access (same data)
name = get(f"@{entity.ecs_id}.name")
```

### Functions Accept Addresses
```python
@CallableRegistry.register("create_report")
def create_report(name: str, score: float) -> Report:
    return Report(name=name, score=score)

# Call with mixed addresses and values
result = CallableRegistry.execute("create_report",
    name=f"@{entity.ecs_id}.name",  # Address (resolved by framework)
    score=95.0                       # Direct value
)
```

**Location transparency**: Functions don't know if data is local, cached, or remote.

## Multi-Entity Returns

### Tuple Returns
```python
@CallableRegistry.register("analyze")
def analyze(entity: Entity) -> Tuple[Analysis, Recommendation]:
    analysis = Analysis(...)
    recommendation = Recommendation(...)
    return analysis, recommendation
```

### Unpacking
```python
result = CallableRegistry.execute("analyze", entity=entity)

# Tuple returns come back as list
if isinstance(result, list):
    analysis, recommendation = result[0], result[1]
```

**Framework tracks sibling relationships** between co-created entities.

## Entity Trees

### Hierarchical Structure
```python
GridMap (root)
  └─ nodes: List[GridNode]
       └─ GridNode
            └─ entities: List[GameEntity]
                 └─ GameEntity
```

### Tree Operations
- `build_entity_tree(root)` → constructs EntityTree graph
- `find_modified_entities(old_tree, new_tree)` → detects changes
- `version_entity(entity)` → creates new version, updates tree

**Entire tree versions atomically** - one change propagates through hierarchy.

## Event System

### Define Events
```python
from abstractions.events.events import CreatedEvent, on, emit

class EntityMovedEvent(CreatedEvent[GameEntity]):
    type: str = "entity.moved"
```

### Register Handlers
```python
@on(EntityMovedEvent)
async def log_movement(event: EntityMovedEvent):
    print(f"Entity {event.subject_id} moved")

# Pattern-based
@on(pattern="entity.*")
def handle_entity_events(event: Event):
    print(f"Entity event: {event.type}")

# Predicate-based
@on(predicate=lambda e: hasattr(e, 'subject_id'))
async def track_all(event: Event):
    print(f"Event: {event.type}")
```

### Emit Events
```python
await emit(EntityMovedEvent(
    subject_type=GameEntity,
    subject_id=entity.ecs_id,
    created_id=entity.ecs_id
))
```

**Events contain references, not data** - lightweight observation layer.

## GridMap Application Patterns

### 1. Entity Mutation
```python
@CallableRegistry.register("move_entity")
def move_entity(entity: GameEntity, position: Tuple[int, int]) -> GameEntity:
    entity.position = position  # Direct mutation
    return entity
```

### 2. Grid Queries
```python
@CallableRegistry.register("get_node_at")
def get_node_at(grid_map: GridMap, position: Tuple[int, int]) -> Optional[GridNode]:
    for node in grid_map.nodes:
        if node.position == position:
            return node
    return None
```

### 3. Complex Transformations
```python
@CallableRegistry.register("move_agent_on_grid")
def move_agent_on_grid(grid_map: GridMap, agent_id: str, target: Tuple[int, int]) -> GridMap:
    # Find and mutate agent
    for node in grid_map.nodes:
        for entity in node.entities:
            if str(entity.ecs_id) == agent_id:
                entity.position = target
                break
    
    # Return mutated grid (framework versions it)
    return grid_map
```

### 4. Result Entities
```python
class Path(Entity):
    start: Tuple[int, int]
    goal: Tuple[int, int]
    waypoints: List[Tuple[int, int]]
    cost: float

@CallableRegistry.register("find_path")
def find_path(grid_map: GridMap, start: Tuple[int, int], goal: Tuple[int, int]) -> Path:
    waypoints = run_astar(grid_map, start, goal)
    return Path(start=start, goal=goal, waypoints=waypoints, cost=len(waypoints))
```

## Key Takeaways for GridMap

1. **Mutate entities directly** - framework handles versioning
2. **Flat lists, linear scans** - simple and correct
3. **Position as property** - not an index
4. **Functions return entities** - even query results
5. **No computed fields** - run functions when needed
6. **Complete state trees** - GridMap contains everything
7. **Atomic versioning** - entire tree versions together

## The Magic

You write:
```python
entity.position = (5, 5)
return entity
```

Framework provides:
- Immutability (original unchanged)
- Versioning (new `ecs_id`)
- Lineage (same `lineage_id`)
- Provenance (tracks transformation)
- Distribution (addressable anywhere)
- Events (observable changes)

**Simple code. Powerful guarantees.**
