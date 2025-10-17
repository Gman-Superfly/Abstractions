# What I Learned From Examples

## Critical Pattern: Entity Mutation

**WRONG (what I suggested):**
```python
def move_entity(entity: GameEntity, position: Tuple[int, int]) -> GameEntity:
    return entity.model_copy(update={"position": position})
```

**CORRECT (from examples):**
```python
def update_gpa(student: Student, new_gpa: float) -> Student:
    student.gpa = new_gpa  # MUTATE IN PLACE
    return student
```

**Key Insight**: Functions **mutate the entity directly** and return it. The framework handles versioning/immutability automatically. Don't use `model_copy()` - just modify fields directly!

## Entity Basics (from 01_basic_entity_transformation.py)

### 1. Define Entities
```python
from abstractions.ecs.entity import Entity

class Student(Entity):
    name: str = ""
    gpa: float = 0.0
```

### 2. Register Functions
```python
from abstractions.ecs.callable_registry import CallableRegistry

@CallableRegistry.register("update_gpa")
def update_gpa(student: Student, new_gpa: float) -> Student:
    student.gpa = new_gpa  # Direct mutation
    return student
```

### 3. Promote to Root & Execute
```python
student = Student(name="Alice", gpa=3.5)
student.promote_to_root()  # Enter distributed entity space

# Execute via registry
result = CallableRegistry.execute("update_gpa", student=student, new_gpa=3.8)

# Result is Union[Entity, List[Entity]]
updated = result if not isinstance(result, list) else result[0]
```

### 4. Automatic Features
- **Versioning**: `student.ecs_id != updated.ecs_id` (different versions)
- **Lineage**: `student.lineage_id == updated.lineage_id` (same entity line)
- **Immutability**: Original `student.gpa` unchanged at 3.5
- **Provenance**: Framework tracks all transformations

## Distributed Addressing (from 02_distributed_addressing.py)

### String-Based Access
```python
from abstractions.ecs.functional_api import get

# Access entity fields via @uuid.field syntax
student_name = get(f"@{student.ecs_id}.name")
student_gpa = get(f"@{student.ecs_id}.gpa")
```

### Functions Accept Addresses
```python
@CallableRegistry.register("create_transcript")
def create_transcript(name: str, gpa: float, courses: List[str]) -> Course:
    return Course(name=f"Transcript for {name}", credits=len(courses))

# Call with mixed addresses and direct values
result = CallableRegistry.execute("create_transcript",
    name=f"@{student.ecs_id}.name",  # Address resolved automatically
    gpa=f"@{student.ecs_id}.gpa",    # Address resolved automatically
    courses=["Math", "Physics"]       # Direct value
)
```

**Key**: Framework automatically resolves `@uuid.field` strings to actual values.

## Multi-Entity Returns (from 03_multi_entity_transformations.py)

### Tuple Returns
```python
@CallableRegistry.register("analyze_performance")
def analyze_performance(student: Student) -> Tuple[Assessment, Recommendation]:
    assessment = Assessment(student_id=str(student.ecs_id), performance_level="high")
    recommendation = Recommendation(student_id=str(student.ecs_id), action="advanced")
    return assessment, recommendation
```

### Unpacking Results
```python
result = CallableRegistry.execute("analyze_performance", student=alice)

# Tuple returns come back as a list
if isinstance(result, list):
    assessment, recommendation = result[0], result[1]
```

**Key**: Tuple returns are unpacked into a list. Framework tracks sibling relationships.

## Complex Workflows (from 04_distributed_grade_processing.py)

### Batch Processing
```python
@CallableRegistry.register("analyze_student_performance")
def analyze_student_performance(student: Student, grades: List[Grade]) -> GradeAnalysis:
    # Filter grades for this student
    student_grades = [g for g in grades if g.student_id == student.student_id]
    
    # Calculate metrics
    weighted_gpa = sum(g.grade_points for g in student_grades) / len(student_grades)
    
    # Return analysis entity
    return GradeAnalysis(
        student_id=student.student_id,
        weighted_gpa=weighted_gpa,
        risk_level="low" if weighted_gpa >= 3.5 else "high"
    )
```

### Cascading Transformations
```python
# Step 1: Analyze each student
analyses = [CallableRegistry.execute("analyze_student_performance", 
                                     student=s, grades=all_grades) 
            for s in students]

# Step 2: Generate report from analyses
report = CallableRegistry.execute("generate_semester_report",
                                  analyses=analyses,
                                  statistics=stats,
                                  semester="Fall 2024")
```

## Event System (from 06_event_system_working.py)

### Define Events
```python
from abstractions.events.events import CreatedEvent, on, emit

class StudentCreatedEvent(CreatedEvent[Student]):
    type: str = "student.created"
```

### Register Handlers
```python
@on(StudentCreatedEvent)
async def log_student_creation(event: StudentCreatedEvent):
    print(f"Student created: {event.subject_id}")

# Pattern-based
@on(pattern="student.*")
def handle_all_student_events(event: Event):
    print(f"Student event: {event.type}")

# Predicate-based
@on(predicate=lambda e: hasattr(e, 'subject_id'))
async def track_all_entities(event: Event):
    print(f"Entity event: {event.type}")
```

### Emit Events
```python
await emit(StudentCreatedEvent(
    subject_type=Student,
    subject_id=student.id,
    created_id=student.id
))
```

## Key Patterns for GridMap

### 1. Entity Mutation Pattern
```python
@CallableRegistry.register("move_entity")
def move_entity(entity: GameEntity, position: Tuple[int, int]) -> GameEntity:
    entity.position = position  # DIRECT MUTATION
    return entity
```

### 2. Grid Queries Return Entities
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
    # Find agent in grid
    for node in grid_map.nodes:
        for entity in node.entities:
            if str(entity.ecs_id) == agent_id and isinstance(entity, Agent):
                # Mutate agent position
                entity.position = target
                break
    
    # Return mutated grid (framework handles versioning)
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
    # Run A* algorithm
    waypoints = run_astar(grid_map, start, goal)
    
    # Return result as entity
    return Path(start=start, goal=goal, waypoints=waypoints, cost=len(waypoints))
```

## Critical Takeaways

1. **Mutate entities directly** - framework handles immutability
2. **Always promote_to_root()** before using entities in registry
3. **Execute returns Union[Entity, List[Entity]]** - handle both cases
4. **Tuple returns become lists** - unpack accordingly
5. **Use @uuid.field for distributed addressing** - framework resolves automatically
6. **Events are async** - use `await emit()` and `async def` handlers
7. **Functions are pure transformations** - no side effects, just return new/modified entities

## For GridMap Implementation

```python
# entities.py
class GameEntity(Entity):
    position: Tuple[int, int]
    walkable: bool
    transparent: bool

class Agent(GameEntity):
    walkable: bool = True
    transparent: bool = True
    speed: int
    sight: int

# functions.py
@CallableRegistry.register("move_entity")
def move_entity(entity: GameEntity, position: Tuple[int, int]) -> GameEntity:
    entity.position = position  # Direct mutation!
    return entity

@CallableRegistry.register("is_position_walkable")
def is_position_walkable(grid_map: GridMap, position: Tuple[int, int]) -> bool:
    node = get_node_at(grid_map, position)
    if not node:
        return False
    return all(e.walkable for e in node.entities)
```
