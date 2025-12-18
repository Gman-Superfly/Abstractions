"""
Test P4: Function Registration & Execution

Tests function registration with various parameter types and return patterns.
"""

from typing import List, Tuple
from abstractions.ecs.entity import Entity
from abstractions.ecs.callable_registry import CallableRegistry
from pydantic import Field


# Test entities
class SimpleEntity(Entity):
    name: str
    value: int = Field(default=0)


def test_p4_1_register_simple_function():
    """P4.1: Register and execute function with entity parameter."""
    print("\n=== P4.1: Register Simple Function ===")
    
    @CallableRegistry.register("simple_function")
    def simple_function(entity: SimpleEntity) -> SimpleEntity:
        entity.value = entity.value * 2
        return entity
    
    entity = SimpleEntity(name="test", value=5)
    entity.promote_to_root()
    
    print(f"Before: value={entity.value}")
    
    result = CallableRegistry.execute("simple_function", entity=entity)
    updated = result if not isinstance(result, list) else result[0]
    
    print(f"After: value={updated.value}")
    print(f"Original unchanged: {entity.value}")
    
    assert updated.value == 10
    assert entity.value == 5  # Original unchanged
    
    print("✅ P4.1 PASSED")
    return True


def test_p4_2_function_with_multiple_parameters():
    """P4.2: Function with entity + primitive parameters."""
    print("\n=== P4.2: Function with Multiple Parameters ===")
    
    @CallableRegistry.register("multi_param")
    def multi_param(entity: SimpleEntity, multiplier: int, name: str) -> SimpleEntity:
        entity.value = entity.value * multiplier
        entity.name = name
        return entity
    
    entity = SimpleEntity(name="old", value=3)
    entity.promote_to_root()
    
    print(f"Before: name={entity.name}, value={entity.value}")
    
    result = CallableRegistry.execute("multi_param", 
                                      entity=entity, 
                                      multiplier=4, 
                                      name="new")
    updated = result if not isinstance(result, list) else result[0]
    
    print(f"After: name={updated.name}, value={updated.value}")
    
    assert updated.value == 12
    assert updated.name == "new"
    
    print("✅ P4.2 PASSED")
    return True


def test_p4_3_function_returning_new_entity():
    """P4.3: Function creates and returns new entity."""
    print("\n=== P4.3: Function Returning New Entity ===")
    
    @CallableRegistry.register("create_entity")
    def create_entity(name: str, value: int) -> SimpleEntity:
        return SimpleEntity(name=name, value=value)
    
    print(f"Creating entity with name='created', value=100")
    
    result = CallableRegistry.execute("create_entity", name="created", value=100)
    entity = result if not isinstance(result, list) else result[0]
    
    print(f"Created: name={entity.name}, value={entity.value}")
    print(f"Has ecs_id: {entity.ecs_id is not None}")
    
    assert entity.name == "created"
    assert entity.value == 100
    assert entity.ecs_id is not None
    
    print("✅ P4.3 PASSED")
    return True


def test_p4_4_function_returning_tuple():
    """P4.4: Function returns multiple entities as tuple."""
    print("\n=== P4.4: Function Returning Tuple ===")
    
    @CallableRegistry.register("create_pair")
    def create_pair(value1: int, value2: int) -> Tuple[SimpleEntity, SimpleEntity]:
        e1 = SimpleEntity(name="first", value=value1)
        e2 = SimpleEntity(name="second", value=value2)
        return e1, e2
    
    print(f"Creating pair with values 10, 20")
    
    result = CallableRegistry.execute("create_pair", value1=10, value2=20)
    
    print(f"Result type: {type(result)}")
    print(f"Result is list: {isinstance(result, list)}")
    
    # Tuple returns as list
    assert isinstance(result, list)
    assert len(result) == 2
    
    print(f"First: name={result[0].name}, value={result[0].value}")
    print(f"Second: name={result[1].name}, value={result[1].value}")
    
    assert result[0].name == "first"
    assert result[0].value == 10
    assert result[1].name == "second"
    assert result[1].value == 20
    
    print("✅ P4.4 PASSED")
    return True


def run_all_tests():
    """Run all P4 tests."""
    print("=" * 60)
    print("TESTING P4: FUNCTION REGISTRATION & EXECUTION")
    print("=" * 60)
    
    tests = [
        test_p4_1_register_simple_function,
        test_p4_2_function_with_multiple_parameters,
        test_p4_3_function_returning_new_entity,
        test_p4_4_function_returning_tuple,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"P4 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
