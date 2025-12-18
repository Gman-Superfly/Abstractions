"""
Test P11: All Execution Paths - Comprehensive Derivation Tracking Test

Tests ALL 5 execution paths to verify derivation tracking works correctly.
This test will FAIL before the fix and PASS after the fix.
"""

from abstractions.ecs.entity import Entity
from abstractions.ecs.callable_registry import CallableRegistry
from pydantic import Field
from typing import Tuple


class SimpleEntity(Entity):
    """Simple entity for testing."""
    name: str
    value: int = 0


def test_p11_path2_no_inputs():
    """PATH 2: Function with no inputs."""
    print("\n=== PATH 2: No Inputs ===")
    
    @CallableRegistry.register("create_from_nothing")
    def create_from_nothing() -> SimpleEntity:
        """No parameters at all."""
        return SimpleEntity(name="created", value=42)
    
    result = CallableRegistry.execute("create_from_nothing")
    entity = result if not isinstance(result, list) else result[0]
    
    print(f"Entity: {entity.name}")
    print(f"  derived_from_function: {entity.derived_from_function}")
    print(f"  derived_from_execution_id: {entity.derived_from_execution_id}")
    
    # Expected after fix
    expected_function = "create_from_nothing"
    expected_has_execution_id = entity.derived_from_execution_id is not None
    
    if entity.derived_from_function == expected_function and expected_has_execution_id:
        print("✅ PATH 2: PASS")
        return True
    else:
        print(f"❌ PATH 2: FAIL")
        print(f"   derived_from_function: expected '{expected_function}', got '{entity.derived_from_function}'")
        print(f"   derived_from_execution_id: expected UUID, got {entity.derived_from_execution_id}")
        return False


def test_p11_path3_transactional_single():
    """PATH 3: Transactional with Entity input, single return."""
    print("\n=== PATH 3: Transactional Single Entity ===")
    
    @CallableRegistry.register("mutate_entity")
    def mutate_entity(entity: SimpleEntity, new_value: int) -> SimpleEntity:
        """Has Entity input, returns single Entity."""
        entity.value = new_value
        return entity
    
    # Create input
    input_entity = SimpleEntity(name="test", value=10)
    input_entity.promote_to_root()
    
    result = CallableRegistry.execute("mutate_entity", entity=input_entity, new_value=20)
    output = result if not isinstance(result, list) else result[0]
    
    print(f"Output: {output.name}, value={output.value}")
    print(f"  derived_from_function: {output.derived_from_function}")
    print(f"  derived_from_execution_id: {output.derived_from_execution_id}")
    
    # Expected after fix
    # Note: This actually goes through PATH 1 (single_entity_with_config) because it has
    # both Entity and primitive parameters, but should still track as "mutate_entity"
    expected_function = "mutate_entity"
    expected_has_execution_id = output.derived_from_execution_id is not None
    
    if output.derived_from_function == expected_function and expected_has_execution_id:
        print(f"✅ PATH 3: PASS")
        return True
    else:
        print(f"❌ PATH 3: FAIL")
        print(f"   derived_from_function: expected '{expected_function}', got '{output.derived_from_function}'")
        print(f"   derived_from_execution_id: expected UUID, got {output.derived_from_execution_id}")
        return False


def test_p11_path3_transactional_multi():
    """PATH 3: Transactional with Entity input, multi return (should already work)."""
    print("\n=== PATH 3: Transactional Multi Entity ===")
    
    @CallableRegistry.register("split_entity")
    def split_entity(entity: SimpleEntity) -> Tuple[SimpleEntity, SimpleEntity]:
        """Has Entity input, returns tuple."""
        return (
            SimpleEntity(name=f"{entity.name}_a", value=entity.value),
            SimpleEntity(name=f"{entity.name}_b", value=entity.value * 2)
        )
    
    # Create input
    input_entity = SimpleEntity(name="test", value=10)
    input_entity.promote_to_root()
    
    result = CallableRegistry.execute("split_entity", entity=input_entity)
    entities = result if isinstance(result, list) else [result]
    
    print(f"Created {len(entities)} entities")
    
    all_pass = True
    for i, entity in enumerate(entities):
        print(f"\nEntity {i}: {entity.name}")
        print(f"  derived_from_function: {entity.derived_from_function}")
        print(f"  derived_from_execution_id: {entity.derived_from_execution_id}")
        print(f"  output_index: {entity.output_index}")
        print(f"  sibling_output_entities: {entity.sibling_output_entities}")
        
        if entity.derived_from_function != "split_entity":
            all_pass = False
        if entity.derived_from_execution_id is None:
            all_pass = False
        if entity.output_index != i:
            all_pass = False
        if len(entity.sibling_output_entities) != 1:  # Should have 1 sibling
            all_pass = False
    
    if all_pass:
        print("\n✅ PATH 3 Multi: PASS (should already work)")
        return True
    else:
        print("\n❌ PATH 3 Multi: FAIL (regression!)")
        return False


def test_p11_path45_borrowing_single():
    """PATH 4/5: Borrowing with primitive inputs, single return."""
    print("\n=== PATH 4/5: Borrowing Single Entity ===")
    
    @CallableRegistry.register("create_from_primitives")
    def create_from_primitives(name: str, value: int) -> SimpleEntity:
        """No Entity inputs, just primitives."""
        return SimpleEntity(name=name, value=value)
    
    result = CallableRegistry.execute("create_from_primitives", name="test", value=42)
    entity = result if not isinstance(result, list) else result[0]
    
    print(f"Entity: {entity.name}")
    print(f"  derived_from_function: {entity.derived_from_function}")
    print(f"  derived_from_execution_id: {entity.derived_from_execution_id}")
    
    # Expected after fix
    expected_function = "create_from_primitives"
    expected_has_execution_id = entity.derived_from_execution_id is not None
    
    if entity.derived_from_function == expected_function and expected_has_execution_id:
        print("✅ PATH 4/5 Single: PASS")
        return True
    else:
        print(f"❌ PATH 4/5 Single: FAIL")
        print(f"   derived_from_function: expected '{expected_function}', got '{entity.derived_from_function}'")
        print(f"   derived_from_execution_id: expected UUID, got {entity.derived_from_execution_id}")
        return False


def test_p11_path45_borrowing_multi():
    """PATH 4/5: Borrowing with primitive inputs, multi return (should already work)."""
    print("\n=== PATH 4/5: Borrowing Multi Entity ===")
    
    @CallableRegistry.register("create_pair_from_primitives")
    def create_pair_from_primitives(name1: str, name2: str) -> Tuple[SimpleEntity, SimpleEntity]:
        """No Entity inputs, returns tuple."""
        return (
            SimpleEntity(name=name1, value=1),
            SimpleEntity(name=name2, value=2)
        )
    
    result = CallableRegistry.execute("create_pair_from_primitives", name1="first", name2="second")
    entities = result if isinstance(result, list) else [result]
    
    print(f"Created {len(entities)} entities")
    
    all_pass = True
    for i, entity in enumerate(entities):
        print(f"\nEntity {i}: {entity.name}")
        print(f"  derived_from_function: {entity.derived_from_function}")
        print(f"  derived_from_execution_id: {entity.derived_from_execution_id}")
        print(f"  output_index: {entity.output_index}")
        print(f"  sibling_output_entities: {entity.sibling_output_entities}")
        
        if entity.derived_from_function != "create_pair_from_primitives":
            all_pass = False
        if entity.derived_from_execution_id is None:
            all_pass = False
        if entity.output_index != i:
            all_pass = False
        if len(entity.sibling_output_entities) != 1:  # Should have 1 sibling
            all_pass = False
    
    if all_pass:
        print("\n✅ PATH 4/5 Multi: PASS (should already work)")
        return True
    else:
        print("\n❌ PATH 4/5 Multi: FAIL (regression!)")
        return False


def run_all_tests():
    """Run all P11 tests."""
    print("=" * 70)
    print("TESTING P11: ALL EXECUTION PATHS - DERIVATION TRACKING")
    print("=" * 70)
    print("\nThis test verifies that ALL execution paths set derivation tracking.")
    print("BEFORE FIX: Single-entity paths will FAIL")
    print("AFTER FIX: All paths should PASS")
    print("=" * 70)
    
    results = {
        "PATH 2: No Inputs (Single)": test_p11_path2_no_inputs(),
        "PATH 3: Transactional (Single)": test_p11_path3_transactional_single(),
        "PATH 3: Transactional (Multi)": test_p11_path3_transactional_multi(),
        "PATH 4/5: Borrowing (Single)": test_p11_path45_borrowing_single(),
        "PATH 4/5: Borrowing (Multi)": test_p11_path45_borrowing_multi(),
    }
    
    print("\n" + "=" * 70)
    print("P11 RESULTS:")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} passed")
    
    # Check for regressions (multi-entity should always work)
    regressions = []
    if not results["PATH 3: Transactional (Multi)"]:
        regressions.append("PATH 3 Multi")
    if not results["PATH 4/5: Borrowing (Multi)"]:
        regressions.append("PATH 4/5 Multi")
    
    if regressions:
        print(f"\n⚠️  REGRESSIONS DETECTED: {', '.join(regressions)}")
        print("   Multi-entity paths should ALWAYS work!")
    
    print("=" * 70)
    
    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
