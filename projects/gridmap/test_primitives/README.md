# Primitives Testing

## Overview

This directory contains incremental tests for framework primitives needed by GridMap.

## Test Files

### Phase 1: Basics
- **test_p1_entity_creation.py** - Entity creation, promotion, collections
- **test_p2_entity_hierarchies.py** - Nested structures, tree building
- **test_p3_mutation_versioning.py** - Direct mutation, versioning

### Phase 2: Functions (To Create)
- **test_p4_function_registration.py** - Function registration and execution
- **test_p5_collection_manipulation.py** - List operations in trees

### Phase 3: Advanced (To Create)
- **test_p6_distributed_addressing.py** - @uuid.field addressing
- **test_p7_tree_operations.py** - attach(), detach()

## Running Tests

### Run individual test file:
```bash
python test_p1_entity_creation.py
```

### Run all tests (once created):
```bash
python run_all_tests.py
```

## Test Strategy

1. **Run P1 first** - Basic entity operations
2. **Run P2 next** - Hierarchies (depends on P1)
3. **Run P3 next** - Mutation (depends on P1, P2)
4. **Then P4-P7** - Advanced features

## Success Criteria

Each test file should:
- ✅ Pass all assertions
- ✅ Print clear output
- ✅ Handle errors gracefully
- ✅ Exit with code 0 on success, 1 on failure

## Notes

- Tests are isolated - each can run independently
- Tests use simple print statements for visibility
- Tests verify both behavior and framework assumptions
