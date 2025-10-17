"""
Test P9: Dict and Set Fields in Entities

Tests that entities can have Dict and Set fields, which we need for graph representations.
"""

from typing import Dict, Set, List, Tuple
from abstractions.ecs.entity import Entity
from abstractions.ecs.callable_registry import CallableRegistry
from pydantic import Field


class GraphEntity(Entity):
    """Entity with Dict and Set fields (like we need for pathfinding graphs)."""
    name: str
    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = Field(default_factory=dict)
    edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = Field(default_factory=set)


def test_p9_1_dict_field_in_entity():
    """P9.1: Entity can have Dict field."""
    print("\n=== P9.1: Dict Field in Entity ===")
    
    # Create entity with dict field
    graph = GraphEntity(
        name="test_graph",
        adjacency={
            (0, 0): [(0, 1), (1, 0)],
            (0, 1): [(0, 0), (1, 1)],
            (1, 0): [(0, 0), (1, 1)],
            (1, 1): [(0, 1), (1, 0)]
        }
    )
    
    graph.promote_to_root()
    
    print(f"Graph name: {graph.name}")
    print(f"Adjacency dict size: {len(graph.adjacency)}")
    print(f"Node (0,0) neighbors: {graph.adjacency[(0, 0)]}")
    
    assert len(graph.adjacency) == 4
    assert (0, 1) in graph.adjacency[(0, 0)]
    
    print("✅ P9.1 PASSED")
    return True


def test_p9_2_set_field_in_entity():
    """P9.2: Entity can have Set field."""
    print("\n=== P9.2: Set Field in Entity ===")
    
    # Create entity with set field
    graph = GraphEntity(
        name="test_graph",
        edges={
            ((0, 0), (0, 1)),
            ((0, 0), (1, 0)),
            ((0, 1), (1, 1)),
            ((1, 0), (1, 1))
        }
    )
    
    graph.promote_to_root()
    
    print(f"Graph name: {graph.name}")
    print(f"Edges set size: {len(graph.edges)}")
    print(f"Contains edge (0,0)->(0,1): {((0, 0), (0, 1)) in graph.edges}")
    
    assert len(graph.edges) == 4
    assert ((0, 0), (0, 1)) in graph.edges
    
    print("✅ P9.2 PASSED")
    return True


def test_p9_3_function_returns_entity_with_dict():
    """P9.3: Function can return entity with Dict field."""
    print("\n=== P9.3: Function Returns Entity with Dict ===")
    
    @CallableRegistry.register("create_graph")
    def create_graph(name: str, size: int) -> GraphEntity:
        """Create a simple grid graph."""
        adjacency = {}
        
        # Create simple 2x2 grid adjacency
        for y in range(size):
            for x in range(size):
                pos = (x, y)
                neighbors = []
                
                # Add 4-directional neighbors
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        neighbors.append((nx, ny))
                
                adjacency[pos] = neighbors
        
        return GraphEntity(name=name, adjacency=adjacency)
    
    # Execute via registry
    result = CallableRegistry.execute("create_graph", name="grid_2x2", size=2)
    graph = result if not isinstance(result, list) else result[0]
    
    print(f"Created graph: {graph.name}")
    print(f"Adjacency dict size: {len(graph.adjacency)}")
    print(f"Node (0,0) neighbors: {graph.adjacency[(0, 0)]}")
    print(f"Node (1,1) neighbors: {graph.adjacency[(1, 1)]}")
    
    assert len(graph.adjacency) == 4  # 2x2 grid
    assert len(graph.adjacency[(0, 0)]) == 2  # Corner has 2 neighbors
    assert len(graph.adjacency[(1, 1)]) == 2  # Corner has 2 neighbors
    
    print("✅ P9.3 PASSED")
    return True


def test_p9_4_mutate_dict_field():
    """P9.4: Can mutate Dict field in function."""
    print("\n=== P9.4: Mutate Dict Field ===")
    
    @CallableRegistry.register("add_edge")
    def add_edge(graph: GraphEntity, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> GraphEntity:
        """Add an edge to the graph."""
        if from_pos not in graph.adjacency:
            graph.adjacency[from_pos] = []
        
        if to_pos not in graph.adjacency[from_pos]:
            graph.adjacency[from_pos].append(to_pos)
        
        return graph
    
    # Create initial graph
    graph = GraphEntity(
        name="test",
        adjacency={(0, 0): [(0, 1)]}
    )
    graph.promote_to_root()
    
    print(f"Before: (0,0) neighbors = {graph.adjacency[(0, 0)]}")
    
    # Add edge via registry
    result = CallableRegistry.execute("add_edge", 
                                      graph=graph, 
                                      from_pos=(0, 0), 
                                      to_pos=(1, 0))
    updated_graph = result if not isinstance(result, list) else result[0]
    
    print(f"After: (0,0) neighbors = {updated_graph.adjacency[(0, 0)]}")
    print(f"Version changed: {updated_graph.ecs_id != graph.ecs_id}")
    
    # Check mutation worked
    assert len(updated_graph.adjacency[(0, 0)]) == 2
    assert (1, 0) in updated_graph.adjacency[(0, 0)]
    
    # Original unchanged
    assert len(graph.adjacency[(0, 0)]) == 1
    
    print("✅ P9.4 PASSED")
    return True


def run_all_tests():
    """Run all P9 tests."""
    print("=" * 60)
    print("TESTING P9: DICT AND SET FIELDS IN ENTITIES")
    print("=" * 60)
    
    tests = [
        test_p9_1_dict_field_in_entity,
        test_p9_2_set_field_in_entity,
        test_p9_3_function_returns_entity_with_dict,
        test_p9_4_mutate_dict_field,
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
    print(f"P9 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
