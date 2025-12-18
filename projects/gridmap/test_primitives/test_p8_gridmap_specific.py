"""
Test P8: GridMap-Specific Patterns

Tests the specific pattern of moving entities between nodes in a grid structure.
This is the core GridMap operation: remove from one node, add to another.
"""

from typing import List, Tuple
from abstractions.ecs.entity import Entity
from abstractions.ecs.callable_registry import CallableRegistry
from pydantic import Field


# GridMap-like entities
class GameEntity(Entity):
    """Entity that can exist in a grid node."""
    position: Tuple[int, int]
    name: str


class GridNode(Entity):
    """A node in the grid containing entities."""
    position: Tuple[int, int]
    entities: List[GameEntity] = Field(default_factory=list)


class GridMap(Entity):
    """The complete grid containing all nodes."""
    nodes: List[GridNode] = Field(default_factory=list)
    width: int
    height: int


def test_p8_1_move_entity_between_nodes():
    """P8.1: Move entity from one node to another (core GridMap operation)."""
    print("\n=== P8.1: Move Entity Between Nodes ===")
    
    @CallableRegistry.register("move_entity_between_nodes")
    def move_entity_between_nodes(grid_map: GridMap, entity_name: str, target_pos: Tuple[int, int]) -> GridMap:
        """Move entity from current node to target node."""
        # Find entity and current node
        entity = None
        current_node = None
        
        for node in grid_map.nodes:
            for e in node.entities:
                if e.name == entity_name:
                    entity = e
                    current_node = node
                    break
            if entity:
                break
        
        if entity and current_node:
            # Find target node
            target_node = next((n for n in grid_map.nodes if n.position == target_pos), None)
            
            if target_node and target_node != current_node:
                print(f"  Moving {entity.name} from {current_node.position} to {target_node.position}")
                
                # 1. Remove from current node
                current_node.entities.remove(entity)
                print(f"    Removed from node {current_node.position}, now has {len(current_node.entities)} entities")
                
                # 2. Update entity position
                entity.position = target_pos
                print(f"    Updated entity position to {entity.position}")
                
                # 3. Add to target node
                target_node.entities.append(entity)
                print(f"    Added to node {target_node.position}, now has {len(target_node.entities)} entities")
        
        return grid_map
    
    # Create a 2x2 grid
    grid_map = GridMap(nodes=[], width=2, height=2)
    
    # Create nodes
    node_00 = GridNode(position=(0, 0), entities=[])
    node_01 = GridNode(position=(0, 1), entities=[])
    node_10 = GridNode(position=(1, 0), entities=[])
    node_11 = GridNode(position=(1, 1), entities=[])
    
    grid_map.nodes.extend([node_00, node_01, node_10, node_11])
    
    # Create entity at (0, 0)
    agent = GameEntity(position=(0, 0), name="agent1")
    node_00.entities.append(agent)
    
    # Promote to root
    grid_map.promote_to_root()
    
    print(f"Initial state:")
    print(f"  Agent at: {agent.position}")
    print(f"  Node (0,0) has {len(node_00.entities)} entities")
    print(f"  Node (1,1) has {len(node_11.entities)} entities")
    print(f"  GridMap ecs_id: {grid_map.ecs_id}")
    
    # Move entity from (0,0) to (1,1)
    result = CallableRegistry.execute("move_entity_between_nodes", 
                                      grid_map=grid_map, 
                                      entity_name="agent1", 
                                      target_pos=(1, 1))
    
    updated_grid = result if not isinstance(result, list) else result[0]
    
    print(f"\nAfter move:")
    print(f"  Original grid ecs_id: {grid_map.ecs_id}")
    print(f"  Updated grid ecs_id: {updated_grid.ecs_id}")
    print(f"  Different ecs_id: {updated_grid.ecs_id != grid_map.ecs_id}")
    
    # Find nodes in updated grid
    updated_node_00 = next((n for n in updated_grid.nodes if n.position == (0, 0)), None)
    updated_node_11 = next((n for n in updated_grid.nodes if n.position == (1, 1)), None)
    
    print(f"  Updated node (0,0) has {len(updated_node_00.entities)} entities")
    print(f"  Updated node (1,1) has {len(updated_node_11.entities)} entities")
    
    if len(updated_node_11.entities) > 0:
        moved_entity = updated_node_11.entities[0]
        print(f"  Moved entity position: {moved_entity.position}")
        print(f"  Moved entity name: {moved_entity.name}")
    
    # Verify the move
    assert updated_grid.ecs_id != grid_map.ecs_id  # New version created
    assert len(updated_node_00.entities) == 0  # Removed from source
    assert len(updated_node_11.entities) == 1  # Added to target
    assert updated_node_11.entities[0].name == "agent1"
    assert updated_node_11.entities[0].position == (1, 1)
    
    # Original unchanged
    assert len(node_00.entities) == 1  # Original still has entity
    assert len(node_11.entities) == 0  # Original target still empty
    
    print("✅ P8.1 PASSED")
    print("Note: Entity moved between nodes within same tree, no detach/attach needed")
    return True


def test_p8_2_multiple_entities_in_node():
    """P8.2: Multiple entities can exist in same node."""
    print("\n=== P8.2: Multiple Entities in Node ===")
    
    @CallableRegistry.register("add_entity_to_node")
    def add_entity_to_node(grid_map: GridMap, entity_name: str, position: Tuple[int, int]) -> GridMap:
        """Add a new entity to a node."""
        node = next((n for n in grid_map.nodes if n.position == position), None)
        if node:
            new_entity = GameEntity(position=position, name=entity_name)
            node.entities.append(new_entity)
        return grid_map
    
    # Create grid with one node
    grid_map = GridMap(nodes=[], width=1, height=1)
    node = GridNode(position=(0, 0), entities=[])
    grid_map.nodes.append(node)
    grid_map.promote_to_root()
    
    print(f"Initial: node has {len(node.entities)} entities")
    
    # Add first entity
    result1 = CallableRegistry.execute("add_entity_to_node", 
                                       grid_map=grid_map, 
                                       entity_name="entity1", 
                                       position=(0, 0))
    grid_v1 = result1 if not isinstance(result1, list) else result1[0]
    
    # Add second entity
    result2 = CallableRegistry.execute("add_entity_to_node", 
                                       grid_map=grid_v1, 
                                       entity_name="entity2", 
                                       position=(0, 0))
    grid_v2 = result2 if not isinstance(result2, list) else result2[0]
    
    # Check final state
    final_node = next((n for n in grid_v2.nodes if n.position == (0, 0)), None)
    
    print(f"After adding 2 entities: node has {len(final_node.entities)} entities")
    for e in final_node.entities:
        print(f"  - {e.name}")
    
    assert len(final_node.entities) == 2
    assert final_node.entities[0].name == "entity1"
    assert final_node.entities[1].name == "entity2"
    
    print("✅ P8.2 PASSED")
    return True


def test_p8_3_entity_position_consistency():
    """P8.3: Entity position should match its node position after move."""
    print("\n=== P8.3: Entity Position Consistency ===")
    
    @CallableRegistry.register("move_with_position_update")
    def move_with_position_update(grid_map: GridMap, entity_name: str, target_pos: Tuple[int, int]) -> GridMap:
        """Move entity and ensure position field matches."""
        entity = None
        current_node = None
        
        for node in grid_map.nodes:
            for e in node.entities:
                if e.name == entity_name:
                    entity = e
                    current_node = node
                    break
            if entity:
                break
        
        if entity and current_node:
            target_node = next((n for n in grid_map.nodes if n.position == target_pos), None)
            
            if target_node:
                # Remove from current
                current_node.entities.remove(entity)
                
                # Update position BEFORE adding to new node
                entity.position = target_pos
                
                # Add to target
                target_node.entities.append(entity)
        
        return grid_map
    
    # Setup
    grid_map = GridMap(nodes=[], width=2, height=1)
    node_0 = GridNode(position=(0, 0), entities=[])
    node_1 = GridNode(position=(1, 0), entities=[])
    grid_map.nodes.extend([node_0, node_1])
    
    entity = GameEntity(position=(0, 0), name="test")
    node_0.entities.append(entity)
    grid_map.promote_to_root()
    
    print(f"Before move: entity.position = {entity.position}")
    
    # Move
    result = CallableRegistry.execute("move_with_position_update", 
                                      grid_map=grid_map, 
                                      entity_name="test", 
                                      target_pos=(1, 0))
    updated_grid = result if not isinstance(result, list) else result[0]
    
    # Check consistency
    target_node = next((n for n in updated_grid.nodes if n.position == (1, 0)), None)
    moved_entity = target_node.entities[0]
    
    print(f"After move:")
    print(f"  Node position: {target_node.position}")
    print(f"  Entity position: {moved_entity.position}")
    print(f"  Match: {target_node.position == moved_entity.position}")
    
    assert target_node.position == moved_entity.position
    
    print("✅ P8.3 PASSED")
    return True


def run_all_tests():
    """Run all P8 tests."""
    print("=" * 60)
    print("TESTING P8: GRIDMAP-SPECIFIC PATTERNS")
    print("=" * 60)
    
    tests = [
        test_p8_1_move_entity_between_nodes,
        test_p8_2_multiple_entities_in_node,
        test_p8_3_entity_position_consistency,
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
    print(f"P8 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
