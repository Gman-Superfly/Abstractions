"""
Test Basic Movement

Verify that basic entity movement works with the GridMap structure.
"""

from abstractions.ecs.callable_registry import CallableRegistry
from game_entities import Agent
from grid_init import create_floor_grid, add_agent_to_grid, print_grid
from movement import move_entity_one_step


def test_basic_movement():
    """Test basic one-step movement."""
    print("=" * 60)
    print("TEST: Basic Movement")
    print("=" * 60)
    
    # Create a 5x5 grid with floor tiles
    grid = create_floor_grid(5, 5)
    
    # Add an agent at (2, 2)
    grid = add_agent_to_grid(grid, "agent1", (2, 2))
    
    # Promote to root (required for CallableRegistry)
    grid.promote_to_root()
    
    print("\nInitial state:")
    print_grid(grid)
    
    # Move agent right (2,2) -> (3,2)
    print("Moving agent right...")
    result = CallableRegistry.execute("move_entity_one_step", 
                                      grid_map=grid, 
                                      entity_name="agent1", 
                                      target_pos=(3, 2))
    grid_v1 = result if not isinstance(result, list) else result[0]
    
    print("\nAfter move right:")
    print_grid(grid_v1)
    print(f"Version changed: {grid_v1.ecs_id != grid.ecs_id}")
    
    # Move agent down (3,2) -> (3,3)
    print("Moving agent down...")
    result = CallableRegistry.execute("move_entity_one_step", 
                                      grid_map=grid_v1, 
                                      entity_name="agent1", 
                                      target_pos=(3, 3))
    grid_v2 = result if not isinstance(result, list) else result[0]
    
    print("\nAfter move down:")
    print_grid(grid_v2)
    print(f"Version changed: {grid_v2.ecs_id != grid_v1.ecs_id}")
    
    # Try invalid move (too far)
    print("Attempting invalid move (too far)...")
    result = CallableRegistry.execute("move_entity_one_step", 
                                      grid_map=grid_v2, 
                                      entity_name="agent1", 
                                      target_pos=(0, 0))
    grid_v3 = result if not isinstance(result, list) else result[0]
    
    print("\nAfter invalid move attempt:")
    print_grid(grid_v3)
    print(f"Version changed: {grid_v3.ecs_id != grid_v2.ecs_id}")
    
    # Verify original unchanged
    print("\nOriginal grid (should be unchanged):")
    print_grid(grid)
    
    print("✅ Basic movement test complete!")


def test_blocked_movement():
    """Test movement blocked by non-walkable terrain."""
    print("\n" + "=" * 60)
    print("TEST: Blocked Movement")
    print("=" * 60)
    
    from game_entities import Wall
    
    # Create a 3x3 grid
    grid = create_floor_grid(3, 3)
    
    # Add agent at (0, 1)
    grid = add_agent_to_grid(grid, "agent1", (0, 1))
    
    # Add wall at (1, 1) - blocking the path
    wall_node = next((n for n in grid.nodes if n.position == (1, 1)), None)
    wall_node.entities.clear()  # Remove floor
    wall = Wall(name="wall_1_1", walkable=False, transparent=False)
    wall_node.entities.append(wall)
    
    grid.promote_to_root()
    
    print("\nInitial state (wall at 1,1):")
    print_grid(grid)
    
    # Try to move into wall
    print("Attempting to move into wall...")
    result = CallableRegistry.execute("move_entity_one_step", 
                                      grid_map=grid, 
                                      entity_name="agent1", 
                                      target_pos=(1, 1))
    grid_v1 = result if not isinstance(result, list) else result[0]
    
    print("\nAfter blocked move attempt:")
    print_grid(grid_v1)
    print(f"Version changed: {grid_v1.ecs_id != grid.ecs_id}")
    
    print("✅ Blocked movement test complete!")


def test_diagonal_movement():
    """Test diagonal movement (Chebyshev distance)."""
    print("\n" + "=" * 60)
    print("TEST: Diagonal Movement")
    print("=" * 60)
    
    # Create a 5x5 grid
    grid = create_floor_grid(5, 5)
    
    # Add agent at (2, 2)
    grid = add_agent_to_grid(grid, "agent1", (2, 2))
    
    grid.promote_to_root()
    
    print("\nInitial state:")
    print_grid(grid)
    
    # Move diagonally (2,2) -> (3,3)
    print("Moving agent diagonally (down-right)...")
    result = CallableRegistry.execute("move_entity_one_step", 
                                      grid_map=grid, 
                                      entity_name="agent1", 
                                      target_pos=(3, 3))
    grid_v1 = result if not isinstance(result, list) else result[0]
    
    print("\nAfter diagonal move:")
    print_grid(grid_v1)
    print(f"Version changed: {grid_v1.ecs_id != grid.ecs_id}")
    
    print("✅ Diagonal movement test complete!")


if __name__ == "__main__":
    test_basic_movement()
    test_blocked_movement()
    test_diagonal_movement()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE!")
    print("=" * 60)
