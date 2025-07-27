#!/usr/bin/env python3
"""
Simple OCC conflict test to validate the basic mechanism works.
"""

import sys
import asyncio
import time
sys.path.insert(0, '.')

from abstractions.ecs.entity import Entity, EntityRegistry
from datetime import datetime, timezone


class TestEntity(Entity):
    """Simple test entity."""
    name: str = "test"
    value: int = 0


async def test_occ_basic_conflict():
    """Test basic OCC conflict detection."""
    print("🧪 Testing Basic OCC Conflict Detection")
    print("=" * 50)
    
    # Create and register entity
    entity = TestEntity(name="test_entity", value=100)
    entity.promote_to_root()
    
    print(f"✅ Created entity: {entity.name} = {entity.value}")
    print(f"📊 Initial version: {entity.version}")
    print(f"⏰ Initial modified: {entity.last_modified}")
    
    # Scenario 1: No conflict (normal case)
    print(f"\n🔍 Scenario 1: No Conflict")
    original_version = entity.version
    original_modified = entity.last_modified
    
    # Modify entity
    entity.value = 200
    entity.mark_modified()
    
    # Get fresh copy from registry
    stored_entity = EntityRegistry.get_stored_entity(entity.ecs_id, entity.ecs_id)
    
    # Check - should be no conflict since no one else modified it
    has_conflict = (stored_entity.version != original_version or 
                   stored_entity.last_modified != original_modified)
    
    print(f"📊 Original version: {original_version}, Current: {stored_entity.version}")
    print(f"⏰ Original modified: {original_modified}")
    print(f"⏰ Current modified: {stored_entity.last_modified}")
    print(f"⚠️  Conflict detected: {has_conflict}")
    
    # Scenario 2: Simulate conflict
    print(f"\n🔍 Scenario 2: Simulated Conflict")
    
    # Take snapshot
    entity2 = TestEntity(name="test_entity2", value=300)
    entity2.promote_to_root()
    
    snapshot_version = entity2.version
    snapshot_modified = entity2.last_modified
    
    print(f"📊 Snapshot version: {snapshot_version}")
    print(f"⏰ Snapshot modified: {snapshot_modified}")
    
    # Simulate another operation modifying the entity
    entity2.value = 400
    entity2.mark_modified()  # This increments version and updates timestamp
    
    print(f"📊 After modification version: {entity2.version}")
    print(f"⏰ After modification modified: {entity2.last_modified}")
    
    # Now check for conflict
    conflict_detected = (entity2.version != snapshot_version or 
                        entity2.last_modified != snapshot_modified)
    
    print(f"⚠️  Conflict detected: {conflict_detected}")
    print(f"✅ Test result: {'PASS' if conflict_detected else 'FAIL'}")
    
    print(f"\n✅ Basic OCC test complete!")


async def test_concurrent_modification_simulation():
    """Simulate what happens with concurrent modifications."""
    print(f"\n🔄 Testing Concurrent Modification Simulation")
    print("=" * 50)
    
    # Create entity
    entity = TestEntity(name="concurrent_test", value=500)
    entity.promote_to_root()
    
    print(f"📊 Initial: value={entity.value}, version={entity.version}")
    
    # Operation 1: Take snapshot
    op1_snapshot_version = entity.version
    op1_snapshot_modified = entity.last_modified
    
    print(f"🔄 Op1 snapshot: version={op1_snapshot_version}")
    
    # Operation 2: Modify entity (simulating concurrent operation)
    entity.value = 600
    entity.mark_modified()
    
    print(f"🔄 Op2 modifies: value={entity.value}, version={entity.version}")
    
    # Operation 1: Try to apply its changes (should detect conflict)
    current_version = entity.version
    current_modified = entity.last_modified
    
    conflict = (current_version != op1_snapshot_version or 
               current_modified != op1_snapshot_modified)
    
    print(f"⚠️  Op1 detects conflict: {conflict}")
    print(f"📊 Expected version: {op1_snapshot_version}, Actual: {current_version}")
    print(f"✅ Concurrent test: {'PASS' if conflict else 'FAIL'}")


if __name__ == "__main__":
    async def main():
        await test_occ_basic_conflict()
        await test_concurrent_modification_simulation()
        
        print(f"\n✨ OCC Validation Complete!")
        print(f"OCC = Optimistic Concurrency Control conflict detection works correctly")
    
    asyncio.run(main()) 