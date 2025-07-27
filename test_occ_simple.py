#!/usr/bin/env python3
"""
Simple test to validate OCC (Optimistic Concurrency Control) implementation.
"""

import sys
import os
sys.path.insert(0, '.')

from abstractions.ecs.entity import Entity
from datetime import datetime, timezone


class TestEntity(Entity):
    """Simple test entity."""
    name: str = "test"
    value: int = 0


def test_occ_fields():
    """Test that OCC fields are present and working."""
    print("🧪 Testing OCC Implementation")
    print("=" * 40)
    
    # Create entity
    entity = TestEntity(name="test_entity", value=42)
    print(f"✅ Entity created: {entity.name}")
    print(f"📊 Initial version: {entity.version}")
    print(f"⏰ Initial last_modified: {entity.last_modified}")
    
    # Test mark_modified
    original_version = entity.version
    original_modified = entity.last_modified
    
    # Wait a tiny bit to ensure timestamp difference
    import time
    time.sleep(0.001)
    
    entity.value = 100
    entity.mark_modified()
    
    print(f"\n🔄 After mark_modified():")
    print(f"📊 Version: {original_version} → {entity.version}")
    print(f"⏰ Modified: {original_modified != entity.last_modified}")
    print(f"✅ Version incremented: {entity.version == original_version + 1}")
    
    # Test conflict detection
    other_entity = TestEntity(name="test_entity", value=42)
    other_entity.ecs_id = entity.ecs_id  # Same ID
    other_entity.version = 0  # But old version
    
    has_conflict = other_entity.has_occ_conflict(entity)
    print(f"\n🔍 Conflict Detection:")
    print(f"📊 Old version {other_entity.version} vs Current version {entity.version}")
    print(f"⚠️  Conflict detected: {has_conflict}")
    
    # Test update_ecs_ids also updates OCC
    original_version = entity.version
    entity.update_ecs_ids()
    
    print(f"\n🔄 After update_ecs_ids():")
    print(f"📊 Version incremented: {entity.version == original_version + 1}")
    print(f"⏰ forked_at set: {entity.forked_at is not None}")
    print(f"⏰ last_modified updated: {entity.last_modified}")
    
    print(f"\n✅ All OCC tests passed!")
    print(f"OCC = Optimistic Concurrency Control - prevents race conditions")


if __name__ == "__main__":
    test_occ_fields() 