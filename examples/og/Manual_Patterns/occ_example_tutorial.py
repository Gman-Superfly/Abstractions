"""
OCC (Optimistic Concurrency Control) Example for Abstractions Framework

This example demonstrates how to use the new OCC fields (version, last_modified) 
to prevent race conditions when multiple operations modify the same entity.

OCC = Optimistic Concurrency Control - allows concurrent access while detecting conflicts before commit.
"""

import asyncio
import time
from typing import Optional
from uuid import UUID

from abstractions.ecs.entity import Entity, EntityRegistry
from abstractions.ecs.functional_api import get, put


class BankAccount(Entity):
    """Example entity with balance that needs OCC protection."""
    account_number: str = ""
    balance: float = 0.0
    transaction_count: int = 0


class OCCOperationResult:
    """Result of an OCC-protected operation."""
    def __init__(self, success: bool, retries: int = 0, error: Optional[str] = None):
        self.success = success
        self.retries = retries
        self.error = error


async def deposit_money_with_occ(account_id: UUID, amount: float, max_retries: int = 5) -> OCCOperationResult:
    """
    Deposit money to account with OCC (Optimistic Concurrency Control) protection.
    
    OCC = prevents race conditions when multiple deposits happen simultaneously.
    
    Args:
        account_id: The account entity ID
        amount: Amount to deposit
        max_retries: Maximum retry attempts on conflicts
        
    Returns:
        OCCOperationResult with success status and retry count
    """
    retries = 0
    
    while retries <= max_retries:
        # Step 1: Get current account state and snapshot OCC fields
        account = EntityRegistry.get_stored_entity(account_id, account_id)
        if not account:
            return OCCOperationResult(False, retries, "Account not found")
        
        # OCC Snapshot - capture current version and modification time
        original_version = account.version
        original_modified = account.last_modified
        
        print(f"💰 DEPOSIT ATTEMPT {retries + 1}: Account {account.account_number}")
        print(f"   📊 Current Balance: ${account.balance:.2f}")
        print(f"   🔢 OCC Version: {original_version}")
        print(f"   ⏰ OCC Modified: {original_modified}")
        
        # Step 2: Simulate some processing time (where conflicts can occur)
        await asyncio.sleep(0.1)  # Simulate network delay, validation, etc.
        
        # Step 3: Perform the business logic
        new_balance = account.balance + amount
        account.balance = new_balance
        account.transaction_count += 1
        
        # Step 4: Mark as modified (updates OCC fields)
        account.mark_modified()  # Increments version, updates last_modified
        
        print(f"   💵 New Balance: ${account.balance:.2f}")
        print(f"   🔢 New OCC Version: {account.version}")
        
        # Step 5: Check for OCC conflicts before committing
        current_account = EntityRegistry.get_stored_entity(account_id, account_id)
        if current_account and account.has_occ_conflict(current_account):
            retries += 1
            
            print(f"   ⚠️  OCC CONFLICT DETECTED!")
            print(f"   📊 Expected Version: {original_version}, Actual: {current_account.version}")
            print(f"   ⏰ Expected Modified: {original_modified}, Actual: {current_account.last_modified}")
            
            if retries > max_retries:
                return OCCOperationResult(False, retries, "Max retries exceeded")
            
            # Exponential backoff before retry
            backoff_time = 0.01 * (2 ** retries)
            print(f"   🔄 Retrying in {backoff_time:.3f}s...")
            await asyncio.sleep(backoff_time)
            continue
        
        # Step 6: Commit the changes (no conflict detected)
        put(account)
        print(f"   ✅ DEPOSIT SUCCESS: ${amount:.2f} deposited")
        print(f"   📊 Final Balance: ${account.balance:.2f}")
        print(f"   🔢 Final Version: {account.version}")
        
        return OCCOperationResult(True, retries)
    
    return OCCOperationResult(False, retries, "Unexpected failure")


async def withdraw_money_with_occ(account_id: UUID, amount: float, max_retries: int = 5) -> OCCOperationResult:
    """
    Withdraw money from account with OCC protection.
    
    Similar to deposit but with overdraft protection.
    """
    retries = 0
    
    while retries <= max_retries:
        account = EntityRegistry.get_stored_entity(account_id, account_id)
        if not account:
            return OCCOperationResult(False, retries, "Account not found")
        
        # OCC Snapshot
        original_version = account.version
        original_modified = account.last_modified
        
        print(f"💸 WITHDRAWAL ATTEMPT {retries + 1}: Account {account.account_number}")
        print(f"   📊 Current Balance: ${account.balance:.2f}")
        print(f"   🔢 OCC Version: {original_version}")
        
        # Check for sufficient funds
        if account.balance < amount:
            return OCCOperationResult(False, retries, "Insufficient funds")
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Perform withdrawal
        new_balance = account.balance - amount
        account.balance = new_balance
        account.transaction_count += 1
        account.mark_modified()
        
        print(f"   💵 New Balance: ${account.balance:.2f}")
        
        # OCC conflict check
        current_account = EntityRegistry.get_stored_entity(account_id, account_id)
        if current_account and account.has_occ_conflict(current_account):
            retries += 1
            print(f"   ⚠️  OCC CONFLICT on withdrawal - retrying...")
            
            if retries > max_retries:
                return OCCOperationResult(False, retries, "Max retries exceeded")
            
            await asyncio.sleep(0.01 * (2 ** retries))
            continue
        
        # Commit withdrawal
        put(account)
        print(f"   ✅ WITHDRAWAL SUCCESS: ${amount:.2f} withdrawn")
        print(f"   📊 Final Balance: ${account.balance:.2f}")
        
        return OCCOperationResult(True, retries)
    
    return OCCOperationResult(False, retries, "Unexpected failure")


async def concurrent_operations_demo():
    """
    Demonstrate concurrent operations with OCC protection.
    
    Shows how OCC prevents race conditions when multiple operations
    modify the same account simultaneously.
    """
    print("\n🏦 OCC (Optimistic Concurrency Control) Demo")
    print("=" * 60)
    
    # Create test account
    account = BankAccount(
        account_number="ACC-001", 
        balance=1000.0,
        transaction_count=0
    )
    account.promote_to_root()
    
    print(f"📈 Initial Account State:")
    print(f"   Account: {account.account_number}")
    print(f"   Balance: ${account.balance:.2f}")
    print(f"   Version: {account.version}")
    print(f"   Modified: {account.last_modified}")
    
    # Simulate concurrent operations
    print(f"\n🚀 Starting Concurrent Operations...")
    
    # Launch multiple deposits and withdrawals simultaneously
    tasks = [
        deposit_money_with_occ(account.ecs_id, 100.0),    # +100
        deposit_money_with_occ(account.ecs_id, 50.0),     # +50
        withdraw_money_with_occ(account.ecs_id, 75.0),    # -75
        deposit_money_with_occ(account.ecs_id, 200.0),    # +200
        withdraw_money_with_occ(account.ecs_id, 25.0),    # -25
    ]
    
    # Wait for all operations to complete
    results = await asyncio.gather(*tasks)
    
    # Analyze results
    print(f"\n📊 Operation Results:")
    successful_ops = sum(1 for r in results if r.success)
    total_retries = sum(r.retries for r in results)
    
    print(f"   ✅ Successful Operations: {successful_ops}/{len(results)}")
    print(f"   🔄 Total Retries: {total_retries}")
    
    for i, result in enumerate(results):
        op_type = "DEPOSIT" if i in [0, 1, 3] else "WITHDRAWAL"
        amount = [100.0, 50.0, 75.0, 200.0, 25.0][i]
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        print(f"   {op_type} ${amount:.2f}: {status} (retries: {result.retries})")
    
    # Final account state
    final_account = EntityRegistry.get_stored_entity(account.ecs_id, account.ecs_id)
    if final_account:
        print(f"\n📈 Final Account State:")
        print(f"   Balance: ${final_account.balance:.2f}")
        print(f"   Transactions: {final_account.transaction_count}")
        print(f"   Final Version: {final_account.version}")
        print(f"   Final Modified: {final_account.last_modified}")
        
        # Calculate expected balance
        expected_balance = 1000.0 + 100.0 + 50.0 - 75.0 + 200.0 - 25.0  # 1250.0
        if abs(final_account.balance - expected_balance) < 0.01:
            print(f"   ✅ Balance verification: PASSED (expected ${expected_balance:.2f})")
        else:
            print(f"   ❌ Balance verification: FAILED (expected ${expected_balance:.2f})")


async def simple_occ_example():
    """Simple example showing basic OCC usage."""
    print("\n🔍 Simple OCC Example")
    print("=" * 40)
    
    # Create entity
    account = BankAccount(account_number="SIMPLE-001", balance=500.0)
    account.promote_to_root()
    
    print(f"Initial: Balance=${account.balance:.2f}, Version={account.version}")
    
    # Modify and check OCC fields
    account.balance += 100.0
    account.mark_modified()  # Updates OCC fields
    
    print(f"After modification: Balance=${account.balance:.2f}, Version={account.version}")
    print(f"Last modified: {account.last_modified}")
    
    # Demonstrate conflict detection
    other_account = BankAccount(account_number="SIMPLE-001", balance=500.0)
    other_account.ecs_id = account.ecs_id  # Same entity ID
    other_account.version = 0  # But old version
    
    has_conflict = other_account.has_occ_conflict(account)
    print(f"Conflict detected: {has_conflict} (old version {other_account.version} vs current {account.version})")


if __name__ == "__main__":
    async def main():
        await simple_occ_example()
        await concurrent_operations_demo()
        
        print(f"\n✨ OCC Demo Complete!")
        print(f"OCC = Optimistic Concurrency Control prevents race conditions in concurrent entity modifications.")
    
    asyncio.run(main()) 