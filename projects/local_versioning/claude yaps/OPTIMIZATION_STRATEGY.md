# Optimization Strategy: Two-Phase Approach

## 🎯 The Real Problem

**Current Performance** (with CallableRegistry):
- 10×10 (111 entities): **151 ms/op** (7,589x slower than direct mutation)
- 50×20 (1,051 entities): **1,526 ms/op** (~75,000x slower than direct mutation)
- 50×50 (2,551 entities): **~3,500 ms/op** (~175,000x slower than direct mutation)

**This is unacceptably slow even with "full validation" enabled.**

---

## 📋 Two-Phase Strategy

### Phase 1: Core Optimizations (Make the system fast)
**Goal**: Get to <10ms per operation for 1,000+ entities

**Focus**: Optimize the fundamental operations that happen on EVERY execution:
1. Tree building (currently O(N) every time)
2. Diff computation (currently O(N) every time)
3. Tree copying (deep copy overhead)
4. Field introspection (repeated type checking)
5. List lookups (O(N) linear scans)

**Expected Speedup**: 100-1,000x

### Phase 2: Lazy/Local Optimizations (Make it even faster)
**Goal**: Get to <1ms per operation for localized changes

**Focus**: Skip unnecessary work when we control the workflow:
1. Lazy divergence checking (skip when safe)
2. Partial versioning (version only modified subtrees)
3. Reattachment pattern (avoid full tree validation)

**Expected Speedup**: Additional 10-100x on top of Phase 1

---

## 🔥 Phase 1 Priorities (Do These First)

### P0: Profiling & Measurement
**Why**: Need to know where the actual time is spent

**Actions**:
1. Run `profile_pipeline.py` to measure each step
2. Identify the #1 bottleneck (likely tree building or diff)
3. Measure per-entity costs

**Expected**: Will reveal if it's tree building, diff, or copying

### P1: Quick Wins (1-2 days)
**Why**: Low-hanging fruit with high impact

**Actions**:
1. **Early exit for divergence checks**
   - Add `has_any_differences()` that returns on first diff
   - Skip full diff computation when we just need yes/no
   - **Expected: 10x speedup for divergence checks**

2. **Cache field metadata**
   - Class-level cache for Pydantic field info
   - Avoid repeated type introspection
   - **Expected: 2-3x speedup for tree building**

3. **Shallow copies for read-only ops**
   - Use `model_copy(deep=False)` where possible
   - Only deep copy when mutating
   - **Expected: 5x speedup for tree retrieval**

### P2: Structural Changes (3-5 days)
**Why**: Fundamental algorithmic improvements

**Actions**:
1. **Dict vs List benchmark**
   - Test `nodes: Dict[Tuple[int,int], Node]` vs `nodes: List[Node]`
   - Measure tree building and lookup performance
   - **Expected: 10-100x speedup for lookups**

2. **Incremental tree updates**
   - Only rebuild modified subtrees
   - Reuse unchanged parts of tree
   - **Expected: 100x+ speedup for small changes**

3. **Lazy ancestry computation**
   - Compute paths on demand, not upfront
   - Cache results
   - **Expected: 2-5x speedup for tree building**

### P3: Algorithm Improvements (5-7 days)
**Why**: Optimize the diff and versioning algorithms

**Actions**:
1. **Optimize diff algorithm**
   - Early exit on first difference
   - Skip unnecessary comparisons
   - Use cached edge sets
   - **Expected: 10-50x speedup for diffs**

2. **Reduce redundant data**
   - Use `Field(exclude=True)` for computed fields
   - Avoid storing redundant indexes
   - **Expected: 2-3x speedup, lower memory**

---

## 📊 Expected Results After Phase 1

### Conservative Estimate
**From**: 1,526 ms/op (1,051 entities)  
**To**: 10-50 ms/op (1,051 entities)  
**Speedup**: 30-150x

### Optimistic Estimate
**From**: 1,526 ms/op (1,051 entities)  
**To**: 1-10 ms/op (1,051 entities)  
**Speedup**: 150-1,500x

### Why This Matters
- Makes the system **usable** even with full validation
- Provides a solid foundation for Phase 2 optimizations
- Reduces the "wasteful operations" overhead

---

## 🚀 Phase 2: Lazy/Local Optimizations

**Only start after Phase 1 is complete and we have <10ms baseline.**

### Lazy Divergence Checking
**What**: Skip divergence check when we control execution flow

**Implementation**:
```python
CallableRegistry.execute(
    "move_agent",
    gridmap=map,
    skip_divergence_check=True  # ← Skip check
)
```

**Expected**: 2x additional speedup (eliminates one tree build + diff)

### Partial Versioning with Reattachment
**What**: Version only modified subtrees, reattach to parent

**Implementation**:
```python
@CallableRegistry.register(reattach_outputs={...})
def move_agent_local(source_node, agent, target_node) -> (node, node):
    # Returns detached nodes
    return source_node, target_node

# System reattaches to gridmap, versions only 4 entities
```

**Expected**: 10-100x additional speedup for localized changes

### Greedy Versioning
**What**: Skip full diff when we know what changed

**Implementation**:
```python
EntityRegistry.version_entity_partial(
    gridmap,
    modified_child_ids={node1.ecs_id, node2.ecs_id},
    skip_full_diff=True
)
```

**Expected**: Enables partial versioning, reduces overhead

---

## 🎯 Success Metrics

### Phase 1 Target
- **10×10 (111 entities)**: <5 ms/op (currently 151 ms)
- **50×20 (1,051 entities)**: <10 ms/op (currently 1,526 ms)
- **50×50 (2,551 entities)**: <20 ms/op (currently 3,500 ms)

### Phase 2 Target (after Phase 1)
- **10×10 (111 entities)**: <1 ms/op
- **50×20 (1,051 entities)**: <2 ms/op
- **50×50 (2,551 entities)**: <5 ms/op

### Final Goal
**Enable real-time simulation of 10,000+ entity worlds at 60+ FPS**
- 60 FPS = 16.67 ms per frame
- 100 agent moves per frame = 0.167 ms per move
- **Need ~10,000x total speedup from current baseline**

---

## 📝 Immediate Next Steps

1. **Run profiling script**:
   ```bash
   python profile_pipeline.py
   ```

2. **Analyze results** to identify #1 bottleneck

3. **Implement top 3 quick wins**:
   - Early exit for divergence
   - Field metadata caching
   - Shallow copies

4. **Benchmark improvements** after each change

5. **Iterate** until Phase 1 targets are met

6. **Then** move to Phase 2 (lazy/local optimizations)

---

## 🔍 Key Insights

1. **The system is doing too much work** - 5×O(N) operations per move
2. **Lists are slow** - O(N) lookups for every node access
3. **Deep copies are expensive** - Copying entire trees unnecessarily
4. **Field introspection is repeated** - No caching of type metadata
5. **Diff is too thorough** - No early exit options

**All of these can be fixed without changing the core architecture.**

---

## 📚 Documentation Created

1. **`CORE_OPTIMIZATION_OPPORTUNITIES.md`** - Detailed analysis of all optimization opportunities
2. **`profile_pipeline.py`** - Profiling script to measure bottlenecks
3. **`OPTIMIZATION_STRATEGY.md`** - This document (high-level strategy)

**Next**: Run profiling and start implementing optimizations!
