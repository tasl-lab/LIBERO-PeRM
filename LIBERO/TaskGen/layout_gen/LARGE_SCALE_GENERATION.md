# Large-Scale Layout Generation Guide

## Quick Start

### Run 500 layouts in background (survives logout):
```bash
cd /home/jingyu/P_Bench/P_Bench/LIBERO

nohup python -m TaskGen.layout_gen.test_layout_gen_with_llm \
  --test basic \
  --count 500 \
  --metrics metrics_500.json \
  > layout_gen.log 2>&1 &

# Check progress
tail -f layout_gen.log

# Check the background job
jobs
ps aux | grep test_layout_gen_with_llm
```

### Run smaller test first:
```bash
# Test with 10 layouts first
python -m TaskGen.layout_gen.test_layout_gen_with_llm --test basic --count 10 --metrics test_metrics.json
```

## Disk Space Estimation

Each BDDL file is approximately **2-4 KB**:
- 10 layouts ≈ 20-40 KB
- 100 layouts ≈ 200-400 KB  
- 500 layouts ≈ 1-2 MB
- 1000 layouts ≈ 2-4 MB

**Metrics JSON file** grows with number of runs:
- Stores position data + success/failure info for each run
- 500 runs ≈ 500-1000 KB
- 1000 runs ≈ 1-2 MB

**Total for 500 layouts: ~2-3 MB** (very reasonable!)

## Output Files

### BDDL Files
Saved to: `dataset/pref_bddl_run<N>/P_BENCH_SCENE_*.bddl`
```
dataset/
  pref_bddl_run1/
  pref_bddl_run2/
  ...
  pref_bddl_run500/
```

### Metrics JSON
Contains:
```json
{
  "test_name": "Multi-Preference Layout Generation",
  "start_time": "2025-10-13T...",
  "end_time": "2025-10-13T...",
  "target_count": 500,
  "completed_count": 500,
  "success_count": 485,
  "failure_count": 15,
  "statistics": {
    "red_bowl": {
      "mean_x": 0.05,
      "mean_y": -0.10,
      "std_x": 0.08,
      "std_y": 0.06,
      "min_x": -0.15,
      "max_x": 0.20,
      "min_y": -0.22,
      "max_y": 0.15,
      "count": 485
    },
    ...
  },
  "runs": [
    {
      "run_number": 1,
      "timestamp": "...",
      "success": true,
      "objects_placed": {
        "red_bowl": [0.05, -0.10],
        "tomato_sauce": [0.15, 0.03],
        ...
      }
    },
    {
      "run_number": 2,
      "success": false,
      "failure_reason": "ValueError: collision detected",
      "traceback": "..."
    },
    ...
  ]
}
```

## Monitoring Progress

### Check log file:
```bash
tail -f layout_gen.log
```

### Check metrics file (updates every 10 runs):
```bash
# Pretty print current progress
python -c "import json; m=json.load(open('metrics_500.json')); print(f\"Progress: {m['completed_count']}/{m['target_count']} | Success: {m['success_count']} | Failed: {m['failure_count']}\")"
```

### Kill if needed:
```bash
# Find process ID
ps aux | grep test_layout_gen_with_llm

# Kill it
kill <PID>
```

## Expected Runtime

With GPT-5 latency ~1-3 seconds per LLM call:
- Each layout: 2 preferences + ~5-7 objects = ~7-9 LLM calls
- Per layout time: ~10-30 seconds
- 500 layouts: **~1.5 - 4 hours**

Progress is saved incrementally every 10 runs, so you can resume if interrupted.

## Success Rate

Expected success rate: **80-95%**

Common failure reasons:
1. **Collision detection**: LLM places objects too close (most common)
2. **API errors**: Temporary GPT-5 API issues (rare)
3. **Out of bounds**: LLM coordinates outside table bounds (rare, auto-corrected)

Failed runs are logged with failure reasons in metrics JSON for debugging.

---

## OBSERVED PERFORMANCE & FAILURE ANALYSIS

### Actual Runtime (Measured)
⚠️ **Significantly slower than expected:**
- **Per LLM call: ~39 seconds** (vs expected 1-3 sec)
- **Per layout: ~5 minutes** (vs expected 10-30 sec)
- **500 layouts: ~42 hours** (vs expected 1.5-4 hours)

**Bottleneck:** GPT-5 API latency (20x slower than estimated)

### Actual Success Rate (Measured)
📊 **First 20 runs analysis:**
- ✅ **Success: 65%** (13/20)
- ❌ **Failure: 35%** (7/20)
- **All failures: `red_bowl` collision detection**

Much lower than expected 80-95% success rate.

---

## Collision Detection Standards & Failure Root Cause

### Current Collision Detection Configuration

**Method: 2D AABB (Axis-Aligned Bounding Box)**
```python
COLLISION_MARGIN = 0.02  # 2cm minimum clearance between objects
```

**Detection Process:**
1. New object requested at position (x, y)
2. Calculate bounding box from object size:
   - `min = (x - width/2, y - height/2)`
   - `max = (x + width/2, y + height/2)`
3. Expand box by margin: ±0.02m in all directions
4. Check overlap with ALL existing object boxes
5. **If collision → raise ValueError, object SKIPPED (no retry)**

**LLM is provided:**
- All existing object positions (x, y)
- All object sizes (width × height)
- Table bounds: x ∈ [-0.25, 0.25], y ∈ [-0.25, 0.25]
- Explicit instruction: "Maintain at least 0.02m clearance"

---

### Why LLM Still Generates Collisions (35% failure rate)

**Root Causes:**

1. **Imperfect Spatial Math**
   - LLM approximates distances, doesn't compute exact geometry
   - Misjudges clearance by 1-3cm (critical when margin = 2cm)
   - Example: Thinks 0.025m gap is safe, actual gap < 0.02m

2. **Workspace Density**
   - 7+ objects in 0.5m × 0.5m space (0.25m² total)
   - Each object ~0.05-0.15m (large relative to workspace)
   - Limited free space → high collision probability

3. **Sequential Placement = Crowding**
   - Objects added one-by-one
   - `red_bowl` added LAST (after 6-7 objects)
   - Workspace most crowded when red_bowl attempts placement
   - Highest failure probability for last objects

4. **No Retry Logic (Critical Gap)**
   ```python
   # pref_layout_gen.py lines 802-807
   try:
       position = object_locating(constraint, layout, obj_name)
       layout.object_add(obj_name, position)  # Collision check here
   except ValueError as e:
       print(f"  ✗ Failed to add {obj_name}: {e}")
       # ⚠️ Object PERMANENTLY SKIPPED - no retry!
   ```

5. **Strict Test Validation**
   ```python
   # test_layout_gen_with_llm.py line 196
   if "red_bowl" not in personalized_layout.object_init_position:
       return None  # Hard failure
   ```
   **ANY missing required object = entire test fails**

---

### Failure Pattern Breakdown

**Observed in first 20 runs:**

| Run | Status | Issue |
|-----|--------|-------|
| 1 | ❌ FAIL | red_bowl collision |
| 2 | ✅ SUCCESS | - |
| 3 | ❌ FAIL | red_bowl collision |
| 4 | ❌ FAIL | red_bowl collision |
| 5 | ❌ FAIL | red_bowl collision |
| ... | ... | ... |

**Pattern:**
- 7 failures, ALL due to `red_bowl` not added
- Console shows: `"✗ Failed to add red_bowl: insert 'red_bowl'@(x,y) conflicting..."`
- Layout returned without red_bowl
- Test detects missing red_bowl → returns None → counted as failure

**Why red_bowl specifically?**
- Added last in sequence
- Workspace most crowded
- No retry on collision
- First failure causes cascade (strict validation)

---

### Proposed Fixes (Future Work)

1. **Batch LLM Generation**
   - Single LLM call outputs ALL object positions
   - Global optimization vs. greedy sequential
   - Can balance overall layout density

2. **Hybrid Approach**
   - LLM: semantic regions ("center-left, accessible")
   - Algorithm: exact collision-free position within region
   - Combines semantic reasoning + geometric precision

3. **Iterative Refinement Loop**
   - LLM generates layout
   - Collision checker provides feedback
   - LLM adjusts problematic positions
   - Repeat until success (max 3 iterations)

---

## Analysis

After generation, analyze metrics:

```python
import json
import numpy as np

# Load metrics
with open('metrics_500.json') as f:
    metrics = json.load(f)

# Print summary
print(f"Success rate: {metrics['success_count']/metrics['target_count']*100:.1f}%")

# Analyze position variance
for obj_name, stats in metrics['statistics'].items():
    print(f"\n{obj_name}:")
    print(f"  Position variance: x={stats['std_x']:.3f}, y={stats['std_y']:.3f}")
    print(f"  High variance? {'YES' if stats['std_x'] > 0.05 else 'NO'}")
```

## Cleanup

To remove generated files:
```bash
# Remove BDDL directories
rm -rf dataset/pref_bddl_run*

# Remove metrics
rm metrics*.json

# Remove logs
rm layout_gen.log
```
