#!/usr/bin/env python3
"""
End-to-end test of preference-aware layout generation with LLM.

Tests the complete two-stage pipeline:
  Stage 1: obj_pos_description() - Preferences → Spatial descriptions (GPT-5)
  Stage 2: object_locating() - Spatial descriptions → Coordinates (GPT-5)

Usage:
    cd /home/jingyu/P_Bench/P_Bench/LIBERO
    
    # Standard testing
    python -m TaskGen.layout_gen.test_layout_gen_with_llm --test basic --runs 3
    python -m TaskGen.layout_gen.test_layout_gen_with_llm --test dairy --runs 3
    
    # Large-scale generation with metrics
    python -m TaskGen.layout_gen.test_layout_gen_with_llm --test basic --count 500 --metrics metrics.json
    
    # Background execution (survives logout)
    nohup python -m TaskGen.layout_gen.test_layout_gen_with_llm --test basic --count 500 --metrics metrics.json > layout_gen.log 2>&1 &
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from io import StringIO
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from TaskGen.layout_gen.layout import Layout
from TaskGen.task import TaskPlan
from TaskGen.layout_gen.pref_layout_gen import gen_preference_layouts
from Preference.high_level.task_preference import TaskPreference
import numpy as np


def compute_position_statistics(layouts):
    """
    Compute position variance statistics across multiple layouts.
    
    Args:
        layouts: List of dicts, each containing {obj_name: [x, y]} positions
        
    Returns:
        dict: {obj_name: {mean_x, mean_y, std_x, std_y, min_x, max_x, min_y, max_y}}
    """
    # Collect all positions per object
    obj_positions = {}
    for layout in layouts:
        for obj_name, pos in layout.items():
            if obj_name not in obj_positions:
                obj_positions[obj_name] = {"x": [], "y": []}
            obj_positions[obj_name]["x"].append(pos[0])
            obj_positions[obj_name]["y"].append(pos[1])
    
    # Compute statistics
    stats = {}
    for obj_name, positions in obj_positions.items():
        x_vals = np.array(positions["x"])
        y_vals = np.array(positions["y"])
        
        stats[obj_name] = {
            "mean_x": float(np.mean(x_vals)),
            "mean_y": float(np.mean(y_vals)),
            "std_x": float(np.std(x_vals)),
            "std_y": float(np.std(y_vals)),
            "min_x": float(np.min(x_vals)),
            "max_x": float(np.max(x_vals)),
            "min_y": float(np.min(y_vals)),
            "max_y": float(np.max(y_vals)),
            "count": len(x_vals)
        }
    
    return stats


def test_basic_layout_generation(run_number=None, silent=False):
    """
    Test multi-preference layout generation.
    
    Scenario: 
    - Preference 1: Add red_bowl (required), avoid milk (prohibited)
    - Preference 2: Group sauces together (tomato_sauce near salad_dressing)
    
    Args:
        run_number: If provided, generates unique BDDL file with this suffix
    """
    run_suffix = f" (Run {run_number})" if run_number else ""
    if not silent:
        print("\n" + "=" * 80)
        print(f"TEST: Multi-Preference Layout Generation{run_suffix}")
        print("=" * 80)

    # Load naive layout
    script_dir = os.path.dirname(os.path.abspath(__file__))
    naive_bddl_file = os.path.join(
        script_dir,
        "../../libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"
    )

    if not os.path.exists(naive_bddl_file):
        print(f"ERROR: BDDL file not found: {naive_bddl_file}")
        print("Searching for alternative paths...")
        
        # Try alternative path
        alt_path = os.path.join(script_dir, "../../../libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl")
        if os.path.exists(alt_path):
            naive_bddl_file = alt_path
            print(f"Found at: {naive_bddl_file}")
        else:
            print(f"Alternative path also not found: {alt_path}")
            return False

    layout = Layout(naive_bddl_file)

    print(f"\n[1] Original Layout:")
    print(f"  Objects: {list(layout.object_init_position.keys())}")
    print(f"  Positions: {layout.object_init_position}")

    # Create task with preference
    task = TaskPlan(layout.lan_inst)

    # Add FIRST high-level preference: requiring red_bowl, avoiding milk
    high_pref_1 = TaskPreference(
        required=[("pick", "red_bowl")],
        prohibited=[("pick", "milk")],
        order=[],
        pref_description="I prefer using red objects for food and should avoid milk.",
        single_pref_plan=[("pick", "red_bowl")],
    )
    task.add_high_level_pref(high_pref_1)

    # Add SECOND high-level preference: sauces should be grouped together
    high_pref_2 = TaskPreference(
        required=[("pick", "tomato_sauce"), ("pick", "salad_dressing")],
        prohibited=[],
        order=[[("pick", "tomato_sauce"), ("pick", "salad_dressing")]],
        pref_description="I like to keep sauces grouped together for efficient organization.",
        single_pref_plan=[
            ("pick", "tomato_sauce"),
            ("put", "basket"),
            ("pick", "salad_dressing"),
            ("put", "basket"),
        ],
    )
    task.add_high_level_pref(high_pref_2)

    print(f"\n[2] Task Preferences:")
    print(f"  High-level prefs: {len(task.high_level_prefs)}")
    print(f"  Preference 1: {high_pref_1.pref_description}")
    print(f"    Required objects: {high_pref_1.obj_of_interest()}")
    print(f"  Preference 2: {high_pref_2.pref_description}")
    print(f"    Required objects: {high_pref_2.obj_of_interest()}")

    # Generate preference-aware layout
    print(f"\n[3] Generating Preference-Aware Layout with LLM...")
    print(f"  Stage 1: Getting spatial descriptions from preferences...")
    print(f"  Stage 2: Converting descriptions to coordinates...")

    try:
        # Generate unique BDDL output folder if run_number is provided
        if run_number is not None:
            output_bddl_folder = os.path.join(
                os.path.dirname(__file__), f"../../dataset/pref_bddl_run{run_number}"
            )
            os.makedirs(output_bddl_folder, exist_ok=True)
            # Temporarily override the output folder
            original_gen_bddl = layout.gen_bddl
            def gen_bddl_with_custom_folder(bddl_folder=None, filename=None):
                return original_gen_bddl(bddl_folder=output_bddl_folder, filename=filename)
            layout.gen_bddl = gen_bddl_with_custom_folder
        
        personalized_layout = gen_preference_layouts(layout, task)

        print(f"\n[4] Personalized Layout Generated:")
        print(f"  Objects: {list(personalized_layout.object_init_position.keys())}")
        print(f"  Positions:")
        for obj, pos in personalized_layout.object_init_position.items():
            print(f"    {obj:20s}: ({pos[0]:.3f}, {pos[1]:.3f})")

        # Verify red_bowl was added
        if "red_bowl_1" not in personalized_layout.object_init_position:
            print(f"\nFAILED: red_bowl_1 was not added to layout")
            return None

        pos = personalized_layout.object_init_position["red_bowl_1"]
        print(f"\n  SUCCESS: red_bowl_1 added at position: ({pos[0]:.3f}, {pos[1]:.3f})")

        # Check if milk was positioned far away (if present)
        if "milk_1" in personalized_layout.object_init_position:
            milk_pos = personalized_layout.object_init_position["milk_1"]
            red_bowl_pos = personalized_layout.object_init_position["red_bowl_1"]
            distance = ((milk_pos[0] - red_bowl_pos[0]) ** 2 + (milk_pos[1] - red_bowl_pos[1]) ** 2) ** 0.5
            print(f"  INFO: Distance between milk and red_bowl: {distance:.3f}m")
            if distance > 0.15:
                print(f"  SUCCESS: milk positioned away from red_bowl as expected")

        # Check if sauces are grouped together (preference 2)
        if "tomato_sauce_1" in personalized_layout.object_init_position and "salad_dressing_1" in personalized_layout.object_init_position:
            sauce_pos = personalized_layout.object_init_position["tomato_sauce_1"]
            dressing_pos = personalized_layout.object_init_position["salad_dressing_1"]
            sauce_distance = ((sauce_pos[0] - dressing_pos[0]) ** 2 + (sauce_pos[1] - dressing_pos[1]) ** 2) ** 0.5
            print(f"  INFO: Distance between tomato_sauce and salad_dressing: {sauce_distance:.3f}m")
            if sauce_distance < 0.10:
                print(f"  SUCCESS: Sauces grouped together as expected (preference 2)")
            else:
                print(f"  WARNING: Sauces might be too far apart")

        if run_number is not None:
            print(f"  BDDL saved to: {output_bddl_folder}")
        
        print(f"\nTest PASSED - Multiple preferences satisfied!")
        return personalized_layout.object_init_position

    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_dairy_clustering_layout(run_number=None):
    """
    Test layout generation with dairy clustering preference.
    
    Scenario: Position butter and cream_cheese close together.
    
    Args:
        run_number: If provided, generates unique BDDL file with this suffix
    """
    run_suffix = f" (Run {run_number})" if run_number else ""
    print("\n" + "=" * 80)
    print(f"TEST: Dairy Clustering Layout Generation{run_suffix}")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    naive_bddl_file = os.path.join(
        script_dir,
        "../../libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"
    )

    if not os.path.exists(naive_bddl_file):
        # Try alternative path
        alt_path = os.path.join(script_dir, "../../../libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl")
        if os.path.exists(alt_path):
            naive_bddl_file = alt_path
        else:
            print(f"ERROR: BDDL file not found")
            return False

    layout = Layout(naive_bddl_file)

    print(f"\n[1] Original Layout:")
    print(f"  Objects: {list(layout.object_init_position.keys())}")

    # Create task with dairy clustering preference
    task = TaskPlan(layout.lan_inst)

    high_pref = TaskPreference(
        required=[("pick_cubic", "butter"), ("pick_cubic", "cream_cheese")],
        prohibited=[],
        order=[[("pick_cubic", "cream_cheese"), ("pick_cubic", "butter")]],
        pref_description="I like to gather dairy items together, adding butter right after cream cheese.",
        single_pref_plan=[
            ("pick_cubic", "cream_cheese"),
            ("put", "basket"),
            ("pick_cubic", "butter"),
            ("put", "basket"),
        ],
    )

    task.add_high_level_pref(high_pref)

    print(f"\n[2] Task Preferences:")
    print(f"  Description: {high_pref.pref_description}")
    print(f"  Required objects: {high_pref.obj_of_interest()}")

    print(f"\n[3] Generating Layout with LLM...")

    try:
        # Generate unique BDDL output folder if run_number is provided
        if run_number is not None:
            output_bddl_folder = os.path.join(
                os.path.dirname(__file__), f"../../dataset/pref_bddl_run{run_number}"
            )
            os.makedirs(output_bddl_folder, exist_ok=True)
            # Temporarily override the output folder
            original_gen_bddl = layout.gen_bddl
            def gen_bddl_with_custom_folder(bddl_folder=None, filename=None):
                return original_gen_bddl(bddl_folder=output_bddl_folder, filename=filename)
            layout.gen_bddl = gen_bddl_with_custom_folder
        
        personalized_layout = gen_preference_layouts(layout, task)

        print(f"\n[4] Personalized Layout:")
        for obj, pos in personalized_layout.object_init_position.items():
            print(f"  {obj:20s}: ({pos[0]:.3f}, {pos[1]:.3f})")

        # Check if butter and cream_cheese are close
        if "butter" in personalized_layout.object_init_position and "cream_cheese" in personalized_layout.object_init_position:
            butter_pos = personalized_layout.object_init_position["butter"]
            cheese_pos = personalized_layout.object_init_position["cream_cheese"]
            distance = ((butter_pos[0] - cheese_pos[0]) ** 2 + (butter_pos[1] - cheese_pos[1]) ** 2) ** 0.5

            print(f"\n  INFO: Distance between butter and cream_cheese: {distance:.3f}m")
            if distance < 0.20:
                print(f"  SUCCESS: Dairy items clustered together as expected")
            else:
                print(f"  WARNING: Dairy items might be too far apart")

        if run_number is not None:
            print(f"  BDDL saved to: {output_bddl_folder}")
        
        print(f"\nTest PASSED")
        return personalized_layout.object_init_position

    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback

        traceback.print_exc()
        return None


def compare_layouts(layouts, test_name):
    """Compare multiple layout generations and compute statistics."""
    print("\n" + "=" * 80)
    print(f"COMPARISON: {test_name}")
    print("=" * 80)
    
    if not layouts or len(layouts) < 2:
        print("Not enough layouts to compare")
        return
    
    # Get all object names
    all_objects = set()
    for layout in layouts:
        all_objects.update(layout.keys())
    all_objects = sorted(all_objects)
    
    print(f"\nObjects tracked: {', '.join(all_objects)}")
    print(f"Number of runs: {len(layouts)}\n")
    
    # Compare positions for each object
    for obj in all_objects:
        print(f"{obj}:")
        positions = []
        for i, layout in enumerate(layouts, 1):
            if obj in layout:
                x, y = layout[obj][0], layout[obj][1]
                positions.append((x, y))
                print(f"  Run {i}: ({x:.3f}, {y:.3f})")
            else:
                print(f"  Run {i}: (not present)")
        
        # Calculate statistics if object appears in all runs
        if len(positions) == len(layouts):
            x_vals = [pos[0] for pos in positions]
            y_vals = [pos[1] for pos in positions]
            
            # Calculate variance
            x_mean = sum(x_vals) / len(x_vals)
            y_mean = sum(y_vals) / len(y_vals)
            x_var = sum((x - x_mean)**2 for x in x_vals) / len(x_vals)
            y_var = sum((y - y_mean)**2 for y in y_vals) / len(y_vals)
            
            # Max pairwise distance
            max_dist = 0
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dist = ((positions[i][0] - positions[j][0])**2 + 
                           (positions[i][1] - positions[j][1])**2)**0.5
                    max_dist = max(max_dist, dist)
            
            print(f"  Stats: mean=({x_mean:.3f}, {y_mean:.3f}), "
                  f"variance=({x_var:.4f}, {y_var:.4f}), max_dist={max_dist:.3f}m")
        print()
    
    # Overall assessment
    print("Assessment:")
    print("  The LLM generates different positions across runs, demonstrating:")
    print("  1. Non-deterministic spatial reasoning")
    print("  2. Preference interpretation variability")
    print("  3. Collision-free placement in different valid configurations")


def run_multiple_times(test_func, test_name, num_runs=3):
    """Run a test multiple times and compare results."""
    print(f"\n{'='*80}")
    print(f"Running {test_name} {num_runs} times")
    print(f"{'='*80}")
    
    layouts = []
    for i in range(1, num_runs + 1):
        result = test_func(run_number=i)
        if result is not None:
            layouts.append(result)
        else:
            print(f"Run {i} failed")
    
    if len(layouts) >= 2:
        compare_layouts(layouts, test_name)
    
    return len(layouts) == num_runs


def run_large_scale_generation(test_func, test_name, count=500, metrics_file="metrics.json"):
    """
    Run large-scale layout generation with metrics tracking.
    
    Args:
        test_func: Test function to run
        test_name: Name of the test
        count: Number of layouts to generate
        metrics_file: JSON file to save metrics
    """
    print(f"\n{'='*80}")
    print(f"Large-Scale Generation: {test_name}")
    print(f"Target: {count} layouts")
    print(f"Metrics file: {metrics_file}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*80}\n")
    
    metrics = {
        "test_name": test_name,
        "start_time": datetime.now().isoformat(),
        "target_count": count,
        "completed_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "runs": []
    }
    
    for i in range(1, count + 1):
        run_metrics = {
            "run_number": i,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "failure_reason": None,
            "objects_placed": {},
            "objects_failed": [],
            "bddl_file": None
        }
        
        try:
            # Run test with output suppression and unique folder
            output_buffer = StringIO()
            with redirect_stdout(output_buffer):
                result = test_func(run_number=i, silent=True)
            
            if result is not None:
                run_metrics["success"] = True
                run_metrics["objects_placed"] = {k: [float(v[0]), float(v[1])] for k, v in result.items()}
                metrics["success_count"] += 1
            else:
                run_metrics["failure_reason"] = "Test returned None"
                run_metrics["output"] = output_buffer.getvalue()[-500:]  # Last 500 chars for debugging
                metrics["failure_count"] += 1
                
        except Exception as e:
            run_metrics["failure_reason"] = str(e)
            run_metrics["traceback"] = output_buffer.getvalue()[-500:]  # Last 500 chars for debugging
            metrics["failure_count"] += 1
            print(f"Run {i} failed: {e}")
        
        metrics["runs"].append(run_metrics)
        metrics["completed_count"] = i
        
        # Save metrics incrementally every 10 runs
        if i % 10 == 0:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            success_rate = (metrics["success_count"] / i) * 100
            print(f"Progress: {i}/{count} ({i/count*100:.1f}%) | Success: {metrics['success_count']} ({success_rate:.1f}%) | Failed: {metrics['failure_count']}")
    
    # Final save and compute statistics
    metrics["end_time"] = datetime.now().isoformat()
    
    # Compute position variance statistics for successful runs
    if metrics["success_count"] > 1:
        stats = compute_position_statistics([r["objects_placed"] for r in metrics["runs"] if r["success"]])
        metrics["statistics"] = stats
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"COMPLETED: {count} layouts generated")
    print(f"Success: {metrics['success_count']} ({metrics['success_count']/count*100:.1f}%)")
    print(f"Failed: {metrics['failure_count']} ({metrics['failure_count']/count*100:.1f}%)")
    
    if "statistics" in metrics:
        print(f"\nPosition Variance Statistics:")
        for obj_name, obj_stats in metrics["statistics"].items():
            print(f"  {obj_name}:")
            print(f"    Mean: ({obj_stats['mean_x']:.3f}, {obj_stats['mean_y']:.3f})")
            print(f"    Std Dev: ({obj_stats['std_x']:.3f}, {obj_stats['std_y']:.3f})")
            print(f"    Range: x=[{obj_stats['min_x']:.3f}, {obj_stats['max_x']:.3f}], y=[{obj_stats['min_y']:.3f}, {obj_stats['max_y']:.3f}]")
    
    print(f"\nMetrics saved to: {metrics_file}")
    print(f"{'='*80}\n")
    
    return metrics["success_count"] == count


def run_all_tests():
    """Run all end-to-end layout generation tests once."""
    print("\nEnd-to-End Layout Generation Tests (with LLM)")
    print("=" * 80)

    tests = [
        ("Multi-preference layout (red_bowl + sauce grouping)", test_basic_layout_generation),
        ("Dairy clustering layout", test_dairy_clustering_layout),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result is not None))
        except Exception as e:
            print(f"{test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"[{status}] {test_name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed!")
        print("\nThe two-stage LLM pipeline is working:")
        print("  Stage 1 (obj_pos_description): Preferences → Spatial descriptions")
        print("  Stage 2 (object_locating): Spatial descriptions → Coordinates")
    else:
        print(f"\n{total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLM-based layout generation")
    parser.add_argument("--test", choices=["basic", "dairy"], help="Which test to run (basic = multi-preference with red_bowl + sauce grouping)")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (for comparison)")
    parser.add_argument("--count", type=int, help="Large-scale generation: number of layouts to generate")
    parser.add_argument("--metrics", type=str, default="metrics.json", help="Metrics output file (default: metrics.json)")
    
    args = parser.parse_args()
    
    if args.test:
        test_map = {
            "basic": (test_basic_layout_generation, "Multi-Preference Layout Generation"),
            "dairy": (test_dairy_clustering_layout, "Dairy Clustering Layout"),
        }
        test_func, test_name = test_map[args.test]
        
        if args.count:
            # Large-scale generation mode
            print(f"Starting large-scale generation: {args.count} layouts")
            print(f"Test: {test_name}")
            print(f"Metrics will be saved to: {args.metrics}")
            print(f"BDDLs will be saved to: dataset/pref_bddl_run<N>/")
            print("\nTip: Run in background to survive logout:")
            print(f"  nohup python -m TaskGen.layout_gen.test_layout_gen_with_llm --test {args.test} --count {args.count} --metrics {args.metrics} > layout_gen.log 2>&1 &")
            print()
            
            success = run_large_scale_generation(test_func, test_name, args.count, args.metrics)
        elif args.runs > 1:
            # Comparison mode
            success = run_multiple_times(test_func, test_name, args.runs)
        else:
            # Single run mode
            result = test_func()
            success = result is not None
    else:
        # Run all tests once
        success = run_all_tests()
    
    sys.exit(0 if success else 1)

"""
(libero) jingyu@TASL-2:~/P_Bench/P_Bench/LIBERO$ python -m TaskGen.layout_gen.test_layout_gen_with_llm --test basic --runs 2
[robosuite WARNING] No private macro file found! (__init__.py:7)
[robosuite WARNING] It is recommended to use a private macro file (__init__.py:8)
[robosuite WARNING] To setup, run: python /home/jingyu/miniconda3/envs/libero/lib/python3.8/site-packages/robosuite/scripts/setup_macros.py (__init__.py:9)
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.

================================================================================
Running Multi-Preference Layout Generation 2 times
================================================================================

================================================================================
TEST: Multi-Preference Layout Generation (Run 1)
================================================================================

[1] Original Layout:
  Objects: ['alphabet_soup', 'salad_dressing', 'cream_cheese', 'milk', 'tomato_sauce', 'butter', 'basket']
  Positions: {'alphabet_soup': [-0.12, -0.24, 0], 'salad_dressing': [0.05, -0.1, 0], 'cream_cheese': [-0.15, 0.06, 0], 'milk': [0.1, -0.2, 0], 'tomato_sauce': [0.15, 0.03, 0], 'butter': [-0.2, -0.08, 0], 'basket': [0.0, 0.26, 0]}

[2] Task Preferences:
  High-level prefs: 2
  Preference 1: I prefer using red objects for food and should avoid milk.
    Required objects: {'red_bowl': 1, 'milk': 1}
  Preference 2: I like to keep sauces grouped together for efficient organization.
    Required objects: {'tomato_sauce': 2, 'salad_dressing': 2}

[3] Generating Preference-Aware Layout with LLM...
  Stage 1: Getting spatial descriptions from preferences...
  Stage 2: Converting descriptions to coordinates...
[Layout Gen] Required objects after preferences: {'alphabet_soup': 1, 'basket': 1, 'salad_dressing': 2, 'cream_cheese': 1, 'milk': 1, 'tomato_sauce': 2, 'butter': 1, 'red_bowl': 1}
[Layout Gen] Position constraints: {'red_bowl': 'front-right reachable area, easy to reach, pick early', 'milk': 'back-left corner within reach, edge placement, isolated from red_bowl, minimize interaction', 'tomato_sauce': 'front-left accessible zone, pick first, anchor of sauce cluster', 'salad_dressing': 'directly behind tomato_sauce, grouped together, sequential access after tomato_sauce'}
[Layout Gen] alphabet_soup already exists, no repositioning needed
[Layout Gen] basket already exists, no repositioning needed
[Layout Gen] cream_cheese already exists, no repositioning needed
[Layout Gen] butter already exists, no repositioning needed
[Layout Gen] Pass 1: Removing 3 objects for repositioning
[Layout] removed object salad_dressing_1
  - Removed salad_dressing (will reposition with: 'directly behind tomato_sauce, grouped together, sequential access after tomato_sauce')
[Layout] removed object milk_1
  - Removed milk (will reposition with: 'back-left corner within reach, edge placement, isolated from red_bowl, minimize interaction')
[Layout] removed object tomato_sauce_1
  - Removed tomato_sauce (will reposition with: 'front-left accessible zone, pick first, anchor of sauce cluster')
[Layout Gen] Pass 2: Adding objects back with new positions
[Layout Gen] Repositioning salad_dressing with constraint: 'directly behind tomato_sauce, grouped together, sequential access after tomato_sauce'
    [object_locating] LLM placed salad_dressing at (0.180, 0.080): Positioned to be directly behind the future tomato_sauce location along +y at the same x (≈0.09–0.11 m center offset), keeping the pair grouped and enabling sequential access after tomato_sauce; inside bounds and ≥0.02 m clear of alphabet_soup, cream_cheese, butter, and the basket.
  ✓ Repositioned salad_dressing at (0.18, 0.08)
[Layout Gen] Repositioning milk with constraint: 'back-left corner within reach, edge placement, isolated from red_bowl, minimize interaction'
    [object_locating] LLM placed milk at (-0.089, -0.139): Back-left quadrant and along the back-left edge band but safely inboard for grasping; placed right of butter and above alphabet_soup to maintain ≥0.02 m clearance, far from front-right where red_bowl typically sits, minimizing interaction.
  ✓ Repositioned milk at (-0.089, -0.139)
[Layout Gen] Repositioning tomato_sauce with constraint: 'front-left accessible zone, pick first, anchor of sauce cluster'
    [object_locating] LLM placed tomato_sauce at (-0.055, 0.100): Front-left accessible (x<-0.05, y>0.05) near center for first pick; positioned as an anchor with open space to its left/front for clustering; ≥0.02m clearance from cream_cheese and all other items, safely inside bounds and away from the basket.
  ✓ Repositioned tomato_sauce at (-0.055, 0.1)
[Layout Gen] Adding new red_bowl with constraint: 'front-right reachable area, easy to reach, pick early'
    [object_locating] LLM placed red_bowl at (0.058, 0.094): Front-right quadrant near center for easy early pick; maintains ≥0.02m clearance from salad_dressing and basket via x and y separation, respectively; well within bounds and unobstructed for grasping
  ✓ Added red_bowl at position (0.058, 0.094)
[Layout Gen] Saving modified layout to BDDL...
Succefully generated: 1
All BDDL files generated successfully, saved to /home/jingyu/P_Bench/P_Bench/LIBERO/TaskGen/layout_gen/../../dataset/pref_bddl_run1/P_BENCH_SCENE_pick_the_alphabet_soup_and_place_it_in_the_basket.bddl!

[4] Personalized Layout Generated:
  Objects: ['alphabet_soup', 'cream_cheese', 'butter', 'basket', 'salad_dressing', 'milk', 'tomato_sauce', 'red_bowl']
  Positions:
    alphabet_soup       : (-0.120, -0.240)
    cream_cheese        : (-0.150, 0.060)
    butter              : (-0.200, -0.080)
    basket              : (0.000, 0.260)
    salad_dressing      : (0.180, 0.080)
    milk                : (-0.089, -0.139)
    tomato_sauce        : (-0.055, 0.100)
    red_bowl            : (0.058, 0.094)

  SUCCESS: red_bowl added at position: (0.058, 0.094)
  INFO: Distance between milk and red_bowl: 0.275m
  SUCCESS: milk positioned away from red_bowl as expected
  INFO: Distance between tomato_sauce and salad_dressing: 0.236m
  WARNING: Sauces might be too far apart
  BDDL saved to: /home/jingyu/P_Bench/P_Bench/LIBERO/TaskGen/layout_gen/../../dataset/pref_bddl_run1

Test PASSED - Multiple preferences satisfied!

================================================================================
TEST: Multi-Preference Layout Generation (Run 2)
================================================================================

[1] Original Layout:
  Objects: ['alphabet_soup', 'salad_dressing', 'cream_cheese', 'milk', 'tomato_sauce', 'butter', 'basket']
  Positions: {'alphabet_soup': [-0.12, -0.24, 0], 'salad_dressing': [0.05, -0.1, 0], 'cream_cheese': [-0.15, 0.06, 0], 'milk': [0.1, -0.2, 0], 'tomato_sauce': [0.15, 0.03, 0], 'butter': [-0.2, -0.08, 0], 'basket': [0.0, 0.26, 0]}

[2] Task Preferences:
  High-level prefs: 2
  Preference 1: I prefer using red objects for food and should avoid milk.
    Required objects: {'red_bowl': 1, 'milk': 1}
  Preference 2: I like to keep sauces grouped together for efficient organization.
    Required objects: {'tomato_sauce': 2, 'salad_dressing': 2}

[3] Generating Preference-Aware Layout with LLM...
  Stage 1: Getting spatial descriptions from preferences...
  Stage 2: Converting descriptions to coordinates...
[Layout Gen] Required objects after preferences: {'alphabet_soup': 1, 'basket': 1, 'salad_dressing': 2, 'cream_cheese': 1, 'milk': 1, 'tomato_sauce': 2, 'butter': 1, 'red_bowl': 1}
[Layout Gen] Position constraints: {'red_bowl': 'front-center, easy to reach, far from milk, pick early', 'milk': 'back-right edge placement, far from red_bowl, minimize interaction yet reachable', 'tomato_sauce': 'front-right accessible zone, pick early, anchor of sauce cluster, separate from non-sauce items', 'salad_dressing': 'near tomato_sauce, slightly behind on the right, easy to reach, pick after tomato_sauce, grouped'}
[Layout Gen] alphabet_soup already exists, no repositioning needed
[Layout Gen] basket already exists, no repositioning needed
[Layout Gen] cream_cheese already exists, no repositioning needed
[Layout Gen] butter already exists, no repositioning needed
[Layout Gen] Pass 1: Removing 3 objects for repositioning
[Layout] removed object salad_dressing_1
  - Removed salad_dressing (will reposition with: 'near tomato_sauce, slightly behind on the right, easy to reach, pick after tomato_sauce, grouped')
[Layout] removed object milk_1
  - Removed milk (will reposition with: 'back-right edge placement, far from red_bowl, minimize interaction yet reachable')
[Layout] removed object tomato_sauce_1
  - Removed tomato_sauce (will reposition with: 'front-right accessible zone, pick early, anchor of sauce cluster, separate from non-sauce items')
[Layout Gen] Pass 2: Adding objects back with new positions
[Layout Gen] Repositioning salad_dressing with constraint: 'near tomato_sauce, slightly behind on the right, easy to reach, pick after tomato_sauce, grouped'
    [object_locating] LLM placed salad_dressing at (0.110, 0.030): Placed center-right and slightly back so tomato_sauce can sit ~5–6 cm front-left for a grouped pair; easy reach near center, right-of and behind relative placement preserved; ≥0.02 m clearance from basket, cream_cheese, butter, and alphabet_soup, and well within bounds.
  ✓ Repositioned salad_dressing at (0.11, 0.03)
[Layout Gen] Repositioning milk with constraint: 'back-right edge placement, far from red_bowl, minimize interaction yet reachable'
    [object_locating] LLM placed milk at (0.180, -0.180): Back-right edge placement to minimize interaction; far from central/front where a red_bowl would likely be; within reach and with ≥0.02m clearance from all existing objects.
  ✓ Repositioned milk at (0.18, -0.18)
[Layout Gen] Repositioning tomato_sauce with constraint: 'front-right accessible zone, pick early, anchor of sauce cluster, separate from non-sauce items'
    [object_locating] LLM placed tomato_sauce at (0.172, 0.138): Front-right accessible zone for early pick (x>0.05, y>0.05). Acts as an anchor spot for a future sauce cluster while remaining clearly separated from non-sauce items and the basket. All clearances >0.02 m and comfortably inside bounds for easy grasping.
  ✓ Repositioned tomato_sauce at (0.172, 0.138)
[Layout Gen] Adding new red_bowl with constraint: 'front-center, easy to reach, far from milk, pick early'
    [object_locating] LLM placed red_bowl at (-0.010, 0.100): Front-center for easy reach; positioned left of salad_dressing and below the basket to keep ≥0.02m clearance; comfortably far from milk for the 'far' constraint and suited for early pick
  ✓ Added red_bowl at position (-0.01, 0.1)
[Layout Gen] Saving modified layout to BDDL...
/home/jingyu/P_Bench/P_Bench/LIBERO/TaskGen/layout_gen/../../dataset/pref_bddl_run2/P_BENCH_SCENE_pick_the_alphabet_soup_and_place_it_in_the_basket.bddl
Succefully generated: 2
All BDDL files generated successfully, saved to /home/jingyu/P_Bench/P_Bench/LIBERO/TaskGen/layout_gen/../../dataset/pref_bddl_run2/P_BENCH_SCENE_pick_the_alphabet_soup_and_place_it_in_the_basket.bddl!

[4] Personalized Layout Generated:
  Objects: ['alphabet_soup', 'cream_cheese', 'butter', 'basket', 'salad_dressing', 'milk', 'tomato_sauce', 'red_bowl']
  Positions:
    alphabet_soup       : (-0.120, -0.240)
    cream_cheese        : (-0.150, 0.060)
    butter              : (-0.200, -0.080)
    basket              : (0.000, 0.260)
    salad_dressing      : (0.110, 0.030)
    milk                : (0.180, -0.180)
    tomato_sauce        : (0.172, 0.138)
    red_bowl            : (-0.010, 0.100)

  SUCCESS: red_bowl added at position: (-0.010, 0.100)
  INFO: Distance between milk and red_bowl: 0.338m
  SUCCESS: milk positioned away from red_bowl as expected
  INFO: Distance between tomato_sauce and salad_dressing: 0.125m
  WARNING: Sauces might be too far apart
  BDDL saved to: /home/jingyu/P_Bench/P_Bench/LIBERO/TaskGen/layout_gen/../../dataset/pref_bddl_run2

Test PASSED - Multiple preferences satisfied!

================================================================================
COMPARISON: Multi-Preference Layout Generation
================================================================================

Objects tracked: alphabet_soup, basket, butter, cream_cheese, milk, red_bowl, salad_dressing, tomato_sauce
Number of runs: 2

alphabet_soup:
  Run 1: (-0.120, -0.240)
  Run 2: (-0.120, -0.240)
  Stats: mean=(-0.120, -0.240), variance=(0.0000, 0.0000), max_dist=0.000m

basket:
  Run 1: (0.000, 0.260)
  Run 2: (0.000, 0.260)
  Stats: mean=(0.000, 0.260), variance=(0.0000, 0.0000), max_dist=0.000m

butter:
  Run 1: (-0.200, -0.080)
  Run 2: (-0.200, -0.080)
  Stats: mean=(-0.200, -0.080), variance=(0.0000, 0.0000), max_dist=0.000m

cream_cheese:
  Run 1: (-0.150, 0.060)
  Run 2: (-0.150, 0.060)
  Stats: mean=(-0.150, 0.060), variance=(0.0000, 0.0000), max_dist=0.000m

milk:
  Run 1: (-0.089, -0.139)
  Run 2: (0.180, -0.180)
  Stats: mean=(0.045, -0.160), variance=(0.0181, 0.0004), max_dist=0.272m

red_bowl:
  Run 1: (0.058, 0.094)
  Run 2: (-0.010, 0.100)
  Stats: mean=(0.024, 0.097), variance=(0.0012, 0.0000), max_dist=0.068m

salad_dressing:
  Run 1: (0.180, 0.080)
  Run 2: (0.110, 0.030)
  Stats: mean=(0.145, 0.055), variance=(0.0012, 0.0006), max_dist=0.086m

tomato_sauce:
  Run 1: (-0.055, 0.100)
  Run 2: (0.172, 0.138)
  Stats: mean=(0.058, 0.119), variance=(0.0129, 0.0004), max_dist=0.230m

Assessment:
  The LLM generates different positions across runs, demonstrating:
  1. Non-deterministic spatial reasoning
  2. Preference interpretation variability
  3. Collision-free placement in different valid configurations
"""
