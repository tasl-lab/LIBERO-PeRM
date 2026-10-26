#!/usr/bin/env python3
"""
Simple test script to verify pref_layout_gen is runnable.

This demonstrates the naive implementation that always adds a red_bowl
at a random location to any layout.

cd /home/jingyu/P_Bench/P_Bench/LIBERO/TaskGen/layout_gen
python test_pref_layout_gen.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from TaskGen.layout_gen.layout import Layout
from TaskGen.task import TaskPlan
from TaskGen.layout_gen.pref_layout_gen import (
    gen_preference_layouts,
    create_naive_test_preference,
)
from Preference.low_level.distance import AvoidToObjectPreference


def test_basic_layout_generation():
    """
    Test that we can generate a layout with red_bowl added.

    NOTE: BDDL file generation may fail - that's expected for now.
    We're testing the core functionality of modifying layouts based on preferences.
    """
    print("=" * 80)
    print("TEST: Basic Layout Generation with red_bowl")
    print("=" * 80)
    print("NOTE: BDDL generation failure is expected - we're testing object insertion")
    print()

    # Load naive layout
    script_dir = os.path.dirname(os.path.abspath(__file__))
    naive_bddl_file = os.path.join(script_dir, "../../libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl")

    if not os.path.exists(naive_bddl_file):
        print(f"ERROR: BDDL file not found at calculated path: {naive_bddl_file}")
        return False
    
    layout = Layout(naive_bddl_file)

    print("\n[1/3] Original Naive Layout:")
    print(f"  Objects: {list(layout.object_init_position.keys())}")
    print(f"  Task: {layout.lan_inst}")

    # Create task with test preference
    task = TaskPlan(layout.lan_inst)
    create_naive_test_preference(task)  # Adds requirement for red_bowl

    print("\n[2/3] Task Preferences:")
    print(f"  High-level prefs: {len(task.high_level_prefs)}")
    if task.high_level_prefs:
        test_objs = task.high_level_prefs[0].obj_of_interest()
        print(f"  Required objects: {test_objs}")

    # Generate personalized layout
    print("\n[3/3] Generating Personalized Layout...")
    personalized_layout = gen_preference_layouts(layout, task)

    print("\n✓ Personalized Layout Generated:")
    print(f"  Objects: {list(personalized_layout.object_init_position.keys())}")

    # Verify red_bowl was added
    if "red_bowl_1" not in personalized_layout.object_init_position:
        print("\n✗ FAIL: red_bowl_1 was not added to layout")
        return False

    pos = personalized_layout.object_init_position["red_bowl_1"]
    print(f"  ✓ red_bowl_1 added at position: {pos}")

    # Check if BDDL file was actually generated (optional for now)
    if personalized_layout.bddl_file and os.path.exists(personalized_layout.bddl_file):
        print(f"  ✓ BONUS: BDDL file generated: {personalized_layout.bddl_file}")
    else:
        print(f"  ⚠ BDDL file generation failed (expected)")

    print("\n" + "=" * 80)
    print("SUCCESS: Core functionality works - red_bowl added to layout!")
    print("=" * 80)
    return True


def test_layout_generation_with_low_level_pref():
    """
    Test that we can generate a layout with a low-level distance preference.
    """
    print("=" * 80)
    print("TEST: Layout Generation with Low-Level Preference")
    print("=" * 80)

    # 1. Load naive layout
    script_dir = os.path.dirname(os.path.abspath(__file__))
    naive_bddl_file = os.path.join(script_dir, "../../libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl")
    if not os.path.exists(naive_bddl_file):
        print(f"ERROR: BDDL file not found at calculated path: {naive_bddl_file}")
        return False
    layout = Layout(naive_bddl_file)
    print(f"  Loaded naive layout with objects: {list(layout.object_init_position.keys())}")

    # 2. Create a task and add a low-level preference
    task = TaskPlan(layout.lan_inst)
    low_pref = AvoidToObjectPreference(
        target_obj="red_bowl",
        distance_metric="greater_than",
        threshold=0.3,
        pref_description="keep gripper far away from the red bowl",
    )
    task.add_low_level_pref(low_pref)
    print(f"  Added low-level preference: {low_pref}")

    # 3. Generate the personalized layout
    print("  Generating personalized layout...")
    personalized_layout = gen_preference_layouts(layout, task)
    print(f"  Generated layout with objects: {list(personalized_layout.object_init_position.keys())}")

    # 4. Verify that the new object was added
    if "red_bowl_1" not in personalized_layout.object_init_position:
        print("\n✗ FAIL: red_bowl_1 was not added to layout based on low-level preference")
        return False

    pos = personalized_layout.object_init_position["red_bowl_1"]
    print(f"  ✓ SUCCESS: red_bowl_1 added at position: {pos}")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success_basic = test_basic_layout_generation()
    success_low_level = test_layout_generation_with_low_level_pref()
    sys.exit(0 if success_basic and success_low_level else 1)
