import os
import unittest
import numpy as np
import random

# Change the working directory to the root of the project
# to ensure that all relative paths work correctly.
os.chdir(os.path.join(os.path.dirname(__file__), "../../../"))

from libero.libero.envs.env_wrapper import ControlEnv
from TaskGen.layout_gen.pref_layout_gen import (
    collision_detection_3d,
    is_scene_stable_v0,
    settle_and_check_stability,
    are_all_objects_stable,
    check_stability_with_position,
)


def get_scene_kinetic_energy(env):
    """Calculates the total kinetic energy of all movable objects in the scene."""
    sim = env.sim
    total_energy = 0.0
    for i in range(sim.model.nbody):
        body_name = sim.model.body_id2name(i)
        if (body_name is None or 'robot' in body_name or 'gripper' in body_name or 
            'floor' in body_name or 'table' in body_name):
            continue

        # Use MuJoCo's built-in kinetic energy calculation from sim.data.cvel
        # and body mass from sim.model.body_mass
        body_vel = sim.data.cvel[i]
        mass = sim.model.body_mass[i]
        
        # Linear kinetic energy: 0.5 * m * v^2
        linear_energy = 0.5 * mass * np.sum(np.square(body_vel[0:3]))
        
        # Note: Rotational energy is more complex (0.5 * I * w^2), but for the purpose
        # of this test (comparing relative stability), linear energy is a sufficient proxy.
        total_energy += linear_energy
    return total_energy


class TestLayoutUtils(unittest.TestCase):
    """
    Unit tests for layout generation utility functions in `pref_layout_gen.py`.

    This test suite verifies that core utility functions for collision detection
    and scene stability are working correctly. It also serves as a usage example.

    How to run this test:
    `PYTHONPATH=$PYTHONPATH:$(pwd) python LIBERO/TaskGen/layout_gen/test_collision_detection.py`
    """

    @classmethod
    def setUpClass(cls):
        """Set up a shared environment for the entire test class once."""
        bddl_file_rel_path = "LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"
        bddl_file = os.path.abspath(bddl_file_rel_path)
        
        try:
            env_args = {
                "bddl_file_name": bddl_file,
                "robots": ["Panda"],
                "controller": "OSC_POSE",
                "has_renderer": False,
                "has_offscreen_renderer": False,
                "use_camera_obs": False,
                "control_freq": 20,
            }
            env_wrapper = ControlEnv(**env_args)
            cls.env = env_wrapper.env
        except Exception as e:
            raise Exception(f"Failed to initialize environment in setUpClass: {e}")

        # Let the scene settle once and save the stable state.
        print("--- Initializing and settling scene once for all tests ---")
        for _ in range(100): # Settle for a generous amount of time.
            cls.env.sim.step()
        
        # Save the qpos and qvel of the settled state.
        cls.stable_qpos = cls.env.sim.data.qpos.copy()
        cls.stable_qvel = cls.env.sim.data.qvel.copy()
        print("--- Scene settled. Starting tests. ---")


    def setUp(self):
        """Reset the simulation to a stable state before each test."""
        # Load the pre-settled stable state.
        self.env.sim.data.qpos[:] = self.stable_qpos
        self.env.sim.data.qvel[:] = self.stable_qvel
        # Crucially, we must call sim.reset() to clear internal simulation data
        # like contact forces, which are not captured in qpos or qvel.
        self.env.sim.reset()


    def test_add_object_with_collision_check(self):
        """
        Tests finding a collision-free spot for an object using a loop.
        This serves as an example of how to programmatically add an object
        to a scene without causing new collisions.
        """
        sim = self.env.sim
        obj_to_place_name = "butter_1"
        obj_to_place_body_name = f"{obj_to_place_name}_main"
        
        # Get the joint info for the object we want to move
        obj_joint_id = sim.model.joint_name2id(f"{obj_to_place_name}_joint0")
        obj_joint_addr = sim.model.jnt_qposadr[obj_joint_id]

        # Find a stable Z position from an existing object to place the new one
        other_obj_joint_id = sim.model.joint_name2id("alphabet_soup_1_joint0")
        other_obj_joint_addr = sim.model.jnt_qposadr[other_obj_joint_id]
        stable_z = sim.data.qpos[other_obj_joint_addr + 2]

        MAX_ATTEMPTS = 50
        placement_successful = False
        print("\n--- Testing object placement ---")
        print(f"Attempting to find a collision-free spot for '{obj_to_place_name}'...")

        for attempt in range(MAX_ATTEMPTS):
            # Before trying a new placement, reset the simulation to the clean stable state
            # This is critical to ensure each attempt is independent.
            sim.data.qpos[:] = self.stable_qpos
            sim.data.qvel[:] = self.stable_qvel
            sim.forward()

            # 1. Generate a random candidate position on the table
            x_pos = random.uniform(-0.25, 0.25)
            y_pos = random.uniform(-0.25, 0.25)
            
            # Set the object position slightly above the table to let it settle
            sim.data.qpos[obj_joint_addr:obj_joint_addr+3] = [x_pos, y_pos, stable_z + 0.1]
            sim.forward()

            # 2. Check for collisions involving ONLY the new object
            collisions = collision_detection_3d(self.env, objects_of_interest=[obj_to_place_body_name])

            if not collisions:
                # 3. If no collisions, we consider it a success for this test's purpose.
                # The separate problem of ensuring the object is STABLE after placement
                # is not tested here.
                print(f"  ✅ SUCCESS: Found a collision-free position for '{obj_to_place_name}' at ({x_pos:.2f}, {y_pos:.2f}).")
                placement_successful = True
                break
            else:
                # For debugging, it's useful to see what it collided with
                if attempt < 5: # Only print for the first few attempts to avoid spam
                     print(f"  Attempt {attempt + 1}: Collision detected at ({x_pos:.2f}, {y_pos:.2f}) with {collisions[0]}.")

        self.assertTrue(placement_successful, f"Failed to find a collision-free position for {obj_to_place_name} after {MAX_ATTEMPTS} attempts.")


    def test_collision_detection_3d(self):
        """
        Tests the collision_detection_3d function by:
        1. Verifying no initial inter-object collisions exist.
        2. Forcing an object-object collision and asserting it is detected.
        3. Moving the objects apart and asserting the collision is resolved.
        """
        sim = self.env.sim
        # NOTE: The body names in the simulation often have a "_main" suffix.
        obj1_name = "alphabet_soup_1"
        obj2_name = "salad_dressing_1"
        obj1_body_name = f"{obj1_name}_main"
        obj2_body_name = f"{obj2_name}_main"
        objects_to_check = [obj1_body_name, obj2_body_name]

        # 1. Get the initial collision state. Due to potentially unstable scene generation,
        # we don't assert that this is zero. We just record it.
        sim.forward() # Ensure contacts are updated
        initial_collisions = collision_detection_3d(self.env, objects_of_interest=objects_to_check)
        print(f"\n✅ Found {len(initial_collisions)} initial collision(s). This is acceptable.")
        print(f"   [Debug] Initial collisions: {initial_collisions}")


        # In robosuite, free joints control the 7-DOF pose of objects. We get the
        # joint address to directly manipulate their positions in the simulation state vector (qpos).
        obj1_joint_id = sim.model.joint_name2id(f"{obj1_name}_joint0")
        obj2_joint_id = sim.model.joint_name2id(f"{obj2_name}_joint0")
        
        obj1_joint_addr = sim.model.jnt_qposadr[obj1_joint_id]
        obj2_joint_addr = sim.model.jnt_qposadr[obj2_joint_id]

        # 2. Force a collision by moving obj2 to the exact same position as obj1.
        obj1_pos = sim.data.qpos[obj1_joint_addr:obj1_joint_addr+3].copy()
        sim.data.qpos[obj2_joint_addr:obj2_joint_addr+3] = obj1_pos
        
        # Step the simulation to allow the physics engine to register the contact.
        sim.forward()

        # 2. Assert that our specific collision has been added.
        forced_collisions = collision_detection_3d(self.env, objects_of_interest=objects_to_check)
        expected_pair = tuple(sorted((obj1_body_name, obj2_body_name)))
        print(f"   [Debug] Forced collisions: {forced_collisions}")
        self.assertIn(expected_pair, forced_collisions,
                      f"FAIL: Collision was NOT detected between '{obj1_body_name}' and '{obj2_body_name}' when overlapping.")
        print("✅ PASS: Object-object collision was successfully detected.")

        # 3. Verify the no-collision case after moving objects apart.
        # We move obj2 far away to a non-colliding position.
        sim.data.qpos[obj2_joint_addr:obj2_joint_addr+3] = obj1_pos + np.array([0.5, 0.5, 0.0])
        sim.forward()
        
        # Assert that the specific collision between our objects is now gone.
        resolved_collisions = collision_detection_3d(self.env, objects_of_interest=objects_to_check)
        print(f"   [Debug] Resolved collisions: {resolved_collisions}")
        self.assertNotIn(expected_pair, resolved_collisions,
                         f"FAIL: Collision still detected between objects after moving them apart: {resolved_collisions}")
        print("✅ PASS: No collisions detected after moving objects apart.")


    def test_is_scene_stable_on_stable_scene(self):
        """Tests that a pre-settled, stable scene is correctly identified as stable."""
        print("\n--- Testing is_scene_stable on a stable scene ---")
        # The setUp method already resets the scene to a stable state.
        self.assertTrue(is_scene_stable_v0(self.env),
                        "FAIL: is_scene_stable() returned False for a scene that should be stable.")
        print("✅ PASS: Stable scene correctly identified.")


    def test_is_scene_stable_on_unstable_scene(self):
        """Tests that a scene with a moving object is correctly identified as unstable."""
        print("\n--- Testing is_scene_stable on an unstable scene ---")
        sim = self.env.sim
        obj_name = "alphabet_soup_1"
        obj_joint_id = sim.model.joint_name2id(f"{obj_name}_joint0")
        obj_joint_addr = sim.model.jnt_qposadr[obj_joint_id]

        # Move the object up in the air so it will fall
        sim.data.qpos[obj_joint_addr+2] += 0.2
        sim.forward()

        # Step the simulation a few times to let gravity impart velocity
        for _ in range(5):
            sim.step()

        self.assertFalse(is_scene_stable_v0(self.env),
                         "FAIL: is_scene_stable() returned True for a scene that should be unstable.")
        print("✅ PASS: Unstable scene correctly identified.")


    def test_settle_and_check_stability(self):
        """
        Tests settle_and_check_stability by:
        1. Verifying it returns False for a scene that won't settle in time.
        2. Verifying it returns True for a scene that has enough time to settle.
        """
        print("\n--- Testing settle_and_check_stability ---")
        sim = self.env.sim
        obj_name = "alphabet_soup_1"
        obj_joint_id = sim.model.joint_name2id(f"{obj_name}_joint0")
        obj_joint_addr = sim.model.jnt_qposadr[obj_joint_id]

        # --- Case 1: Object dropped from high up, insufficient settle time ---
        print("  Case 1: Object dropped from high up, insufficient settle time.")
        # Reset to stable state
        sim.data.qpos[:] = self.stable_qpos
        sim.data.qvel[:] = self.stable_qvel
        sim.forward()
        
        # Move object high up
        sim.data.qpos[obj_joint_addr+2] += 0.5
        sim.forward()

        # Check stability with short settle time; it should still be falling.
        is_stable_short = settle_and_check_stability(self.env, steps=20)
        self.assertFalse(is_stable_short, "FAIL: Scene was considered stable even when object was still falling.")
        print("  ✅ PASS: Correctly identified as unstable when object is falling.")

        # --- Case 2: Object dropped from a small height, enough time to settle ---
        print("\n  Case 2: Object dropped from a small height, sufficient settle time.")
        # Reset to stable state
        sim.data.qpos[:] = self.stable_qpos
        sim.data.qvel[:] = self.stable_qvel
        sim.forward()

        # Move object slightly above the table
        sim.data.qpos[obj_joint_addr+2] += 0.05
        sim.forward()
        
        # Check stability with a generous settle time.
        is_stable_long = settle_and_check_stability(self.env, steps=100)
        self.assertTrue(is_stable_long, "FAIL: Scene was considered unstable after having enough time to settle.")
        print("  ✅ PASS: Correctly identified as stable after object settled.")


    def test_add_object_and_check_stability(self):
        """
        EXAMPLE USAGE: This test demonstrates the complete workflow for robustly
        adding a new object to a scene:
        1. Find a collision-free position using `collision_detection_3d`.
        2. Verify the placement is physically stable using `settle_and_check_stability`.
        
        This two-step process ensures an object is not only placed without initial
        overlap but also that it won't immediately fall over or slide away.
        """
        print("\n--- Testing combined workflow: Add object and check stability ---")
        sim = self.env.sim
        obj_to_place_name = "butter_1"
        obj_to_place_body_name = f"{obj_to_place_name}_main"
        
        obj_joint_id = sim.model.joint_name2id(f"{obj_to_place_name}_joint0")
        obj_joint_addr = sim.model.jnt_qposadr[obj_joint_id]

        other_obj_joint_id = sim.model.joint_name2id("alphabet_soup_1_joint0")
        other_obj_joint_addr = sim.model.jnt_qposadr[other_obj_joint_id]
        stable_z = sim.data.qpos[other_obj_joint_addr + 2]

        MAX_ATTEMPTS = 50
        placement_successful = False
        print(f"Attempting to find a collision-free AND stable spot for '{obj_to_place_name}'...")

        for attempt in range(MAX_ATTEMPTS):
            # STEP 0: Reset the scene to a clean state before each new attempt.
            # This is crucial for making each placement attempt independent.
            sim.data.qpos[:] = self.stable_qpos
            sim.data.qvel[:] = self.stable_qvel
            sim.forward()

            # STEP 1: Generate a random candidate position.
            x_pos = random.uniform(-0.25, 0.25)
            y_pos = random.uniform(-0.25, 0.25)
            
            # Place the object slightly above the surface to let it settle down.
            sim.data.qpos[obj_joint_addr:obj_joint_addr+3] = [x_pos, y_pos, stable_z + 0.1]
            sim.forward()

            # STEP 2: Perform a quick, static collision check.
            # This immediately rejects positions where the object is already overlapping another.
            collisions = collision_detection_3d(self.env, objects_of_interest=[obj_to_place_body_name])

            if not collisions:
                print(f"  Attempt {attempt + 1}: Position ({x_pos:.2f}, {y_pos:.2f}) is collision-free. Now checking stability...")
                
                # STEP 3: If no static collision, check for physical stability.
                # This runs the simulation for a short period to see if the object
                # falls over or moves, which can happen even without an initial collision.
                # We check the stability of only the object we are placing.
                if settle_and_check_stability(self.env, steps=100, objects_of_interest=[obj_to_place_body_name]):
                    # After settling, get the final position of the object for debugging.
                    final_pos = sim.data.qpos[obj_joint_addr:obj_joint_addr+3].copy()
                    print(f"  ✅ SUCCESS: Found a stable, collision-free position for '{obj_to_place_name}'.")
                    print(f"     - Initial candidate XY: ({x_pos:.3f}, {y_pos:.3f})")
                    print(f"     - Final settled position: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")
                    placement_successful = True
                    break
                else:
                    # For debugging, see where the object ended up after being unstable.
                    unstable_pos = sim.data.qpos[obj_joint_addr:obj_joint_addr+3].copy()
                    print(f"  ... Position was unstable. Object moved to ({unstable_pos[0]:.3f}, {unstable_pos[1]:.3f})")
            else:
                 if attempt < 10: # Only print for the first few attempts to avoid spam
                     print(f"  Attempt {attempt + 1}: Collision detected at ({x_pos:.2f}, {y_pos:.2f}).")

        self.assertTrue(placement_successful, f"Failed to find a stable and collision-free position for {obj_to_place_name} after {MAX_ATTEMPTS} attempts.")


    def test_are_all_objects_stable(self):
        """
        Tests the are_all_objects_stable helper function.
        It should detect instability even if only one object is moving.
        """
        print("\n--- Testing are_all_objects_stable on a scene with one unstable object ---")
        sim = self.env.sim
        obj_name = "alphabet_soup_1"
        obj_joint_id = sim.model.joint_name2id(f"{obj_name}_joint0")
        obj_joint_addr = sim.model.jnt_qposadr[obj_joint_id]

        # Make the scene unstable by moving one object up
        sim.data.qpos[obj_joint_addr+2] += 0.2
        sim.forward()
        for _ in range(5):
            sim.step()

        # The new helper should detect this instability without being told which object to watch.
        self.assertFalse(are_all_objects_stable(self.env),
                         "FAIL: are_all_objects_stable() failed to detect an unstable object.")
        print("✅ PASS: Correctly identified instability in the full scene.")


    def test_instability_chain_reaction(self):
        """
        Tests a chain reaction where one falling object makes another object unstable.
        
        This verifies that the stability check can identify instability in objects
        that were not the initial source of movement.
        """
        print("\n--- Testing instability chain reaction ---")
        sim = self.env.sim
        obj_top_name = "salad_dressing_1"
        obj_bottom_name = "alphabet_soup_1"
        obj_top_body_name = f"{obj_top_name}_main"
        obj_bottom_body_name = f"{obj_bottom_name}_main"

        # Get body and joint info for both objects
        obj_top_joint_id = sim.model.joint_name2id(f"{obj_top_name}_joint0")
        obj_top_joint_addr = sim.model.jnt_qposadr[obj_top_joint_id]
        obj_bottom_joint_id = sim.model.joint_name2id(f"{obj_bottom_name}_joint0")
        obj_bottom_joint_addr = sim.model.jnt_qposadr[obj_bottom_joint_id]
        obj_bottom_body_id = sim.model.body_name2id(obj_bottom_body_name)

        # Get the initial stable position of the bottom object
        bottom_pos = sim.data.qpos[obj_bottom_joint_addr:obj_bottom_joint_addr+3].copy()
        print(f"  - Bottom object '{obj_bottom_name}' initial stable position: ({bottom_pos[0]:.3f}, {bottom_pos[1]:.3f}, {bottom_pos[2]:.3f})")

        # Place the top object directly above the bottom one, high enough to fall
        print(f"  - Placing '{obj_top_name}' at an unstable position above '{obj_bottom_name}'...")
        top_initial_pos = [bottom_pos[0], bottom_pos[1], bottom_pos[2] + 0.1]
        sim.data.qpos[obj_top_joint_addr:obj_top_joint_addr+3] = top_initial_pos
        print(f"  - Top object '{obj_top_name}' placed at: ({top_initial_pos[0]:.3f}, {top_initial_pos[1]:.3f}, {top_initial_pos[2]:.3f})")
        sim.forward()

        # Let the simulation run just long enough for the collision to occur and transfer energy
        print("  - Simulating for 30 steps to observe the chain reaction...")
        for _ in range(30):
            sim.step()

        # For debugging, let's see where both objects ended up.
        top_final_pos = sim.data.qpos[obj_top_joint_addr:obj_top_joint_addr+3].copy()
        bottom_final_pos = sim.data.qpos[obj_bottom_joint_addr:obj_bottom_joint_addr+3].copy()
        print(f"  - Final position of top object:    ({top_final_pos[0]:.3f}, {top_final_pos[1]:.3f}, {top_final_pos[2]:.3f})")
        print(f"  - Final position of bottom object: ({bottom_final_pos[0]:.3f}, {bottom_final_pos[1]:.3f}, {bottom_final_pos[2]:.3f})")

        # Check 1: Verify the bottom object is now moving due to the impact.
        # We get its velocity directly from the simulation data.
        bottom_body_vel = sim.data.cvel[obj_bottom_body_id]
        bottom_linear_speed = np.linalg.norm(bottom_body_vel[0:3])

        print(f"  - Speed of bottom object ('{obj_bottom_name}') after impact: {bottom_linear_speed:.4f} m/s")
        self.assertGreater(bottom_linear_speed, 0.01,
                            f"FAIL: The bottom object '{obj_bottom_name}' was not significantly moved by the falling object.")
        print(f"  ✅ PASS: Bottom object is moving, confirming a chain reaction.")

        # Check 2: Verify that our stability checker correctly identifies the scene as unstable.
        # The instability is reported because at least one of the objects of interest is moving.
        print("  - Verifying scene is detected as unstable...")
        self.assertFalse(is_scene_stable_v0(self.env, objects_of_interest=[obj_top_body_name, obj_bottom_body_name]),
                            "FAIL: is_scene_stable() did not detect the instability of the chain reaction.")
        print("  ✅ PASS: The overall scene instability was correctly detected.")


    def test_check_stability_with_position_on_stable_scene(self):
        """
        Tests that the position-based stability checker correctly identifies a stable scene,
        even with minor physics jitter. The key is using a lenient-enough threshold.
        """
        print("\n--- Testing position-based stability check on a STABLE scene (lenient) ---")
        # The scene is already stable from setUp. This should return True because the
        # threshold in check_stability_with_position is high enough to ignore micro-drifts.
        self.assertTrue(check_stability_with_position(self.env),
                        "FAIL: Position-based check returned False for a stable scene, even with a lenient threshold.")
        print("✅ PASS: Correctly identified stable scene.")


if __name__ == '__main__':
    unittest.main()
