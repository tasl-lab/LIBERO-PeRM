"""Unit tests for MotionGen.gen_metadata.GenMetadataCollector — pure logic
only (no sim): plan attribution, boundary bookkeeping, gripper-event scan,
speed derivation, language templating, final_raw passthrough."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from MotionGen.gen_metadata import GenMetadataCollector, _subtask_language


def _fake_hdf5_dict(n, low_pref_rewards=None):
    t = np.arange(n, dtype=float)
    ee = np.stack([t * 0.01, np.zeros(n), np.ones(n)], axis=1)  # 1cm/step in x
    actions = np.zeros((n, 7))
    return {
        "obs": {"ee_pos": ee, "ee_ori": np.zeros((n, 3))},
        "actions": list(actions),
        "low_pref_rewards": low_pref_rewards or {},
    }


PLAN = [["pick", "butter_1"], ["put", "basket_1"]]


class TestPlanAttribution:
    def test_move_to_belongs_to_next_plan_entry(self):
        c = GenMetadataCollector(task_plan=PLAN)
        c.on_segment("move_to", "butter_1", 10)
        c.on_segment("pick", "butter_1", 30)
        c.on_segment("move_to", "basket_1", 45)
        c.on_segment("put", "basket_1", 60)
        idx = [s["plan_idx"] for s in c._segments]
        assert idx == [0, 0, 1, 1]

    def test_sim_name_conversion_still_consumes(self):
        # loop tail often passes to_sim_env_name-converted targets
        c = GenMetadataCollector(task_plan=[["pick", "butter_1"], ["put", "basket_1"]])
        from Utility.tool_func import to_sim_env_name
        c.on_segment("pick", to_sim_env_name("butter_1"), 20)
        assert c._plan_ptr == 1

    def test_empty_segment_ignored(self):
        c = GenMetadataCollector(task_plan=PLAN)
        c.on_segment("move_to", "butter_1", 0)
        assert c._segments == []

    def test_dynamic_skills_inherit_current_plan_idx(self):
        c = GenMetadataCollector(task_plan=PLAN)
        c.on_segment("lift_up", None, 5)           # avoid-rewrite inserts these
        c.on_segment("move_to_away", "butter_1", 15)
        c.on_segment("pick", "butter_1", 30)
        assert [s["plan_idx"] for s in c._segments] == [0, 0, 0]
        assert c._plan_ptr == 1


class TestFinalize:
    def test_speed_and_goal_image(self):
        c = GenMetadataCollector(task_plan=PLAN)
        c.on_segment("pick", "butter_1", 10)
        c.on_segment("put", "basket_1", 20)
        meta = c.finalize(_fake_hdf5_dict(20))
        segs = meta["segments"]
        assert len(segs) == 2
        assert abs(segs[0]["avg_speed"] - 0.01) < 1e-9     # 1cm/step fake data
        assert segs[0]["goal_image_step"] == 9
        assert meta["subtasks"][0]["language"] == "pick up the butter"
        assert meta["subtasks"][1]["language"] == "put the butter on the basket"
        assert meta["num_steps"] == 20

    def test_bounds_clamped_to_recorded_steps(self):
        c = GenMetadataCollector(task_plan=PLAN)
        c.on_segment("pick", "butter_1", 25)   # claims more than recorded
        meta = c.finalize(_fake_hdf5_dict(20))
        assert meta["segments"][-1]["end"] == 20

    def test_gripper_events_and_release_height(self):
        c = GenMetadataCollector(task_plan=PLAN)
        c.on_segment("pick", "butter_1", 6)
        d = _fake_hdf5_dict(6)
        g = [-1, -1, 1, 1, -1, -1]  # close at t=2, open at t=4
        d["actions"] = [np.array([0, 0, 0, 0, 0, 0, gi], dtype=float) for gi in g]
        meta = c.finalize(d)
        ev = meta["events"]
        assert ev["gripper_close_steps"] == [2]
        assert ev["gripper_open_steps"] == [4]
        assert abs(ev["releases"][0]["height"] - 1.0) < 1e-9  # fake ee z = 1.0

    def test_final_raw_uses_pref_objects(self):
        class FakePref:
            preference_type = "speed_range"
            def final_raw(self, mean_reward=None):
                return 123.0
        c = GenMetadataCollector(task_plan=PLAN)
        c.on_segment("pick", "butter_1", 5)
        d = _fake_hdf5_dict(5, low_pref_rewards={"speed_range": [1.0] * 5})
        meta = c.finalize(d, low_pref2apply=[FakePref()])
        assert meta["low_pref_final_raw"]["speed_range"] == 123.0

    def test_avoid_waypoints_recorded(self):
        c = GenMetadataCollector(task_plan=PLAN)
        c.record_avoid_waypoint(np.array([0.1, 0.2, 0.3]))
        c.on_segment("pick", "butter_1", 5)
        meta = c.finalize(_fake_hdf5_dict(5))
        assert meta["low_prefs_applied"]["avoid_waypoints"] == [[0.1, 0.2, 0.3]]


class TestLanguage:
    @pytest.mark.parametrize("skill,target,carried,expected", [
        ("pick", "butter_1", None, "pick up the butter"),
        ("pick_right_edge", "akita_black_bowl_1", None,
         "pick up the akita black bowl by its right edge"),
        ("pick_from_side", "porcelain_mug_1", None,
         "pick up the porcelain mug from the side"),
        ("put", "plate_1", "butter_1", "put the butter on the plate"),
        ("put", "white_cabinet_1_cabinet_bottom", "akita_black_bowl_1",
         "put the akita black bowl in the bottom drawer of the white cabinet"),
        ("put_right_edge", "white_cabinet_1_cabinet_bottom", "akita_black_bowl_1",
         "put the akita black bowl at the right edge of the bottom drawer of the white cabinet"),
        ("open_drawer", "white_cabinet_1", None, "open the white cabinet"),
        ("open_drawer", "wooden_cabinet_1_cabinet_top", None,
         "open the top drawer of the wooden cabinet"),
        ("turn_on_stove", "flat_stove_1", None, "turn on the flat stove"),
    ])
    def test_templates(self, skill, target, carried, expected):
        assert _subtask_language(skill, target, carried) == expected

    def test_stored_language_is_pref_agnostic(self):
        # variant adverbials never reach meta["subtasks"][i]["language"];
        # the actual plan target (region suffix) is kept (2026-07-13 decision)
        plan = [["pick_right_edge", "akita_black_bowl_1"],
                ["put_right_edge", "white_cabinet_1_cabinet_bottom"]]
        c = GenMetadataCollector(task_plan=plan)
        c.on_segment("pick_right_edge", "akita_black_bowl_1", 10)
        c.on_segment("put_right_edge", "white_cabinet_1_cabinet_bottom", 20)
        meta = c.finalize(_fake_hdf5_dict(20))
        assert meta["subtasks"][0]["language"] == "pick up the akita black bowl"
        assert meta["subtasks"][1]["language"] == (
            "put the akita black bowl in the bottom drawer of the white cabinet")
        # skill field still records the variant for anyone who needs it
        assert meta["subtasks"][0]["skill"] == "pick_right_edge"


class TestPrefCritical:
    def _finalize(self, plan, naive, low=None, high=None, with_moves=False):
        c = GenMetadataCollector(task_plan=plan, naive_plan=naive,
                                 low_pref_entries=low or [],
                                 high_pref_desc=high)
        end = 0
        for skill, obj in plan:
            if with_moves:
                end += 4
                c.on_segment("move_to", obj, end)
            end += 6
            c.on_segment(skill, obj, end)
        return c.finalize(_fake_hdf5_dict(end))

    def test_low_pref_transport_plus_plan_diff(self):
        # 1-4_034 shape: avoid + edge variants → both subtasks critical
        # (edge variants count as pref expression, 2026-07-02 decision)
        meta = self._finalize(
            [["pick_right_edge", "akita_black_bowl_1"],
             ["put_right_edge", "white_cabinet_1_cabinet_bottom"]],
            [["pick", "akita_black_bowl_1"], ["put", "white_cabinet_1"]],
            low=[{"type": "avoid_to_object", "description": "keep away"}])
        assert [s["pref_critical"] for s in meta["subtasks"]] == [1, 1]
        assert meta["subtasks"][0]["pref_critical_by_pref"] == {"avoid_to_object": 1}

    def test_move_scope_pref_marks_every_move_phase(self):
        # schema v4: avoid/height/rotation are "move" scope — gen shapes ALL
        # move_to's (approach included), so every MOVE phase is critical and
        # no MANIPULATION phase is
        meta = self._finalize(
            [["pick", "bowl_1"], ["put", "plate_1"]],
            [["pick", "bowl_1"], ["put", "plate_1"]],
            low=[{"type": "avoid_to_object", "description": "d"}],
            with_moves=True)
        assert [(s["phase"], s["pref_critical"]) for s in meta["subtasks"]] == [
            ("move", 1), ("manipulate", 0), ("move", 1), ("manipulate", 0)]

    def test_transport_scope_pref_marks_carry_move_only(self):
        # straight_path stays "transport": the gen knob only fires with an
        # object in hand, so just the carrying MOVE phase (put step) is marked
        meta = self._finalize(
            [["pick", "bowl_1"], ["put", "plate_1"]],
            [["pick", "bowl_1"], ["put", "plate_1"]],
            low=[{"type": "straight_path", "description": "d"}],
            with_moves=True)
        assert [(s["phase"], s["pref_critical"]) for s in meta["subtasks"]] == [
            ("move", 0), ("manipulate", 0), ("move", 1), ("manipulate", 0)]

    def test_move_phase_split_language_and_bounds(self):
        meta = self._finalize(
            [["pick_right_edge", "akita_black_bowl_1"],
             ["put_right_edge", "white_cabinet_1_cabinet_bottom"]],
            [["pick", "akita_black_bowl_1"], ["put", "white_cabinet_1"]],
            low=[{"type": "avoid_to_object", "description": "keep away"}],
            with_moves=True)
        subs = meta["subtasks"]
        assert [(s["plan_idx"], s["phase"], s["skill"]) for s in subs] == [
            (0, "move", "move_to"), (0, "manipulate", "pick_right_edge"),
            (1, "move", "move_to"), (1, "manipulate", "put_right_edge")]
        assert subs[0]["language"] == "move to the akita black bowl"
        # plain "move to X" even while carrying — the carried object was
        # already named by the preceding pick subtask
        assert subs[2]["language"] == ("move to the bottom drawer of the "
                                       "white cabinet")
        assert [(s["start"], s["end"]) for s in subs] == [
            (0, 4), (4, 10), (10, 14), (14, 20)]
        # diff flags both phases of both steps; transport adds the carry move
        assert [s["pref_critical"] for s in subs] == [1, 1, 1, 1]

    def test_global_scope_marks_all(self):
        meta = self._finalize(
            [["pick", "bowl_1"], ["put", "plate_1"]],
            [["pick", "bowl_1"], ["put", "plate_1"]],
            low=[{"type": "speed_range", "description": "d"}])
        assert [s["pref_critical"] for s in meta["subtasks"]] == [1, 1]

    def test_high_pref_object_substitution(self):
        # 3-12_036_2 shape: pick a different object; put target only refined
        # (white_cabinet_1 → …_cabinet_bottom is granularity, not pref)
        meta = self._finalize(
            [["pick", "small_orange_juice_1"],
             ["put", "white_cabinet_1_cabinet_bottom"]],
            [["pick", "wine_bottle_1"], ["put", "white_cabinet_1"]],
            high="Prefer boxed objects")
        assert [s["pref_critical_by_pref"]["task_pref"]
                for s in meta["subtasks"]] == [1, 0]
        assert meta["high_pref"] == {"description": "Prefer boxed objects"}
        assert meta["naive_plan"] == [["pick", "wine_bottle_1"],
                                      ["put", "white_cabinet_1"]]

    def test_inserted_subtask_is_diff(self):
        meta = self._finalize(
            [["open_drawer", "cab_1"], ["pick", "bowl_1"],
             ["put", "cab_1_cabinet_top"]],
            [["pick", "bowl_1"], ["put", "cab_1"]],
            high="open the drawer first")
        assert [s["pref_critical_by_pref"]["task_pref"]
                for s in meta["subtasks"]] == [1, 0, 0]

    def test_empty_high_pref_desc_treated_as_absent(self):
        # pipeline passes pref_description="" for series-1 missions; the diff
        # must still be attributed to the low pref keys, not task_pref
        meta = self._finalize(
            [["pick_right_edge", "bowl_1"], ["put", "plate_1"]],
            [["pick", "bowl_1"], ["put", "plate_1"]],
            low=[{"type": "avoid_to_object", "description": "d"}],
            high="")
        assert meta["subtasks"][0]["pref_critical_by_pref"] == {"avoid_to_object": 1}
        assert "task_pref" not in meta["subtasks"][0]["pref_critical_by_pref"]
        assert meta["high_pref"] is None

    def test_no_pref_all_zero(self):
        meta = self._finalize(
            [["pick", "bowl_1"], ["put", "plate_1"]],
            [["pick", "bowl_1"], ["put", "plate_1"]])
        assert all(s["pref_critical"] == 0 for s in meta["subtasks"])
        assert all(s["pref_critical_by_pref"] == {} for s in meta["subtasks"])
