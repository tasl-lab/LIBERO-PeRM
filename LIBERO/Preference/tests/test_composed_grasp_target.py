"""Grasp scoring must follow object choice without selecting by score."""
from types import SimpleNamespace
import numpy as np
from Preference.low_level.skill_pose import GraspOffsetPreference


class Data:
    def __init__(self):
        self.pos = {'gripper0_eef': np.array([0.97, 0., 0.]),
                    'base_main': np.array([0., 0., 0.]),
                    'yellow_main': np.array([1., 0., 0.])}
        self.gap = .08

    def get_body_xpos(self, name):
        return self.pos[name]

    def get_body_xmat(self, name):
        return np.eye(3)

    def get_joint_qpos(self, name):
        return self.gap/2 if name.endswith('joint1') else -self.gap/2


def setup(multiple=True):
    data = Data()
    env = SimpleNamespace(sim=SimpleNamespace(data=data), robots=[])
    pref = GraspOffsetPreference(target_obj='base', target_objs=['base', 'yellow'] if multiple else None,
        axis='x', frame='world', signed=True, criterion_threshold=-.02, criterion_direction=-1)
    pref.reset(env)
    return data, env, pref


def lift(data, env, pref, obj):
    pref.get_reward(env)
    data.gap = .02
    pref.get_reward(env)
    data.pos[obj+'_main'][2] = .03
    pref.get_reward(env)


def test_composed_object_change_fires_same_grasp_criterion():
    data, env, pref = setup()
    lift(data, env, pref, 'yellow')
    assert pref.target_obj == 'yellow'
    assert np.isclose(pref.final_raw(), -.03)
    assert pref.satisfaction(pref.final_raw()) == 100
    data, env, original = setup(False)
    lift(data, env, original, 'yellow')
    assert original.final_raw() is None


def test_first_grasp_is_scored_even_when_later_object_would_pass():
    data, env, pref = setup()
    data.pos['gripper0_eef'][0] = .04
    lift(data, env, pref, 'base')
    data.pos['gripper0_eef'][0] = .97
    data.pos['yellow_main'][2] = .03
    pref.get_reward(env)
    assert pref.target_obj == 'base'
    assert pref.satisfaction(pref.final_raw()) == 0


def test_reset_does_not_keep_previous_object_choice():
    data, env, pref = setup()
    lift(data, env, pref, 'yellow')
    data.pos['yellow_main'][2] = 0
    data.pos['gripper0_eef'][0] = -.03
    data.gap = .08
    pref.reset(env)
    assert pref.final_raw() is None
    lift(data, env, pref, 'base')
    assert pref.target_obj == 'base'
    assert pref.satisfaction(pref.final_raw()) == 100
