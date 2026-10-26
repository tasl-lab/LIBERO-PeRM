from robosuite.utils.binding_utils import MjSim, MjSimState
import h5py
import numpy as np

def load_xml_and_states(hdf5_path, demo_key = "demo_0"):
    with h5py.File(hdf5_path, "r") as f:
        model_xml = f[f"data/{demo_key}"].attrs["model_file"]
        flat_states = np.array(f["data/demo_0/states"][:])
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode("utf-8") 
    return model_xml, flat_states

def load_actions(hdf5_path, demo_key = "demo_0"):
    with h5py.File(hdf5_path, "r") as f:
        actions = np.array(f[f"data/{demo_key}/actions"][:])
    return actions

def load_obs_dict(hdf5_path, demo_key = "demo_0"):
    with h5py.File(hdf5_path, "r") as f:
        obs_dict = {}
        for key in f[f"data/{demo_key}/obs"].keys():
            obs_dict[key] = np.array(f[f"data/demo_0/obs/{key}"][:])
    return obs_dict

def build_sim_from_xml(model_xml):
    sim = MjSim.from_xml_string(model_xml)
    return sim

def set_flatten_state(sim, flat_state):
    sim.set_state(MjSimState.from_flattened(flat_state, sim))
    sim.forward()

class HDF5_Player():
    def __init__(self, hdf5_path):
        self.model_xml, self.flat_states = load_xml_and_states(hdf5_path)
        self.sim = build_sim_from_xml(self.model_xml)
        self.obs = load_obs_dict(hdf5_path)
        self.actions = load_actions(hdf5_path)
        self.length = self.flat_states.shape[0]
    
    def get_body_pos_at_time(self, body_name, t):
        bid = self.sim.model.body_name2id(body_name)
        set_flatten_state(self.sim, self.flat_states[t])
        return self.sim.data.body_xpos[bid]
    
    def get_ee_state(self, t):
        return self.obs["ee_states"][t]
    
    def get_action(self, t):
        return self.actions[t]

    def get_agentview(self, t):
        return self.obs["agentview_rgb"][t]

    def get_frontview(self, t):
        return self.obs["frontview_rgb"][t]
