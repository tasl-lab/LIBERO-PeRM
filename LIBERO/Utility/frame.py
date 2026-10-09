def get_img_from_obs(obs,key):
    if key not in obs.keys():
        raise ValueError(f"The key {key} is not in the observation dict")
    else:
        img=obs[key][::-1, ::-1]
    return img

