"""
Read LIBERO object footprint as [len_x, len_y] (meters) from MJCF.
Directory layout (given):
  <root>/<object>/<object>.xml
Where roots are:
  .../libero/libero/assets/stable_hope_objects
  .../libero/libero/assets/stable_scanned_objects
  .../libero/libero/assets/turbosquid_objects
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict
import math
import numpy as np

# =========================
# Config
# =========================

# If your MJCF uses quat="w x y z" (matches your sample), keep True.
# If it uses "x y z w", set to False.
QUAT_IS_WXYZ = True

# Optional: map BDDL names to asset folder names if they differ.
ALIAS_MAP: Dict[str, str] = {
    # "alphabet_soup": "soup_can",
}

# Optional: fallback when XML can't yield a footprint (rare).
# Values are FULL sizes (len_x, len_y), not halfsizes.
# Based on typical real-world object dimensions
FALLBACK_FOOTPRINT: Dict[str, Tuple[float, float]] = {
    # Cans/cylinders (soups, sauces)
    "alphabet_soup": (0.065, 0.065),  # Soup can ~6.5cm diameter
    "tomato_sauce": (0.065, 0.065),
    "bbq_sauce": (0.060, 0.060),
    "salad_dressing": (0.070, 0.070),
    
    # Boxes/cubes (dairy, etc)
    "butter": (0.10, 0.08),  # Butter box
    "cream_cheese": (0.08, 0.08),  # Cream cheese box
    "milk": (0.08, 0.08),  # Milk carton
    "chocolate_pudding": (0.09, 0.06),
    "cookies": (0.12, 0.09),
    "macaroni_and_cheese": (0.10, 0.08),
    "orange_juice": (0.08, 0.08),
    "popcorn": (0.10, 0.08),
    
    # Bottles
    "ketchup": (0.065, 0.065),
    "wine_bottle": (0.075, 0.075),
    
    # Books
    "black_book": (0.15, 0.12),
    "yellow_book": (0.15, 0.12),
    
    # Bowls/mugs
    "red_bowl": (0.12, 0.12),
    "akita_black_bowl": (0.12, 0.12),
    "glazed_rim_porcelain_ramekin": (0.10, 0.10),
    "porcelain_mug": (0.08, 0.08),
    "red_coffee_mug": (0.08, 0.08),
    "white_yellow_mug": (0.08, 0.08),
    
    # Plates
    "plate": (0.20, 0.20),
    "blue_plate": (0.20, 0.20),
    "red_plate": (0.20, 0.20),
    
    # Containers
    "basket": (0.20, 0.15),
    "bowl_drainer": (0.25, 0.20),
    "desk_caddy": (0.10, 0.10),
    "chefmate_8_frypan": (0.20, 0.20),
    "wooden_tray": (0.25, 0.20),
}

COMPLEX_OBJECT_MAP =  {"basin_faucet": "../../Utility/discardable/BASIN_FAUCET_visualize_all_available_object.bddl", 
                       "faucet": "", 
                       "flat_stove": "../../Utility/discardable/FLAT_STOVE_visualize_all_available_object.bddl", 
                       "microwave": "../../Utility/discardable/MICROWAVE_visualize_all_available_object.bddl",
                       "short_cabinet": "../../Utility/discardable/SHORT_CABINET_visualize_all_available_object.bddl", 
                       "short_fridge": "../../Utility/discardable/SHORT_FRIDGE_visualize_all_available_object.bddl", 
                       "slide_cabinet": "../../Utility/discardable/SLIDE_CABINET_visualize_all_available_object.bddl", 
                       "white_cabinet": "../../Utility/discardable/WHITE_CABINET_visualize_all_available_object.bddl", 
                       "window": "../../Utility/discardable/WINDOW_visualize_all_available_object.bddl", 
                       "wooden_cabinet": "../../Utility/discardable/WOODEN_CABINET_visualize_all_available_object.bddl"}

# =========================
# Name utils
# =========================

_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<idx>\d+)$")

def strip_instance_suffix(name: str) -> str:
    """Strip only a trailing _<digits> (e.g., 'mug_12' -> 'mug')."""
    m = _SUFFIX_RE.match(name)
    return m.group("base") if m else name

def parse_floats(s: Optional[str]) -> List[float]:
    if not s: return []
    return [float(x) for x in s.replace(",", " ").split() if x.strip()]

# =========================
# Paths
# =========================

def find_assets_roots() -> List[str]:
    """
    Find the three object-root directories relative to this file:
    <project_root>/libero/libero/assets/<category>_objects
    Returns the subset that actually exists; raises if none exist.
    """
    import os
    current_dir = os.path.dirname (os.path.abspath (__file__))

    # 回溯到 LIBERO 根目录（TaskGen 同级）
    project_root = os.path.abspath (os.path.join (current_dir, "../.."))


    candidates = [
        os.path.join(project_root, "libero", "libero", "assets", "stable_hope_objects"),
        os.path.join(project_root, "libero", "libero", "assets", "stable_scanned_objects"),
        os.path.join(project_root, "libero", "libero", "assets", "turbosquid_objects"),
    ]
    roots = [p for p in candidates if os.path.isdir(p)]
    if not roots:
        raise FileNotFoundError(
            "No LIBERO asset roots found under:\n  " + "\n  ".join(candidates)
        )
    return roots

def find_assets_xml_files(object):
    current_dir = os.path.dirname (os.path.abspath (__file__))

    project_root = os.path.abspath (os.path.join (current_dir, "../.."))

    path = os.path.join(project_root, "libero", "libero", "assets", "articulated_objects", f"{object}.xml")
    return path

def resolve_object_xml(obj_type: str, roots: List[str]) -> Optional[str]:
    """
    Strict layout: <root>/<obj>/<obj>.xml
    Returns first existing path or None.
    """
    if obj_type in COMPLEX_OBJECT_MAP:
        p = find_assets_xml_files(obj_type)
        if os.path.isfile(p):
            return p
    for r in roots:
        p = os.path.join(r, obj_type, f"{obj_type}.xml")
        if os.path.isfile(p):
            return p
    return None

# =========================
# Geometry helpers
# =========================


import numpy as np
import math
import xml.etree.ElementTree as ET

def parse_quat(q_str):
    if not q_str:
        return np.array([1, 0, 0, 0], dtype=float)
    vals = [float(x) for x in q_str.split()]
    if len(vals) == 4:
        return np.array(vals, dtype=float)
    return np.array([1, 0, 0, 0], dtype=float)

def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])

def apply_transform(pos_parent, quat_parent, pos_child, quat_child):
    """
    combine body and geom contained in the body
    """
    R_p = quat_to_mat(quat_parent)
    p_global = pos_parent + R_p.dot(pos_child)
    # Hamilton multiplication
    w1,x1,y1,z1 = quat_parent
    w2,x2,y2,z2 = quat_child
    quat_global = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    return p_global, quat_global

def get_geom_aabb(geom_elem, pos_global, quat_global):
    from math import cos, sin
    half = geom_xy_half(geom_elem)
    if not half:
        return None
    cx, cy = pos_global[:2]
    hx, hy = half
    R = quat_to_mat(quat_global)
    xs = (-hx, hx)
    ys = (-hy, hy)
    minx = miny = 1e9
    maxx = maxy = -1e9
    for xi in xs:
        for yi in ys:
            X = R[0,0]*xi + R[0,1]*yi
            Y = R[1,0]*xi + R[1,1]*yi
            minx = min(minx, X)
            maxx = max(maxx, X)
            miny = min(miny, Y)
            maxy = max(maxy, Y)
    return (cx + minx, cy + miny), (cx + maxx, cy + maxy)

def accumulate_aabb(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No <worldbody> found in {xml_path}")

    world_min = None
    world_max = None

    def recurse_body(body_elem, pos_parent, quat_parent):
        nonlocal world_min, world_max
        pos = np.array([0.,0.,0.], dtype=float)
        if body_elem.get("pos"):
            pos = np.array([float(x) for x in body_elem.get("pos").split()])
        quat = parse_quat(body_elem.get("quat"))
        pos_global, quat_global = apply_transform(pos_parent, quat_parent, pos, quat)

        # go through geoms in this body
        for geom in body_elem.findall("geom"):
            aabb = get_geom_aabb(geom, pos_global, quat_global)
            if aabb:
                (xmin, ymin), (xmax, ymax) = aabb
                if world_min is None:
                    world_min, world_max = (xmin, ymin), (xmax, ymax)
                else:
                    world_min = (min(world_min[0], xmin), min(world_min[1], ymin))
                    world_max = (max(world_max[0], xmax), max(world_max[1], ymax))

        # recursively update boundary
        for child in body_elem.findall("body"):
            recurse_body(child, pos_global, quat_global)

    recurse_body(worldbody, np.zeros(3), np.array([1,0,0,0]))

    if world_min is None:
        raise RuntimeError(f"No geom found in {xml_path}")
    return [world_max[0] - world_min[0], world_max[1] - world_min[1]]


def _aabb_from_center_len_yaw(cx, cy, len_x, len_y, yaw):
    hx, hy = len_x/2.0, len_y/2.0
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    hx_eff = c*hx + s*hy
    hy_eff = s*hx + c*hy
    return (cx - hx_eff, cy - hy_eff), (cx + hx_eff, cy + hy_eff)

def quat_to_rot_matrix(q: List[float]) -> List[List[float]]:
    if QUAT_IS_WXYZ:
        w, x, y, z = q[0], q[1], q[2], q[3]
    else:
        x, y, z, w = q[0], q[1], q[2], q[3]
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    return [
        [ww + xx - yy - zz, 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     ww - xx + yy - zz, 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     ww - xx - yy + zz],
    ]

def rotate_box_xy_half(hx: float, hy: float, hz: float, quat: Optional[List[float]]) -> Tuple[float, float]:
    """Rotate an axis-aligned box (half-sizes) by quat, project to XY, return half-sizes."""
    if quat is None:
        return abs(hx), abs(hy)
    R = quat_to_rot_matrix(quat)
    xs = (-hx, hx); ys = (-hy, hy); zs = (-hz, hz)
    minx = miny = 1e9; maxx = maxy = -1e9
    for xi in xs:
        for yi in ys:
            for zi in zs:
                X = R[0][0]*xi + R[0][1]*yi + R[0][2]*zi
                Y = R[1][0]*xi + R[1][1]*yi + R[1][2]*zi
                minx = min(minx, X); maxx = max(maxx, X)
                miny = min(miny, Y); maxy = max(maxy, Y)
    return (maxx - minx)/2.0, (maxy - miny)/2.0

def is_pure_visual(geom: ET.Element) -> bool:
    """
    Heuristics: ignore visual-only geoms.
    - group="1" OR (contype="0" and conaffinity="0")
    """
    contype = geom.get("contype")
    conaff  = geom.get("conaffinity")
    group   = geom.get("group")
    return (group == "1") or (contype == "0" and conaff == "0")

def geom_xy_half(geom: ET.Element) -> Optional[Tuple[float, float]]:
    """
    Return (hx, hy) for one <geom>, projected to XY (consider local quat).
    Types supported: box / cylinder / capsule / sphere / ellipsoid / mesh / convex
    Ignored: plane / hfield and pure-visual geoms.
    """
    typ = (geom.get("type") or "box").lower()
    if typ in ("plane", "hfield"):
        return None
    if is_pure_visual(geom):
        return None

    size_attr  = geom.get("size")
    scale_attr = geom.get("scale") or geom.get("scales")
    quat_attr  = geom.get("quat")
    quat = parse_floats(quat_attr) if quat_attr else None

    if typ == "box":
        if not size_attr: return None
        sx, sy, sz = parse_floats(size_attr)[:3]
        return rotate_box_xy_half(abs(sx), abs(sy), abs(sz), quat)

    if typ in ("cylinder", "capsule"):
        if not size_attr: return None
        vals = parse_floats(size_attr)
        r = abs(vals[0])
        h = abs(vals[1]) if len(vals) > 1 else r
        return rotate_box_xy_half(r, r, h/2.0, quat)

    if typ == "sphere":
        if not size_attr: return None
        r = abs(parse_floats(size_attr)[0])
        return (r, r)

    if typ == "ellipsoid":
        if not size_attr: return None
        a, b, c = parse_floats(size_attr)[:3]
        return rotate_box_xy_half(abs(a), abs(b), abs(c), quat)

    if typ in ("mesh", "convex"):
        # Use only if it participates in collision and has a scale/size.
        if not (scale_attr or size_attr):
            return None
        sx, sy, sz = (parse_floats(scale_attr) or parse_floats(size_attr))[:3]
        return rotate_box_xy_half(abs(sx), abs(sy), abs(sz), quat)

    # Fallback: if unknown type but has size, treat as box
    if size_attr:
        sx, sy, sz = parse_floats(size_attr)[:3]
        return rotate_box_xy_half(abs(sx), abs(sy), abs(sz), quat)
    return None

def accumulate_world_xy_aabb(world_min: Optional[Tuple[float,float]],
                             world_max: Optional[Tuple[float,float]],
                             center_xy: Tuple[float,float],
                             half_xy: Tuple[float,float]) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    cx, cy = center_xy
    hx, hy = half_xy
    xmin = cx - hx; xmax = cx + hx
    ymin = cy - hy; ymax = cy + hy
    if world_min is None:
        return (xmin, ymin), (xmax, ymax)
    return (min(world_min[0], xmin), min(world_min[1], ymin)), \
           (max(world_max[0], xmax), max(world_max[1], ymax))


def collision_detection_2d(new_min, new_max, existing_objects, margin=0.02):
	"""
	In dense layout setting, we need to check whether two objects overlap in 2D top-down view.
	I.e. project the 3D bounding box to 2D plane, and check whether two 2D bounding boxes overlap.
	"""
	x_min, y_min = new_min[0] - margin, new_min[1] - margin
	x_max, y_max = new_max[0] + margin, new_max[1] + margin

	for obj in existing_objects:
		ox_min, oy_min = obj["min"]
		ox_max, oy_max = obj["max"]

		if not (x_max < ox_min or x_min > ox_max or y_max < oy_min or y_min > oy_max):
			return True, obj["name"]
	return False, None


# =========================
# Main API
# =========================
fixure={"wooden_cabinet","flat_stove","moka_pot"}

def get_object_size(object_or_type: str) -> List[float]:
    """
    Return [len_x, len_y] (meters), the top-down footprint of the object.
    - Strips trailing _<digits>, applies alias, then loads <root>/<obj>/<obj>.xml
    - Parses collision-relevant geoms (consider pos & quat), merges world XY AABB.
    - Falls back to FALLBACK_FOOTPRINT if assets not found
    """
    base = strip_instance_suffix(object_or_type)
    obj = ALIAS_MAP.get(base, base)


    #Zhixu: @tianze, here it cannot process the "wooden_cabinet"
    # Try to find XML assets, but fall back to FALLBACK_FOOTPRINT if not found
    try:
        roots = find_assets_roots()
        xml_path = resolve_object_xml(obj, roots)
        if not xml_path:
            fb = FALLBACK_FOOTPRINT.get(obj)
            if fb:
                return [float(fb[0]), float(fb[1])]
            return None
    except FileNotFoundError:
        # Asset directories not found, use fallback
        fb = FALLBACK_FOOTPRINT.get(obj)
        if fb:
            return [float(fb[0]), float(fb[1])]
        raise FileNotFoundError(
            f"No assets found and no fallback defined for '{obj}'"
        )
    
    if obj in COMPLEX_OBJECT_MAP:
        return accumulate_aabb(xml_path)

    root = ET.parse(xml_path).getroot()
    world_min = world_max = None
    used_any = False
    for geom in root.iter("geom"):
        half = geom_xy_half(geom)
        if not half:
            continue
        pos = parse_floats(geom.get("pos"))
        cx = float(pos[0]) if len(pos) >= 1 else 0.0
        cy = float(pos[1]) if len(pos) >= 2 else 0.0
        world_min, world_max = accumulate_world_xy_aabb(world_min, world_max, (cx, cy), half)
        used_any = True

    if used_any and world_min and world_max:
        lx = world_max[0] - world_min[0]
        ly = world_max[1] - world_min[1]
        return [abs(float(lx)), abs(float(ly))]

    fb = FALLBACK_FOOTPRINT.get(obj)
    if fb:
        return [float(fb[0]), float(fb[1])]
    raise RuntimeError(
        f"No usable collision geoms for '{obj}' in {xml_path} "
        f"(likely only has visual meshes without size/scale)."
    )


def get_object_size_3d (env,object_or_type: str) -> List [float]:
    """
    Return [len_x, len_y] (meters), the top-down footprint of the object.
    - Strips trailing _<digits>, applies alias, then loads <root>/<obj>/<obj>.xml
    - Parses collision-relevant geoms (consider pos & quat), merges world XY AABB.
    - Falls back to FALLBACK_FOOTPRINT if assets not found
    """
    if object_or_type.endswith("_main"):
        object_or_type=object_or_type[:-5]
    x,y=get_object_size (object_or_type)

    if not object_or_type.endswith("_main") and object_or_type.endswith("_1"):
        object_or_type = object_or_type + "_main"
    target_xpos = env.sim.data.get_body_xpos (object_or_type).copy ()
    half_z=target_xpos[2]
    if half_z>0.8:
        half_z-=0.8
    elif half_z>0.42:
        half_z-=0.42

    return [x,y,half_z*2]



