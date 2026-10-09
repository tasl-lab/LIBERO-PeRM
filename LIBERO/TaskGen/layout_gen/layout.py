"""
First we generate a naive layout, or just use LIBERO existing layout.
FYI: a bddl file is equivalent to a layout file.
Then we use the naive layout to generate more layouts, in which there will be more personalized tasks.
E.g. If we have a yellow book in naive layout, we can generate a layout with a yellow book, a black book and a box.
"""

import os
import copy
import logging
import numpy as np
import math
import re

logger = logging.getLogger("MotionGen")
from typing import Tuple, Dict, List, Optional
import libero.libero.envs.bddl_utils as BDDLUtils
from TaskGen.layout_gen.utils import (
    get_object_size,
    strip_instance_suffix,
    _aabb_from_center_len_yaw,
    collision_detection_2d,
)
from Utility.frame import get_img_from_obs
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)

from libero.libero.utils.task_generation_utils import (
    register_task_info,
    get_task_info,
    generate_bddl_from_task_info,
)


def _to_float(x):
    if isinstance(x, (tuple, list)):
        return float(x[0])
    return float(x)


class Layout:
    """
    Represents a robot task environment layout with objects and their positions.
    
    IMPORTANT LIMITATION: This class currently only supports ONE instance per object type.
    The object_init_position dict uses base object names (e.g., "milk") as keys, not 
    instance names (e.g., "milk_1", "milk_2"). If a BDDL file contains multiple instances
    of the same type (e.g., milk_1, milk_2), only the LAST instance's position is stored.
    
    TODO: Redesign object_init_position to support multiple instances:
          - Change from Dict[str, List[float]] to Dict[str, Dict[int, List[float]]]
          - Example: {"milk": {1: [x1, y1, yaw1], 2: [x2, y2, yaw2]}}
          - Update all methods (object_add, object_remove, object_rotate) to handle instance IDs
    """
    def __init__(self, bddl_file: str = None, lan_inst: str = None,bddl_workspace: str=None):
        self.existing_objects=None
        self.layout_id = None
        if bddl_file is not None:
            self.bddl_file = bddl_file
            self.parsed_problem = BDDLUtils.robosuite_parse_problem(self.bddl_file)
            self.problem_info=BDDLUtils.get_problem_info(self.bddl_file)
            # 直接提取 gen_bddl 需要的字段
            self.object_num_info = self._extract_object_num_info()
            self.fixture_num_info = self._extract_fixture_num_info()
            self.object_init_position = self._extract_object_init_position()

            self.workspace_name = None
            self.bddl_workspace = bddl_workspace
            self.object_set= set(self.object_init_position.keys())
            self.init_state,self.regions = self._extract_init_state_and_region()

            self.goal_state = [
                tuple(item) if isinstance(item, list) else item
                for item in self.parsed_problem.get("goal_state", [])
            ]

            self.objects_of_interest = self.parsed_problem.get("obj_of_interest", [])
            if lan_inst is not None:
                self.lan_inst = lan_inst
            else:
                self.lan_inst = " ".join(
                    self.parsed_problem.get("language_instruction", [])
                )
            self.obj2add={}
            self.existing_objects = self.object_init_position.copy ()
        else:
            # Generate a naive layout from scratch
            raise NotImplementedError(
                "Currently we only support using existing bddl file to generate layout. Please provide a bddl_file."
            )
    def get_hdf5_filename(self) -> str:
        assert isinstance (self.lan_inst, str) and len (self.lan_inst) > 0, \
            "lan_inst must be a non-empty string"

        # 1. lowercase
        name = self.lan_inst.lower ()

        # 2. remove leading/trailing spaces
        name = name.strip ()

        # 3. replace any whitespace (space, tab, newline) with single underscore
        name = re.sub (r"\s+", "_", name)

        return name

    def set_layout_id(self, layout_id: int):
        self.layout_id = layout_id
    def get_layout_id(self) -> int:
        if self.layout_id is not None:
            return self.layout_id
        else:
            raise ValueError("Layout ID has not been set.")


    def set_lan_inst(self, new_lan_inst: str):
        self.lan_inst = new_lan_inst



    def _extract_init_state_and_region (self) -> Tuple [list, dict]:
        """
        Extract and normalize both init_state and regions.

        规则：
          1. 对于 ("On", obj, region) 的条目：
             - region 改名为 "{obj}_init_region"
             - 同时复制该 region 的信息到新名字下
          2. 其他谓词不变
          3. 未出现在 On 中的 region 原样拷贝

        Returns:
            tuple: (normalized_init_state, normalized_regions)
        """
        bddl_init_state = self.parsed_problem.get ("initial_state", [])
        raw_regions = self.parsed_problem.get ("regions", {})
        normalized_regions = {}
        normalized_init = []

        def _normalize_region_info(reg_info: dict) -> dict:
            """只保留 target / ranges / yaw_rotation，并修正格式"""
            new_info = {}
            for key in ["target", "ranges", "yaw_rotation"]:
                if key not in reg_info:
                    continue
                val = reg_info[key]

                # ranges → [tuple(...)]
                if key == "ranges":
                    normalized_ranges = []
                    for r in val:
                        if isinstance(r, (list, tuple)):
                            normalized_ranges.append(tuple(r))
                    new_info[key] = normalized_ranges

                # yaw_rotation → [(low, high)]
                elif key == "yaw_rotation":
                    normalized_yaw = []
                    if isinstance(val, (list, tuple)):
                        if len(val) == 2 and not isinstance(val[0], (list, tuple)):
                            normalized_yaw.append(tuple(val))
                        else:
                            for y in val:
                                if isinstance(y, (list, tuple)):
                                    normalized_yaw.append(tuple(y))
                    new_info[key] = normalized_yaw

                else:
                    new_info[key] = val
            return new_info

        # 记录哪些 region 被重命名过
        renamed_map = {}

        for item in bddl_init_state:
            if isinstance (item, list):
                item = tuple (item)

            # 只修改 On 开头的 predicate
            if len (item) == 3 and item [0].lower () == "on":
                pred, obj, old_region = item

                # 拷贝原 region 内容到新 region
                if old_region in raw_regions:
                    new_region = f"{obj}_init_region"
                    target=raw_regions [old_region]["target"]
                    if not target.endswith("_table"): #Zhixu: not put on table,keep the old region
                        normalized_init.append (item)
                        continue
                    if self.workspace_name is None:#ZhixuL: set workspace_name as the target of the first init state region
                        self.workspace_name =target
                    # normalized_regions [new_region] = raw_regions [old_region].copy ()
                    normalized_regions [new_region] = _normalize_region_info (raw_regions [old_region])
                    new_init_region=f"{target}_{new_region}"
                    normalized_init.append ((pred, obj, new_init_region))
                    renamed_map [old_region] = new_region
                else:
                    print (f"[Layout][WARN] region '{old_region}' not found for '{obj}'")
            else:
                normalized_init.append (item)
        # 其余 region 原样复制（除非它已经被改名）
        for reg_name, reg_info in raw_regions.items ():
            if reg_name not in renamed_map:
                target = reg_info["target"]
                prefix = f"{target}_"
                new_name = reg_name
                if target and reg_name.startswith (prefix):
                    new_name = reg_name [len (prefix):]  # 去掉前缀
                # normalized_regions [new_name] = reg_info
                normalized_regions [new_name] = _normalize_region_info (reg_info)

        # 可选：提醒有哪些被改名
        if renamed_map:
            logger.debug(f"[Layout] Renamed regions: {renamed_map}")
        else:
            print ("[Layout] No regions renamed from init_state.")

        return normalized_init, normalized_regions

    def _extract_object_num_info(self) -> dict:
        """每个 object 默认只有一个实例"""
        return {k: len(v) for k, v in self.parsed_problem.get("objects", {}).items()}

    def _extract_fixture_num_info(self) -> dict:
        return {k: len(v) for k, v in self.parsed_problem.get("fixtures", {}).items()}

    def _extract_object_init_position(self) -> dict:
        """
        Extract object positions from BDDL initial_state.
        
        For each ("on", obj, region) in initial_state, extracts the region center as position.
        
        LIMITATION: If multiple instances of same type exist (e.g., milk_1, milk_2), 
        only the LAST instance's position is stored because we strip instance suffixes
        and use base names as dict keys. This means milk_2 overwrites milk_1.
        
        TODO: Track instance IDs to support multiple instances per object type.
        
        Returns:
            dict: {base_object_name: [x, y, yaw]}
        """
        obj_pos = {}

        for item in self.parsed_problem.get("initial_state", []):
            if len(item) == 2:
                    continue   # 或者 raise / log，看你需求
            pred, obj, region= item
            if pred.lower() == "on" and region in self.parsed_problem["regions"]:
                reg = self.parsed_problem["regions"][region]
                if "ranges" not in reg or len(reg["ranges"]) == 0:
                    print(f"[Layout][WARN] No ranges found for region '{region}' of object '{obj}'")
                    continue
                ranges = self.parsed_problem["regions"][region]["ranges"][0]

                # ranges = [x_min, y_min, x_max, y_max]
                x = (ranges[0] + ranges[2]) / 2
                y = (ranges[1] + ranges[3]) / 2
                yaw = 0.0
                if "yaw_rotation" in reg:
                    a, b = reg["yaw_rotation"]
                    a = _to_float(a)
                    b = _to_float(b)
                    yaw = a if abs(a - b) < 1e-6 else 0.5 * (a + b)
                #Zhixu: here use the name with suffix, e.g. apple_1, milk_2
                # obj_name = strip_instance_suffix(                 obj                )  # "alphabet_soup_1" -> "alphabet_soup"
                # WARNING: If obj_name already exists, this OVERWRITES previous position
                obj_pos[obj] = [x, y, yaw]
        return obj_pos



    def object_add(
        self,
        obj_name: str,
        position_xy: Tuple[float, float],
        yaw: Tuple[float,float] = (-np.pi, np.pi),
        margin: float = 0.02,
        force_insert: bool = False,
    ) :
        """
        Add a new object instance to the layout.
        
        LIMITATION: object_init_position can only store ONE position per object type.
        When adding a new instance, the position is stored under the base type name,
        overwriting any previous position for that type. This means we can't track
        positions for multiple instances like milk_1 and milk_2 separately.
        
        TODO: Support multiple instances per type by tracking instance-specific positions.
        
        Args:
            obj_name: Object instance name (e.g., "milk_1")
            position_xy: (x, y) coordinates for the object center
            margin: Collision margin for overlap detection
            
        Returns:
            bool: True if added successfully, False if collision detected
        """

        size_xy = get_object_size(obj_name)
        obj_type= strip_instance_suffix(obj_name)

        if not size_xy or len(size_xy) < 2:
            print(f"get_object_size('{obj_name}') does not return valid [len_x, len_y]")
            return False, ""
        len_x, len_y = float(size_xy[0]), float(size_xy[1])

        nx, ny = float(position_xy[0]), float(position_xy[1])
        new_min = (nx - len_x / 2.0, ny - len_y / 2.0)
        new_max = (nx + len_x / 2.0, ny + len_y / 2.0)

        # print(f"self.existing_objects: {self.existing_objects}")
        existing_positions = self.existing_objects.copy()

        existing_objects = []
        for t, xy in existing_positions.items():
            ex, ey = float(xy[0]), float(xy[1])
            t_size = get_object_size(t) or size_xy
            tx, ty = float(t_size[0]), float(t_size[1])
            existing_objects.append(
                {
                    "name": t,
                    "min": (ex - tx / 2.0, ey - ty / 2.0),
                    "max": (ex + tx / 2.0, ey + ty / 2.0),
                }
            )
        collision, collision_with=collision_detection_2d (new_min, new_max, existing_objects, margin=margin)
        if collision and not force_insert:
            print(f"insert '{obj_name}'@({nx:.3f},{ny:.3f}) conflicting with current existing object with margin being {margin:.3f}")
            return False, collision_with

        cnt = int(self.object_num_info.get(obj_type, 0)) + 1
        self.object_num_info[obj_type] = cnt
        # new_inst_name = f"{obj_type}_{cnt}"

        #x,y,yaw
        self.obj2add[obj_name]=[nx, ny, yaw]
        self.existing_objects[obj_name] = [nx, ny, yaw]
        self.object_set.add(obj_name)


        region_name = f"{self.workspace_name}_{obj_name}_init_region"
        self.init_state.append(("On", obj_name, region_name))  # Must be tuple, not list

        return True, ""

    def object_rotate(self, obj_name, delta_yaw, margin=0.02):
        """
        rotate a existing object, with a given yaw, in range of [-pi, pi]
        """
        if obj_name not in self.existing_objects:
            raise KeyError(f"{obj_name} not found in layout.")
        x, y, yaw = self.existing_objects
        new_yaw = (yaw + delta_yaw + math.pi) % (2 * math.pi) - math.pi

        try:
            len_x, len_y = get_object_size(obj_name)
        except Exception:
            base = obj_name.split("_")[0]
            len_x, len_y = get_object_size(base)

        new_min, new_max = _aabb_from_center_len_yaw(x, y, len_x, len_y, new_yaw)

        existing_objs = []
        for other_name, pose in self.existing_objects.items():
            if other_name == obj_name:
                continue
            ox, oy, oyaw = pose
            try:
                olen_x, olen_y = get_object_size(other_name)
            except Exception:
                base = strip_instance_suffix(other_name)
                olen_x, olen_y = get_object_size(base)
            omin, omax = _aabb_from_center_len_yaw(ox, oy, olen_x, olen_y, oyaw)
            existing_objs.append({"name": other_name, "min": omin, "max": omax})

        # 检查碰撞
        collision, _ = collision_detection_2d(new_min, new_max, existing_objs, margin)
        if collision:
            print(
                f"[rotate_object] Rotation of {obj_name} blocked: collision detected."
            )
            return False

        # 若无碰撞，则更新 yaw
        self.existing_objects[obj_name][2] = new_yaw
        print(f"[rotate_object] {obj_name} rotated to yaw={new_yaw:.3f} rad.")
        return True

    def object_remove(self, obj_name: str):
        """
        Remove an object instance from the layout.
        
        LIMITATION: Since object_init_position only tracks one position per type,
        removing ANY instance of a type (e.g., milk_1, milk_2) removes THE SAME 
        stored position. We can't selectively remove positions for specific instances.
        
        TODO: Track instance-specific positions to properly support multi-instance removal.
        
        Args:
            obj_name: Object instance name (e.g., "milk_1") or base name (e.g., "milk")
        """
        base_name = strip_instance_suffix(obj_name)  # apple_1 -> apple

        if base_name in self.object_num_info:
            self.object_num_info[base_name] -= 1
            if self.object_num_info[base_name] <= 0:
                del self.object_num_info[base_name]

        if obj_name not in self.existing_objects.keys():
            print(f"[Layout] object {obj_name} not found in layout.")

            return
        else:
            del self.existing_objects[obj_name]

        self.init_state = [t for t in self.init_state if obj_name not in t]
        # self.goal_state = [t for t in self.goal_state if obj_name not in t] #Zhixu: goal status is not editable in default


        if obj_name in self.objects_of_interest:
            self.objects_of_interest.remove(obj_name)
        self.object_set.remove (obj_name)
        print(f"[Layout] removed object {obj_name}")

    def print_layout(self):
        output_dict = dict(
            lan_inst=self.lan_inst,
            object_num_info=self.object_num_info,
            fixture_num_info=self.fixture_num_info,
            object_init_position=self.existing_objects,
            init_state=self.init_state,
            goal_state=self.goal_state,
            objects_of_interest=self.objects_of_interest,
            regions=self.regions,
        )
        print(f"\n===== Layout Info =====\n")

        for k, v in output_dict.items():
            if k=="regions":
                print(f"{k}:")
                for rk,rv in v.items():
                    print(f"  {rk}: {rv}")
            else:
                print(f"{k}: {v}\n")

    def gen_bddl(
        self,
        bddl_folder: str = "../dataset/bddl",
        layout_id:str="",
    ):
        """
        Generate bddl files, given object_num_info, fixture_num_info, object_init_position, init_state, goal_state, objects_of_interest, lan_inst.
        In default, we assume each object has only one instance.
        Example:
        object_num_info = {
                                                        "apple": 1,
                                                        "banana": 1,
                                                        "bread": 1,
                                                }
        fixture_num_info = {
                                                        "kitchen_table": 1,
                                                        "study_table": 1,
                                                        "wooden_cabinet": 1,
                                                }
        goal_states=[
                                # ("On", f"{name}_1", "target_region")
                        ],
        init_state = [
                                ("On", "apple_1", "apple_1_init_region"),
                                ]
        """
        object_num_info = self.object_num_info
        fixture_num_info = self.fixture_num_info
        object_init_position = self.existing_objects
        obj2add=self.obj2add
        init_state = self.init_state
        goal_state = self.goal_state

        objects_of_interest = self.objects_of_interest
        lan_inst = self.lan_inst
        regions=self.regions
        bddl_workspace=self.bddl_workspace



        class PBenchScene(InitialSceneTemplates):

            def __init__(self):
                # 生成时缓存 object 列表（按名字稳定排序，保证可复现）
                self._all_object_info = sorted(list(object_num_info.keys()))
                self._all_fixture_info = sorted(list(fixture_num_info.keys()))
                self._obj_init_position = (
                    object_init_position  # Keep as dict for .items() usage
                )
                self.obj2add = obj2add
                self._regions = regions

                # Use 'floor' as workspace_name to match LIBERO's floor manipulation setup
                # This prevents the system from trying to look up affordances for 'floor'
                super().__init__(
                    # workspace_name="living",  # Match the fixture type
                    workspace_name=bddl_workspace,
                    fixture_num_info=fixture_num_info,  # Pass dict, not list
                    object_num_info=object_num_info,  # Pass dict, not list
                )

            def define_regions(self):
                for key,value in self._regions.items():
                    self.regions.update({key:value})

                region_half_len = 0.025
                for obj_name, (xi, yi,yaw) in self.obj2add.items():
                    t=self.get_region_dict(
                            region_centroid_xy=[xi, yi],
                            region_name=f"{obj_name}_init_region",
                            # target_name="floor",  # Use 'floor' to match workspace_name
                            region_half_len=region_half_len,
                            yaw_rotation=yaw,  # ZhixuL: for now just ignore the yaw_rotation
                        )
                    self.regions.update(t)


                self.xy_region_kwargs_list = (
                    get_xy_region_kwargs_list_from_regions_info(self.regions)
                )



            @property
            def init_states(self):
                return init_state
        decorator = register_mu(scene_type="p_bench_type")
        decorator(PBenchScene)  # register the scene
        lan_inst= "L"+layout_id+lan_inst
        register_task_info(
            language=lan_inst,
            scene_name=f"p_bench_scene",
            objects_of_interest=objects_of_interest,
            goal_states=goal_state,
        )
        if bddl_folder is not None:
            os.makedirs(bddl_folder, exist_ok=True)

        # If filename is provided, write BDDL directly (simpler approach)
        # if filename:
        #     output_path = os.path.join(bddl_folder, filename)
        #     self._write_bddl_file_directly(
        #         output_path,
        #         object_num_info,
        #         fixture_num_info,
        #         object_init_position,
        #         init_state,
        #         goal_state,
        #         objects_of_interest,
        #         lan_inst,
        #     )
        #     self.bddl_file = output_path
        #     print(f"✓ BDDL file written successfully to: {output_path}")


        bddl_file_names, failures = generate_bddl_from_task_info(folder=bddl_folder)
        if not failures:
            print(
                f"All BDDL files generated successfully, saved to {bddl_file_names[0]}!"
            )
            self.bddl_file = bddl_file_names[0]
        else:
            print(f"Failed to generate BDDL files for {failures} tasks!")
            print("Debugging: Check task_generation_utils.py for details")

    def _write_bddl_file_directly(
        self,
        output_path,
        object_num_info,
        fixture_num_info,
        object_init_position,
        init_state,
        goal_state,
        objects_of_interest,
        lan_inst,
    ):
        """
        Write BDDL file directly without using the task generation system.
        This is a simpler approach that formats the BDDL manually.
        """
        with open(output_path, "w") as f:
            # Header
            f.write("(define (problem LIBERO_Floor_Manipulation)\n")
            f.write("  (:domain robosuite)\n")
            f.write(f"  (:language {lan_inst})\n")

            # Regions
            f.write("    (:regions\n")
            region_half_len = 0.025
            for obj_name, (x, y) in object_init_position.items():
                x_min, x_max = x - region_half_len, x + region_half_len
                y_min, y_max = y - region_half_len, y + region_half_len
                f.write(f"      ({obj_name}_init_region\n")
                f.write(f"          (:target floor)\n")
                f.write(f"          (:ranges (\n")
                f.write(f"              ({x_min} {y_min} {x_max} {y_max})\n")
                f.write(f"            )\n")
                f.write(f"          )\n")
                f.write(f"      )\n")
            f.write("    )\n\n")

            # Fixtures
            f.write("  (:fixtures\n")
            for fixture_type, count in fixture_num_info.items():
                f.write(f"    {fixture_type} - {fixture_type}\n")
            f.write("  )\n\n")

            # Objects
            f.write("  (:objects\n")
            for obj_type, count in object_num_info.items():
                for i in range(1, count + 1):
                    f.write(f"    {obj_type}_{i} - {obj_type}\n")
            f.write("  )\n\n")

            # Objects of interest
            f.write("  (:obj_of_interest\n")
            for obj in objects_of_interest:
                f.write(f"    {obj}\n")
            f.write("  )\n\n")

            # Initial state
            f.write("  (:init\n")
            for predicate, obj, region in init_state:
                f.write(f"    ({predicate} {obj} {region})\n")
            f.write("  )\n\n")

            # Goal state
            f.write("  (:goal (and\n")
            for predicate, *args in goal_state:
                args_str = " ".join(str(arg) for arg in args)
                f.write(f"    ({predicate} {args_str})\n")
            f.write("    )\n")
            f.write("  )\n")
            f.write(")\n")

    def get_obs_image(self, path=None):
        env_args = {
            "bddl_file_name": self.bddl_file,
            "camera_heights": 512,
            "camera_widths": 512,
        }
        env = OffScreenRenderEnv(**env_args)
        obs = env.reset()
        img= get_img_from_obs(obs, key="agentview_image")
        if path is not None:
            import cv2
            cv2.imwrite(path, img)
        env.close()
        return img

    def get_object_dict(self):
        return self.object_num_info


    def copy (self) -> "Layout":
        """
        Create a deep copy of the current Layout instance.
        The returned layout is independent — modifying it
        will NOT affect the original.

        Returns:
            Layout: a deep-copied Layout instance
        """
        # 1️⃣ 创建一个新的空 Layout（不通过 __init__ 加载 bddl）
        new_layout = object.__new__ (Layout)
        new_layout.layout_id = self.layout_id
        # 2️⃣ 手动复制所有必要属性（确保每个属性互不影响）
        new_layout.bddl_file = self.bddl_file
        new_layout.parsed_problem = copy.deepcopy (self.parsed_problem)
        new_layout.problem_info = copy.deepcopy (self.problem_info)
        new_layout.object_num_info = copy.deepcopy (self.object_num_info)
        new_layout.fixture_num_info = copy.deepcopy (self.fixture_num_info)
        new_layout.object_init_position = copy.deepcopy (self.existing_objects)
        new_layout.workspace_name = self.workspace_name
        new_layout.bddl_workspace = self.bddl_workspace
        new_layout.object_set = copy.deepcopy (self.object_set)
        new_layout.init_state = copy.deepcopy (self.init_state)
        new_layout.goal_state = copy.deepcopy (self.goal_state)
        new_layout.objects_of_interest = copy.deepcopy (self.objects_of_interest)
        new_layout.lan_inst = str (self.lan_inst)
        new_layout.obj2add = copy.deepcopy (self.obj2add)
        new_layout.regions = copy.deepcopy (self.regions)
        new_layout.problem_info = copy.deepcopy (self.problem_info)
        new_layout.object_num_info = copy.deepcopy (self.object_num_info)
        new_layout.fixture_num_info = copy.deepcopy (self.fixture_num_info)
        new_layout.problem_info = copy.deepcopy (self.problem_info)
        new_layout.existing_objects = copy.deepcopy (self.existing_objects)
        new_layout.object_init_position = copy.deepcopy (self.object_init_position)

        # 3️⃣ 返回新的 Layout
        return new_layout


if __name__ == "__main__":
    bddl_file = "../../libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"
    # Zhixu: here we just use libero's layout for naive layout
    # Later we can implement our own naive layout generator
    layout = Layout(bddl_file)
    layout.print_layout()
    img = layout.get_obs_image()
    # import cv2
    # cv2.imwrite("naive_layout.png", img)
