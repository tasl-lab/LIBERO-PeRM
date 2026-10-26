import os
import re
import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string

import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

from libero.libero.envs.base_object import register_object
from robosuite.models.objects.primitive.cylinder import CylinderObject
from robosuite.models.objects.primitive.box import BoxObject
from robosuite.models.objects.primitive.ball import BallObject
def get_preset_obj():
    return {
        "box": [
              # 5个普通尺寸变化的盒子（-0.5~0.5倍）
              (0.0125, 0.0125, 0.0125),  # -0.5倍
              (0.02, 0.02, 0.02),        # -0.2倍
              (0.025, 0.025, 0.025),     # 原始尺寸
              (0.03, 0.03, 0.03),        # +0.2倍
              (0.0375, 0.0375, 0.0375),  # +0.5倍
              # 5个高度增加的盒子 - 已与圆柱体对齐
              (0.025, 0.025, 0.16),      # 与圆柱体高度0.16对齐
              (0.025, 0.025, 0.2),       # 与圆柱体高度0.2对齐
              (0.025, 0.025, 0.25),      # 与圆柱体高度0.25对齐
              (0.025, 0.025, 0.3),       # 与圆柱体高度0.3对齐
              (0.025, 0.025, 0.35)       # 与圆柱体高度0.35对齐
          ],
        "cylinder": [
            # 5个普通尺寸变化的圆柱（-0.5~0.5倍）
            (0.025, 0.058),             # -0.5倍
            (0.04, 0.08),              # -0.2倍
            (0.03, 0.05),               # 原始尺寸
            (0.06, 0.12),              # +0.2倍
            (0.075, 0.15),             # +0.5倍
            # 5个高度增加的圆柱（高度0.5~2.5倍）- 已调整
            (0.05, 0.16),              # 高度+0.6倍
            (0.05, 0.2),               # 高度+1.0倍
            (0.05, 0.25),              # 高度+1.5倍
            (0.05, 0.3),               # 高度+2.0倍
            (0.05, 0.35)               # 高度+2.5倍
        ],
        "ball": [
            # 10个球体尺寸（-0.5~0.5倍）
            (0.025,),                  # -0.5倍
            (0.03,),                   # -0.4倍
            (0.035,),                  # -0.3倍
            (0.04,),                   # -0.2倍
            (0.045,),                  # -0.1倍
            (0.05,),                   # 原始尺寸
            (0.055,),                  # +0.1倍
            (0.06,),                   # +0.2倍
            (0.065,),                  # +0.3倍
            (0.075,)                   # +0.5倍
        ],
        "airball": [
            # 10个空中球体尺寸（-0.5~0.5倍）
            (0.025,),                  # -0.5倍
            (0.03,),                   # -0.4倍
            (0.035,),                  # -0.3倍
            (0.04,),                   # -0.2倍
            (0.045,),                  # -0.1倍
            (0.05,),                   # 原始尺寸
            (0.055,),                  # +0.1倍
            (0.06,),                   # +0.2倍
            (0.065,),                  # +0.3倍
            (0.075,)                   # +0.5倍
        ]
    }
### Zhixu: Examples
# class HopeBaseObject(MujocoXMLObject):
#     def __init__(self, name, obj_name):
#         super().__init__(
#             os.path.join(
#                 str(absolute_path),
#                 f"assets/stable_hope_objects/{obj_name}/{obj_name}.xml",
#             ),
#             name=name,
#             joints=[dict(type="free", damping="0.0005")],
#             obj_type="all",
#             duplicate_collision_geoms=False,
#         )
#         self.category_name = "_".join(
#             re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
#         ).lower()
#         self.rotation = (np.pi / 2, np.pi / 2)
#         self.rotation_axis = "x"
#
#         self.object_properties = {"vis_site_names": {}}
#
#
# @register_object
# class AlphabetSoup(HopeBaseObject):
#     def __init__(self, name="alphabet_soup", obj_name="alphabet_soup"):
#         super().__init__(name, obj_name)
#         self.rotation_axis = "z"
### Zhixu: Examples

#Zhixu: Here we define all objects types at once
@register_object
class BallA(BallObject):
    def __init__(self,
                 name="ball",
                 ball_id=0,
                 ):
        size=get_preset_obj()["ball"][ball_id]

        #Zhixu: Change the size of the cylinder here
        super().__init__(
            name=name,
            size=size,
        )

        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        # self.rotation_axis = "x"
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BallB(BallObject):
    def __init__(self,
                 name="ball",
                 ball_id=1,
                 ):
        size=get_preset_obj()["ball"][ball_id]

        #Zhixu: Change the size of the cylinder here
        super().__init__(
            name=name,
            size=size,
        )

        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        # self.rotation_axis = "x"
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BallC(BallObject):
    def __init__(self,
                 name="ball",
                 ball_id=2,
                 ):
        size=get_preset_obj()["ball"][ball_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BallD(BallObject):
    def __init__(self,
                 name="ball",
                 ball_id=3,
                 ):
        size=get_preset_obj()["ball"][ball_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BallE(BallObject):
    def __init__(self,
                 name="ball",
                 ball_id=4,
                 ):
        size=get_preset_obj()["ball"][ball_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BallF(BallObject):
    def __init__(self,
                 name="ball",
                 ball_id=5,
                 ):
        size=get_preset_obj()["ball"][ball_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BallG(BallObject):
    def __init__(self,
                 name="ball",
                 ball_id=6,
                 ):
        size=get_preset_obj()["ball"][ball_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BallH(BallObject):
    def __init__(self,
                 name="ball",
                 ball_id=7,
                 ):
        size=get_preset_obj()["ball"][ball_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BallI(BallObject):
    def __init__(self,
                 name="ball",
                 ball_id=8,
                 ):
        size=get_preset_obj()["ball"][ball_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BallJ(BallObject):
    def __init__(self,
                 name="ball",
                 ball_id=9,
                 ):
        size=get_preset_obj()["ball"][ball_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class CylinderA(CylinderObject):
    def __init__(self,
                 name="cylinder",
                 cylinder_id=0,
                 ):
        size=get_preset_obj()["cylinder"][cylinder_id]
        #Zhixu: Change the size of the cylinder here
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            # "x": (-np.pi/2, -np.pi/2),
            # "x": (0,0),
            "z": (0,0),
        }

        # self.rotation_axis = "x"
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class CylinderB(CylinderObject):
    def __init__(self,
                 name="cylinder",
                 cylinder_id=1,
                 ):
        size=get_preset_obj()["cylinder"][cylinder_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            # "x": (-np.pi/2, -np.pi/2),
            "z": (0,0),
        }

        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class CylinderC(CylinderObject):
    def __init__(self,
                 name="cylinder",
                 cylinder_id=2,
                 ):
        size=get_preset_obj()["cylinder"][cylinder_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            # "x": (-np.pi/2, -np.pi/2),
            "z": (0,0),
        }

        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class CylinderD(CylinderObject):
    def __init__(self,
                 name="cylinder",
                 cylinder_id=3,
                 ):
        size=get_preset_obj()["cylinder"][cylinder_id]
        super().__init__(
            name=name,
            size=size,
        )
        # self.rotation= None
        self.rotation = {
            # "x": (-np.pi/2, -np.pi/2),
            "z": (0,0),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class CylinderE(CylinderObject):
    def __init__(self,
                 name="cylinder",
                 cylinder_id=4,
                 ):
        size=get_preset_obj()["cylinder"][cylinder_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
        #     "x": (-np.pi/2, -np.pi/2),
            "z": (0,0),
        }
        # self.rotation= None
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class CylinderF(CylinderObject):
    def __init__(self,
                 name="cylinder",
                 cylinder_id=5,
                 ):
        size=get_preset_obj()["cylinder"][cylinder_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation= None
        self.rotation = {
            # "x": (-np.pi/2, -np.pi/2),
            "z": (0,0),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class CylinderG(CylinderObject):
    def __init__(self,
                 name="cylinder",
                 cylinder_id=6,
                 ):
        size=get_preset_obj()["cylinder"][cylinder_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            # "x": (-np.pi/2, -np.pi/2),
            # "z": (0,0),
        }
        # self.rotation= None
        self.rotation_axis = "x"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class CylinderH(CylinderObject):
    def __init__(self,
                 name="cylinder",
                 cylinder_id=7,
                 ):
        size=get_preset_obj()["cylinder"][cylinder_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "z": (0,0),
        }
        # self.rotation= None
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class CylinderI(CylinderObject):
    def __init__(self,
                 name="cylinder",
                 cylinder_id=8,
                 ):
        size=get_preset_obj()["cylinder"][cylinder_id]
        super().__init__(
            name=name,
            size=size,
        )

        self.rotation = {
            # "x": (-np.pi/2, -np.pi/2),
            "z": (0,0),
        }
        # self.rotation= None
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class CylinderJ(CylinderObject):
    def __init__(self,
                 name="cylinder",
                 cylinder_id=9,
                 ):
        size=get_preset_obj()["cylinder"][cylinder_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            # "x": (-np.pi/2, -np.pi/2),
            "z": (0,0),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BoxA(BoxObject):
    def __init__(self,
                 name="box",
                 box_id=0,
                 ):
        size=get_preset_obj()["box"][box_id]
        #Zhixu: Change the size of the box here
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        # self.rotation_axis = "x"
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BoxB(BoxObject):
    def __init__(self,
                 name="box",
                 box_id=1,
                 ):
        size=get_preset_obj()["box"][box_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BoxC(BoxObject):
    def __init__(self,
                 name="box",
                 box_id=2,
                 ):
        size=get_preset_obj()["box"][box_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BoxD(BoxObject):
    def __init__(self,
                 name="box",
                 box_id=3,
                 ):
        size=get_preset_obj()["box"][box_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BoxE(BoxObject):
    def __init__(self,
                 name="box",
                 box_id=4,
                 ):
        size=get_preset_obj()["box"][box_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BoxF(BoxObject):
    def __init__(self,
                 name="box",
                 box_id=5,
                 ):
        size=get_preset_obj()["box"][box_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BoxG(BoxObject):
    def __init__(self,
                 name="box",
                 box_id=6,
                 ):
        size=get_preset_obj()["box"][box_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BoxH(BoxObject):
    def __init__(self,
                 name="box",
                 box_id=7,
                 ):
        size=get_preset_obj()["box"][box_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BoxI(BoxObject):
    def __init__(self,
                 name="box",
                 box_id=8,
                 ):
        size=get_preset_obj()["box"][box_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

@register_object
class BoxJ(BoxObject):
    def __init__(self,
                 name="box",
                 box_id=9,
                 ):
        size=get_preset_obj()["box"][box_id]
        super().__init__(
            name=name,
            size=size,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
        }
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
