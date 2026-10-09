# Preference/__init__.py

# 导入低层偏好模块
from .low_level.distance import *
from .low_level.motion_preference import *
from .low_level.speed import *
# Note: relationship preferences (OnTop / In / InContact / Opened / Closed)
# are no longer auto-imported — they are task-state predicates rather than
# motion preferences. motion_gen.py imports the classes directly when
# needed for skill-level isinstance checks.


# 导入高层偏好模块（如果有）
from .high_level.task_preference import *