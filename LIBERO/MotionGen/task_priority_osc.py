"""
Task-Priority OSC Patch: 位置第一优先级，姿态在位置零空间内执行。

用法：
    from MotionGen.task_priority_osc import enable_task_priority_osc, disable_task_priority_osc
    enable_task_priority_osc(env)   # 开启：旋转时 TCP 位置不动
    disable_task_priority_osc(env)  # 恢复：默认 OSC 行为

原理：
    当前 OSC: τ = J_full.T @ [λ_pos @ F; λ_ori @ τ_ori]  ← 耦合
    改为:
        τ_pos = J_pos.T @ λ_pos @ F_pos                   ← 位置主任务
        N_pos = I - M_inv @ J_pos.T @ λ_pos @ J_pos       ← 位置零空间投影
        τ_ori = N_pos.T @ J_ori.T @ λ_ori @ τ_ori         ← 姿态在零空间内
        τ = τ_pos + τ_ori + gravity_comp + nullspace_torques
"""

import numpy as np
from robosuite.utils.control_utils import opspace_matrices, nullspace_torques, orientation_error


def _task_priority_run_controller(self):
    """替换 OSC 的 run_controller，使用任务优先级控制。"""
    self.update()

    # 计算目标位置
    if self.interpolator_pos is not None:
        if self.interpolator_pos.order == 1:
            desired_pos = self.interpolator_pos.get_interpolated_goal()
        else:
            desired_pos = np.array(self.goal_pos)
    else:
        desired_pos = np.array(self.goal_pos)

    # 计算目标姿态误差
    if self.interpolator_ori is not None:
        self.relative_ori = orientation_error(self.ee_ori_mat, self.ori_ref)
        ori_error = self.interpolator_ori.get_interpolated_goal()
    else:
        desired_ori = np.array(self.goal_ori)
        ori_error = orientation_error(desired_ori, self.ee_ori_mat)

    # 位置/姿态误差 → 期望力/力矩
    position_error = desired_pos - self.ee_pos
    vel_pos_error = -self.ee_pos_vel
    desired_force = np.multiply(position_error, self.kp[0:3]) + np.multiply(vel_pos_error, self.kd[0:3])

    vel_ori_error = -self.ee_ori_vel
    desired_torque = np.multiply(ori_error, self.kp[3:6]) + np.multiply(vel_ori_error, self.kd[3:6])

    # 计算操作空间矩阵
    lambda_full, lambda_pos, lambda_ori, ns_matrix = opspace_matrices(
        self.mass_matrix, self.J_full, self.J_pos, self.J_ori
    )

    # ===== 任务优先级控制 =====
    # Task 1 (最高优先级): 位置控制
    tau_pos = self.J_pos.T @ (lambda_pos @ desired_force)

    # 位置任务的零空间投影矩阵
    mass_matrix_inv = np.linalg.inv(self.mass_matrix)
    J_pos_bar = mass_matrix_inv @ self.J_pos.T @ lambda_pos  # 动态一致伪逆 (7×3)
    N_pos = np.eye(self.J_full.shape[1]) - J_pos_bar @ self.J_pos  # 7×7 零空间投影

    # Task 2 (低优先级): 姿态控制，投影到位置零空间
    tau_ori = N_pos.T @ self.J_ori.T @ (lambda_ori @ desired_torque)

    # 合并 + 重力补偿
    self.torques = tau_pos + tau_ori + self.torque_compensation

    # 关节零空间力矩（维持初始关节构型）
    self.torques += nullspace_torques(
        self.mass_matrix, ns_matrix, self.initial_joint, self.joint_pos, self.joint_vel
    )

    super(type(self), self).run_controller()
    return self.torques


# 保存原始方法
_original_run_controller = None


def enable_task_priority_osc(env):
    """启用任务优先级 OSC（位置零空间内做姿态控制）。"""
    global _original_run_controller
    controller = env.robots[0].controller
    _original_run_controller = controller.run_controller.__func__
    import types
    controller.run_controller = types.MethodType(_task_priority_run_controller, controller)
    print("[TaskPriorityOSC] Enabled: position-priority, orientation in nullspace")


def disable_task_priority_osc(env):
    """恢复原始 OSC 控制器。"""
    global _original_run_controller
    if _original_run_controller is not None:
        controller = env.robots[0].controller
        import types
        controller.run_controller = types.MethodType(_original_run_controller, controller)
        _original_run_controller = None
        print("[TaskPriorityOSC] Disabled: restored default OSC")
