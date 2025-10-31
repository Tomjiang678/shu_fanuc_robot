#constraints.py
import numpy as np
from scipy.spatial.transform import Rotation as R
# -------------------------
# 姿态误差
# -------------------------
def orientation_error(R_current, q_target):
    R_target = R.from_quat(q_target).as_matrix()
    R_err = R_target.T @ R_current
    rotvec = R.from_matrix(R_err).as_rotvec()
    return np.linalg.norm(rotvec)  # 弧度误差
# -------------------------
# J2-J3 非线性约束
# -------------------------
def get_joint_limits(q):
    joint_angle_max = np.pi * np.array([180, 145, 0, 190, 125, 360]) / 180
    joint_angle_min = np.pi * np.array([-180, -100, 0, -190, -125, -360]) / 180

    J2_deg = np.rad2deg(q[1])

    # J3 最小值
    if J2_deg < 74:
        J3_min = -J2_deg - 68
    else:
        J3_min = 0.0101 * J2_deg**2 - 1.8202 * J2_deg - 62.682

    # J3 最大值
    if J2_deg < -65:
        J3_max = 280
    else:
        J3_max = 215 - J2_deg

    joint_angle_min[2] = np.deg2rad(J3_min)
    joint_angle_max[2] = np.deg2rad(J3_max)
    return joint_angle_min, joint_angle_max