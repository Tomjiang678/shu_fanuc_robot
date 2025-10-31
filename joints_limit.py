import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def load_robot_joint_data(path: str):
    """
    读取机械臂关节限制与目标点序列（单位均为弧度、米）

    返回:
        dict 包含:
            - max_speeds, max_accs, max_decs: (6,) 弧度制
            - joint_angle_min, joint_angle_max: (6,) 弧度制
            - init_joint_angles: (6,) 弧度制
            - target_tcp_pos: (3,) 末端目标位置 (m)
            - target_tcp_rot: (3,3) 末端目标旋转矩阵
            - target_tcp_quat: (4,) 末端目标四元数 (x, y, z, w)
    """

    # ------------------- 限制参数读取 -------------------
    df_limits = pd.read_csv(path, header=None, skiprows=1, nrows=1)
    joint_limits = df_limits.iloc[0, :18].values

    max_speeds = np.pi * joint_limits[0::3] / 180   # deg/s → rad/s
    max_accs   = np.pi * joint_limits[1::3] / 180   # deg/s² → rad/s²
    max_decs   = np.pi * joint_limits[2::3] / 180   # deg/s² → rad/s²

    # ------------------- 静态关节角限制 -------------------
    joint_angle_max = np.pi * np.array([180, 145, 0, 190, 125, 360]) / 180
    joint_angle_min = np.pi * np.array([-180, -100, 0, -190, -125, -360]) / 180

    '''初始关节角度与目标TCP位姿可以自己定义'''
    # ------------------- 初始关节角度 -------------------
    init_joint_angles = np.pi * np.array([-96.29, 48.63, -20.11, 17.18, 20.97, -16.1]) / 180

    # ------------------- 目标 TCP 位姿 -------------------
    target_tcp_pos = np.array([2084.82, 360.77, 1270.22]) * 0.001  # mm → m
    target_tcp_quat = np.array([0, 0.707107, 0, 0.707107])  # [x, y, z, w]
    target_tcp_rot = R.from_quat(target_tcp_quat).as_matrix()

    # ------------------- 打包返回 -------------------
    data = {
        "max_speeds": max_speeds,
        "max_accs": max_accs,
        "max_decs": max_decs,
        "joint_angle_min": joint_angle_min,
        "joint_angle_max": joint_angle_max,
        "init_joint_angles": init_joint_angles,
        "target_tcp_pos": target_tcp_pos,
        "target_tcp_rot": target_tcp_rot,
        "target_tcp_quat": target_tcp_quat,
    }

    return data

