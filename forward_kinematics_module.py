# forward_kinematics_module.py
import numpy as np
from scipy.spatial.transform import Rotation as R  
from ikpy.chain import Chain
# 提前加载 URDF
assembleRobot = Chain.from_urdf_file("lrmate200id7l.urdf")

# 定义基座变换
r_base = R.from_euler("z", 180, degrees=True).as_matrix()
t_base = np.array([1.756, -0.00416, 1.0])

def forward_kinematics(q):
    """
    输入: q = [q1,...,q6] 弧度
    输出: TCP在基坐标系下的位置 [x,y,z] 米
    """
    j1, j2, j3, j4, j5, j6 = q
    j3 += j2  

    joint_angles_all = np.zeros(9)
    joint_angles_all[1:7] = [j1, j2, j3, j4, j5, j6]

    tcp_matrix = assembleRobot.forward_kinematics(joint_angles_all)
    tcp_position = tcp_matrix[:3, 3]
    tcp_orientation = tcp_matrix[:3, :3]   # 旋转矩阵

    tcp_position_vc = r_base @ tcp_position + t_base
    tcp_orientation_vc = r_base @ tcp_orientation

    return tcp_position_vc, tcp_orientation_vc # 返回位置和姿态
