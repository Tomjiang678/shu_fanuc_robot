#display.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def apply_joint_correction(q):
    """保持和 forward_kinematics_module 一致：j3 += j2"""
    q_mod = q.copy()
    q_mod[2] += q_mod[1]
    return q_mod

def animate_chain(chain, traj, target=None, save_path="trajectory.mp4", interval=100):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # ----------------------------
    # 基座变换矩阵 (Z 轴旋转 180°, 平移到指定位置)
    # ----------------------------
    base_transform = np.eye(4)
    base_transform[:3, 3] = [1.756, -0.00416, 1.0]  # 平移
    Rz = np.array([
        [-1, 0, 0],
        [ 0,-1, 0],
        [ 0, 0, 1]
    ])  # 绕 Z 轴 180°
    base_transform[:3, :3] = Rz
    base_inv = np.linalg.inv(base_transform)

    # target 变换到基座系
    if target is not None:
        target_h = np.ones(4)
        target_h[:3] = target
        target_in_base = (base_inv @ target_h)[:3]
    else:
        target_in_base = None

    # ----------------------------
    # 更新函数
    # ----------------------------
    def update(frame):
        ax.cla()
        # 设置坐标轴
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)
        ax.set_zlim(0, 2)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"Frame {frame}/{len(traj)}")

        # 画机械臂
        q_mod = apply_joint_correction(traj[frame])
        q_full = np.concatenate(([0], q_mod, [0, 0]))
        chain.plot(q_full, ax=ax, show=False)  # ⚡ 机械臂不动

        # 画 target
        if target_in_base is not None:
            ax.scatter(target_in_base[0], target_in_base[1], target_in_base[2],
                       c="r", s=80, label="Target")

        ax.legend()

    ani = FuncAnimation(fig, update, frames=len(traj), interval=interval)
    ani.save(save_path, writer="ffmpeg")
    plt.show()
    plt.close(fig)
    print(f"✅ 动画已保存到 {save_path}")
