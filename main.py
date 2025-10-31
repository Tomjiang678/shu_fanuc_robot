# main.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from optimizer import pso_de_optimize, quintic_interpolation, fitness  # optimizer里已有fitness
from forward_kinematics_module import forward_kinematics
from constraints import orientation_error, get_joint_limits
from joints_limit import load_robot_joint_data
from display import animate_chain
from ikpy.chain import Chain
# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块
# -------------------------
# 包装PSO+DE，用于记录每代指标
# -------------------------
def pso_de_optimize_with_history(q0, target_pos, target_quat, vmax, amax,
                                 n_particles=60, n_iter=200):
    dim = len(q0) + 1
    x = np.random.uniform(-np.pi, np.pi, (n_particles, dim))
    x[:, -1] = np.random.uniform(0.5, 5.0, n_particles)
    v = np.zeros_like(x)

    pbest = x.copy()
    pbest_val = np.array([fitness(q, q0, target_pos, target_quat, vmax, amax) for q in x])
    gbest = pbest[np.argmin(pbest_val)]
    gbest_val = np.min(pbest_val)

    history_loss = []
    history_pos_err = []
    history_ori_err = []

    w, c1, c2 = 0.7, 1.5, 1.5
    DE_interval = 10

    for it in range(n_iter):
        r1, r2 = np.random.rand(n_particles, dim), np.random.rand(n_particles, dim)
        v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        x = x + v

        # 限制角度
        for i in range(n_particles):
            q_min, q_max = get_joint_limits(x[i, :-1])
            x[i, :-1] = np.clip(x[i, :-1], q_min, q_max)
        x[:, -1] = np.clip(x[:, -1], 0.1, 10.0)

        # 计算适应度
        vals = np.array([fitness(q, q0, target_pos, target_quat, vmax, amax) for q in x])
        better = vals < pbest_val
        pbest[better] = x[better]
        pbest_val[better] = vals[better]

        if np.min(vals) < gbest_val:
            gbest = x[np.argmin(vals)]
            gbest_val = np.min(vals)

        # DE增强
        if it % DE_interval == 0:
            # 假设你的optimizer.py里有de_mutation函数
            from optimizer import de_mutation
            x, vals = de_mutation(x, vals, q0, target_pos, target_quat, vmax, amax)
            better = vals < pbest_val
            pbest[better] = x[better]
            pbest_val[better] = vals[better]
            if np.min(vals) < gbest_val:
                gbest = x[np.argmin(vals)]
                gbest_val = np.min(vals)

        # 记录指标
        q_goal = gbest[:-1]
        T = gbest[-1]
        pos, R_current = forward_kinematics(q_goal)
        pos_err = np.linalg.norm(pos - target_pos)
        ori_err = orientation_error(R_current, target_quat)
        history_loss.append(gbest_val)
        history_pos_err.append(pos_err)
        history_ori_err.append(ori_err)

    return gbest[:-1], gbest[-1], gbest_val, history_loss, history_pos_err, history_ori_err


if __name__ == "__main__":
    # CSV路径
    csv_path = "assemblerobot_joint_position.csv"
    data = load_robot_joint_data(csv_path)

    q0 = data["init_joint_angles"]
    vmax = data["max_speeds"]
    amax = data["max_accs"]
    target_pos = data["target_tcp_pos"]
    target_quat = data["target_tcp_quat"]

    # -------------------------
    # PSO+DE轨迹优化
    # -------------------------
    print("开始优化轨迹...")
    q_goal, T_opt, best_val, hist_loss, hist_pos_err, hist_ori_err = pso_de_optimize_with_history(
        q0, target_pos, target_quat, vmax, amax, n_particles=60, n_iter=200
    )
    print("优化完成！")
    print("最优关节角:", q_goal)
    print("最优总时间:", T_opt)
    print("最优适应度:", best_val)

    # -------------------------
    # 绘制迭代指标变化
    # -------------------------
    plt.figure()
    plt.plot(hist_loss, label="Fitness / Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("全局最优适应度变化")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(hist_pos_err, label="Position Error")
    plt.xlabel("Iteration")
    plt.ylabel("Position Error [m]")
    plt.title("位置误差变化")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(hist_ori_err, label="Orientation Error")
    plt.xlabel("Iteration")
    plt.ylabel("Orientation Error [rad]")
    plt.title("姿态误差变化")
    plt.grid()
    plt.legend()
    plt.show()

    # -------------------------
    # 生成轨迹
    # -------------------------
    traj, t = quintic_interpolation(q0, q_goal, steps=100, T=T_opt)

    # -------------------------
    # 可视化关节角轨迹
    # -------------------------
    plt.figure(figsize=(10, 6))
    for i in range(traj.shape[1]):
        plt.plot(t, traj[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Angle [rad]')
    plt.title('Optimized Joint Trajectory')
    plt.legend()
    plt.grid()
    plt.show()

    # -------------------------
    # 可视化速度和加速度
    # -------------------------
    vel = np.gradient(traj, t, axis=0)
    acc = np.gradient(vel, t, axis=0)

    plt.figure(figsize=(10, 6))
    for i in range(traj.shape[1]):
        plt.plot(t, vel[:, i], label=f'Joint {i+1} Velocity')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [rad/s]')
    plt.title('Joint Velocities')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    for i in range(traj.shape[1]):
        plt.plot(t, acc[:, i], label=f'Joint {i+1} Acceleration')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [rad/s²]')
    plt.title('Joint Accelerations')
    plt.legend()
    plt.grid()
    plt.show()

    # -------------------------
    # 动画展示
    # -------------------------
    chain_path = "lrmate200id7l.urdf"
    chain = Chain.from_urdf_file(chain_path)
    animate_chain(chain, traj, target=target_pos, save_path="optimized_trajectory.mp4", interval=100)
