import numpy as np
from forward_kinematics_module import forward_kinematics
from constraints import orientation_error, get_joint_limits

# -------------------------
# 五次多项式轨迹生成 (带总时间T)
# -------------------------
def quintic_interpolation(q0, qf, steps=50, T=1.0):
    t = np.linspace(0, T, steps)     
    tau = t / T                      
    traj = []
    for i in range(len(q0)):
        a0 = q0[i]
        a1 = 0
        a2 = 0
        a3 = 10*(qf[i] - q0[i])
        a4 = -15*(qf[i] - q0[i])
        a5 = 6*(qf[i] - q0[i])
        qi = a0 + a1*tau + a2*tau**2 + a3*tau**3 + a4*tau**4 + a5*tau**5
        traj.append(qi)
    return np.array(traj).T, t   # shape: [steps, n_joints]

# -------------------------
# 适应度函数
# -------------------------
def fitness(candidate, q0, target_pos, target_quat, vmax, amax):
    q_goal = candidate[:-1]      
    T = abs(candidate[-1]) + 0.1 

    # 角度限制
    q_min, q_max = get_joint_limits(q_goal)
    q_goal = np.clip(q_goal, q_min, q_max)

    # 正向运动学
    pos, R_current = forward_kinematics(q_goal)

    # 位置 & 姿态误差
    pos_err = np.linalg.norm(pos - target_pos)
    ori_err = orientation_error(R_current, target_quat)

    # 轨迹生成
    traj, t = quintic_interpolation(q0, q_goal, steps=50, T=T)

    # 数值微分近似速度/加速度
    vel = np.gradient(traj, t, axis=0)
    acc = np.gradient(vel, t, axis=0)

    # 速度/加速度超限惩罚
    violation = 0.0
    for j in range(len(q_goal)):
        violation += np.sum(np.maximum(np.abs(vel[:, j]) - vmax[j], 0))
        violation += np.sum(np.maximum(np.abs(acc[:, j]) - amax[j], 0))

    # 平滑性惩罚
    smooth_penalty = np.sum(np.diff(traj, axis=0)**2)

    return 2*pos_err + 0.3*ori_err + 0.01*smooth_penalty + 100*violation + 0.1*T

# -------------------------
# DE变异函数
# -------------------------
def de_mutation(x, fitness_vals, q0, target_pos, target_quat, vmax, amax, F=0.8, CR=0.9, n_select=10):
    n_particles, dim = x.shape
    for _ in range(n_select):
        idxs = np.random.choice(n_particles, 3, replace=False)
        r1, r2, r3 = x[idxs[0]], x[idxs[1]], x[idxs[2]]
        trial = r1 + F * (r2 - r3)
        j_rand = np.random.randint(dim)
        for j in range(dim):
            if np.random.rand() > CR and j != j_rand:
                trial[j] = x[idxs[0], j]
        # 约束处理
        q_min, q_max = get_joint_limits(trial[:-1])
        trial[:-1] = np.clip(trial[:-1], q_min, q_max)
        trial[-1] = np.clip(trial[-1], 0.1, 10.0)
        trial_fit = fitness(trial, q0, target_pos, target_quat, vmax, amax)
        worst_idx = np.argmax(fitness_vals)
        if trial_fit < fitness_vals[worst_idx]:
            x[worst_idx] = trial
            fitness_vals[worst_idx] = trial_fit
    return x, fitness_vals

# -------------------------
# PSO + DE轨迹优化
# -------------------------
def pso_de_optimize(q0, target_pos, target_quat, vmax, amax, n_particles=60, n_iter=300):
    dim = len(q0) + 1   # 关节 + 时间
    x = np.random.uniform(-np.pi, np.pi, (n_particles, dim))
    x[:, -1] = np.random.uniform(0.5, 5.0, n_particles)  # 初始T范围
    v = np.zeros_like(x)

    pbest = x.copy()
    pbest_val = np.array([fitness(q, q0, target_pos, target_quat, vmax, amax) for q in x])
    gbest = pbest[np.argmin(pbest_val)]
    gbest_val = np.min(pbest_val)

    w, c1, c2 = 0.7, 1.5, 1.5
    DE_interval = 10  # 每隔10代执行DE

    for it in range(n_iter):
        r1, r2 = np.random.rand(n_particles, dim), np.random.rand(n_particles, dim)
        v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        x = x + v

        # 限制角度范围
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

        # -------- DE增强 --------
        if it % DE_interval == 0:
            x, vals = de_mutation(x, vals, q0, target_pos, target_quat, vmax, amax)
            better = vals < pbest_val
            pbest[better] = x[better]
            pbest_val[better] = vals[better]
            if np.min(vals) < gbest_val:
                gbest = x[np.argmin(vals)]
                gbest_val = np.min(vals)

    q_goal = gbest[:-1]
    T_opt = gbest[-1]
    return q_goal, T_opt, gbest_val

