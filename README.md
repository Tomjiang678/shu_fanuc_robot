# shu_fanuc_robot

**Fanuc 6-Joints Robot Trajectory Optimizer**

---

## 简介 | Description

本项目实现了基于 **PSO+DE（粒子群优化 + 差分进化）** 算法的六关节 Fanuc 机械臂轨迹优化器，能够在关节约束条件下优化末端轨迹的 **平滑性、执行时间** 和 **多目标性能**。

This project implements a **PSO+DE (Particle Swarm Optimization + Differential Evolution)** based trajectory optimizer for a Fanuc 6-joint robot. It can optimize the end-effector trajectory **smoothness, execution time**, and **multi-objective performance** under joint constraints.

---

## 算法特点 | Algorithm Highlights

- **混合优化**：结合 PSO 的全局搜索能力与 DE 的局部变异能力，提高收敛速度，避免陷入局部最优。
- **约束处理**：自动考虑关节角度、速度及加速度限制，保证轨迹可执行。
- **多目标优化**：优化目标包括末端轨迹平滑性、执行时间和能耗，可根据需求加权调整。

- **Hybrid Optimization**: Combines PSO's global search with DE's local mutation to improve convergence and avoid local optima.
- **Constraint Handling**: Joint angle, velocity, and acceleration limits are automatically respected to ensure feasible trajectories.
- **Multi-objective Optimization**: Objectives include end-effector smoothness, execution time, and energy consumption, with configurable weighting.

---

## 使用方法 | Usage

```bash
# 运行轨迹优化
python main.py
```

---

## 依赖 | Dependencies

- Python >= 3.8
- numpy, matplotlib

---

## 许可 | License

MIT License