#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B 档回放管线 v0 的最小可行实现（L1 层）。

职责（对应计划 1.8）：
  1. 读 RoboMIND v1.0 的 HDF5（兼容成功样本 joint_position(T,8) 与
     失败样本 arm(T,6)+hand(T,1) 两种 schema）；
  2. 关节列 → 模型 qpos 的映射（**剔除夹爪列**，这是 L1 最容易错的地方）；
  3. 用各本体的官方模型做正运动学，取末端位姿；
  4. 与记录末端位姿比对，位置用欧氏距离、朝向用测地距离（**分别在位置与朝向两个分量上报告**，
     不合成单一标量）；
  5. 时间基：以帧索引为时间基（v1.0 无时间戳），支持按目标帧率线性重采样。

设计约束（来自已核事实）：
  - RoboMIND 末端朝向是 **RPY 欧拉角**（不是四元数）→ 必须转旋转矩阵后算测地距离；
  - 关节末列可能是夹爪 → 由 caller 显式给出关节列索引，默认剔除末列；
  - 四个本体的关节数不同（Franka 7、UR5e 6、Piper 6、X5 6）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

import h5py
import numpy as np


# ---------------------------------------------------------------- 姿态工具
def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """RPY（固定轴 XYZ，弧度）→ 旋转矩阵，支持 (N,3)。"""
    rpy = np.atleast_2d(rpy)
    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    R = np.empty((rpy.shape[0], 3, 3))
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


def quat_to_matrix(q: np.ndarray, order: str = 'xyzw') -> np.ndarray:
    """四元数 → 旋转矩阵，支持 (N,4)。"""
    q = np.atleast_2d(q).astype(np.float64)
    if order == 'xyzw':
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif order == 'wxyz':
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        raise ValueError(order)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    n[n == 0] = 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """旋转矩阵对的测地距离（度）。逐帧计算，不做欧拉角直接相减。"""
    Rrel = np.einsum('nji,njk->nik', Ra, Rb)
    trace = np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


# ---------------------------------------------------------------- 数据读取
@dataclass
class Episode:
    """一条轨迹的统一表示。"""
    path: str
    n_frames: int
    joints: np.ndarray            # (T, J) 仅关节，已剔除夹爪
    gripper: np.ndarray | None    # (T,) 或 None
    ee_xyz: np.ndarray | None     # (T, 3)
    ee_rpy: np.ndarray | None     # (T, 3) —— 注意是 RPY，不是四元数
    joint_names: list[str] = field(default_factory=list)
    note: str = ''


def load_robomind_v1(path: str, source: str = 'puppet') -> Episode:
    """读 RoboMIND v1.0 的 HDF5，兼容两种 schema。"""
    with h5py.File(path, 'r') as f:
        if f'{source}/joint_position' in f:
            # 成功样本：(T, 8) = 7 关节 + 夹爪
            raw = f[f'{source}/joint_position'][()]
            joints = raw[:, :-1]
            gripper = raw[:, -1]
            note = f'{source}/joint_position 合并列，末列视为夹爪'
        elif f'{source}/arm_joint_position' in f:
            # 失败样本：arm(T,6) + hand(T,1) 分开
            joints = f[f'{source}/arm_joint_position'][()]
            hand = f[f'{source}/hand_joint_position'][()]
            gripper = hand[:, 0] if hand.ndim > 1 else hand
            note = f'{source}/arm_joint_position + hand_joint_position 分开'
        else:
            raise KeyError(f'{path}: 既无 joint_position 也无 arm_joint_position')

        ee_xyz = ee_rpy = None
        for cand in (f'{source}/end_effector', f'{source}/end_effector_left'):
            if cand in f:
                ee = f[cand][()]
                if ee.shape[1] >= 6:
                    ee_xyz, ee_rpy = ee[:, :3], ee[:, 3:6]
                break

        return Episode(path=path, n_frames=int(joints.shape[0]),
                       joints=np.asarray(joints, dtype=np.float64),
                       gripper=None if gripper is None else np.asarray(gripper, dtype=np.float64),
                       ee_xyz=ee_xyz, ee_rpy=ee_rpy, note=note)


def resample_to(ep: Episode, target_n: int) -> Episode:
    """以帧索引为时间基线性重采样（v1.0 无时间戳）。"""
    if ep.n_frames == target_n:
        return ep
    src = np.linspace(0.0, 1.0, ep.n_frames)
    dst = np.linspace(0.0, 1.0, target_n)

    def interp(a):
        if a is None:
            return None
        if a.ndim == 1:
            return np.interp(dst, src, a)
        return np.stack([np.interp(dst, src, a[:, k]) for k in range(a.shape[1])], axis=1)

    return Episode(path=ep.path, n_frames=target_n,
                   joints=interp(ep.joints), gripper=interp(ep.gripper),
                   ee_xyz=interp(ep.ee_xyz), ee_rpy=interp(ep.ee_rpy),
                   joint_names=ep.joint_names,
                   note=ep.note + f'｜重采样 {ep.n_frames}→{target_n}')


# ---------------------------------------------------------------- L1 回放
def l1_residual(mujoco, model, data, ep: Episode, ee_site_or_body: str,
                joint_qpos_idx: Sequence[int]) -> dict:
    """L1：以记录关节角做正运动学，与记录末端位姿比对。

    joint_qpos_idx: 关节在 qpos 中的下标（顺序须与数据列一一对应）
    返回位置残差（m）与朝向残差（度），**两个分量不合并**。
    """
    n = ep.n_frames
    pos_err = np.empty(n)
    rot_err = np.empty(n)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_site_or_body)

    for t in range(n):
        data.qpos[joint_qpos_idx] = ep.joints[t, :len(joint_qpos_idx)]
        mujoco.mj_forward(model, data)
        p = data.xpos[ee_id].copy()
        R_sim = data.xmat[ee_id].reshape(3, 3).copy()
        pos_err[t] = np.linalg.norm(p - ep.ee_xyz[t])

    if ep.ee_rpy is not None:
        # 批量算朝向：先取各帧仿真 R
        R_sim_all = np.empty((n, 3, 3))
        for t in range(n):
            data.qpos[joint_qpos_idx] = ep.joints[t, :len(joint_qpos_idx)]
            mujoco.mj_forward(model, data)
            R_sim_all[t] = data.xmat[ee_id].reshape(3, 3)
        R_rec = rpy_to_matrix(ep.ee_rpy)
        rot_err = geodesic_deg(R_sim_all, R_rec)

    return {
        'n': n,
        'pos_rms_m': float(np.sqrt((pos_err ** 2).mean())),
        'pos_median_m': float(np.median(pos_err)),
        'pos_p95_m': float(np.percentile(pos_err, 95)),
        'rot_rms_deg': float(np.sqrt((rot_err ** 2).mean())),
        'rot_median_deg': float(np.median(rot_err)),
        'rot_p95_deg': float(np.percentile(rot_err, 95)),
        # 明确不提供合并标量
    }


def summarize(d: dict) -> str:
    return (f"n={d['n']} | 位置 RMS={d['pos_rms_m']:.4f} m 中位={d['pos_median_m']:.4f} "
            f"p95={d['pos_p95_m']:.4f} | 朝向 RMS={d['rot_rms_deg']:.2f}° "
            f"中位={d['rot_median_deg']:.2f}° p95={d['rot_p95_deg']:.2f}°")