#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2.1 E1 残差计算：对状态量 npz 做 L1 正运动学比对。

口径（与 1.8／1.9 严格一致，不做改动）：
  · 关节角 → 官方模型正运动学 → 末端位姿
  · 与记录末端位姿比对：**位置用欧氏距离、朝向用测地距离，分两列报告，不合成标量**
  · 常差（坐标系/安装位）按本体一次性扣除：先拟合平移与旋转，报「原始」与「扣后」两个数
  · 阈值读 `ch110/l1_thresholds.json`，**位置与朝向各自取本体的档**

⚠ 关键实现约束（1.2c／1.11 实测得出）：
  · Franka 的 `joint_position` 是 (T,8) = **7 臂 + 1 夹爪，夹爪在索引 7**；
    `end_effector` 是 (T,6) = xyz + RPY。切臂列必须用 n_arm 而不是 shape-1，
    否则会把 joint7 当夹爪丢掉，FK 少一个自由度。
  · 朝向是 **RPY 欧拉角**，必须先转旋转矩阵再算测地距离，不能直接相减。

用法：
  python3 e1_residuals.py --dir <npz目录> --body franka
  python3 e1_residuals.py --dir <npz目录> --body franka --json-out x.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THRESH = os.path.join(HERE, '..', 'ch110', 'l1_thresholds.json')
CACHE = os.path.expanduser('~/Library/Caches/mujoco_menagerie/models')

MODELS = {
    'franka': ('franka_emika_panda-386f680ac533760a', 'panda.xml', 7),
    'ur5': ('universal_robots_ur5e-a92a363fa824f9d6', 'ur5e.xml', 6),
    'piper': ('agilex_piper-f9a26b1ea07606e3', 'piper.xml', 6),
}
N_ARM = {'franka': 7, 'ur5': 6, 'piper': 6, 'arx_x5': 6}

# ARX X5 只有 URDF，走 assets17（1.7 装好的那份，mesh 已在 assets/ 下）
X5_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'assets17', 'X5A.urdf')

# 末端参照 body —— **必须与数据方的定义一致**，否则会得到「恒定但很大」的偏移。
# 这组值来自 1.9 的 probe_fk_identity_multi.py，本轮逐一复验：
#   franka  hand    扣常差 RMS 0.000000 m（常差恒 [0,0,0]）→ 记录值就是它的 FK 输出
#   piper   link6   RMS 5.5 / 7.6 mm（左右臂）  → 与 1.9 的 8–12 mm 同量级
#   arx_x5  link6   RMS 1.68 mm，常差约 [0.097, 0, 0.156]（1.9 记的世界平移符号相反、
#                   数值一致，即「模型→数据」与「数据→模型」的差别）
EEF_BODY = {'franka': 'hand', 'ur5': 'wrist_3_link',
            'piper': 'link6', 'arx_x5': 'link6'}


# ---------------------------------------------------------------- 姿态工具
def rpy_to_matrix(rpy):
    """RPY（固定轴 XYZ，弧度）→ 旋转矩阵，(N,3)。"""
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


def geodesic_deg(Ra, Rb):
    """旋转矩阵对的测地距离（度）。逐帧算，不对欧拉角直接相减。"""
    Rrel = np.einsum('nji,njk->nik', Ra, Rb)
    tr = np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def load_model(body):
    import mujoco
    if body == 'arx_x5':
        p = os.path.abspath(X5_URDF)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        m = mujoco.MjModel.from_xml_path(p)
        return mujoco, m, mujoco.MjData(m)
    d, f, _ = MODELS[body]
    p = os.path.join(CACHE, d, f)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    m = mujoco.MjModel.from_xml_path(p)
    return mujoco, m, mujoco.MjData(m)


def _eef_body_id(m, body):
    """取真正的末端 body —— **按名字查表，不用 `m.nbody - 1`**。

    ⚠ 本轮实测的坑：Panda 的 body 序列末尾是 `left_finger` / `right_finger`，
    `nbody-1` 拿到的是**手指**，不是法兰。用错的后果（600 帧 Franka 样本）：

        用 nbody-1（right_finger）→ 位置残差 3.4e-2 m、朝向 139°
        用 hand（正确）          → 位置残差 4.0e-8 m、朝向 6e-6 °

    差 6 个数量级，但**不报错、形状对、量纲也对**。若不做交叉验证，
    会把 1.9 已确认的「同源档」误判为「不达标」。
    正解由 `EEF_BODY` 表给出，来源是 1.9 并已在本轮复验。
    """
    name = EEF_BODY.get(body)
    names = {m.body(i).name: i for i in range(m.nbody)}
    if name and name in names:
        return names[name]
    # 兜底：按常见命名依次尝试，最后才用末个 body（并告警）
    for cand in ('hand', 'link6', 'tool0', 'ee_link', 'wrist_3_link', 'flange'):
        if cand in names:
            print(f'  [warn] {body}: EEF_BODY 缺失，回退到 {cand}')
            return names[cand]
    print(f'  [warn] {body}: 未匹配到末端 body，回退 nbody-1（可能不准）')
    return m.nbody - 1


def fk_positions(body, q):
    """q (T,n_arm) → 末端位置 (T,3)。取最后一个 body 的位置。"""
    mujoco, m, d = load_model(body)
    eef = _eef_body_id(m, body)
    n = min(q.shape[1], m.nq)
    out = np.empty((len(q), 3))
    for i, row in enumerate(q):
        d.qpos[:] = 0.0
        d.qpos[:n] = row[:n]
        mujoco.mj_kinematics(m, d)
        out[i] = d.xpos[eef]
    return out


def kabsch(A, B):
    """求 R,t 使 R@A_i + t ≈ B_i（最小二乘）。返回 (R, t)。"""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cb - R @ ca
    return R, t


def analyse(npz_path, body):
    z = np.load(npz_path)
    # 成功侧键为 `puppet__joint_position`；**失败侧实测为
    # `puppet__arm_joint_position`(T,8)**（2026-11-09 实测）。
    # 8 列 = 7 臂关节 + 1 夹爪，且第 8 列与 `hand_joint_position` 完全相同
    # （三条样本逐列量程一致），故臂关节取前 n_arm 列。
    key = 'puppet__joint_position'
    if key not in z:
        key = 'puppet__arm_joint_position'
    if key not in z:
        return {'file': os.path.basename(npz_path), 'error': 'no puppet/joint_position 或 arm_joint_position'}
    Q = z[key]
    n_arm = N_ARM[body]
    if Q.shape[1] < n_arm:
        return {'file': os.path.basename(npz_path),
                'error': f'列数 {Q.shape[1]} < 臂关节数 {n_arm}'}
    q = Q[:, :n_arm]

    eekey = 'puppet__end_effector'
    if eekey not in z:
        return {'file': os.path.basename(npz_path), 'error': 'no end_effector'}
    E = z[eekey]
    pos_rec = E[:, :3]
    rot_rec = rpy_to_matrix(E[:, 3:6])

    # 冻结守卫：1.2c 的教训，退化数据不能得出「一致」的假结论
    if float(np.ptp(q, axis=0).max()) < 1e-9:
        return {'file': os.path.basename(npz_path), 'verdict': 'SKIP_FROZEN_JOINTS'}
    if float(np.ptp(pos_rec, axis=0).max()) < 1e-9:
        return {'file': os.path.basename(npz_path), 'verdict': 'SKIP_FROZEN_REFERENCE'}

    T = len(q)
    pos_fk = fk_positions(body, q)

    # 原始残差
    d_raw = np.linalg.norm(pos_fk - pos_rec, axis=1)
    rot_raw = geodesic_deg(np.tile(np.eye(3), (T, 1, 1)), rot_rec)  # 仅作占位说明
    # 朝向：FK 无末端朝向输出（只取位置），故朝向残差以
    # 「记录朝向 vs 记录朝向的首帧」不成立——必须用模型末端 body 的旋转。
    # 这里改为取模型 eef body 的 xmat，需重跑 FK 取旋转。
    rot_fk = fk_rotations(body, q)
    rot_raw = geodesic_deg(rot_fk, rot_rec)

    # 扣常差：位置用 Kabsch（含旋转），朝向用「相对旋转的中位」
    R_c, t_c = kabsch(pos_fk, pos_rec)
    pos_aligned = (R_c @ pos_fk.T).T + t_c
    d_after = np.linalg.norm(pos_aligned - pos_rec, axis=1)

    # 朝向常差：R_fk 与 R_rec 之间差一个**固定旋转**（Franka 实测为绕 z 轴 45°，
    # 与 1.9 记录一致）。求法：逐帧算 R_fk^-1 R_rec 再取「中位旋转」。
    # ⚠ 不能用「矩阵元素和做 SVD」近似——当相对旋转恰为 45° 时，矩阵和的特征结构
    # 会退化，SVD 给出的主方向不对，结果仍差 90°（本轮实际踩到：位置已到 4e-8 m，
    # 朝向却停在 89.99°）。改用四元数特征向量法，数值稳定。
    rel = np.einsum('nji,njk->nik', rot_fk, rot_rec)   # rel = R_fk^T R_rec
    R_med = _mean_rotation(rel)
    # 对齐方向：rel = R_fk^T R_rec ⇒ R_rec = R_fk @ rel ⇒ 用 **右乘** R_fk @ R_med。
    # ⚠ 写成左乘 R_med @ R_fk 会得到 90° 的残余（本轮实际踩到：位置已 4e-8 m，
    # 朝向却卡在 89.99°）。右乘后实测 max 6e-6 °，与 1.9 记录一致。
    rot_after = geodesic_deg(np.einsum('nij,jk->nik', rot_fk, R_med), rot_rec)

    return {
        'file': os.path.basename(npz_path),
        'body': body, 'n_frames': int(T), 'n_arm': int(n_arm),
        'pos_raw_m': round(float(d_raw.max()), 9),
        'pos_after_m': round(float(d_after.max()), 9),
        'pos_after_p95_m': round(float(np.percentile(d_after, 95)), 9),
        'rot_raw_deg': round(float(rot_raw.max()), 6),
        'rot_after_deg': round(float(rot_after.max()), 6),
        'rot_after_p95_deg': round(float(np.percentile(rot_after, 95)), 6),
        'verdict': 'OK',
    }


def fk_rotations(body, q):
    """q (T,n) → 末端 body 的旋转矩阵 (T,3,3)。"""
    mujoco, m, d = load_model(body)
    eef = _eef_body_id(m, body)
    n = min(q.shape[1], m.nq)
    out = np.empty((len(q), 3, 3))
    for i, row in enumerate(q):
        d.qpos[:] = 0.0
        d.qpos[:n] = row[:n]
        mujoco.mj_kinematics(m, d)
        out[i] = d.xmat[eef].reshape(3, 3)
    return out


def _mean_rotation(Rs):
    """一组旋转矩阵的平均旋转（四元数特征向量法，数值稳定）。

    做法：每个 R → 四元数 → 堆成 4×4 矩阵 → 取最大特征值对应的特征向量 → 归一化回旋转。
    比「矩阵元素求和再 SVD」稳健：后者在相对旋转接近 45° 时特征结构退化。
    """
    n = len(Rs)
    Q = np.empty((n, 4))
    for i, R in enumerate(Rs):
        tr = np.trace(R)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            Q[i] = [0.25 * s,
                    (R[2, 1] - R[1, 2]) / s,
                    (R[0, 2] - R[2, 0]) / s,
                    (R[1, 0] - R[0, 1]) / s]
        else:
            m = np.argmax(np.diag(R))
            if m == 0:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                Q[i] = [(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                        (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
            elif m == 1:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                Q[i] = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                        0.25 * s, (R[1, 2] + R[2, 1]) / s]
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                Q[i] = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                        (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    # 统一半球（避免 q 与 -q 抵消）
    ref = Q[0]
    sign = np.sign(Q @ ref)
    sign[sign == 0] = 1
    Q = Q * sign[:, None]
    A = Q.T @ Q
    w, V = np.linalg.eigh(A)
    q = V[:, np.argmax(w)]
    q = q / np.linalg.norm(q)
    w_, x_, y_, z_ = q
    return np.array([
        [1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_)],
        [2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - x_ * w_)],
        [2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)],
    ])


def quat_to_matrix(quats):
    """四元数 [qw,qx,qy,qz]（RoboDojo 约定）→ 旋转矩阵 (N,3,3)。

    ⚠ **必须先做符号连续化**。q 与 −q 表示同一旋转，但原始数据里四元数会
    在相邻帧间翻符号（同一姿态写成 q 又写成 −q）。单帧转矩阵本身没问题
    （矩阵相同），可一旦**跨帧做统计**（取平均旋转、逐帧比残差），
    翻符号的帧会出现异常。实测 `put_objects_into_basket/episode_0000007`：
    朝向残差 max 达 **180.000°**，而 **P95 只有 0.34°** —— 就是翻符号那一帧。
    做法：与前一帧点积为负则整体取反，保证相邻帧在同一半球。
    """
    q = np.atleast_2d(quats).astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
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


def analyse_robodojo(src, body, side='left'):
    """RoboDojo real 单条样本：`state/{side}_arm_joint_states` + `{side}_ee_poses`。

    支持两种输入：
      · `.hdf5` —— 原始完整文件（键为 `state/<name>`）
      · `.npz`  —— `fetch_robodojo.py` 的 slim 产物（键为 `state__<name>`）

    与 RoboMIND 的两处差异必须区分：
      · 末端位姿是 **四元数**（[qw,qx,qy,qz]），不是 RPY；
      · 是**双臂**，须指定 side（默认 left）。
    """
    if src.endswith('.npz'):
        z = np.load(src)
        kq, kp = f'state__{side}_arm_joint_states', f'state__{side}_ee_poses'
        if kq not in z or kp not in z:
            return {'file': os.path.basename(src), 'error': f'缺 {kq} 或 {kp}'}
        q, P = z[kq], z[kp]
    else:
        import h5py
        with h5py.File(src, 'r') as h:
            kq, kp = f'state/{side}_arm_joint_states', f'state/{side}_ee_poses'
            if kq not in h or kp not in h:
                return {'file': os.path.basename(src), 'error': f'缺 {kq} 或 {kp}'}
            q, P = h[kq][()], h[kp][()]
    n_arm = N_ARM[body]
    q = q[:, :n_arm]
    pos_rec, rot_rec = P[:, :3], quat_to_matrix(P[:, 3:7])

    if float(np.ptp(q, axis=0).max()) < 1e-9:
        return {'file': os.path.basename(src), 'verdict': 'SKIP_FROZEN_JOINTS'}
    if float(np.ptp(pos_rec, axis=0).max()) < 1e-9:
        return {'file': os.path.basename(src), 'verdict': 'SKIP_FROZEN_REFERENCE'}

    pos_fk = fk_positions(body, q)
    rot_fk = fk_rotations(body, q)
    d_raw = np.linalg.norm(pos_fk - pos_rec, axis=1)
    rot_raw = geodesic_deg(rot_fk, rot_rec)

    R_c, t_c = kabsch(pos_fk, pos_rec)
    d_after = np.linalg.norm((R_c @ pos_fk.T).T + t_c - pos_rec, axis=1)

    rel = np.einsum('nji,njk->nik', rot_fk, rot_rec)
    R_med = _mean_rotation(rel)
    rot_after = geodesic_deg(np.einsum('nij,jk->nik', rot_fk, R_med), rot_rec)

    return {
        'file': os.path.basename(src), 'body': body, 'side': side,
        'n_frames': int(len(q)), 'n_arm': int(n_arm),
        'pos_raw_m': round(float(d_raw.max()), 9),
        'pos_after_m': round(float(d_after.max()), 9),
        'pos_after_p95_m': round(float(np.percentile(d_after, 95)), 9),
        'rot_raw_deg': round(float(rot_raw.max()), 6),
        'rot_after_deg': round(float(rot_after.max()), 6),
        'rot_after_p95_deg': round(float(np.percentile(rot_after, 95)), 6),
        'const_offset_m': [round(float(x), 5) for x in
                           ((R_c @ pos_fk.T).T + t_c - pos_rec).mean(0)],
        'verdict': 'OK',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--body', default='franka')
    ap.add_argument('--side', default='left',
                    help='RoboDojo 有双臂；RoboMIND 忽略此参数')
    ap.add_argument('--prefix', default=None,
                    help='只跑文件名以此开头的（同一目录下多本体混放时必须给，'
                         '否则会用错模型算，得到「全都 FAIL」的假结果）')
    ap.add_argument('--json-out', default=None)
    a = ap.parse_args()

    def _pick(pats):
        out = []
        for pat in pats:
            for p in glob.glob(os.path.join(a.dir, pat)):
                if os.path.basename(p).startswith('._'):
                    continue
                if a.prefix and not os.path.basename(p).startswith(a.prefix):
                    continue
                out.append(p)
        return sorted(set(out))

    files = _pick(['*.npz'])
    h5s = _pick(['*.hdf5'])
    # 数据源判别：看 npz 里有没有 RoboDojo 的 `state__*` 键。
    # RoboMIND 的 slim 产物是 `puppet__joint_position` 这种，RoboDojo 是 `state__*`。
    rd_npz = []
    if not h5s:
        for f in files:
            try:
                with np.load(f) as z:
                    if any(k.startswith('state__') for k in z.files):
                        rd_npz.append(f)
            except Exception:
                pass
    is_rd = bool(h5s) or bool(rd_npz)
    srcs = (h5s + rd_npz) if is_rd else files
    if not srcs:
        print(f'{a.dir} 下无匹配文件（prefix={a.prefix}）')
        return 1

    th = json.load(open(THRESH, encoding='utf-8'))['bodies'].get(a.body, {})
    thr_pos = th.get('threshold_pos_m')
    thr_rot = th.get('threshold_rot_deg')
    print(f'本体 {a.body}｜阈值 位置 {thr_pos:.2e} m｜朝向 {thr_rot:.2e} °'
          f'（档：{th.get("tier_pos")}/{th.get("tier_rot")}）')
    print(f'{"文件":<26s} {"帧":>5s} {"位置P95(m)":>13s} {"朝向P95(°)":>13s} {"判定":>8s}')
    print('-' * 74)

    rows = []
    for f in srcs:
        try:
            r = (analyse_robodojo(f, a.body, a.side) if is_rd
                 else analyse(f, a.body))
        except Exception as e:
            r = {'file': os.path.basename(f), 'error': f'{type(e).__name__}: {e}'}
        rows.append(r)
        if 'error' in r:
            print(f'{r["file"][:26]:<26s} {"—":>5s} {"—":>13s} {"—":>13s} {"ERR":>8s}')
        elif r.get('verdict', '').startswith('SKIP'):
            print(f'{r["file"][:26]:<26s} {"—":>5s} {"—":>13s} {"—":>13s} {r["verdict"][:8]:>8s}')
        else:
            # 判据用 **P95 而非 max**，与 1.9 的「取第 95 百分位」口径一致。
            # 理由（实测）：RoboDojo 存在个别帧的四元数退化（qw=0 的 180° 边界，
            # 如 put_objects_into_basket/episode_0000007 第 42 帧），单帧 max 直接
            # 判 FAIL 会把 8 条本来正常的样本打成不通过。P95 对该异常免疫。
            ok = (thr_pos is not None and r['pos_after_p95_m'] <= thr_pos
                  and thr_rot is not None and r['rot_after_p95_deg'] <= thr_rot)
            r['max_flags'] = []
            if r['pos_after_m'] > (thr_pos or 1e9) * 3:
                r['max_flags'].append('pos_max_outlier')
            if r['rot_after_deg'] > (thr_rot or 1e9) * 3:
                r['max_flags'].append('rot_max_outlier')
            r['pass'] = ok
            print(f'{r["file"][:26]:<26s} {r["n_frames"]:>5d} '
                  f'{r["pos_after_p95_m"]:>13.3e} {r["rot_after_p95_deg"]:>13.4f} '
                  f'{"PASS" if ok else "FAIL":>8s}')

    good = [r for r in rows if r.get('verdict') == 'OK']
    if good:
        p = np.array([r['pos_after_p95_m'] for r in good])
        t = np.array([r['rot_after_p95_deg'] for r in good])
        print('-' * 74)
        print(f'汇总 {len(good)} 条可算：位置 P95 跨条中位 {np.median(p):.3e} m｜'
              f'朝向 P95 中位 {np.median(t):.4f} °')
        print(f'通过 {sum(1 for r in good if r.get("pass"))}/{len(good)} 条')

    if a.json_out:
        json.dump(rows, open(a.json_out, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        print(f'已写出 {a.json_out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())