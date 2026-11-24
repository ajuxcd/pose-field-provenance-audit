#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L2 开环物理回放（B 档回放管线 v1，新增 L2 层）。

## L2 是什么（口径来自初稿 3.2／3.4）

「以记录的动作指令驱动仿真，**令物理引擎自行演化**，得到回放轨迹后与真机轨迹比对。」
开环意味着**不根据观测修正**——把记录的关节指令按时序喂进去，让 MuJoCo 自己积分。

残差定义：L2 末端位置 与 记录末端位置 的距离；Δ = e_L2 − e_L1。

## 控制器增益的出处（不自己发明）

位置执行器增益**直接取模型自带的**（MuJoCo Menagerie 官方配置），这是能拿到的最可靠来源：

| 本体 | 关节 | kp | kv | 来源 |
|---|---|---|---|---|
| Franka | j1–j2 | 4500 | 450 | `panda.xml` gainprm/biasprm |
| Franka | j3–j4 | 3500 | 350 | 同上 |
| Franka | j5–j7 | 2000 | 200 | 同上 |
| Piper | j1–j3 | 80 | 5 | `piper.xml` |
| Piper | j4 | 40 | 5 | 同上 |
| Piper | j5–j6 | 10 | 1.5 | 同上 |
| **ARX X5** | 全部 | **100** | **2** | **自设**（URDF 无执行器，须在正文声明） |

X5 自设值取 Piper 同量级（两者都是 6 关节轻量协作臂，Piper 近端 80、远端 10），
取 100／2 作中值。**该值必须在 2.4 的参数扫描里被覆盖**，否则 Δ 的绝对值不可解释。

## 三个必须写在结果里的限制

1. **无物体回放**。v1.0 不含物体位姿（1.5 已核），无法重建场景，故本版**只有臂与地面**，
   不含接触。→ Δ 只反映「臂动力学 + 控制器跟踪」，**接触贡献不在其中**。这一条限定了 Δ 的解释边界。
2. **帧率未知**。v1.0 无时间戳，官方未规定采样率。本文按 **30 Hz** 处理（与 RoboDojo 一致），
   5 秒 = 150 帧。若真实帧率不同，分段长度与子步数等比变化，但残差量级不变。
3. **段首瞬态**。每段独立初始化，段首速度用前两帧有限差分估计；估计不准会在段首制造
   几十毫秒的过渡，故统计时**丢弃每段前 3 帧**。

## 用法

    python3 replay_l2.py --selftest                    # 单条验证（Franka）
    python3 replay_l2.py --dir <npz目录> --body piper --side left
    python3 replay_l2.py --dir ... --json-out x.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser('~/Library/Caches/mujoco_menagerie/models')
X5_URDF = os.path.join(HERE, '..', 'assets17', 'X5A.urdf')

MODELS = {
    'franka': ('franka_emika_panda-386f680ac533760a', 'panda.xml'),
    'ur5': ('universal_robots_ur5e-a92a363fa824f9d6', 'ur5e.xml'),
    'piper': ('agilex_piper-f9a26b1ea07606e3', 'piper.xml'),
}
N_ARM = {'franka': 7, 'ur5': 6, 'piper': 6, 'arx_x5': 6}
EEF_BODY = {'franka': 'hand', 'ur5': 'wrist_3_link',
            'piper': 'link6', 'arx_x5': 'link6'}

# 仅 X5 用（URDF 无执行器）。见文件头说明。
X5_KP, X5_KV = 200.0, 4.0

# **增益缩放（2.4 参数扫描用）**：1.0 = 基准。有执行器的本体缩 gainprm[0] 与
# biasprm[1]、biasprm[2]（位置伺服力的 kp 与 kv）；X5 无执行器，缩 X5_KP/X5_KV。
# 只动增益，其余参数与 2.1 完全一致。
GAIN_SCALE = 1.0
# X5 的 URDF 缺 armature / damping（实测全为 0），直接仿真会在 10 步内发散并被
# MuJoCo 重置——表现为「稳态误差恒定且不随增益变化」。官方 Menagerie 的同类本体
# 都带这两项（Franka armature=0.1/damping=1.0；Piper 0.005/0.0），故补齐。
# 取 0.01 介于两者之间；扫描 0.005–0.05 结果不变（稳态误差 1.58°），说明不敏感。
X5_ARMATURE, X5_DAMPING = 0.01, 0.5

# v1.0 真机数据无时间戳，按 30 Hz 处理（按任务时长推断合理）。
# **仿真侧不同**：RoboMIND 孪生的 `sim_time_seq` 实测 dt = 1/120 s，即 120 Hz。
# 若按 30 Hz 回放 120 Hz 的数据，时间被压 4 倍，臂跟不上，残差会虚高到数百毫米
# （本轮实测 360–400 mm）。故 FPS 必须按数据集给，用 --fps 指定。
FPS = 30.0
SEG_SEC = 5.0         # 初稿：长任务切为不超过五秒的片段
N_SUB = 20            # 每数据帧的物理子步数 → sim dt = 1/(30*20) ≈ 1.667 ms
WARMUP = 3            # 每段丢弃前 3 帧（段首瞬态）


# ---------------------------------------------------------------- 姿态工具
def rpy_to_matrix(rpy):
    rpy = np.atleast_2d(rpy)
    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
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


def quat_to_matrix(quats):
    """[qw,qx,qy,qz] → 旋转矩阵；含符号连续化（见 e1_residuals.py 的同类说明）。"""
    q = np.atleast_2d(quats).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
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


def geodesic_deg(Ra, Rb):
    Rrel = np.einsum('nji,njk->nik', Ra, Rb)
    tr = np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def mean_rotation(Rs):
    """一组旋转的平均（四元数特征向量法）。

    用途：求 sim 与记录之间的**恒定旋转差**（Franka 实测为绕 z 45°，是模型 hand 体
    与数据 RPY 表示之间的约定差，不是物理误差）。**L2 必须与 L1 扣同一类常差**，
    否则 Δ 里会混进 45°。
    """
    Q = np.empty((len(Rs), 4))
    for i, R in enumerate(Rs):
        tr = np.trace(R)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            Q[i] = [0.25 * s, (R[2, 1] - R[1, 2]) / s,
                    (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        else:
            k = int(np.argmax(np.diag(R)))
            if k == 0:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                Q[i] = [(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                        (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
            elif k == 1:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                Q[i] = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                        0.25 * s, (R[1, 2] + R[2, 1]) / s]
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                Q[i] = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                        (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    sign = np.sign(Q @ Q[0])
    sign[sign == 0] = 1
    Q = Q * sign[:, None]
    w, V = np.linalg.eigh(Q.T @ Q)
    a, b, c, e = V[:, np.argmax(w)]
    n = np.sqrt(a * a + b * b + c * c + e * e)
    a, b, c, e = a / n, b / n, c / n, e / n
    return np.array([
        [1 - 2 * (c * c + e * e), 2 * (b * c - e * a), 2 * (b * e + c * a)],
        [2 * (b * c + e * a), 1 - 2 * (b * b + e * e), 2 * (c * e - b * a)],
        [2 * (b * e - c * a), 2 * (c * e + b * a), 1 - 2 * (b * b + c * c)],
    ])


# ---------------------------------------------------------------- 模型与控制
def load_model(body):
    import mujoco
    if body == 'arx_x5':
        p = os.path.abspath(X5_URDF)
        m = mujoco.MjModel.from_xml_path(p)
        m.opt.timestep = 1.0 / (FPS * N_SUB)
        # 补 URDF 缺失的关节属性（见 X5_ARMATURE 注释）
        m.dof_armature[:N_ARM[body]] = X5_ARMATURE
        m.dof_damping[:N_ARM[body]] = X5_DAMPING
        # 夹爪两个 slide DOF 不参与回放，锁死以免其无阻尼发散
        m.dof_armature[N_ARM[body]:] = 0.01
        m.dof_damping[N_ARM[body]:] = 10.0
        # URDF 的 mesh 碰撞体互相碰撞（contype/conaffinity=1），无物体回放不需要，
        # 关掉可消除非凸 mesh 自碰撞造成的数值不稳定。
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        return mujoco, m, mujoco.MjData(m)
    d, f = MODELS[body]
    p = os.path.join(CACHE, d, f)
    m = mujoco.MjModel.from_xml_path(p)
    # 改 timestep 使「每数据帧恰好 N_SUB 步」，避免时基漂移（默认 0.002 会让 30 Hz 差 4%）
    m.opt.timestep = 1.0 / (FPS * N_SUB)
    if GAIN_SCALE != 1.0:
        # 位置执行器：force = gainprm[0]*ctrl + biasprm[1]*qpos + biasprm[2]*qvel，
        # 故 kp 在 gainprm[0] 与 biasprm[1]（符号相反），kv 在 biasprm[2]。
        for a in _arm_actuator_ids(m, N_ARM[body]):
            m.actuator_gainprm[a, 0] *= GAIN_SCALE
            m.actuator_biasprm[a, 1] *= GAIN_SCALE
            m.actuator_biasprm[a, 2] *= GAIN_SCALE
    return mujoco, m, mujoco.MjData(m)


def _eef_id(m, body):
    names = {m.body(i).name: i for i in range(m.nbody)}
    return names.get(EEF_BODY.get(body), m.nbody - 1)


def _arm_actuator_ids(m, n_arm):
    """取前 n_arm 个关节执行器的 id。Franka 的夹爪是 trntype=3，Piper 的夹爪是第 7 个。"""
    import mujoco
    ids = []
    for a in range(m.nu):
        if int(m.actuator_trntype[a]) != 0:   # 0 = mjTRN_JOINT
            continue          # 跳 tendon 型（Franka 夹爪）
        ids.append(a)
        if len(ids) >= n_arm:
            break
    return ids


def _joint_qpos_idx(m, n_arm):
    """前 n_arm 个铰链关节在 qpos 中的下标（X5 用）。"""
    import mujoco
    # jnt_type 是 uint8：0=free 1=ball 2=slide 3=hinge。用整数比较，
    # 与 mjtJoint 枚举比较会因 numpy 标量类型而失配（本轮实际踩到：返回空下标）。
    idx = [j for j in range(m.njnt) if int(m.jnt_type[j]) in (2, 3)]
    return idx[:n_arm]


def replay_segment(mujoco, m, d, q_cmd, body, eef_id, act_ids, qpos_idx):
    """回放一段：q_cmd (T,n_arm) → 末端位置 (T,3) 与 (T,3,3)。

    **开环**：没有任何反馈修正，指令序列一次性给定，引擎自行积分。
    """
    n_arm = q_cmd.shape[1]
    use_act = len(act_ids) > 0
    kp = np.array([X5_KP * GAIN_SCALE] * n_arm) if not use_act else None
    kv = np.array([X5_KV * GAIN_SCALE] * n_arm) if not use_act else None

    # 段首状态
    d.qpos[:] = m.qpos0
    d.qpos[qpos_idx] = q_cmd[0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    T = len(q_cmd)
    pos = np.empty((T, 3))
    rot = np.empty((T, 3, 3))
    pos[0] = d.xpos[eef_id]
    rot[0] = d.xmat[eef_id].reshape(3, 3)

    for t in range(1, T):
        target = q_cmd[t]
        for _ in range(N_SUB):
            if use_act:
                for k, a in enumerate(act_ids):
                    d.ctrl[a] = target[k]
            else:
                q = d.qpos[qpos_idx]
                v = d.qvel[qpos_idx]
                d.qfrc_applied[qpos_idx] = kp * (target - q) - kv * v
            mujoco.mj_step(m, d)
        pos[t] = d.xpos[eef_id]
        rot[t] = d.xmat[eef_id].reshape(3, 3)
    return pos, rot


# ---------------------------------------------------------------- 数据读取
def load_episode(path, body, side='left'):
    """返回 dict(joints, ee_xyz, ee_rot) 或 None。兼容 npz 两种 schema。"""
    n_arm = N_ARM[body]
    if path.endswith('.npz'):
        z = np.load(path)
        if f'state__{side}_arm_joint_states' in z:          # RoboDojo
            q = z[f'state__{side}_arm_joint_states'][()]
            P = z[f'state__{side}_ee_poses'][()]
            return {'joints': q[:, :n_arm], 'ee_xyz': P[:, :3],
                    'ee_rot': quat_to_matrix(P[:, 3:7])}
        # RoboMIND 成功侧是 `puppet__joint_position`；**失败侧实测是
        # `puppet__arm_joint_position`(T,8)**（2026-11-09 实测，第 8 列与
        # `hand_joint_position` 相同，是夹爪重复列）。
        # **按后缀找键**，不写死 `puppet__` 前缀：仿真侧存的是 `franka__joint_position`
        # 与 `franka__end_effector`（7 列 = xyz + 四元数，不是真机的 6 列 RPY）。
        # ⚠ **不能直接 next(...) 取第一个匹配键**：RoboMIND 成功侧的 npy 里
        # `master__joint_position` 排在 `puppet__joint_position` 之前（zip 内字母序），
        # 且 master 侧没有 `master__end_effector`，于是函数返回 None——90 条全部读不出，
        # 而报错形式是「静默的 0 条」。2026-11-09 实际踩到（噪声实验时发现）。
        # 且即便 master 有 end_effector 也不能用：puppet 与 master 是同源的两侧，
        # 逐列差 0.006–0.036 rad（主从跟踪误差），本文一律取 puppet。
        def _pick(z):
            for k in ('puppet__joint_position', 'puppet__arm_joint_position',
                      'franka__joint_position', 'franka__arm_joint_position'):
                if k in z:
                    return k
            return next((k for k in z.files
                         if (k.endswith(('__joint_position', '__arm_joint_position'))
                             and not k.startswith('master__'))), None)

        jkey = _pick(z)
        if jkey is not None:
            prefix = jkey.split('__')[0]
            ek = f'{prefix}__end_effector'
            if ek in z:
                q = z[jkey][()]
                E = z[ek][()]
                # **双臂数据**：joint_position 是 (T,16) = 左7+左夹爪+右7+右夹爪，
                # end_effector 是 (T,12) = 左 xyz+RPY + 右 xyz+RPY。
                # 若不加区分地取前 7 列，会回放一支**没在动的臂**——
                # 本轮实测 `right_slide_*` 是右手任务、左臂速度恒为 0，
                # 于是 L2 残差只有 0.3 mm，看着像"完美回放"，实为口径错误。
                if q.shape[1] >= 16 and E.shape[1] >= 12:
                    if side == 'right':
                        return {'joints': q[:, 8:15], 'ee_xyz': E[:, 6:9],
                                'ee_rot': rpy_to_matrix(E[:, 9:12]),
                                'dual': True}
                    return {'joints': q[:, :7], 'ee_xyz': E[:, :3],
                            'ee_rot': rpy_to_matrix(E[:, 3:6]),
                            'dual': True}
                # 6 列 = xyz + RPY（真机）；7 列 = xyz + 四元数（仿真侧，Isaac Sim 导出）。
                if E.shape[1] >= 7:
                    return {'joints': q[:, :n_arm], 'ee_xyz': E[:, :3],
                            'ee_rot': quat_to_matrix(E[:, 3:7]), 'dual': False}
                return {'joints': q[:, :n_arm], 'ee_xyz': E[:, :3],
                        'ee_rot': rpy_to_matrix(E[:, 3:6]), 'dual': False}
        return None
    import h5py
    with h5py.File(path, 'r') as h:
        if f'state/{side}_arm_joint_states' in h:
            q = h[f'state/{side}_arm_joint_states'][()]
            P = h[f'state/{side}_ee_poses'][()]
            return {'joints': q[:, :n_arm], 'ee_xyz': P[:, :3],
                    'ee_rot': quat_to_matrix(P[:, 3:7])}
        if 'puppet/joint_position' in h:
            q = h['puppet/joint_position'][()]
            E = h['puppet/end_effector'][()]
            return {'joints': q[:, :n_arm], 'ee_xyz': E[:, :3],
                    'ee_rot': rpy_to_matrix(E[:, 3:6])}
    return None


def analyse(path, body, side='left', seg_frames=None):
    # 双臂任务按任务名前缀选臂：left_* 用左臂、right_* 用右臂、
    # both_* 两臂都跑取较大者（协作任务两臂都在动，取较大更能反映失配）。
    if side == 'auto':
        base = os.path.basename(path).lower()
        if base.startswith('right_'):
            side = 'right'
        elif base.startswith('both_'):
            a = analyse(path, body, 'left', seg_frames)
            b = analyse(path, body, 'right', seg_frames)
            if a.get('verdict') != 'OK':
                return b
            if b.get('verdict') != 'OK':
                return a
            return a if a['pos_p95_m'] >= b['pos_p95_m'] else b
        else:
            side = 'left'
    ep = load_episode(path, body, side)
    if ep is None:
        return {'file': os.path.basename(path), 'error': 'schema 未识别'}

    mujoco, m, d = load_model(body)
    eef_id = _eef_id(m, body)
    act_ids = _arm_actuator_ids(m, N_ARM[body])
    qpos_idx = _joint_qpos_idx(m, N_ARM[body])

    T = len(ep['joints'])
    L = seg_frames or int(FPS * SEG_SEC)
    d_pos, d_rot, d_rot_raw, n_used, n_seg = [], [], [], 0, 0
    segs = []
    for s in range(0, T - WARMUP - 1, L):
        e = min(s + L, T)
        if e - s < WARMUP + 2:
            break
        pos, rot = replay_segment(mujoco, m, d, ep['joints'][s:e],
                                  body, eef_id, act_ids, qpos_idx)
        segs.append((s, e, pos, rot))

    # **位置常差**：sim 与记录的安装位差（E1 已逐本体量过：Franka 为 0、
    # ARX X5 约 184 mm、Piper 十几毫米）。不扣的话 L2 残差被常差主导，
    # Δ = e_L2 − e_L1 就不可比（L1 是扣过常差的）。
    # 做法与 E1 一致：用整条轨迹的均值位移**一次性**估计，所有段共用同一偏移。
    if segs:
        off = np.concatenate([p[WARMUP:] - ep['ee_xyz'][s:e][WARMUP:]
                              for s, e, p, _ in segs]).mean(0)
    else:
        off = np.zeros(3)

    d_pos_end, d_rot_end = [], []
    for s, e, pos, rot in segs:
        sl = slice(WARMUP, e - s)
        seg_d = np.linalg.norm(pos[sl] - off - ep['ee_xyz'][s:e][sl], axis=1)
        d_pos.append(seg_d)
        # **段末残差**：只取该段最后 10% 的帧，代表「该段累积到的最坏失配」。
        # 这个口径与段数无关——把「所有帧混算 P95」的做法会因段数多而被系统性拉低，
        # 详见 NOTES_l2.md 的「坑五」。
        k = max(1, int(round(len(seg_d) * 0.10)))
        d_pos_end.append(seg_d[-k:])
        # 朝向的段末残差与位置同 k（同段、同末 10%）。**必须在算完 rot 之后再取**，
        # 否则 d_rot_end 一直是空列表，np.concatenate 会抛
        # 「need at least one array to concatenate」——本轮先加了输出字段却没加追加，
        # 直接把三个 e2_*_l2.json 写成了错误记录（已重跑还原）。
        # 朝向：扣「模型 body 与数据表示」的恒定旋转差后才是可比量。
        # 不扣的话 Franka 会停在 45°（1.8 已查明是 hand 体与 RPY 约定的差）。
        rot_rec = ep['ee_rot'][s:e][sl]
        rel = np.einsum('nji,njk->nik', rot[sl], rot_rec)
        R_med = mean_rotation(rel)
        aligned = np.einsum('nij,jk->nik', rot[sl], R_med)
        seg_r = geodesic_deg(aligned, rot_rec)
        d_rot.append(seg_r)
        d_rot_end.append(seg_r[-k:])
        d_rot_raw.append(geodesic_deg(rot[sl], rot_rec))
        n_used += (e - s) - WARMUP
        n_seg += 1
    if not d_pos:
        return {'file': os.path.basename(path), 'verdict': 'TOO_SHORT',
                'n_frames': int(T)}

    p = np.concatenate(d_pos)
    r = np.concatenate(d_rot)
    return {
        'file': os.path.basename(path), 'body': body, 'side': side,
        'n_frames': int(T), 'n_segments': n_seg, 'n_used': int(n_used),
        'ctrl': ('actuator' if act_ids
                 else f'pd(kp={X5_KP * GAIN_SCALE:g},kv={X5_KV * GAIN_SCALE:g})'),
        'gain_scale': GAIN_SCALE,
        'armature': None if act_ids else X5_ARMATURE,
        'pos_offset_m': [round(float(x), 5) for x in off],
        'pos_median_m': float(np.median(p)),
        'pos_p95_m': float(np.percentile(p, 95)),
        'pos_max_m': float(p.max()),
        'pos_end_median_m': float(np.median(np.concatenate(d_pos_end))),
        # **逐段**的段末残差中位数（每段末 10% 帧的中位数）。给 3.x 的
        # 「段级重采样 + 按轨迹聚类 bootstrap」用：段在回放时独立初始化，
        # 可作独立观测单元，但同一轨迹的段仍相关，故重抽必须以轨迹为簇。
        'seg_pos_end_m': [float(np.median(x)) for x in d_pos_end],
        'seg_rot_end_deg': [float(np.median(x)) for x in d_rot_end],
        'rot_median_deg': float(np.median(r)),
        # 朝向的段末残差：与位置同口径，供图 2 的两层对照使用。
        # 不给这个字段的话，图 2 会变成「位置用段末、朝向用全帧 P95」，同图两套口径。
        'rot_end_median_deg': float(np.median(np.concatenate(d_rot_end))),
        'rot_p95_deg': float(np.percentile(r, 95)),
        'rot_max_deg': float(r.max()),
        'rot_raw_median_deg': float(np.median(np.concatenate(d_rot_raw))),
        'verdict': 'OK',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=None)
    ap.add_argument('--body', default='franka')
    ap.add_argument('--side', default='left',
                    help='left/right/auto（auto 按任务名前缀选臂，双臂数据必须用）')
    ap.add_argument('--prefix', default=None)
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 条（0=全部）')
    ap.add_argument('--json-out', default=None)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--gain-scale', type=float, default=1.0,
                    help='位置增益缩放倍数（2.4 参数扫描用，1.0=基准）')
    ap.add_argument('--fps', type=float, default=30.0,
                    help='数据帧率：真机 30（默认），RoboMIND 孪生 120')
    a = ap.parse_args()

    global GAIN_SCALE, FPS
    GAIN_SCALE = a.gain_scale
    FPS = a.fps

    if a.selftest:
        p = os.path.join(os.environ.get('PFPA_DATA_ROOT', 'data'), 'state_e1',
                        'benchmark1_0_compressed__slide_close_drawer__000.npz')
        if not os.path.exists(p):
            print('缺少 selftest 样本:', p)
            return 1
        r = analyse(p, 'franka')
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 0

    files = []
    for pat in ('*.npz', '*.hdf5'):
        for f in glob.glob(os.path.join(a.dir, pat)):
            if os.path.basename(f).startswith('._'):
                continue
            if a.prefix and not os.path.basename(f).startswith(a.prefix):
                continue
            files.append(f)
    files.sort()
    if a.limit:
        files = files[:a.limit]
    if not files:
        print('无匹配文件')
        return 1

    print(f'{a.body}｜{len(files)} 条｜{SEG_SEC:.0f} s 分段｜{N_SUB} 子步/帧｜FPS={FPS:.0f}')
    print(f'{"文件":<40s} {"帧":>5s} {"段":>3s} {"位置P95(m)":>12s} {"位置max(m)":>11s} {"朝向P95(°)":>11s}')
    print('-' * 88)
    rows = []
    for f in files:
        try:
            r = analyse(f, a.body, a.side)
        except Exception as e:
            r = {'file': os.path.basename(f), 'error': f'{type(e).__name__}: {e}'}
        rows.append(r)
        if 'error' in r or r.get('verdict') != 'OK':
            print(f'{r["file"][:40]:<40s} {"—":>5s} {"—":>3s} {"—":>12s} {"—":>11s} '
                  f'{r.get("error", r.get("verdict", ""))[:10]:>11s}')
        else:
            print(f'{r["file"][:40]:<40s} {r["n_frames"]:>5d} {r["n_segments"]:>3d} '
                  f'{r["pos_p95_m"]:>12.4e} {r["pos_max_m"]:>11.4e} {r["rot_p95_deg"]:>11.4f}')

    ok = [r for r in rows if r.get('verdict') == 'OK']
    if ok:
        p95 = np.array([r['pos_p95_m'] for r in ok])
        print('-' * 88)
        print(f'汇总 {len(ok)} 条：位置 P95 中位 {np.median(p95):.4e} m｜'
              f'P90 {np.percentile(p95, 90):.4e} m')
    if a.json_out:
        json.dump(rows, open(a.json_out, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        print(f'写出 {a.json_out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())