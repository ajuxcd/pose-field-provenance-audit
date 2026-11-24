#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 FMB 的 insert 轨迹作为**第五批（接触类）**送进本文四步判定。

## 为什么值得单独跑一批

前四批（RoboMIND Franka 成功／失败侧、RoboDojo 的 Piper／ARX X5）都是
**非接触**轨迹。审稿人自然要问：判据在接触类数据上还成立吗？接触会不会成为
混淆项？FMB 是现成的答案——它是插入装配任务，全程有真实接触，且带末端力／力矩
读数，可用力信号独立核验「接触确实发生过」。

## 口径（与初稿算法 1 逐项一致，不另立标准）

  S0 零方差预检：位置极差 ≤ 1e-9 m 且关节量程 > 0.1 rad → 字段无信息
  S2 量级：r_L1 = 各条「Kabsch 扣常差后 P95 残差」的中位数 ≤ 1e-6 m → 模型正运动学输出
  S3 常差稳定性：常差模长中位 > 10×CAP 且方向离散度 < 50% → 实机标定值
  S4 噪声敏感性：σ = 1e-6 rad 注入后 η = 中位/P95 之比；η>5 → A，η<2 → B，否则存疑

唯一的口径补充：FMB 与本体的**工具坐标系约定不同**，故 S2 之前不做任何人工对齐，
一律交给 Kabsch——这正是本文 E1 的做法，也保证「恒定位姿偏移会被吸收」这一性质
（见 sr49_51_judge_eval.py 的 B1 真值说明）。

用法：python3 judge_fmb.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser('~/Library/Caches/mujoco_menagerie/models')
PANDA = os.path.join(CACHE, 'franka_emika_panda-386f680ac533760a', 'panda.xml')
DATA = os.environ.get('FMB_DST', 'data')

N_ARM = 7
CAP = 1.0e-6
S0_FLOOR_POS, S0_JOINT_RANGE = 1e-9, 0.1
S3_MAG_MULT, S3_DISP = 10.0, 0.50
S4_HI, S4_LO = 5.0, 2.0
SIGMA = 1e-6
SEED = 20261113


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def align_residual(pos_fk, pos_rec):
    """与 E1 同口径：Kabsch 扣常差后取 P95 残差，返回 (p95, 平移常差)。"""
    R_c, t_c = kabsch(pos_fk, pos_rec)
    d = np.linalg.norm((R_c @ pos_fk.T).T + t_c - pos_rec, axis=1)
    return float(np.percentile(d, 95)), np.asarray(t_c, float)


class FK:
    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(PANDA)
        self.d = mujoco.MjData(self.m)
        names = {self.m.body(i).name: i for i in range(self.m.nbody)}
        self.eef = names['hand']
        self.idx = [j for j in range(self.m.njnt)
                    if int(self.m.jnt_type[j]) in (2, 3)][:N_ARM]

    def __call__(self, q):
        out = np.empty((len(q), 3))
        for i, row in enumerate(q):
            self.d.qpos[:] = self.m.qpos0
            self.d.qpos[self.idx] = row[:N_ARM]
            self.mj.mj_kinematics(self.m, self.d)
            out[i] = self.d.xpos[self.eef]
        return out


def main() -> int:
    fk = FK()
    rows = []
    for fp in sorted(glob.glob(os.path.join(DATA, '*.npy'))):
        d = np.load(fp, allow_pickle=True).item()
        q = np.asarray(d['obs/q'], float)[:, :N_ARM]
        rec = np.asarray(d['obs/tcp_pose'], float)[:, :3]
        F = np.linalg.norm(np.asarray(d['obs/tcp_force'], float), axis=1)
        Tq = np.linalg.norm(np.asarray(d['obs/tcp_torque'], float), axis=1)
        p95, t_c = align_residual(fk(q), rec)
        rows.append(dict(
            file=os.path.basename(fp), N=int(len(q)),
            r_l1_m=p95, off_m=[float(x) for x in t_c],
            force_peak_N=float(F.max()), force_p50_N=float(np.median(F)),
            force_last_N=float(F[-1]), torque_peak_Nm=float(Tq.max()),
            z_drop_mm=float((rec[0, 2] - rec[-1, 2]) * 1e3),
            joint_range=float(np.ptp(q, axis=0).max()),
            info=d['object_info']))

    n = len(rows)
    print(f'FMB insert 轨迹 {n} 条（Franka / Menagerie 官方模型 / Kabsch 口径）\n')

    # ── S0 ────────────────────────────────────────────────────
    ptp_pos = max(float(np.ptp(np.load(os.path.join(DATA, r['file']),
                                     allow_pickle=True).item()
                              ['obs/tcp_pose'][:, :3], axis=0).max())
                  for r in rows)
    jr = max(r['joint_range'] for r in rows)
    s0_hit = ptp_pos <= S0_FLOOR_POS and jr > S0_JOINT_RANGE
    print(f'S0 零方差预检：位置极差 {ptp_pos:.3e} m（阈 {S0_FLOOR_POS:.0e}），'
          f'关节量程 {jr:.3f} rad（阈 {S0_JOINT_RANGE}）→ '
          f'{"命中：字段无信息" if s0_hit else "未命中，继续"}')

    # ── S2 ────────────────────────────────────────────────────
    r_l1 = float(np.median([r['r_l1_m'] for r in rows]))
    print(f'S2 量级：r_L1 中位 {r_l1:.4e} m（阈 {CAP:.0e} m）→ '
          f'{"模型正运动学输出" if r_l1 <= CAP else "不达标，继续"}')

    # ── S3 ────────────────────────────────────────────────────
    offs = np.array([r['off_m'] for r in rows])
    mags = np.linalg.norm(offs, axis=1)
    med = float(np.median(mags))
    disp = float(max(np.std(offs[:, k]) for k in range(3)) / med) if med > 0 else np.inf
    ok_s3 = (med > S3_MAG_MULT * CAP) and (disp < S3_DISP)
    print(f'S3 常差稳定性：模长中位 {med*1e3:.2f} mm'
          f'（阈 >{S3_MAG_MULT*CAP*1e3:.4f} mm），方向离散度 {disp*100:.1f}%'
          f'（阈 <{S3_DISP*100:.0f}%）→ {"实机标定值" if ok_s3 else "不达标，继续"}')

    # ── S4 ────────────────────────────────────────────────────
    base, noisy = [], []
    rng = np.random.default_rng(SEED)
    for r in rows:
        d = np.load(os.path.join(DATA, r['file']), allow_pickle=True).item()
        q = np.asarray(d['obs/q'], float)[:, :N_ARM]
        rec = np.asarray(d['obs/tcp_pose'], float)[:, :3]
        base.append(align_residual(fk(q), rec)[0])
        noisy.append(align_residual(fk(q + rng.normal(0, SIGMA, q.shape)),
                                    rec)[0])
    b, nn = float(np.median(base)), float(np.median(noisy))
    eta = nn / b if b > 0 else np.inf
    lab_s4 = ('模型正运动学输出' if eta > S4_HI else
              '实机标定值' if eta < S4_LO else '来源存疑')
    print(f'S4 噪声敏感性：η = {eta:.4f}（>{S4_HI:.0f} → A，<{S4_LO:.0f} → B）'
          f'→ {lab_s4}')

    label = ('字段无信息' if s0_hit else
             '模型正运动学输出' if r_l1 <= CAP else
             '实机标定值' if ok_s3 else lab_s4)
    by = ('S0' if s0_hit else 'S2' if r_l1 <= CAP else 'S3' if ok_s3 else 'S4')
    print(f'\n★ FMB 批次判定：{label}（由 {by} 定出）')

    # ── 接触强度与残差的关系（判据是否被接触混淆） ──────────────
    f = np.array([r['force_peak_N'] for r in rows])
    e = np.array([r['r_l1_m'] for r in rows])
    def spearman(a, b_):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b_)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])
    print(f'\n接触强度：力峰值中位 {np.median(f):.2f} N'
          f'（四分位 {np.percentile(f,25):.2f}–{np.percentile(f,75):.2f}），'
          f'末帧力中位 {np.median([r["force_last_N"] for r in rows]):.2f} N')
    print(f'r_L1 中位 {np.median(e)*1e3:.3f} mm'
          f'（区间 {e.min()*1e3:.3f}–{e.max()*1e3:.3f}），'
          f'r_L1 与力峰值的 Spearman ρ = {spearman(e, f):+.3f}')

    out = os.path.join(HERE, 'judge_fmb.json')
    with open(out, 'w') as fh:
        json.dump(dict(criteria=dict(CAP=CAP, S0_FLOOR_POS=S0_FLOOR_POS,
                                     S0_JOINT_RANGE=S0_JOINT_RANGE,
                                     S3_MAG_MULT=S3_MAG_MULT, S3_DISP=S3_DISP,
                                     S4_HI=S4_HI, S4_LO=S4_LO, SIGMA=SIGMA),
                       summary=dict(n=n, s0_hit=s0_hit, r_l1_m=r_l1,
                                    off_med_m=med, disp=disp, eta=eta,
                                    label=label, by=by,
                                    rho_force_rl1=spearman(e, f),
                                    force_peak_med_N=float(np.median(f))),
                       per_traj=rows), fh, ensure_ascii=False, indent=1)
    print(f'\n→ {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
