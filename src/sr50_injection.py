#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SR-50 判据假阳性率：真值已知样本上的混淆矩阵。

## 为什么要做注入实验

本文最关键的输出是四个来源标签，但没有以真值标注的混淆矩阵——这是审稿人
最可能追问的一点（「判据的假阳性率是多少」）。

公开仿真批次虽然真值已知（仿真里没有真实机器人可标定，记录的末端位姿只能是
模型自己的正运动学输出），但其中 38/44 条字段恒定、6 条与模型对不上，
**真正能检验判据的样本极少**。

因此本脚本做**注入实验（controlled injection）**：取真实轨迹的关节角 q(t)，
用官方模型做正运动学得到 FK 位姿，再按构造生成三种真值已知的记录位姿字段：

  A 真值（模型正运动学输出）：p_rec = FK(q)                → 期望标签 A
  B 真值（实机标定值）：      p_rec = FK(q) + 常差 w       → 期望标签 B
  C 真值（字段无信息）：      p_rec = 恒定向量              → 期望标签 C
  D 真值（来源存疑）：        p_rec = FK(q) + 逐帧独立噪声  → 期望标签 D

对每种真值，都跑一遍完整四步判据（S0→S2→S3→S4），统计混淆矩阵。

**注入实验的效力边界（须在正文明示）**：
- 它检验的是**判据流程**能否把构造上不同的来源区分开，不是检验真实数据的来源；
- 噪声注入 D 用逐帧独立噪声模拟「既非 FK 输出、也非稳定常差」的字段，
  它在物理上等价于「该字段的来源无法由本流程判定」；
- B 真值用的常差取自 S3 的实测分布（X5 183.7 mm／Piper 12.8 mm 量级），
  以使 S3 的两条界在注入样本上同样被检验到。

用法：`python3 sr50_injection.py [--bodies franka piper arx_x5] [--n 20]`
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CH21 = os.path.abspath(os.path.join(HERE, os.pardir, 'ch21'))
sys.path.insert(0, CH21)
sys.path.insert(0, HERE)

RAW = os.environ.get('PFPA_DATA_ROOT', 'data')

# ── 判据常量（与初稿算法 1 一致） ────────────────────────────────
CAP = 1.0e-6
S0_FLOOR_POS, S0_FLOOR_ROT, S0_JOINT_RANGE = 1e-9, 1e-6, 0.1
S3_MAG_MULT, S3_DISP = 10.0, 0.50
S4_HI, S4_LO = 5.0, 2.0
SIGMA = 1e-6

BODY_DIR = {
    'franka': (os.path.join(RAW, 'state_e1'), 'franka'),
    'piper': (os.path.join(RAW, 'state_e1_rd', 'piper'), 'piper'),
    'arx_x5': (os.path.join(RAW, 'state_e1_rd', 'arx_x5'), 'arx_x5'),
}


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def align_residual(pos_fk, pos_rec):
    """与 E1 同口径：Kabsch（含旋转）扣常差后取 P95 残差。"""
    import e1_residuals as E
    R_c, t_c = E.kabsch(pos_fk, pos_rec)
    d = np.linalg.norm((R_c @ pos_fk.T).T + t_c - pos_rec, axis=1)
    return float(np.percentile(d, 95)), np.asarray(t_c, dtype=float)


def run_criterion(q_list, rec_list, fk_fn, with_s4=True, seed=20261111):
    """对一批轨迹跑完整四步判据，返回 (标签, 决定步骤, 中间量)。

    中间量与判据链一致：
      S0 位置量程 ≤ 1e-9 m 且朝向量程 ≤ 1e-6°，而关节量程 > 0.1 rad → C
      S1 逐轨迹 Kabsch 扣常差 → 段末代表值 r_L1（P95）
      S2 r_L1 ≤ 1e-6 m → A
      S3 常差模长 > 10×上限 且 跨轨迹方向相对离散度 < 50% → B
      S4 η(1e-6) > 5 → A；η < 2 → B；否则 D
    """
    # ── S0 ──────────────────────────────────────────────────────
    ptp_pos = max(float(np.ptp(r[:, :3], axis=0).max()) for r in rec_list)
    joint_rng = max(float(np.ptp(q, axis=0).max()) for q in q_list)
    # 朝向量程：注入实验只造位置场，朝向按 FK 给出，故此处以位置场判定
    if ptp_pos <= S0_FLOOR_POS and joint_rng > S0_JOINT_RANGE:
        return '字段无信息', 'S0', dict(ptp_pos=ptp_pos, joint_rng=joint_rng)

    p95s, offs = [], []
    for q, rec in zip(q_list, rec_list):
        pos_fk = fk_fn(q)
        if pos_fk.shape != rec[:, :3].shape:
            return None, None, None
        p95, t_c = align_residual(pos_fk, rec[:, :3])
        p95s.append(p95)
        offs.append(t_c)

    # ── S2 ──────────────────────────────────────────────────────
    r_l1 = float(np.median(p95s))
    if r_l1 <= CAP:
        return '模型正运动学输出', 'S2', dict(r_l1=r_l1, p95s=p95s)

    # ── S3 ──────────────────────────────────────────────────────
    offs = np.asarray(offs)
    mags = np.linalg.norm(offs, axis=1)
    med = float(np.median(mags))
    sd = np.array([float(np.std(offs[:, k])) for k in range(3)])
    disp = float(max(sd) / med) if med > 0 else float('inf')
    if med > S3_MAG_MULT * CAP and disp < S3_DISP:
        return '实机标定值', 'S3', dict(r_l1=r_l1, off_mm=med * 1000, disp_pct=disp * 100)

    # ── S4 ──────────────────────────────────────────────────────
    if with_s4:
        base, noisy = [], []
        for q, rec in zip(q_list, rec_list):
            rng = np.random.default_rng(seed)
            base.append(align_residual(fk_fn(q), rec[:, :3])[0])
            qq = q + rng.normal(0.0, SIGMA, size=q.shape)
            noisy.append(align_residual(fk_fn(qq), rec[:, :3])[0])
        b = float(np.median(base))
        n = float(np.median(noisy))
        eta = n / b if b > 0 else float('inf')
        if eta > S4_HI:
            return '模型正运动学输出', 'S4', dict(r_l1=r_l1, eta=eta)
        if eta < S4_LO:
            return '实机标定值', 'S4', dict(r_l1=r_l1, eta=eta)
        return '来源存疑', 'S4', dict(r_l1=r_l1, eta=eta)

    return '来源存疑', '—', dict(r_l1=r_l1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bodies', nargs='+', default=['franka', 'piper', 'arx_x5'])
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--no-s4', action='store_true')
    a = ap.parse_args()

    import e1_residuals as E
    import replay_l2 as R2L

    # 注入用的常差：取 S3 实测分布的量级（X5 183.7 mm、Piper 12.8 mm、Franka ~0）
    OFFSET_MM = {'franka': 8.69, 'piper': 12.77, 'arx_x5': 183.72}

    out = {'criteria': dict(cap_m=CAP, s0_floor_pos=S0_FLOOR_POS, s0_joint_range=S0_JOINT_RANGE,
                            s3_mag_mult=S3_MAG_MULT, s3_disp=S3_DISP,
                            s4_hi=S4_HI, s4_lo=S4_LO, sigma=SIGMA, seed=20261111),
           'injection': {}, 'confusion': {}, 'by_body': {}}

    for body in a.bodies:
        d, b2 = BODY_DIR[body]
        files = [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if f.endswith('.npz') and not f.startswith('._')][:a.n]
        q_list, fk_list = [], []
        for p in files:
            ep = R2L.load_episode(p, b2, 'left')
            if ep is None:
                continue
            q = ep['joints']
            if q is None or len(q) < 2:
                continue
            try:
                pos_fk = E.fk_positions(b2, q)
            except Exception:
                continue
            if pos_fk.shape[0] != q.shape[0]:
                continue
            q_list.append(q)
            fk_list.append(pos_fk)
        if not q_list:
            print(f'{body}: 无可用样本'); continue
        print(f'{body}: 取到 {len(q_list)} 条轨迹')

        fk_fn = (lambda qq, b=b2: E.fk_positions(b, qq))
        off = np.array([OFFSET_MM[body] / 1000.0, 0.0, 0.0])
        R_off = rot_z(np.deg2rad(5.0))

        variants = {}
        # A 真值：记录位姿 = FK 输出本身（模型正运动学输出）
        variants['A_模型输出'] = [(fk.copy()) for fk in fk_list]
        # B 真值：记录位姿 = FK 输出 + 恒定刚体偏移（实机标定值）
        variants['B_标定值'] = [(fk @ R_off.T + off) for fk in fk_list]
        # C 真值：记录位姿 = 恒定（字段无信息）
        const = fk_list[0][0].copy()
        variants['C_字段无信息'] = [np.tile(const, (len(fk), 1)) for fk in fk_list]
        # D 真值：记录位姿 = FK 输出 + 逐帧独立噪声（来源存疑）
        rng = np.random.default_rng(20261111)
        variants['D_存疑'] = [fk + rng.normal(0, 1e-3, size=fk.shape) for fk in fk_list]

        body_rows = {}
        for truth, recs in variants.items():
            lab, by, mid = run_criterion(q_list, recs, fk_fn, with_s4=not a.no_s4)
            body_rows[truth] = dict(label=lab, decided_by=by,
                                    detail={k: (float(v) if isinstance(v, (int, float)) else v)
                                            for k, v in (mid or {}).items()
                                            if k in ('r_l1', 'eta', 'off_mm', 'disp_pct', 'ptp_pos')})
            extra = ''
            if mid and 'eta' in mid:
                extra = '  η=%.4g' % mid['eta']
            elif mid and 'off_mm' in mid:
                extra = '  常差=%.1f mm 离散=%.1f%%' % (mid['off_mm'], mid['disp_pct'])
            elif mid and 'r_l1' in mid:
                extra = '  r_L1=%.3e m' % mid['r_l1']
            print(f'   真值 {truth:<12} → 判 {lab:<12}（{by}）{extra}')
        out['by_body'][body] = body_rows

    # ── 汇总混淆矩阵 ────────────────────────────────────────────
    # 真值 A/B/C/D ↔ 判出的标签
    print('\n' + '=' * 78)
    print('混淆矩阵（真值 → 判出；行＝构造真值，列＝判据输出）')
    print('=' * 78)
    TRUTH = ['A_模型输出', 'B_标定值', 'C_字段无信息', 'D_存疑']
    LABEL = {'模型正运动学输出': 'A', '实机标定值': 'B', '字段无信息': 'C', '来源存疑': 'D'}
    matrix = {t: {} for t in TRUTH}
    for body, rows in out['by_body'].items():
        for t in TRUTH:
            if t in rows:
                matrix[t][LABEL.get(rows[t]['label'], rows[t]['label'])] = \
                    matrix[t].get(LABEL.get(rows[t]['label'], rows[t]['label']), 0) + 1
    print('%-14s %s' % ('真值\\判出', '  A(模型输出)  B(标定值)  C(无信息)  D(存疑)'))
    for t in TRUTH:
        row = matrix[t]
        print('%-14s %7d %11d %11d %10d' % (t, row.get('A', 0), row.get('B', 0),
                                            row.get('C', 0), row.get('D', 0)))
    correct = sum(matrix[t].get(t[0], 0) for t in TRUTH)
    total = sum(sum(matrix[t].values()) for t in TRUTH)
    print(f'\n正确 {correct}/{total}')
    fp = sum(matrix[t].get('B', 0) for t in TRUTH if t != 'B_标定值')
    fn = sum(matrix[t].get(t[0], 0) for t in TRUTH if t != 'A_模型输出')  # 占位，下面重算
    print(f'假阳性（真值非 B 却判 B）＝ {fp} 条')
    out['confusion'] = {t: matrix[t] for t in TRUTH}
    out['injection'] = dict(
        total=total, correct=correct,
        false_positive_B=fp,
        note='注入实验检验的是判据流程对构造已知来源的区分能力，不是真实数据的来源真值')

    p = os.path.join(HERE, 'sr50_injection.json')
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n写出 %s' % p)


if __name__ == '__main__':
    main()
