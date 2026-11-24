#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SR-49／SR-50／SR-51 判据链评估：消融、混淆矩阵、朴素基线对照。

三件事一次算完，共用同一套判据实现（与初稿算法 1 一致）：
  SR-49 消融：四种配置（S2／S0+S2／S0+S2+S3／全四步）在四批真实数据上的标签
  SR-50 混淆矩阵：真值由构造给定的注入样本上，判据的假阳性/假阴性
  SR-51 朴素基线：「只看残差量级」这一最朴素做法，与完整四步在同一批样本上并列

## 注入实验的构造（每种真值都物理自洽）

判据问的是「记录的末端位姿字段是否与该本体的官方模型同源」，因此真值须按
**与 FK 的关系**来构造，而不是随手加噪声：

  A 真值（模型正运动学输出） p_rec = FK(q)
      记录位姿就是模型自己的正运动学输出。
  B1 真值（模型输出·工具点约定不同） p_rec = FK(q) + 恒定刚体偏移
      仍是同一模型的输出，只是工具坐标系/安装位姿不同。
      **恒定刚体偏移会被 Kabsch 完全吸收**，故 r_L1 仍落浮点量级，
      判据应输出 A —— 这是正确结果，不是漏判。
  B2 真值（实机标定值） p_rec = FK(q + δq)，δq 为恒定关节零位偏差
      零位偏差使位姿误差随位形变化，单个刚体变换无法消解，
      这正是 3.1 节所述的 L1 典型失败，判据应输出 B。
  C 真值（字段无信息） p_rec = 恒定向量 → S0 应输出 C
  D 真值（来源存疑） p_rec = 与 FK 无关的独立位姿序列
      既非模型输出、也非标定值；判据对它无依据可判。

用法：`python3 sr49_51_judge_eval.py [--n 20]`
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
S0_FLOOR_POS, S0_JOINT_RANGE = 1e-9, 0.1
S3_MAG_MULT, S3_DISP = 10.0, 0.50
S4_HI, S4_LO = 5.0, 2.0
SIGMA = 1e-6
SEED = 20261111

BODY_DIR = {
    'franka': (os.path.join(RAW, 'state_e1'), 'franka'),
    'piper': (os.path.join(RAW, 'state_e1_rd', 'piper'), 'piper'),
    'arx_x5': (os.path.join(RAW, 'state_e1_rd', 'arx_x5'), 'arx_x5'),
}
# 四批真实数据的判据输入（与 SR-42／SR-43 实测一致）
REAL = {
    'Franka 成功侧': dict(l1_mm=0.000033, off_mm=None, disp_pct=None, eta=60.58544184585754),
    'Franka 失败侧': dict(l1_mm=0.713, off_mm=0.173, disp_pct=130.14, eta=0.9985108022955176),
    'ARX X5':        dict(l1_mm=2.96, off_mm=183.716, disp_pct=0.48, eta=1.0000575917942414),
    'Piper':         dict(l1_mm=6.73, off_mm=12.769, disp_pct=16.86, eta=1.0000232474455668),
}


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def align_residual(pos_fk, pos_rec):
    """与 E1 同口径：Kabsch（含旋转）扣常差后取 P95 残差，并返回拟合平移。"""
    import e1_residuals as E
    R_c, t_c = E.kabsch(pos_fk, pos_rec)
    d = np.linalg.norm((R_c @ pos_fk.T).T + t_c - pos_rec, axis=1)
    return float(np.percentile(d, 95)), np.asarray(t_c, dtype=float)


def run_criterion(q_list, rec_list, fk_fn, steps, with_s4=True, seed=SEED):
    """按给定的步骤集合跑判据链，返回 (标签, 决定步骤, 中间量)。"""
    # ── S0 零方差预检 ───────────────────────────────────────────
    if 'S0' in steps:
        ptp_pos = max(float(np.ptp(r[:, :3], axis=0).max()) for r in rec_list)
        joint_rng = max(float(np.ptp(q, axis=0).max()) for q in q_list)
        if ptp_pos <= S0_FLOOR_POS and joint_rng > S0_JOINT_RANGE:
            return '字段无信息', 'S0', dict(ptp_pos=ptp_pos)

    p95s, offs = [], []
    for q, rec in zip(q_list, rec_list):
        pos_fk = fk_fn(q)
        if pos_fk.shape != rec[:, :3].shape:
            return None, None, None
        p95, t_c = align_residual(pos_fk, rec[:, :3])
        p95s.append(p95)
        offs.append(t_c)

    # ── S2 量级 ─────────────────────────────────────────────────
    r_l1 = float(np.median(p95s))
    if 'S2' in steps and r_l1 <= CAP:
        return '模型正运动学输出', 'S2', dict(r_l1=r_l1)

    # ── S3 常差稳定性 ───────────────────────────────────────────
    if 'S3' in steps:
        offs = np.asarray(offs)
        mags = np.linalg.norm(offs, axis=1)
        med = float(np.median(mags))
        sd = np.array([float(np.std(offs[:, k])) for k in range(3)])
        disp = float(max(sd) / med) if med > 0 else float('inf')
        if med > S3_MAG_MULT * CAP and disp < S3_DISP:
            return '实机标定值', 'S3', dict(r_l1=r_l1, off_mm=med * 1000, disp_pct=disp * 100)

    # ── S4 噪声敏感性 ───────────────────────────────────────────
    if 'S4' in steps and with_s4:
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

    if 'S2' in steps and r_l1 > CAP:
        # 朴素基线的出口：量级不达标却无其它判据可用
        return '实机标定值', 'S2-量级外推', dict(r_l1=r_l1)
    return '来源存疑', '—', dict(r_l1=r_l1)


def load_batch(body, n):
    import e1_residuals as E
    import replay_l2 as R2L
    d, b2 = BODY_DIR[body]
    files = [os.path.join(d, f) for f in sorted(os.listdir(d))
             if f.endswith('.npz') and not f.startswith('._')][:n]
    out = []
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
        out.append((q, pos_fk))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    a = ap.parse_args()

    import e1_residuals as E
    import replay_l2 as R2L

    result = {
        'criteria': dict(cap_m=CAP, s0_floor_pos=S0_FLOOR_POS, s0_joint_range=S0_JOINT_RANGE,
                         s3_mag_mult=S3_MAG_MULT, s3_disp=S3_DISP,
                         s4_hi=S4_HI, s4_lo=S4_LO, sigma=SIGMA, seed=SEED),
        'sr49_ablation': {}, 'sr50_injection': {}, 'sr51_naive': {},
    }

    # ══════════════════════════════════════════════════════════
    # SR-49 消融：四批真实数据
    # ══════════════════════════════════════════════════════════
    CFGS = [('C1 仅S2', ['S2']), ('C2 S0+S2', ['S0', 'S2']),
            ('C3 S0+S2+S3', ['S0', 'S2', 'S3']), ('C4 全四步', ['S0', 'S2', 'S3', 'S4'])]
    print('=' * 92)
    print('SR-49 判据消融：四批真实数据在四种配置下的标签')
    print('=' * 92)
    print('%-14s %-11s %-11s %-13s %-11s %s' %
          ('批次', 'r_L1', '常差模长', '方向离散度', 'η', ''))
    for name, v in REAL.items():
        off = ('%8.3f mm' % v['off_mm']) if v['off_mm'] is not None else '        —'
        disp = ('%6.2f %%' % v['disp_pct']) if v['disp_pct'] is not None else '      —'
        print('%-14s %8.4f mm %s %s %10.4f' % (name, v['l1_mm'], off, disp, v['eta']))
    print('\n%-14s' % '批次', end='')
    for cn, _ in CFGS:
        print(' %-20s' % cn, end='')
    print()
    for name, v in REAL.items():
        row = {}
        print('%-14s' % name, end='')
        for cn, steps in CFGS:
            # 用实测输入直接复算各配置下的标签
            r_l1_m = v['l1_mm'] / 1000.0
            lab, by = '来源存疑', '—'
            if 'S2' in steps and r_l1_m <= CAP:
                lab, by = '模型正运动学输出', 'S2'
            elif 'S3' in steps and v['off_mm'] is not None and \
                    v['off_mm'] / 1000.0 > S3_MAG_MULT * CAP and v['disp_pct'] / 100.0 < S3_DISP:
                lab, by = '实机标定值', 'S3'
            elif 'S4' in steps:
                lab = '模型正运动学输出' if v['eta'] > S4_HI else (
                    '实机标定值' if v['eta'] < S4_LO else '来源存疑')
                by = 'S4'
            elif 'S2' in steps:
                lab, by = '实机标定值', 'S2-量级外推'
            row[cn] = dict(label=lab, decided_by=by)
            print(' %-20s' % ('%s(%s)' % (lab, by)), end='')
        print()
        result['sr49_ablation'][name] = row

    # ══════════════════════════════════════════════════════════
    # SR-50 注入实验 + SR-51 朴素基线对照
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 92)
    print('SR-50／SR-51 注入实验：真值由构造给定，比较完整四步与朴素基线')
    print('=' * 92)

    TRUTHS = ['A_模型输出', 'B1_模型输出工具点不同', 'B2_实机标定值',
              'C_字段无信息', 'D_来源存疑']
    rows = {}
    for body in ('franka', 'piper', 'arx_x5'):
        batch = load_batch(body, a.n)
        if not batch:
            print(f'{body}: 无可用样本'); continue
        q_list = [q for q, _ in batch]
        fk_list = [fk for _, fk in batch]
        fk_fn = (lambda qq, b=body: E.fk_positions(b, qq))
        print(f'\n── {body}：{len(q_list)} 条轨迹 ──')

        rng = np.random.default_rng(SEED)
        # B1：恒定刚体偏移（工具点约定不同）
        off_b1 = np.array([0.10, 0.0, 0.0])
        R_b1 = rot_z(np.deg2rad(5.0))
        # B2：恒定关节零位偏差
        delta_q = np.zeros(q_list[0].shape[1])
        delta_q[:] = 0.02          # 每关节 0.02 rad 恒定零位偏差
        delta_q[-1] = 0.0
        # D：与 FK 无关的独立位姿序列
        const = fk_list[0][0].copy()

        variants = {
            'A_模型输出': [fk.copy() for fk in fk_list],
            'B1_模型输出工具点不同': [fk @ R_b1.T + off_b1 for fk in fk_list],
            'B2_实机标定值': [fk_fn(q + delta_q) for q in q_list],
            'C_字段无信息': [np.tile(const, (len(fk), 1)) for fk in fk_list],
            'D_来源存疑': [const + rng.normal(0, 0.05, size=(len(fk), 3)) for fk in fk_list],
        }

        for truth, recs in variants.items():
            full, fby, fmid = run_criterion(q_list, recs, fk_fn, ['S0', 'S2', 'S3', 'S4'])
            naive, nby, _ = run_criterion(q_list, recs, fk_fn, ['S2'], with_s4=False)
            rows.setdefault(truth, {})[body] = dict(
                full_label=full, full_by=fby, naive_label=naive, naive_by=nby,
                r_l1_mm=(fmid or {}).get('r_l1', 0) * 1000 if fmid else None,
                eta=(fmid or {}).get('eta'), off_mm=(fmid or {}).get('off_mm'),
                disp_pct=(fmid or {}).get('disp_pct'))
            ex = ''
            if fmid and 'eta' in fmid:
                ex = 'η=%.4g' % fmid['eta']
            elif fmid and 'off_mm' in fmid:
                ex = '常差=%.1f mm 离散=%.1f%%' % (fmid['off_mm'], fmid['disp_pct'])
            elif fmid and 'r_l1' in fmid:
                ex = 'r_L1=%.3e m' % fmid['r_l1']
            print('  真值 %-24s → 完整四步: %-14s(%s)  朴素基线: %-14s(%s)  %s'
                  % (truth, full, fby, naive, nby, ex))

    # ── 混淆矩阵 ────────────────────────────────────────────────
    print('\n' + '-' * 92)
    print('混淆矩阵（真值 → 完整四步判出；每个真值在三个本体各一次）')
    print('-' * 92)
    LAB = {'模型正运动学输出': 'A', '实机标定值': 'B', '字段无信息': 'C', '来源存疑': 'D'}
    EXPECT = {'A_模型输出': 'A', 'B1_模型输出工具点不同': 'A', 'B2_实机标定值': 'B',
              'C_字段无信息': 'C', 'D_来源存疑': 'D'}
    mat_full, mat_naive = {}, {}
    for t in TRUTHS:
        mf = {k: 0 for k in 'ABCD'}
        mn = {k: 0 for k in 'ABCD'}
        for b, r in rows.get(t, {}).items():
            mf[LAB.get(r['full_label'], '?')] = mf.get(LAB.get(r['full_label'], '?'), 0) + 1
            mn[LAB.get(r['naive_label'], '?')] = mn.get(LAB.get(r['naive_label'], '?'), 0) + 1
        mat_full[t] = mf
        mat_naive[t] = mn
    print('%-26s %-22s %s' % ('真值', '完整四步 A/B/C/D', '朴素基线 A/B/C/D'))
    for t in TRUTHS:
        f, n = mat_full[t], mat_naive[t]
        print('%-26s %-22s %s' % (t, '/'.join(str(f[k]) for k in 'ABCD'),
                                  '/'.join(str(n[k]) for k in 'ABCD')))
    cor_f = sum(mat_full[t][EXPECT[t]] for t in TRUTHS)
    cor_n = sum(mat_naive[t][EXPECT[t]] for t in TRUTHS)
    tot = sum(sum(mat_full[t].values()) for t in TRUTHS)
    print(f'\n完整四步正确 {cor_f}/{tot}；朴素基线正确 {cor_n}/{tot}')
    fp_f = sum(mat_full[t]['B'] for t in TRUTHS if EXPECT[t] != 'B')
    fp_n = sum(mat_naive[t]['B'] for t in TRUTHS if EXPECT[t] != 'B')
    print(f'假阳性（真值非 B 却判 B）：完整四步 {fp_f} 条；朴素基线 {fp_n} 条')
    # D 真值被误判为 B 是 S4 的固有局限，单列
    d_as_b = mat_full['D_来源存疑']['B']
    print(f'其中 D 真值（与 FK 无关的独立序列）被判 B：{d_as_b} 条 —— '
          'S4 的 η≈1 只能说明「残差与 FK 无关」，无法区分标定值与其它独立量')

    result['sr50_injection'] = dict(
        truths=TRUTHS, expected=EXPECT,
        rows=rows, confusion_full=mat_full, confusion_naive=mat_naive,
        correct_full=cor_f, correct_naive=cor_n, total=tot,
        false_positive_full=fp_f, false_positive_naive=fp_n,
        d_misjudged_as_B=d_as_b,
        caveat='注入实验检验判据流程对构造已知来源的区分能力，不是真实数据的来源真值')
    result['sr51_naive'] = dict(
        definition='朴素基线＝只看扣常差后的残差量级：≤同源档上限判模型正运动学输出，否则判实机标定值；不做 S0／S3／S4',
        confusion=mat_naive, correct=cor_n, false_positive=fp_n)

    p = os.path.join(HERE, 'sr49_51_judge_eval.json')
    json.dump(result, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n写出 %s' % p)


if __name__ == '__main__':
    main()
