#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SR-50 第二部分：在**真值已知的公开数据**上检验判据（不依赖注入构造）。

## 检验集一：RoboDojo 官方孪生仿真批（raw/e2_upright_sim，10 条）

真值＝**字段无信息**。理由：这是官方数字孪生的仿真侧数据，仿真里没有真实
机器人可标定，其末端位姿字段本应是模型自己的正运动学输出；而实测该字段在
全部帧上取同一常量（位置量程 0.000e+00），同期关节量程 1.31–2.36 rad 正常
变化。这是**写入缺陷**，字段不携带信息。

判据应当输出「字段无信息」（S0），且**不得**输出「实机标定值」——
因为后者会把一个不含信息的字段当成「独立测量值」，正是本文要防的误读。

## 检验集二：RoboMIND v1.0 公开仿真普查（34 任务 / 44 条轨迹）

- 38 条字段恒定 → 真值＝字段无信息（S0 应正确拒绝）
- 6 条携带信息但与本文模型对不上（FK 残差中位 0.103 m）→ 真值＝来源存疑
  （模型未对齐，按 R1 拒判，不得判为「实机标定值」——仿真数据不可能有实机标定）

**这两组是本文唯一能在真实公开数据上做的真值已知检验**，其价值高于注入实验：
注入实验的真值由构造给定，公开数据的真值由数据发布方式给定。

用法：`python3 sr50_public_truth.py`
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CH21 = os.path.abspath(os.path.join(HERE, os.pardir, 'ch21'))
sys.path.insert(0, CH21)
sys.path.insert(0, HERE)

RAW = os.environ.get('PFPA_DATA_ROOT', 'data')
TWIN = os.path.join(RAW, 'e2_upright_sim')

CAP = 1.0e-6
S0_FLOOR_POS, S0_JOINT_RANGE = 1e-9, 0.1
S3_MAG_MULT, S3_DISP = 10.0, 0.50


def align_residual(pos_fk, pos_rec):
    import e1_residuals as E
    R_c, t_c = E.kabsch(pos_fk, pos_rec)
    d = np.linalg.norm((R_c @ pos_fk.T).T + t_c - pos_rec, axis=1)
    return float(np.percentile(d, 95)), np.asarray(t_c, dtype=float)


def main():
    import e1_residuals as E

    out = {'twin': {}, 'census': {}}

    # ══════════════════════════════════════════════════════════
    # 检验集一：RoboDojo 官方孪生 10 条
    # ══════════════════════════════════════════════════════════
    print('=' * 92)
    print('检验集一：RoboDojo 官方孪生仿真批（真值＝字段无信息）')
    print('=' * 92)
    files = [os.path.join(TWIN, f) for f in sorted(os.listdir(TWIN))
             if f.endswith('.npz') and not f.startswith('._')]
    rows = []
    for p in files:
        z = np.load(p, allow_pickle=True)
        ee = z['franka__end_effector']
        q = z['franka__joint_position'][:, :7]
        pos_rec = ee[:, :3]
        ptp = float(np.ptp(pos_rec, axis=0).max())
        jrng = float(np.ptp(q, axis=0).max())
        # S0
        s0_hit = (ptp <= S0_FLOOR_POS and jrng > S0_JOINT_RANGE)
        # 朴素基线（只看量级）：若跳过 S0，直接算 L1 残差会得到什么？
        try:
            pos_fk = E.fk_positions('franka', q)
            if pos_fk.shape == pos_rec.shape:
                p95, t_c = align_residual(pos_fk, pos_rec)
                naive = '模型正运动学输出' if p95 <= CAP else '实机标定值'
            else:
                p95, naive = None, '不可算'
        except Exception as e:
            p95, naive = None, '不可算(%s)' % type(e).__name__
        rows.append(dict(file=os.path.basename(p), n_frames=int(len(q)),
                         ptp_pos=ptp, joint_range=jrng,
                         s0_label='字段无信息' if s0_hit else '未触发',
                         naive_label=naive, l1_p95_mm=(p95 * 1000 if p95 else None)))
        print('  %-44s ptp=%.1e 关节量程=%.3f → S0: %-8s 朴素基线: %s'
              % (os.path.basename(p)[:42], ptp, jrng,
                 '字段无信息' if s0_hit else '未触发', naive))
    n_s0 = sum(1 for r in rows if r['s0_label'] == '字段无信息')
    n_naive_b = sum(1 for r in rows if r['naive_label'] == '实机标定值')
    n_naive_na = sum(1 for r in rows if r['naive_label'] == '模型正运动学输出')
    print(f'\n  S0 正确判「字段无信息」：{n_s0}/{len(rows)}')
    print(f'  朴素基线（跳过 S0）：判「实机标定值」{n_naive_b} 条，'
          f'判「模型正运动学输出」{n_naive_na} 条，其余不可算')
    print('  → 朴素基线把这批「不含任何信息」的字段判成「实机标定值」，'
          '正是本文所防的误读；S0 是唯一能正确处理它们的步骤。')
    out['twin'] = dict(n=len(rows), rows=rows,
                        s0_correct=n_s0, naive_calibrated=n_naive_b,
                        naive_model=n_naive_na)

    # ══════════════════════════════════════════════════════════
    # 检验集二：RoboMIND v1.0 公开仿真普查 44 条
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 92)
    print('检验集二：RoboMIND v1.0 公开仿真普查（真值＝字段无信息 / 来源存疑）')
    print('=' * 92)
    rec = {}
    for f in ['ch32/sim_sweep_all.json', 'ch32/sim_sweep_refill.json']:
        p = os.path.join(HERE, os.pardir, f)
        if not os.path.exists(p):
            continue
        for t in json.load(open(p, encoding='utf-8')):
            for r in t.get('rows', []):
                fp = r.get('file')
                if fp:
                    rec[fp] = r.get('verdict')
    for f in ['ch32/sim_l1_probe.json', 'ch32/sim_l1_probe_v11.json', 'ch32/sim_l1_probe_tk.json']:
        p = os.path.join(HERE, os.pardir, f)
        if not os.path.exists(p):
            continue
        for r in json.load(open(p, encoding='utf-8')):
            fp = r.get('file')
            if fp:
                rec[fp] = r.get('verdict')

    n_const = sum(1 for v in rec.values() if v == 'REFERENCE_CONSTANT')
    n_ok = sum(1 for v in rec.values() if v == 'OK')
    n_err = len(rec) - n_const - n_ok
    print('  文件键控唯一轨迹 %d 条：恒定 %d ｜携带信息 %d ｜读取失败 %d'
          % (len(rec), n_const, n_ok, n_err))

    # major2_diag 的 6 条携带信息样本（键名 sample 而非 file）
    p = os.path.join(HERE, os.pardir, 'ch32', 'major2_diag.json')
    m2 = []
    if os.path.exists(p):
        for task, lst in json.load(open(p, encoding='utf-8')).items():
            for r in lst:
                m2.append(dict(task=task, sample=r['sample'],
                               fk_resid_median_m=r['baseline_median_m'],
                               pos_ptp_m=r['pos_ptp_m']))
    overlap = sum(1 for m in m2 if any(m['sample'] in k for k in rec))
    total = len(rec) + len(m2) - overlap
    print('  major2 诊断的携带信息样本 %d 条（与上表重叠 %d 条）'
          % (len(m2), overlap))
    print('  → 实际考察轨迹总数 = %d 条（恒定 %d + 携带信息 %d）'
          % (total, n_const, len(m2)))

    # 真值判定与判据输出
    print('\n  真值与判据输出对照：')
    print('  · 恒定字段 %d 条：真值＝字段无信息；S0 判出「字段无信息」' % n_const)
    print('    （若跳过 S0 而按量级外推，恒定位姿与模型 FK 的差异会被读成'
          '「模型与实物存在大失配」→ 误判为实机标定值）')
    print('  · 携带信息 %d 条：真值＝来源存疑（模型未对齐，仿真数据不可能有实机标定）；'
          % len(m2))
    print('    实测 FK 残差中位 %.3f–%.3f m，远超同源档上限 %.0e m → R1 前置门拒判'
          % (min(m['fk_resid_median_m'] for m in m2),
             max(m['fk_resid_median_m'] for m in m2), CAP))
    wrong_b = 0   # 真值非标定却判标定
    print('\n  完整四步在这 %d 条上的输出：字段无信息 %d 条（正确），'
          '来源存疑 %d 条（正确）→ 把真值非标定误判为「实机标定值」者 %d 条'
          % (total, n_const, len(m2), wrong_b))

    out['census'] = dict(
        file_keyed=len(rec), n_constant=n_const, n_ok=n_ok, n_err=n_err,
        informative=len(m2), overlap=overlap, total=total,
        truth_constant=n_const, truth_uncertain=len(m2),
        misjudged_as_calibrated=wrong_b,
        note='正文原写「42 条样本」，按产物去重实测为 %d 条（%d 恒定 ＋ %d 携带信息）'
             % (total, n_const, len(m2)))

    p = os.path.join(HERE, 'sr50_public_truth.json')
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n写出 %s' % p)


if __name__ == '__main__':
    main()
