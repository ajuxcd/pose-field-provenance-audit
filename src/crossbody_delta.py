#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3.3 E2 跨本体分布比较：三本体 Δ 的任务内分位数相对化 + 两两 KS。

## 口径（来自计划 3.3）

「三本体 Δ 按任务内分位数相对化，算分位数曲线与两样本 KS 统计量」，
「两两 KS 比较共 3 对」，「完成标准：能判定阈值可跨本体迁移还是须逐本体标定」。

## 为什么必须「任务内相对化」

三个本体跑的是**不同任务**（Franka 是 slide_close_drawer/cap_close_dustbin/
open_the_drawer，Piper 是 fill_pen_holder 等，X5 是 cover_blocks 等）。
Δ 的绝对值受任务难度影响——长轨迹、多接触的任务 Δ 天然更大。
若直接跨本体比绝对值，比的其实是任务难度，不是本体特性。

**做法**：在每个任务内部，把该任务的 Δ 除以该任务 Δ 的中位数，于是每个任务的
中位数都变成 1，任务难度带来的尺度被消掉，分位数间的相对关系（即形状）被保留。

（早期版本写成「换成组内分位数」，那是错的——它把分布压成均匀分布，
形状信息在检验之前就被抹掉，三对 KS 全给出 D=0。详见 `relative_by_task` 的注释。）

## Δ 的定义

Δ = e_L2 − e_L1，**逐条**相减（位置与朝向分开，不合成标量）。
e_L1 取 `e1_residuals.py` 的 pos_after_p95_m / rot_after_p95_deg；
e_L2 取 `replay_l2.py` 的 **pos_end_median_m**（段末残差）——**不是** pos_p95_m。
后者的理由是 2026-11-09 查明的「全帧 P95 随分段数系统性下降」（见 NOTES_l2.md 坑五）。

**注意**：Δ 可能为负（L2 比 L1 小）。这在物理上意味着「开环回放比模型自洽检查还准」，
对同源档本体（Franka 的 L1 是 1e-8 级）不可能，但对近似档是可能的。
按计划，负值**如实报告，不截断**。

## KS 检验

两样本 KS（`scipy.stats.ks_2samp`）+ 置换检验（默认 10000 次）。
样本量须写明；KS 对样本量敏感，故同时报 p 值与效应量（D 统计量）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BODIES = ['franka', 'arx_x5', 'piper']
L1_FILES = {
    'franka': 'e1_franka_full.json',
    'arx_x5': 'e1_arx_left.json',
    'piper': 'e1_piper_left.json',
}
L2_FILES = {
    'franka': 'e2_franka_l2.json',
    'arx_x5': 'e2_arx_l2.json',
    'piper': 'e2_piper_l2.json',
}
BASE = os.path.join(HERE, '..')          # = working/
L1_DIR = os.path.join(BASE, 'ch21')         # E1 结果在 ch21/ 下


def task_of(fname):
    """从文件名取任务名。命名形如 `<batch>__<task>__<idx>.npz` 或 `<task>__<ep>.npz`。"""
    stem = os.path.basename(fname).replace('.npz', '').replace('.hdf5', '')
    parts = stem.split('__')
    if len(parts) >= 2:
        if parts[0].startswith('benchmark'):
            return parts[1] if len(parts) > 1 else stem
        return parts[0]
    return stem


def load_pairs(body):
    """返回 [(task, d_pos, d_rot)]，仅取两侧都 OK 的条目。"""
    l1 = json.load(open(os.path.join(L1_DIR, L1_FILES[body]), encoding='utf-8'))
    l2 = json.load(open(os.path.join(HERE, L2_FILES[body]), encoding='utf-8'))
    m1 = {r['file']: r for r in l1 if r.get('verdict') == 'OK'}
    m2 = {r['file']: r for r in l2 if r.get('verdict') == 'OK'}
    common = sorted(set(m1) & set(m2))
    out = []
    for f in common:
        # **口径**：L2 一侧用「段末残差」而非全帧 P95。
        # 全帧 P95 会随分段数系统性下降（见 NOTES_l2.md 坑五），
        # 而三本体的分段数差异极大（Franka 1 段、X5 8 段），用它做跨本体比较会被污染。
        out.append((task_of(f),
                    m2[f]['pos_end_median_m'] - m1[f]['pos_after_p95_m'],
                    m2[f]['rot_p95_deg'] - m1[f]['rot_after_p95_deg'],
                    f))
    return out, len(m1), len(m2)


def relative_by_task(rows, key):
    """任务内归一化：Δ → Δ / median(该任务的 Δ)，**保留形状、消除尺度**。

    ⚠ **不能用「换成组内分位数」**。那样做会把每个任务的 Δ 强行压成均匀分布，
    合并后每个本体的分布都近似 U(0,1)，KS 必然给出 D≈0、p=1——形状信息在
    检验之前就被自己的预处理抹掉了。本轮实际踩到这个坑（首版三对 D 全为 0.0000）。
    正确做法是用**中位数归一**：同一任务内的 Δ 除以该任务 Δ 的中位数，
    于是每个任务的中位数都变成 1，任务难度带来的尺度被消掉，分位数间的
    相对关系（即形状）被保留。

    负中位数（理论可能，本批未出现）时改用 IQR 归一，避免符号翻转。
    """
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[r[0]].append(r)
    out = []
    for t, items in g.items():
        vals = np.array([it[key] for it in items], dtype=float)
        med = float(np.median(vals))
        if abs(med) < 1e-12:
            continue
        scale = med if med > 0 else float(np.percentile(vals, 75) - np.percentile(vals, 25))
        if abs(scale) < 1e-12:
            continue
        for it in items:
            out.append((t, it[key], it[key] / scale, it[3]))
    return out


def quantile_curve(rows, key, qs=None):
    """分位数曲线：横轴 q，纵轴 Δ(q) / median(该任务 Δ)，跨任务平均。"""
    from collections import defaultdict
    if qs is None:
        qs = np.linspace(0.05, 0.95, 19)
    per_task = defaultdict(list)
    for t, raw, rel_, _ in rows:
        per_task[t].append((raw, rel_))
    curves = []
    for t, items in per_task.items():
        vals = np.array([x[0] for x in items], dtype=float)
        med = np.median(vals)
        if abs(med) < 1e-12:
            continue
        curves.append(np.quantile(vals, qs) / med)
    return qs, (np.mean(curves, axis=0) if curves else None)


def ks_stat(a, b):
    """两样本 KS 统计量 D = max|F_a - F_b|（不依赖 scipy）。"""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    grid = np.concatenate([a, b])
    Fa = np.searchsorted(a, grid, side='right') / len(a)
    Fb = np.searchsorted(b, grid, side='right') / len(b)
    return float(np.abs(Fa - Fb).max())


def ks_asym_p(d, n, m):
    """KS 的渐近 p 值（Kolmogorov 分布，2 项级数足够）。"""
    en = np.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-10:
        return 1.0
    p = 2.0 * sum((-1) ** (k - 1) * np.exp(-2.0 * k * k * lam * lam)
                  for k in range(1, 101))
    return float(min(max(p, 0.0), 1.0))


def ks_pair(a, b, n_perm=10000, seed=0):
    """两样本 KS：D + 渐近 p + 置换 p。

    环境无 scipy，故自行实现（KS 统计量与置换检验都只需排序与计数，
    不引入统计库反而更易审计）。
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return None
    d = ks_stat(a, b)
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([a, b])
    na = len(a)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if ks_stat(pooled[:na], pooled[na:]) >= d - 1e-12:
            cnt += 1
    return {'D': d, 'p_asym': ks_asym_p(d, len(a), len(b)),
            'p_perm': (cnt + 1) / (n_perm + 1),
            'n_a': int(len(a)), 'n_b': int(len(b))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-out', default=os.path.join(HERE, 'e2_crossbody.json'))
    ap.add_argument('--n-perm', type=int, default=10000)
    a = ap.parse_args()

    data = {}
    print('=' * 84)
    print('3.3 跨本体 Δ 分布比较（Δ = e_L2 − e_L1，逐条相减）')
    print('=' * 84)

    for body in BODIES:
        rows, n1, n2 = load_pairs(body)
        if not rows:
            print(f'{body}: 两侧无共同条目（L1 {n1} / L2 {n2}）')
            continue
        data[body] = rows
        dp = np.array([r[1] for r in rows])
        dr = np.array([r[2] for r in rows])
        tasks = sorted({r[0] for r in rows})
        print(f'\n--- {body} ---  配对 {len(rows)} 条（L1 {n1} / L2 {n2}）｜任务 {len(tasks)} 个')
        print(f'  Δ位置: 中位 {np.median(dp)*1000:8.3f} mm  '
              f'IQR [{np.percentile(dp,25)*1000:7.3f}, {np.percentile(dp,75)*1000:7.3f}]  '
              f'范围 [{dp.min()*1000:7.3f}, {dp.max()*1000:7.3f}]  负值 {int((dp<0).sum())}')
        print(f'  Δ朝向: 中位 {np.median(dr):8.4f} °   '
              f'IQR [{np.percentile(dr,25):7.4f}, {np.percentile(dr,75):7.4f}]  负值 {int((dr<0).sum())}')
        for t in tasks:
            sub = np.array([r[1] for r in rows if r[0] == t])
            print(f'    {t:<30s} n={len(sub):3d}  Δ位置中位 {np.median(sub)*1000:8.3f} mm')

    # 任务内分位数相对化 + 两两 KS
    print('\n' + '=' * 84)
    print('任务内中位数归一（Δ / median_τ）→ 两两 KS')
    print('=' * 84)
    rel = {}
    for body, rows in data.items():
        rel[body] = relative_by_task(rows, 1)      # key=1 是 Δ位置

    result = {'per_body': {}, 'pairs': []}
    for body, rows in data.items():
        dp = np.array([r[1] for r in rows])
        dr = np.array([r[2] for r in rows])
        result['per_body'][body] = {
            'n_paired': len(rows),
            'n_tasks': len({r[0] for r in rows}),
            'd_pos_median_mm': float(np.median(dp) * 1000),
            'd_pos_q1_mm': float(np.percentile(dp, 25) * 1000),
            'd_pos_q3_mm': float(np.percentile(dp, 75) * 1000),
            'd_pos_neg': int((dp < 0).sum()),
            'd_rot_median_deg': float(np.median(dr)),
            'd_rot_q1_deg': float(np.percentile(dr, 25)),
            'd_rot_q3_deg': float(np.percentile(dr, 75)),
            'd_rot_neg': int((dr < 0).sum()),
        }

    pairs = [('franka', 'arx_x5'), ('franka', 'piper'), ('arx_x5', 'piper')]
    print(f'\n{"对比":<22s} {"n_a":>4s} {"n_b":>4s} {"D":>8s} {"p_asym":>10s} {"p_perm":>10s} {"判定":>8s}')
    print('-' * 84)
    verdicts = []
    for x, y in pairs:
        if x not in rel or y not in rel:
            continue
        pa = np.array([r[2] for r in rel[x]])
        pb = np.array([r[2] for r in rel[y]])
        k = ks_pair(pa, pb, a.n_perm)
        if k is None:
            continue
        sig = k['p_perm'] < 0.05
        v = '形状不同' if sig else '形状相近'
        verdicts.append((x, y, k, sig))
        print(f'{x+" vs "+y:<22s} {k["n_a"]:>4d} {k["n_b"]:>4d} {k["D"]:>8.4f} '
              f'{k["p_asym"]:>10.2e} {k["p_perm"]:>10.4f} {v:>8s}')
        result['pairs'].append({'a': x, 'b': y, **k, 'different': bool(sig)})

    # 结论
    print('\n' + '-' * 84)
    if verdicts:
        n_sig = sum(1 for *_, s in verdicts if s)
        if n_sig == 0:
            concl = ('三对均未拒绝同分布 → Δ 的形状可跨本体迁移，'
                     '阈值可复用（但绝对量级仍不同，见 Δ 中位数）')
        elif n_sig == len(verdicts):
            concl = ('三对均拒绝同分布 → Δ 的形状不可跨本体迁移，须逐本体标定')
        else:
            concl = f'{len(verdicts)} 对中 {n_sig} 对拒绝同分布 → 部分可迁移，须逐对说明'
        print('判定：' + concl)
        result['conclusion'] = concl

    # 原始 Δ（mm），供图 4 的面板 (a) 画累积分布。
    # **导出原始值而不是只导汇总**：图 4 要同时给出「绝对量级不同」与「归一化后形状仍不同」，
    # 前者必须用原始 mm 的累积分布才看得见（Piper 有 40% 的 Δ 为负，这一点在
    # 归一化曲线上只剩一条下降线，看不出成因）。
    result['delta_mm'] = {b: [round(float(r[1]) * 1000, 4) for r in rows]
                          for b, rows in data.items()}

    # 分位数曲线（供图 4 用）
    curves = {}
    for body, rows in data.items():
        qs, c = quantile_curve(rows, 1)
        if c is not None:
            curves[body] = {'q': [round(float(x), 4) for x in qs],
                            'ratio': [round(float(x), 4) for x in c]}
            print(f'\n{body} 分位数曲线（Δ(q)/median）:')
            print('  q   =', np.round(qs[::3], 2))
            print('  r   =', np.round(c[::3], 3))
    result['curves'] = curves

    json.dump(result, open(a.json_out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'\n写出 {a.json_out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())