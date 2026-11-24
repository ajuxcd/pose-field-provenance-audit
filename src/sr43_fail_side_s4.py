#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SR-43 核查：失败侧（RoboMIND Franka failure_episodes）的 S4 能否兜住其来源标签。

## 背景

V4.7 把算法 1 的 S3 由「常差显著且跨轨迹稳定」改写为可执行判据：
「常差模长 > 同源档上限的 10 倍，且跨轨迹常差方向的相对离散度 < 50%」。

但 SR-42 实测发现（`sr42_s3_layer_check.py`）：
- 该量化依据被误取自 **L2 回放层**的 `pos_offset_m`，而 S3 判定的是 **L1 层**常差；
- 在正确的 L1 层上：Piper 12.77 mm/16.9%、ARX X5 183.7 mm/0.5% 均通过，
  但 **Franka 失败侧 0.173 mm/130% 不通过**（常差方向逐轨迹不一致）。

S3 不通过 → 按算法 1 转 S4（噪声敏感性排除法）。而 `noise_falsify_franka.json`
只覆盖成功侧 30 条，**失败侧无 S4 数据**。

本脚本为失败侧补做 S4：向关节角注入 σ 的独立高斯噪声后重算 L1，
按式(6) 计算敏感度 η(σ) = 残差之比：
- η(1×10⁻⁶ rad) > 5 → 判「模型正运动学输出」
- η < 2 → 判「实机标定值」

失败侧残差若对噪声不敏感（η ≈ 1），则与成功侧的 60.6 倍形成镜像，
可判「实机标定值」，标签链即完整闭合。

用法：`python3 sr43_fail_side_s4.py`
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CH21 = os.path.abspath(os.path.join(HERE, '..', 'ch21'))
sys.path.insert(0, CH21)
sys.path.insert(0, HERE)

RAW = os.environ.get('PFPA_DATA_ROOT', 'data')
FAIL_DIR = os.path.join(RAW, 'failure_e3')
CAP = 1e-6
SIGMAS = [0.0, 1e-6, 1e-3]


def main():
    import e1_residuals as E
    import replay_l2 as R2L

    files = [os.path.join(FAIL_DIR, f) for f in sorted(os.listdir(FAIL_DIR))
             if f.endswith('.npz') and not f.startswith('._')]
    print(f'失败侧样本 {len(files)} 条\n')

    rows = []
    for p in files:
        ep = R2L.load_episode(p, 'franka', 'left')
        if ep is None:
            continue
        q = ep['joints']; rec = ep['ee_xyz']
        if q is None or len(q) < 2:
            continue
        out = {}
        for sig in SIGMAS:
            rng = np.random.default_rng(20261111)   # 固定种子，保证可复算
            qq = q + (rng.normal(0.0, sig, size=q.shape) if sig > 0 else 0.0)
            try:
                pos_fk = E.fk_positions('franka', qq)
            except Exception:
                out = None; break
            if pos_fk.shape != rec.shape:
                out = None; break
            # 与 E1 同口径：Kabsch 扣常差后取 P95
            R_c, t_c = E.kabsch(pos_fk, rec)
            d = np.linalg.norm((R_c @ pos_fk.T).T + t_c - rec, axis=1)
            out[sig] = float(np.percentile(d, 95))
        if not out:
            continue
        rows.append({'file': os.path.basename(p), 'p95': out})

    if not rows:
        print('无可用样本'); return 2
    print(f'可用 {len(rows)} 条\n')
    print(f'{"σ (rad)":>12} {"P95 中位 (m)":>16} {"相对 σ=0":>12}')
    base = np.median([r['p95'][0.0] for r in rows])
    for sig in SIGMAS:
        v = np.median([r['p95'][sig] for r in rows])
        print(f'{sig:>12.0e} {v:>16.6e} {v/base:>12.2f}')
    eta = np.median([r['p95'][1e-6] for r in rows]) / base
    print(f'\nη(σ=1×10⁻⁶ rad) 中位 = {eta:.4f}')
    print(f'→ S4 判据：η > 5 判「模型正运动学输出」／η < 2 判「实机标定值」'
          f' → {"判实机标定值（标签链闭合）" if eta < 2 else ("判模型正运动学输出" if eta > 5 else "落入存疑区")}')
    print(f'\n[对照] 成功侧同口径 η = 60.59（ratio，σ=1e-6）')

    json.dump({'n': len(rows), 'sigmas': SIGMAS,
               'p95_median_m': {str(s): float(np.median([r['p95'][s] for r in rows])) for s in SIGMAS},
               'eta_1e-6': float(eta),
               'rows': [{'file': r['file'], 'p95': {str(k): v for k, v in r['p95'].items()}} for r in rows]},
              open(os.path.join(HERE, 'sr43_fail_side_s4.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n写出 sr43_fail_side_s4.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
