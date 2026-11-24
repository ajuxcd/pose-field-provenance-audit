#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SR-42 核查：V4.7 写入 §6.1 的 S3 量化依据是否来自正确的层。

## 疑点

V4.7 在 §6.1 写「S3 的两条界同样取自实测：三本体的跨轨迹常差模长中位分别为
8.69 mm、172.6 mm、7.34 mm，相对同源档上限分别高 8.7×10³、1.7×10⁸、7.3×10³ 倍；
同一批轨迹上常差方向的相对离散度为 10.7%、1.2%、37.9%」。

但这组数字取自 `pipeline/e2_*_l2.json` 的 `pos_offset_m`，按 `replay_l2.py` 第 391 行，
`off = mean(replay_pos − recorded_pos)`，即 **L2 回放层**的逐轨迹偏移。

而算法 1 的 S3 承接 S1/S2（S1 复算 L1、S2 判 r_L1 量级），S3 判定的应是
**L1 层**（模型 FK 输出 vs 记录位姿）的常差。二者不是同一个量：
- Franka 成功侧的 L1 常差恒为 0（记录位姿即模型 FK 输出），而 L2 常差是 8.69 mm。
  → 若把 L2 的数字当作 S3 的依据，则「以 L2 常差论证 L1 判定」是层间错配。

## 本脚本做什么

对 RoboDojo 两本体（Piper／ARX X5）与 RoboMIND Franka 成功侧，**逐轨迹**做 L1 层
Kabsch 对齐，取每次对齐的平移向量作为该轨迹的 L1 常差，统计：
  · 常差模长中位数
  · 跨轨迹方向的相对离散度
再套用 V4.7 写入正文的 S3 判据（模长 > 10×同源档上限 且 相对离散度 < 50%），
看标签是否翻转。

用法：`python3 sr42_s3_layer_check.py [--bodies piper arx_x5 franka]`
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CH21 = os.path.abspath(os.path.join(HERE, '..', 'ch21'))
sys.path.insert(0, CH21)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'pipeline')))

RAW = os.environ.get('PFPA_DATA_ROOT', 'data')
DIRS = {
    'franka': os.path.join(RAW, 'state_e1'),
    'arx_x5': os.path.join(RAW, 'state_e1_rd', 'arx_x5'),
    'piper': os.path.join(RAW, 'state_e1_rd', 'piper'),
    'franka_fail': os.path.join(RAW, 'failure_e3'),
}
E2 = {'franka': 'e2_franka_l2.json', 'arx_x5': 'e2_arx_l2.json', 'piper': 'e2_piper_l2.json'}
CAP = 1e-6          # 同源档位置上限（m）
MULT = 10           # V4.7 的模长判据：> 10× 上限
DISP_MAX = 0.50     # V4.7 的离散度判据：< 50%


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bodies', nargs='+',
                    default=['franka_fail', 'piper', 'arx_x5', 'franka'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--json-out', default=os.path.join(HERE, 'sr42_s3_layer_check.json'))
    a = ap.parse_args()

    import e1_residuals as E
    import replay_l2 as R2L
    import mujoco

    out = {}
    for body in a.bodies:
        real_body = 'franka' if body == 'franka_fail' else body
        files = [os.path.join(DIRS[body], f) for f in sorted(os.listdir(DIRS[body]))
                 if f.endswith('.npz') and not f.startswith('._')]
        if a.limit:
            files = files[:a.limit]
        offs, raw_max = [], []
        for p in files:
            # **用 replay_l2.load_episode 读数据**：它兼容 RoboDojo（state__left_*）
            # 与 RoboMIND（puppet__*）两种 schema；e1_residuals.analyse 只认后者，
            # 对 RoboDojo 一律返回 'no puppet/joint_position'（本轮踩到）。
            ep = R2L.load_episode(p, real_body, 'left')
            if ep is None:
                continue
            q = ep['joints']
            rec = ep['ee_xyz']
            if q is None or len(q) < 2:
                continue
            try:
                pos_fk = E.fk_positions(real_body, q)
            except Exception as e:
                print('  skip', os.path.basename(p), type(e).__name__, e)
                continue
            if pos_fk.shape != rec.shape:
                continue
            R_c, t_c = E.kabsch(pos_fk, rec)
            offs.append(np.asarray(t_c, dtype=float))
            raw_max.append(float(np.linalg.norm(pos_fk - rec, axis=1).max()))
        if not offs:
            print(f'{body}: 无可用样本')
            continue
        offs = np.array(offs)
        mags = np.linalg.norm(offs, axis=1)
        med = float(np.median(mags))
        sd = np.array([float(np.std(offs[:, k])) for k in range(3)])
        disp = float(max(sd) / max(med, 1e-12))
        c1 = med > MULT * CAP
        c2 = disp < DISP_MAX

        # 对照：V4.7 §6.1 引用的那组数字（L2 层）
        e2 = json.load(open(os.path.join(HERE, E2.get(real_body, E2['franka'])), encoding='utf-8'))
        e2offs = np.array([r['pos_offset_m'] for r in e2
                           if r.get('verdict') == 'OK' and r.get('pos_offset_m')], dtype=float)
        e2mags = np.linalg.norm(e2offs, axis=1)
        e2med = float(np.median(e2mags))
        e2sd = np.array([float(np.std(e2offs[:, k])) for k in range(3)])
        e2disp = float(max(e2sd) / max(e2med, 1e-12))

        print(f'\n=== {body} （L1 层逐轨迹 Kabsch，n={len(offs)}）===')
        print(f'  L1 常差模长中位  {med*1e3:9.4f} mm   分量SD {[round(x*1e3,4) for x in sd]} mm')
        print(f'  L1 相对离散度    {disp*100:6.2f} %')
        print(f'  → S3 判据：模长>1e-2 mm {"✓" if c1 else "✗"}；离散度<50% {"✓" if c2 else "✗"}'
              f'  → {"通过（标签不变）" if (c1 and c2) else "不通过（标签会翻转）"}')
        print(f'  [对照] V4.7 §6.1 所引（L2 层）模长中位 {e2med*1e3:.4f} mm，离散度 {e2disp*100:.2f}%')
        print(f'  [对照] 原始（扣常差前）残差 max 中位 {np.median(raw_max)*1e3:.3f} mm')

        out[body] = {
            'n': len(offs),
            'l1_offset_median_mm': med * 1e3,
            'l1_dispersion_pct': disp * 100,
            'l1_raw_max_median_mm': float(np.median(raw_max)) * 1e3,
            'l2_offset_median_mm': e2med * 1e3,
            'l2_dispersion_pct': e2disp * 100,
            's3_passes': bool(c1 and c2),
            'c1_magnitude': bool(c1), 'c2_dispersion': bool(c2),
            'offsets_l1': offs.tolist(),
        }

    json.dump(out, open(a.json_out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n写出 {a.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
