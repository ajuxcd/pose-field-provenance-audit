#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SR-37：资产扰动敏感性（P3）——把「资产来源的影响未分离」拆成可分离的通道。

## 问题

§6.6 声明「资产来源的影响未分离」为效度威胁之一，但只作声明不给区间。
本脚本把它拆成**两条通道**，分别量化：

| 通道 | 扰动对象 | 预期 | 是否可分离 |
|---|---|---|---|
| **惯性通道** | 质量与惯量 ±3%（＝密度估计误差）、惯量单独 ±3%（＝惯量分布估计误差） | 改变动力学与跟踪误差 → Δ 应变化 | **可分离、可量化** |
| **几何通道** | 碰撞/视觉 mesh 顶点等比缩放 ±3%（不改连杆原点，故 FK 不变） | 无接触 → 碰撞几何不参与动力学 | 可扰动，但本版预期为零 |

**场景/物体通道（物体网格、材质、接触参数）本版不可分离**：回放场景只有臂与夹爪、
不存在接触对（§5.4 已声明），四个公开数据集均无逐帧物体位姿（附表 1）。
故本脚本**不能**声称已估计全部资产影响，只能把其中两条通道量化。

## 口径

- **L1 不受影响**：质量、惯量、mesh 顶点都不进入正运动学，故 L1 残差与来源标签不变。
  （连杆原点与关节轴不动，故 FK 严格不变。）
- 样本：每本体取 E2 已跑通的前 30 条，与正文同批。
- Δ = 段末 L2 残差 − 该条 L1 的扣常差后 P95（与 §6.2 同口径）。

用法：`python3 sr37_asset_sensitivity.py [--n 30]`
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.path.join(HERE, '..')

RAW = os.environ.get('PFPA_DATA_ROOT', 'data')
DIRS = {
    'franka': os.path.join(RAW, 'state_e1'),
    'arx_x5': os.path.join(RAW, 'state_e1_rd', 'arx_x5'),
    'piper': os.path.join(RAW, 'state_e1_rd', 'piper'),
}
E2 = {'franka': 'e2_franka_l2.json', 'arx_x5': 'e2_arx_l2.json', 'piper': 'e2_piper_l2.json'}
E1 = {'franka': ('ch21', 'e1_franka_full.json'),
      'arx_x5': ('ch21', 'e1_arx_left.json'),
      'piper': ('ch21', 'e1_piper_left.json')}

CONFIGS = [
    ('base', dict()),
    ('mass097', dict(mass=0.97)),
    ('mass103', dict(mass=1.03)),
    ('inertia097', dict(inertia=0.97)),
    ('inertia103', dict(inertia=1.03)),
    ('geo097', dict(geo=0.97)),
    ('geo103', dict(geo=1.03)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--json-out', default=os.path.join(HERE, 'sr37_asset_sensitivity.json'))
    a = ap.parse_args()

    import replay_l2 as R2
    import mujoco

    samples, l1vals = {}, {}
    for body in ('franka', 'arx_x5', 'piper'):
        e2 = json.load(open(os.path.join(HERE, E2[body]), encoding='utf-8'))
        ok = [r for r in e2 if r.get('verdict') == 'OK'][:a.n]
        samples[body] = [os.path.join(DIRS[body], r['file']) for r in ok]
        d1 = os.path.join(BASE, E1[body][0], E1[body][1])
        l1 = {r['file']: r for r in json.load(open(d1, encoding='utf-8'))
              if r.get('verdict') == 'OK'}
        l1vals[body] = {r['file']: l1.get(r['file'], {}).get('pos_after_p95_m') for r in ok}
        miss = sum(1 for r in ok if r['file'] not in l1)
        print(f'{body}: 样本 {len(ok)} 条，L1 缺失 {miss}')

    orig_load = R2.load_model

    def make_patched(cfg):
        def patched(body):
            mujoco_, m, d = orig_load(body)
            if 'mass' in cfg:
                # 质量与惯量同步缩放＝密度估计误差（同形状、密度不同）。
                # 只改质量不改惯量会引入额外的动力学不一致，不是干净的密度扰动。
                m.body_mass[:] *= cfg['mass']
                m.body_inertia[:] *= cfg['mass']
            if 'inertia' in cfg:
                # 仅惯量缩放＝惯量分布估计误差（质量不变）。
                m.body_inertia[:] *= cfg['inertia']
            if 'geo' in cfg:
                # mesh 顶点等比缩放：改各 mesh 在自身 body 坐标系内的形状尺寸，
                # **不动 body 原点与关节轴**，故正运动学严格不变（L1 不受影响）。
                if m.nmesh > 0:
                    m.mesh_vert[:] *= cfg['geo']
            try:
                mujoco_.mj_setConst(m, d)
            except Exception:
                pass
            return mujoco_, m, d
        return patched

    results = {}
    t_all = time.time()
    for key, cfg in CONFIGS:
        R2.load_model = make_patched(cfg)
        t0 = time.time()
        per_body = {}
        for body in ('franka', 'arx_x5', 'piper'):
            rows = []
            for p in samples[body]:
                try:
                    r = R2.analyse(p, body, 'left')
                except Exception as e:
                    print(f'  ✗ {key}/{os.path.basename(p)} {type(e).__name__}: {e}')
                    continue
                if r.get('verdict') != 'OK':
                    continue
                l1 = l1vals[body].get(os.path.basename(p))
                rows.append({'file': os.path.basename(p),
                             'l2_end_m': r['pos_end_median_m'],
                             'l1_p95_m': l1,
                             'delta_m': (r['pos_end_median_m'] - l1) if l1 is not None else None})
            if rows:
                dl = np.array([x['delta_m'] for x in rows if x['delta_m'] is not None])
                l2 = np.array([x['l2_end_m'] for x in rows])
                per_body[body] = {
                    'n': len(rows),
                    'l2_end_median_mm': float(np.median(l2) * 1e3),
                    'delta_median_mm': float(np.median(dl) * 1e3) if len(dl) else None,
                    'delta_neg': int((dl < 0).sum()) if len(dl) else None,
                }
        results[key] = {'cfg': cfg, 'per_body': per_body,
                        'elapsed_s': round(time.time() - t0, 1)}
        print(f'\n=== {key}（{cfg}）用时 {results[key]["elapsed_s"]} s ===')
        for body, v in per_body.items():
            print(f'  {body:<8} n={v["n"]}  L2 段末中位 {v["l2_end_median_mm"]:8.3f} mm  '
                  f'Δ 中位 {v["delta_median_mm"]:8.3f} mm  负值 {v["delta_neg"]}/{v["n"]}')

    base = results['base']['per_body']
    summary = {}
    for key in results:
        if key == 'base':
            continue
        summary[key] = {}
        for body in ('franka', 'arx_x5', 'piper'):
            b0, b1 = base.get(body), results[key]['per_body'].get(body)
            if not b0 or not b1 or not b0['delta_median_mm'] or not b1['delta_median_mm']:
                continue
            summary[key][body] = {
                'delta_base_mm': b0['delta_median_mm'],
                'delta_new_mm': b1['delta_median_mm'],
                'delta_abs_mm': b1['delta_median_mm'] - b0['delta_median_mm'],
                'rel_pct': (b1['delta_median_mm'] / b0['delta_median_mm'] - 1) * 100
                           if b0['delta_median_mm'] else None,
                'l2_rel_pct': (b1['l2_end_median_mm'] / b0['l2_end_median_mm'] - 1) * 100
                              if b0['l2_end_median_mm'] else None,
                'neg_base': b0['delta_neg'], 'neg_new': b1['delta_neg'],
            }
    print('\n' + '=' * 96)
    print('相对基准的变化（Δ 中位：相对%／绝对 mm）')
    print('=' * 96)
    print(f'{"配置":<12}' + ''.join(f'{b:>24}' for b in ('franka', 'arx_x5', 'piper')))
    for key in summary:
        row = f'{key:<12}'
        for body in ('franka', 'arx_x5', 'piper'):
            e = summary[key].get(body)
            if e and e['rel_pct'] is not None:
                row += f'{e["rel_pct"]:+8.1f}%/{e["delta_abs_mm"]:+7.3f} mm'.rjust(24)
            else:
                row += '—'.rjust(24)
        print(row)

    out = {'n_per_body': a.n, 'results': results, 'summary_vs_base': summary,
           'note': 'L1 为纯 FK，质量/惯量/mesh 顶点均不进入正运动学，故 L1 与来源标签不受本扫描影响；'
                   '场景/物体通道（物体网格、材质、接触参数）因无逐帧物体位姿、无接触对而不可分离，'
                   '本扫描不能替代该效度声明。'}
    json.dump(out, open(a.json_out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n总用时 {time.time()-t_all:.0f} s｜写出 {a.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
