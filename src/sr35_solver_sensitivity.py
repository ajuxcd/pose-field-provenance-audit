#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SR-35：L2 回放的求解器敏感性扫描（P1）。

## 为什么要做

全文只报了一档 `timestep = 1/(FPS×N_SUB)`（N_SUB=20，30 Hz → dt ≈ 1.667 ms），
**没有任何求解器／积分器／迭代次数／摩擦的敏感性扫描**。这是审稿人一句
「把 timestep 减半再看」就能要求补的那一项，也是最便宜可补的一项。

## 关键口径：L1 不参与扫描

L1 是**纯正运动学复算**（FK），与物理引擎、积分器、迭代次数、摩擦全都无关；
因此求解器敏感性只作用于 **L2 与 Δ**。对 L1 另做「引擎交叉验证」是无意义的
（L1 不经引擎），本脚本刻意不做，也建议正文这样声明。

## 九种配置

| 键 | 改动 | 目的 |
|---|---|---|
| base | 基准：N_SUB=20／Euler／iterations=100／pyramidal 锥／摩擦 ×1.0 | 对照 |
| nsub10 | N_SUB=10（dt 加倍） | 时间离散 |
| nsub40 | N_SUB=40（dt 减半） | 时间离散 |
| rk4 | integrator=Euler→RK4 | 积分器 |
| iter50 | iterations 100→50 | 求解器收敛预算 |
| iter200 | iterations 100→200 | 求解器收敛预算 |
| elliptic | 摩擦锥 pyramidal→elliptic | 接触锥 |
| fric08 | 滑动摩擦系数 ×0.8 | 接触参数 |
| fric12 | 滑动摩擦系数 ×1.2 | 接触参数 |

**摩擦锥没有「±20%」**：MuJoCo 的接触锥只有 pyramidal／elliptic 两种离散形式，
故改为「锥形式切换」＋「滑动摩擦系数 ±20%」两项。

## 样本

三本体各 10 条，取 E2 已跑通（verdict=OK）的前 10 条，**与正文同批**，
保证可比。ARX X5 的回放按设计 `mjDSBL_CONTACT`（无物体、无接触），
故其摩擦两项的敏感性**恒为 0**——这不是结果，是设定，须在正文说明。

用法：`python3 sr35_solver_sensitivity.py [--n 10]`
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
    ('nsub10', dict(n_sub=10)),
    ('nsub40', dict(n_sub=40)),
    ('rk4', dict(integrator='RK4')),
    ('iter50', dict(iterations=50)),
    ('iter200', dict(iterations=200)),
    ('elliptic', dict(cone='elliptic')),
    ('fric08', dict(friction=0.8)),
    ('fric12', dict(friction=1.2)),
    ('tol1e6', dict(tolerance=1e-6)),
    ('solver_cg', dict(solver='CG')),
    ('ls_iter10', dict(ls_iterations=10)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10, help='每本体取前 N 条')
    ap.add_argument('--json-out', default=os.path.join(HERE, 'sr35_solver_sensitivity.json'))
    a = ap.parse_args()

    import replay_l2 as R2
    import mujoco

    # ── 选样本（与 E2 同批） ──────────────────────────────────────────
    samples = {}
    l1vals = {}
    for body in ('franka', 'arx_x5', 'piper'):
        e2 = json.load(open(os.path.join(HERE, E2[body]), encoding='utf-8'))
        ok = [r for r in e2 if r.get('verdict') == 'OK'][:a.n]
        samples[body] = [os.path.join(DIRS[body], r['file']) for r in ok]
        d1 = os.path.join(BASE, E1[body][0], E1[body][1])
        l1 = {r['file']: r for r in json.load(open(d1, encoding='utf-8'))
              if r.get('verdict') == 'OK'}
        l1vals[body] = {r['file']: l1.get(r['file'], {}).get('pos_after_p95_m')
                        for r in ok}
        missing = [r['file'] for r in ok if r['file'] not in l1]
        print(f'{body}: 样本 {len(ok)} 条，L1 缺失 {len(missing)} 条'
              f'{"（缺失者不计入 Δ）" if missing else ""}')
    for body, ps in samples.items():
        for p in ps:
            if not os.path.exists(p):
                print('  ⚠ 缺文件', p)
                return 2

    # ── 打补丁：让模型带上求解器参数 ──────────────────────────────────
    orig_load = R2.load_model

    def make_patched(cfg):
        def patched(body):
            mujoco_, m, d = orig_load(body)
            if cfg.get('integrator') == 'RK4':
                m.opt.integrator = int(mujoco.mjtIntegrator.mjINT_RK4)
            if 'iterations' in cfg:
                m.opt.iterations = cfg['iterations']
            if cfg.get('cone') == 'elliptic':
                m.opt.cone = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
            if 'friction' in cfg:
                # 滑动摩擦系数（第 0 列）。ARX X5 禁用了接触，此项对其无作用。
                m.geom_friction[:, 0] *= cfg['friction']
            if 'tolerance' in cfg:
                m.opt.tolerance = cfg['tolerance']
            if cfg.get('solver') == 'CG':
                # 本机 MuJoCo 3.17 的枚举名是 mjSOL_CG（不是 mjSOLVER_CG）
                m.opt.solver = int(mujoco.mjtSolver.mjSOL_CG)
            if 'ls_iterations' in cfg:
                m.opt.ls_iterations = cfg['ls_iterations']
            return mujoco_, m, d
        return patched

    # ── 接触诊断：记录回放过程中的接触对数 ────────────────────────────
    # 「摩擦／接触锥／迭代次数敏感性为零」必须有解释：若回放全程 ncon=0，
    # 那是因为**本文的回放场景不含接触**（无物体位姿，见 §5.4 限定），
    # 接触相关的求解器与参数根本不被激活——这是设定，不是结果。
    contact_stat = {'max': 0, 'total': 0, 'calls': 0}

    def _contact_cb(m, d):
        n = int(d.ncon)
        contact_stat['calls'] += 1
        contact_stat['total'] += n
        if n > contact_stat['max']:
            contact_stat['max'] = n
        return False

    try:
        mujoco.set_mjcb_contact(_contact_cb)
    except Exception as e:
        print('⚠ 接触回调注册失败：', e)

    results = {}
    t_all = time.time()
    for key, cfg in CONFIGS:
        R2.N_SUB = cfg.get('n_sub', 20)
        R2.load_model = make_patched(cfg)
        t0 = time.time()
        contact_stat.update({'max': 0, 'total': 0, 'calls': 0})
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
                d_pos = (r['pos_end_median_m'] - l1) if l1 is not None else None
                rows.append({'file': os.path.basename(p),
                             'l2_end_m': r['pos_end_median_m'],
                             'l2_end_rot_deg': r['rot_end_median_deg'],
                             'l1_p95_m': l1, 'delta_m': d_pos})
            if rows:
                dl = np.array([x['delta_m'] for x in rows if x['delta_m'] is not None])
                l2 = np.array([x['l2_end_m'] for x in rows])
                per_body[body] = {
                    'n': len(rows),
                    'l2_end_median_mm': float(np.median(l2) * 1e3),
                    'delta_median_mm': float(np.median(dl) * 1e3) if len(dl) else None,
                    'delta_neg': int((dl < 0).sum()) if len(dl) else None,
                    'delta_neg_frac': float((dl < 0).mean()) if len(dl) else None,
                    'rows': rows,
                }
        results[key] = {'cfg': cfg, 'per_body': per_body,
                        'elapsed_s': round(time.time() - t0, 1),
                        'contact': dict(contact_stat)}
        print(f'\n=== {key}（{cfg}）用时 {results[key]["elapsed_s"]} s ==='
              f'｜接触对峰值 {contact_stat["max"]}（累计 {contact_stat["total"]}）')
        for body, v in per_body.items():
            dm = v['delta_median_mm']
            print(f'  {body:<8} n={v["n"]}  L2 段末中位 {v["l2_end_median_mm"]:8.3f} mm  '
                  f'Δ 中位 {dm:8.3f} mm  负值 {v["delta_neg"]}/{v["n"]}')

    # ── 相对基准的变化 ────────────────────────────────────────────────
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
                'neg_base': b0['delta_neg_frac'], 'neg_new': b1['delta_neg_frac'],
                'l2_rel_pct': (b1['l2_end_median_mm'] / b0['l2_end_median_mm'] - 1) * 100
                              if b0['l2_end_median_mm'] else None,
            }
    print('\n' + '=' * 92)
    print('相对基准的 Δ 中位变化（%）')
    print('=' * 92)
    hdr = f'{"配置":<10}' + ''.join(f'{b:>22}' for b in ('franka', 'arx_x5', 'piper'))
    print(hdr)
    for key in summary:
        row = f'{key:<10}'
        for body in ('franka', 'arx_x5', 'piper'):
            e = summary[key].get(body)
            row += f'{(f"{e[chr(114)+chr(101)+chr(108)+chr(95)+chr(112)+chr(99)+chr(116)]:+.1f}%" if e and e["rel_pct"] is not None else "—"):>22}'
        print(row)

    out = {'n_per_body': a.n, 'results': results, 'summary_vs_base': summary,
           'contact_diag': {k: v.get('contact') for k, v in results.items()},
           'note': 'L1 为纯 FK，不受求解器影响，故扫描只作用于 L2 与 Δ；'
                   'ARX X5 回放禁用接触（mjDSBL_CONTACT），其摩擦两项敏感性恒为 0。'}
    json.dump(out, open(a.json_out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n总用时 {time.time()-t_all:.0f} s｜写出 {a.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
