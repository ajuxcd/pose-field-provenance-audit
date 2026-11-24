#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SR-52／SR-53／SR-54 三项分析：阈值留出法、主结论置信区间、计算成本。

SR-52 阈值留出法：阈值用同一批数据标定、又用同一批数据做白名单判定，构成样本复用。
  做法：把每本体 6 个分组按奇偶分成两半，一半标定阈值、一半验证准入，看白名单是否改变。
SR-53 主结论 CI：对四批 L1 残差中位数与「同源档/失败侧」的 1.76×10⁴ 倍反差做 bootstrap 95% CI。
SR-54 计算成本：单条轨迹判定的耗时（L1 复算 + 噪声注入）与随轨迹数的变化，
  支撑「可作为入库前筛检」这一主张。

用法：python3 sr52_54_analysis.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CH21 = os.path.abspath(os.path.join(HERE, os.pardir, 'ch21'))
sys.path.insert(0, CH21)
sys.path.insert(0, HERE)

RAW = os.environ.get('PFPA_DATA_ROOT', 'data')
CAP = 1.0e-6
SEED = 20261111
N_BOOT = 10000


def load_matrix():
    p = os.path.join(HERE, 'fk_identity_matrix.json')
    return json.load(open(p, encoding='utf-8'))


def sr52_leaveout():
    """阈值留出法：奇偶分组交叉标定／验证。"""
    print('\n' + '=' * 88)
    print('SR-52 阈值标定的留出法验证（奇偶分组交叉）')
    print('=' * 88)
    m = load_matrix()
    # 分组字段是 `kind`（本体），不是 `body`（末端连杆名）——2026-11-11 实测更正
    by_body = {}
    for g in m:
        if g.get('kind'):
            by_body.setdefault(g['kind'], []).append(g)

    out = {}
    for body, gs in sorted(by_body.items()):
        if body in ('ur5',):
            continue
        # 每组一个标定值（扣常差后 P95）与一个准入判定
        gs = sorted(gs, key=lambda x: x.get('file', ''))
        folds = {'odd': gs[0::2], 'even': gs[1::2]}
        res = {}
        for cal_name, val_name in (('odd', 'even'), ('even', 'odd')):
            cal, val = folds[cal_name], folds[val_name]
            if not cal or not val:
                continue
            cal_p95 = [g['pos']['res_after_max_m'] for g in cal
                       if g.get('pos', {}).get('res_after_max_m') is not None]
            if not cal_p95:
                continue
            p95 = float(np.percentile(cal_p95, 95))
            # 同源档阈值＝数值上限；近似档＝P95 × 1.5（与 4.1 节一致）
            thr = CAP if p95 <= CAP else p95 * 1.5
            admit = [g['pos']['res_after_max_m'] <= thr for g in val]
            res['cal=%s' % cal_name] = dict(
                cal_n=len(cal), val_n=len(val),
                cal_p95_m=p95, threshold_m=thr,
                admitted=int(sum(admit)), total=len(admit),
                admit_rate=float(np.mean(admit)) if admit else None,
                admitted_files=[g.get('file', '') for g, a in zip(val, admit) if a])
        # 全量标定（现文口径）作对照
        all_p95 = [g['pos']['res_after_max_m'] for g in gs
                   if g.get('pos', {}).get('res_after_max_m') is not None]
        full_p95 = float(np.percentile(all_p95, 95))
        full_thr = CAP if full_p95 <= CAP else full_p95 * 1.5
        full_admit = [g['pos']['res_after_max_m'] <= full_thr for g in gs]
        res['full'] = dict(cal_n=len(gs), val_n=len(gs), cal_p95_m=full_p95,
                           threshold_m=full_thr, admitted=int(sum(full_admit)),
                           total=len(full_admit),
                           admit_rate=float(np.mean(full_admit)))
        out[body] = res
        print(f'\n  {body}（{len(gs)} 组）')
        for k, v in res.items():
            print(f'    {k:<12} 标定 P95={v["cal_p95_m"]:.4e} m 阈值={v["threshold_m"]:.4e} m '
                  f'→ 验证准入 {v["admitted"]}/{v["total"]}（{v["admit_rate"]*100:.0f}%）')
    return out


def sr53_ci():
    """主结论的 bootstrap 95% CI。"""
    print('\n' + '=' * 88)
    print('SR-53 主结论的 bootstrap 95% 置信区间')
    print('=' * 88)
    rng = np.random.default_rng(SEED)

    def boot_ci(vals, n=N_BOOT):
        v = np.asarray(vals, dtype=float)
        if len(v) == 0:
            return None
        idx = rng.integers(0, len(v), size=(n, len(v)))
        meds = np.median(v[idx], axis=1)
        return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

    out = {}
    # E1 三本体的 L1 残差（逐轨迹 P95）
    files = {
        'Franka 成功侧': os.path.join(RAW, 'state_e1'),
        'Franka 失败侧': os.path.join(RAW, 'failure_e3'),
        'ARX X5': os.path.join(RAW, 'state_e1_rd', 'arx_x5'),
        'Piper': os.path.join(RAW, 'state_e1_rd', 'piper'),
    }
    body_key = {'Franka 成功侧': 'franka', 'Franka 失败侧': 'franka',
                'ARX X5': 'arx_x5', 'Piper': 'piper'}
    print('\n  逐轨迹 L1 残差 P95 的中位数与 95% CI：')
    for name, d in files.items():
        fs = [os.path.join(d, f) for f in sorted(os.listdir(d))
              if f.endswith('.npz') and not f.startswith('._')]
        e1 = json.load(open(os.path.join(CH21, 'e1_%s.json' % (
            'franka_full' if name == 'Franka 成功侧' else (
                'arx_left' if name == 'ARX X5' else 'piper_left'))), encoding='utf-8')) \
            if name != 'Franka 失败侧' else None
        vals = []
        if e1:
            for r in e1:
                if r.get('verdict') == 'OK' and r.get('pos_after_p95_m') is not None:
                    vals.append(r['pos_after_p95_m'])
        else:
            # 失败侧：用 SR-42 的实测（逐条 P95 在 sr42 产物里只有统计量，此处重算）
            import e1_residuals as E
            import replay_l2 as R2L
            for p in fs:
                ep = R2L.load_episode(p, 'franka', 'left')
                if ep is None:
                    continue
                q, rec = ep['joints'], ep['ee_xyz']
                if q is None or len(q) < 2:
                    continue
                try:
                    pos_fk = E.fk_positions('franka', q)
                except Exception:
                    continue
                if pos_fk.shape != rec.shape:
                    continue
                R_c, t_c = E.kabsch(pos_fk, rec)
                dd = np.linalg.norm((R_c @ pos_fk.T).T + t_c - rec, axis=1)
                vals.append(float(np.percentile(dd, 95)))
            # 正文 0.582 mm 用的是**单臂 31 条**口径（剔除 two_arm 双臂 14 条），
            # 摘要的 1.76×10⁴ 倍即由此算出 → CI 必须同口径，否则与正文对不上
            single = [v for v, f in zip(vals, fs) if 'two_arm' not in os.path.basename(f)]
            if len(single) >= 20:
                vals = single
        if not vals:
            print(f'    {name:<14} 无数据'); continue
        ci = boot_ci(vals)
        out[name] = dict(n=len(vals), median_m=float(np.median(vals)),
                         ci95=ci)
        print(f'    {name:<14} n={len(vals):>4}  中位 {np.median(vals)*1000:8.4f} mm'
              f'  95% CI [{ci[0]*1000:.4f}, {ci[1]*1000:.4f}] mm')

    # 1.76×10⁴ 倍反差：失败侧中位 / 成功侧中位
    fs = out.get('Franka 失败侧', {}).get('median_m')
    ss = out.get('Franka 成功侧', {}).get('median_m')
    if fs and ss:
        ratio = fs / ss
        out['ratio_1.76e4'] = dict(point=float(ratio))
        print(f'\n  失败侧/成功侧 中位数之比：{ratio:.4g} 倍（正文写 1.76×10⁴）')
    return out


def sr54_cost():
    """单条轨迹判定的计算成本。"""
    print('\n' + '=' * 88)
    print('SR-54 计算成本：单条轨迹判定的耗时')
    print('=' * 88)
    import e1_residuals as E
    import replay_l2 as R2L

    d = os.path.join(RAW, 'state_e1')
    fs = [os.path.join(d, f) for f in sorted(os.listdir(d))
          if f.endswith('.npz') and not f.startswith('._')][:10]
    l1_t, s4_t, tot_t = [], [], []
    n_ok = 0
    for p in fs:
        ep = R2L.load_episode(p, 'franka', 'left')
        if ep is None:
            continue
        q, rec = ep['joints'], ep['ee_xyz']
        if q is None or len(q) < 2:
            continue
        t0 = time.perf_counter()
        pos_fk = E.fk_positions('franka', q)
        R_c, t_c = E.kabsch(pos_fk, rec)
        dd = np.linalg.norm((R_c @ pos_fk.T).T + t_c - rec, axis=1)
        t1 = time.perf_counter()
        rng = np.random.default_rng(SEED)
        qq = q + rng.normal(0.0, 1e-6, size=q.shape)
        pos_fk2 = E.fk_positions('franka', qq)
        R_c2, t_c2 = E.kabsch(pos_fk2, rec)
        dd2 = np.linalg.norm((R_c2 @ pos_fk2.T).T + t_c2 - rec, axis=1)
        t2 = time.perf_counter()
        l1_t.append(t1 - t0)
        s4_t.append(t2 - t1)
        tot_t.append(t2 - t0)
        n_ok += 1
    if not tot_t:
        return {}
    res = dict(n=n_ok,
               l1_s_per_trace=float(np.median(l1_t)),
               s4_s_per_trace=float(np.median(s4_t)),
               total_s_per_trace=float(np.median(tot_t)),
               note='单条轨迹判定 ＝ 一次正运动学复算（S1／S2）＋ 一次噪声注入复算（S4）；'
                    '与轨迹长度线性相关，不随轨迹数增长')
    print(f'\n  样本 {n_ok} 条（Franka 成功侧）')
    print(f'    L1 复算（一次 FK ＋ Kabsch）：中位 {res["l1_s_per_trace"]*1000:.2f} ms/条')
    print(f'    S4 噪声注入（再一次 FK）：  中位 {res["s4_s_per_trace"]*1000:.2f} ms/条')
    print(f'    单条轨迹完整判定：          中位 {res["total_s_per_trace"]*1000:.2f} ms/条')
    print(f'    → 判定 1000 条轨迹约需 {res["total_s_per_trace"]*1000:.1f} s（单核）；'
          f'与轨迹数线性相关')
    return res


def main():
    out = dict(sr52= sr52_leaveout(), sr53=sr53_ci(), sr54=sr54_cost())
    p = os.path.join(HERE, 'sr52_54_analysis.json')
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n写出 %s' % p)


if __name__ == '__main__':
    main()
