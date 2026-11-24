# -*- coding: utf-8 -*-
"""段级重采样 ＋ 按轨迹聚类的 bootstrap / 置换（3.x 的统计加固）。

为什么这样做：
  L2 按 5 秒切段且**每段独立初始化**，故段是回放意义上的独立观测单元；
  但同一轨迹内的段仍共享本体、任务与标定，不能当作独立样本。
  因此：**用段做观测单元**（提高分辨率），但**以轨迹为簇做重抽**（保住相关性）。

输出三组数：
  1. 轨迹级 AUC（原口径，n=轨迹数）与其 95% CI；
  2. 段级 AUC（n=段数）与**按轨迹聚类**的 95% CI；
  3. KS 的簇置换 p 值（把轨迹整体打标签，不拆散段）。

用法：python3 cluster_stats.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = HERE


def load(f):
    d = json.load(open(os.path.join(P, f), encoding='utf-8'))
    return [r for r in d if r.get('verdict') == 'OK']


def body_of(name):
    t = name.split('__')[1] if '__' in name else name
    return 'dual' if t.startswith('two_arm') else 'single'


def auc(a, b):
    """P(a > b)：a 为「失败」组，b 为「成功」组。0.5=不可分。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = 0.0
    for x in a:
        c += np.sum(x > b) + 0.5 * np.sum(x == b)
    return float(c / (len(a) * len(b)))


def ks_stat(a, b):
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    grid = np.concatenate([a, b])
    Fa = np.searchsorted(a, grid, side='right') / len(a)
    Fb = np.searchsorted(b, grid, side='right') / len(b)
    return float(np.abs(Fa - Fb).max())


def main():
    succ = load('e2_franka_l2.json')
    fail = [r for r in load('e3_failure_l2.json') if body_of(r['file']) == 'single']
    dual_s = load('e3_dual_success_l2.json')
    dual_f = [r for r in load('e3_failure_l2.json') if body_of(r['file']) == 'dual']
    piper = load('e2_piper_l2.json')
    arx = load('e2_arx_l2.json')

    # ---- 1) 轨迹级（原口径） ----
    es = np.array([r['pos_end_median_m'] for r in succ]) * 1000
    ef = np.array([r['pos_end_median_m'] for r in fail]) * 1000
    a_ep = auc(ef, es)

    rng = np.random.default_rng(7)
    bs = [auc(rng.choice(ef, len(ef), True), rng.choice(es, len(es), True))
          for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])

    # ---- 2) 段级（以轨迹为簇重抽） ----
    seg_s = [np.array(r['seg_pos_end_m']) * 1000 for r in succ]
    seg_f = [np.array(r['seg_pos_end_m']) * 1000 for r in fail]
    n_seg_s = sum(len(x) for x in seg_s)
    n_seg_f = sum(len(x) for x in seg_f)
    a_seg = auc(np.concatenate(seg_f), np.concatenate(seg_s))

    seg_bs = []
    for _ in range(5000):
        idx_s = rng.integers(0, len(seg_s), len(seg_s))
        idx_f = rng.integers(0, len(seg_f), len(seg_f))
        A = np.concatenate([seg_f[i] for i in idx_f])
        B = np.concatenate([seg_s[i] for i in idx_s])
        seg_bs.append(auc(A, B))
    s_lo, s_hi = np.percentile(seg_bs, [2.5, 97.5])

    # ---- 3) KS：簇置换（整条轨迹换标签） ----
    def ks_cluster(ga, gb, n_perm=5000):
        A = [np.array(r['seg_pos_end_m']) * 1000 for r in ga]
        B = [np.array(r['seg_pos_end_m']) * 1000 for r in gb]
        obs = ks_stat(np.concatenate(A), np.concatenate(B))
        pool = A + B
        na = len(A)
        cnt = 0
        for _ in range(n_perm):
            rng.shuffle(pool)
            s = ks_stat(np.concatenate(pool[:na]), np.concatenate(pool[na:]))
            if s >= obs - 1e-12:
                cnt += 1
        return obs, (cnt + 1) / (n_perm + 1)

    print('=== 单臂 Franka：失败 vs 成功 ===')
    print(f'  轨迹级  n=失败 {len(ef)} / 成功 {len(es)}｜AUC = {a_ep:.4f}  CI [{lo:.4f}, {hi:.4f}]')
    print(f'  段  级  n=失败 {n_seg_f} / 成功 {n_seg_s}｜AUC = {a_seg:.4f}  CI [{s_lo:.4f}, {s_hi:.4f}]'
          f'（按轨迹聚类）')
    print(f'  → 段数/轨迹数 = {n_seg_s/len(es):.2f}（成功侧）／{n_seg_f/len(ef):.2f}（失败侧）')
    print()

    print('=== 跨本体 KS（段级 ＋ 簇置换） ===')
    pairs = [('Franka–Piper', succ, piper), ('Franka–X5', succ, arx), ('X5–Piper', arx, piper)]
    for name, g1, g2 in pairs:
        D, p = ks_cluster(g1, g2)
        n1 = sum(len(r['seg_pos_end_m']) for r in g1)
        n2 = sum(len(r['seg_pos_end_m']) for r in g2)
        print(f'  {name:<14s} n={n1}/{n2}  D={D:.4f}  p(簇置换)={p:.4f}')

    print()
    print('=== 四组段末残差（段级中位） ===')
    for nm, g in [('单臂成功', succ), ('单臂失败', fail), ('双臂成功', dual_s), ('双臂失败', dual_f)]:
        allseg = np.concatenate([np.array(r['seg_pos_end_m']) * 1000 for r in g])
        print(f'  {nm}: 轨迹 {len(g):>2d}｜段 {len(allseg):>3d}｜段级中位 {np.median(allseg):6.2f} mm')

    out = {'auc_episode': a_ep, 'auc_episode_ci': [lo, hi],
           'auc_segment': a_seg, 'auc_segment_ci': [s_lo, s_hi],
           'n_ep_succ': len(es), 'n_ep_fail': len(ef),
           'n_seg_succ': n_seg_s, 'n_seg_fail': n_seg_f}
    json.dump(out, open(os.path.join(P, 'cluster_stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n→ cluster_stats.json')


if __name__ == '__main__':
    main()
