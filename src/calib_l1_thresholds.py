#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1.10 · L1 阈值初标（逐本体分档）。

输入：1.9 的 `pipeline/fk_identity_matrix.json`（20 组：Franka 1、UR5 1、
ARX X5 6、Piper 6、Piper X 6）。

为什么不能取一个全局阈值
  L1 的残差性质随本体分三档，跨档差约 5–6 个数量级：
    同源档   Franka：记录位姿就是官方模型 FK 的输出 → 残差 = 数值噪声 4e-8 m
    近似档   ARX X5 / Piper：模型与实物有固定安装差，扣常差后 mm 级
    缺口档   Piper X：缺该型号 URDF，扣常差后仍有 100 mm 且随轨迹漂移 → 不可用
  若统一取 1e-5 m，近似档全部被判失败（白名单空）；
  若统一取 1e-2 m，同源档的 4e-8 残差会被完全放过（假阴性 100%）。
  → **必须逐本体、分档设定**。

阈值定义（每档不同，均取「扣常差后残差分布」的高分位）
  同源档：阈值＝数值上限。取 1e-6 m / 1e-3 deg（实测 4e-8 m / 5.8e-6 deg 的 25 / 170 倍
          余量），用于识别「该本体出现了非数值误差」——L1 在此档只做模型自洽检查。
  近似档：阈值＝P95 × 安全系数 1.5。P95 由实测逐条残差算，安全系数用于容纳
          E1 扩大到 30 条/任务时的分布尾部。
  缺口档：**不设阈值，直接不纳入 E1**。

用法：
  python3 calib_l1_thresholds.py            # 打印标定表
  python3 calib_l1_thresholds.py --json     # 输出配置片段（供 E1 读取）
  python3 calib_l1_thresholds.py --write    # 写入 l1_thresholds.json
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, '..', 'pipeline', 'fk_identity_matrix.json')
OUT = os.path.join(HERE, 'l1_thresholds.json')

# 分档归属（2026-11-06 定；UR5 已按 1.2c 剔除，Piper X 为资产缺口）
#
# ⚠ **位置与朝向必须分开定档**，不能按本体给一个档。
# 判据：扣常差后的改善倍数（raw / after）。
#   · 改善 > 10 倍 → 常差解释了大部分偏差，剩下的才是模型精度 → 「近似档」
#   · 改善 < 3 倍  → 本来就没多少常差可扣，残差接近数值噪声 → 「同源档」
# 实测 ARX X5 正是这种「位置近似、朝向同源」的混合体：
#   位置 1.85e-1 → 2.9e-3 m（**62–101 倍**改善）
#   朝向 8.4e-4 → 5.1e-4 deg（**仅 1.5–2 倍**改善）
# 若按本体统一归为近似档，会给朝向设一个 8.4e-4 deg 的阈值——比 Franka 同源档的
# 1e-3 deg 还松，且远高于它自己的实测值，等于对朝向失去筛除力。
TIER_POS = {
    'franka': 'same-source',
    'arx_x5': 'approximate',      # 位置：62–101 倍改善
    'piper': 'approximate',
    'piper_x': 'asset-gap',
    'ur5': 'excluded-data-defect',
}
TIER_ROT = {
    'franka': 'same-source',
    'arx_x5': 'same-source',      # 朝向：仅 1.5–2 倍改善，残差贴近数值噪声
    'piper': 'approximate',       # 朝向 0.67–0.76 deg，扣常差几乎无改善但绝对值大
    'piper_x': 'asset-gap',
    'ur5': 'excluded-data-defect',
}

# 改善倍数分界（用于自动核验上表是否与数据一致）
IMPROVE_APPROX = 10.0
IMPROVE_SAME = 3.0

# 同源档阈值：数值上限（非统计量）。取「数值地板」与「实测 × 余量」的较大者，
# 保证任何本体都有 ≥3 倍余量，不会因为该本体存储精度略低（如 float32）就误判。
SAME_SOURCE = {'pos_m': 1e-6, 'rot_deg': 1e-3}
SAME_SOURCE_MARGIN = 3.0

# 近似档安全系数
SAFETY = 1.5

# 数值地板：浮点噪声不可能低于此值，防止 P95 被零值拉成 0
FLOOR_POS = 1e-6
FLOOR_ROT = 1e-4


def p95(vals):
    """线性插值 P95（不依赖 numpy 的版本差异）。"""
    v = sorted(vals)
    if not v:
        return None
    k = (len(v) - 1) * 0.95
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()

    rows = json.load(open(MATRIX, encoding='utf-8'))
    by_kind = {}
    for r in rows:
        k = r.get('kind')
        by_kind.setdefault(k, []).append(r)

    conf = {'generated': '2026-11-06', 'source': 'fk_identity_matrix.json',
            'definition': 'P95 of post-offset residuals × 1.5 (approximate tier); '
                          'numeric ceiling (same-source tier); excluded otherwise',
            'bodies': {}}

    print('=' * 108)
    print('1.10 · L1 阈值初标（逐本体 × 位置/朝向分别分档，输入＝1.9 扣常差后残差）')
    print('=' * 108)
    print(f'{"本体":<10s} {"位置档":<17s} {"朝向档":<17s} {"组":>2s}  '
          f'{"P95位置(m)":>11s} {"P95朝向(deg)":>11s}  '
          f'{"阈值位置(m)":>11s} {"阈值朝向(deg)":>11s}  改善(位/朝)')
    print('-' * 108)

    for kind in ('franka', 'arx_x5', 'piper', 'piper_x', 'ur5'):
        rs = by_kind.get(kind, [])
        if not rs:
            continue
        t_pos, t_rot = TIER_POS[kind], TIER_ROT[kind]
        pos = [r['pos']['res_after_max_m'] for r in rs]
        raw_pos = [r['raw']['pos_max_m'] for r in rs]
        rot = [r['rot']['res_after_max_deg'] for r in rs]
        raw_rot = [r['raw']['rot_max_deg'] for r in rs]
        p_pos, p_rot = p95(pos), p95(rot)
        # 改善倍数按**逐组**算再取中位：用算术均值会被极端组带偏
        # （Franka 一组实测 rot 45.0 → 5.8e-6，均值法算出 7.7e6 倍，无意义）
        ratios_pos = [a / b for a, b in zip(raw_pos, pos) if b > 0]
        ratios_rot = [a / b for a, b in zip(raw_rot, rot) if b > 0]
        imp_pos = sorted(ratios_pos)[len(ratios_pos) // 2] if ratios_pos else float('nan')
        imp_rot = sorted(ratios_rot)[len(ratios_rot) // 2] if ratios_rot else float('nan')

        # 位置分档
        if t_pos == 'same-source':
            # 数值地板与「实测最大值 × 余量」取大者，保证 ≥3 倍余量
            th_pos = max(SAME_SOURCE['pos_m'], max(pos) * SAME_SOURCE_MARGIN)
        elif t_pos == 'approximate':
            th_pos = max(p_pos * SAFETY, FLOOR_POS)
        else:
            th_pos = None
        # 朝向分档
        if t_rot == 'same-source':
            th_rot = max(SAME_SOURCE['rot_deg'], max(rot) * SAME_SOURCE_MARGIN)
        elif t_rot == 'approximate':
            th_rot = max(p_rot * SAFETY, FLOOR_ROT)
        else:
            th_rot = None

        note = ''
        if t_pos != t_rot:
            note = '（位置/朝向分属不同档）'
        if th_pos is None:
            note += '不纳入 E1'

        print(f'{kind:<10s} 位置:{t_pos:<14s} 朝向:{t_rot:<14s} {len(rs):>2d}  '
              f'{p_pos:>11.3e} {p_rot:>11.3e}  '
              + (f'{th_pos:>11.3e}' if th_pos else f'{"—":>11s}')
              + (f' {th_rot:>11.3e}' if th_rot else f' {"—":>11s}')
              + f'  {imp_pos:5.1f}x/{imp_rot:5.2f}x {note}')

        conf['bodies'][kind] = {
            'tier_pos': t_pos, 'tier_rot': t_rot, 'n_groups': len(rs),
            'n_frames_min': min(r['n'] for r in rs),
            'p95_pos_m': p_pos, 'p95_rot_deg': p_rot,
            'max_pos_m': max(pos), 'max_rot_deg': max(rot),
            'improve_pos_x': round(imp_pos, 1), 'improve_rot_x': round(imp_rot, 2),
            'threshold_pos_m': th_pos, 'threshold_rot_deg': th_rot,
            'in_e1': t_pos != 'asset-gap' and t_pos != 'excluded-data-defect',
        }

    print('-' * 96)
    print('档间跨度：同源 4.0e-8 m ｜ 近似 1.8e-3–1.4e-2 m ｜ 缺口 1.0e-1 m')
    print('         → 同源与近似的位置阈值差 3 个数量级，近似与缺口再差 1–2 个数量级；')
    print('           统一阈值必错，这是分档的唯一理由。')
    print()
    print('注：UR5 的 1.9 数据（扣常差后 4.8e-4 m / 16.05 deg）在数值上属近似档，')
    print('    但其**成功侧 puppet 字段未写入**（1.2c，冻结率 30/31），无从取得逐条残差，')
    print('    故不参与阈值标定。该行保留在正文作缺陷证据，不作阈值依据。')
    print()
    print('⚠ 朝向上限的量级提醒：Franka 同源档实测 5.8e-6 deg，阈值 1e-3 deg 有 170 倍余量；')
    print('  近似档 P95 达 0.7 deg（Piper），两者差 3 个数量级，同样不可混用。')

    if a.json:
        print(json.dumps(conf, ensure_ascii=False, indent=1))
    if a.write:
        json.dump(conf, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'\n已写出 {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())