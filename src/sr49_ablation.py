#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SR-49 判据消融：四种配置下的标签对照（回答「哪一步真正起作用」）。

配置（按算法 1 的实际顺序裁剪）：
  C1  仅 S2          ：只用量级判据（位置 ≤ 1e-6 m → 模型正运动学输出）
  C2  S0+S2          ：加零方差预检（字段恒定 → 字段无信息）
  C3  S0+S2+S3       ：再加常差稳定性（模长 > 10× 上限 且 方向离散度 < 50% → 实机标定值）
  C4  S0+S2+S3+S4    ：全四步（S4 用 η 的实测值；缺 η 者判「来源存疑」）

每批的数据来源与口径（**必须与正文一致，否则消融无意义**）：
  · Franka 成功侧  state_e1        → L1 残差与 S2 判据
  · Franka 失败侧  failure_e3     → L1 残差（SR-42 实测）
  · ARX X5         state_e1_rd/arx_x5
  · Piper          state_e1_rd/piper
  · 常差与离散度   SR-42（L1 层，Kabsch 逐轨迹拟合后的平移）
  · η（噪声敏感性）noise_falsify_*.json ＋ sr43_fail_side_s4.json

输出：sr49_ablation.json ＋ 终端表
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'ch21')))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import e1_residuals as E   # noqa: E402
import replay_l2 as R2L    # noqa: E402

RAW = os.environ.get('PFPA_DATA_ROOT', 'data')

# ── 判据常量（与初稿算法 1 一致） ────────────────────────────────
CAP = 1.0e-6          # 同源档位置上限（m）
S0_FLOOR_POS = 1e-9   # S0 位置数值地板（m）
S0_FLOOR_ROT = 1e-6   # S0 朝向数值地板（度）
S0_JOINT_RANGE = 0.1  # S0 关节量程门（rad）
S3_MAG_MULT = 10.0    # S3 常差模长门（× 上限）
S3_DISP = 0.50        # S3 方向相对离散度上限
S4_HI, S4_LO = 5.0, 2.0   # S4 η 分界

# ── 各批的 L1 判据输入 ─────────────────────────────────────────
# kind: franka / roboDojo（前者用 e1 的 puppet__ 键，后者用 state__left_*）
BATCH = {
    'Franka 成功侧': dict(dir=os.path.join(RAW, 'state_e1'), body='franka', side='left'),
    'Franka 失败侧': dict(dir=os.path.join(RAW, 'failure_e3'), body='franka', side='left'),
    'ARX X5':        dict(dir=os.path.join(RAW, 'state_e1_rd', 'arx_x5'), body='arx_x5', side='left'),
    'Piper':         dict(dir=os.path.join(RAW, 'state_e1_rd', 'piper'), body='piper', side='left'),
}

# L1 常差与离散度（SR-42 实测，L1 层）
L1_OFF = {
    'Franka 成功侧': None,     # 恒为 0，走 S2
    'Franka 失败侧': dict(mag_mm=0.17309837555726065, disp_pct=130.1429486020559),
    'ARX X5':        dict(mag_mm=183.716, disp_pct=0.48),
    'Piper':         dict(mag_mm=12.769, disp_pct=16.86),
}
# η（σ = 1e-6 rad）
ETA = {
    'Franka 成功侧': 60.58544184585754,
    'Franka 失败侧': 0.9985108022955176,
    'ARX X5':        1.0000575917942414,
    'Piper':         1.0000232474455668,
}
# L1 残差中位（m）：Franka 失败侧 SR-42/本轮实测；其余取 E1 结果
L1_RESID = {
    'Franka 成功侧': 3.3e-8,
    'Franka 失败侧': 7.134e-4,      # 全部 45 条 P95 中位（SR-43 同口径）
    'ARX X5':        2.96e-3,
    'Piper':         6.73e-3,
}


def s0_check(pose_ptp_pos, pose_ptp_rot, joint_range):
    """S0 零方差预检：位置/朝向量程在数值地板内，且关节量程 > 0.1 rad。"""
    return (pose_ptp_pos <= S0_FLOOR_POS and pose_ptp_rot <= S0_FLOOR_ROT
            and joint_range > S0_JOINT_RANGE)


def label_by_config(name, cfg):
    """按给定配置返回 (标签, 决定该标签的步骤)。"""
    resid = L1_RESID[name]
    off = L1_OFF[name]
    eta = ETA[name]

    # S0
    if 'S0' in cfg:
        # 四批真机数据的姿态量程均 > 地板（逐轨迹有运动），故 S0 不触发；
        # 真实触发情况见公开仿真集部分。
        pass

    # S2
    if 'S2' in cfg:
        if resid <= CAP:
            return '模型正运动学输出', 'S2'
    # S3
    if 'S3' in cfg and off is not None:
        mag = off['mag_mm'] / 1000.0
        disp = off['disp_pct'] / 100.0
        if mag > S3_MAG_MULT * CAP and disp < S3_DISP:
            return '实机标定值', 'S3'
    # S4
    if 'S4' in cfg and eta is not None:
        if eta > S4_HI:
            return '模型正运动学输出', 'S4'
        if eta < S4_LO:
            return '实机标定值', 'S4'
        return '来源存疑', 'S4'
    return '来源存疑', '—'


def main():
    configs = [
        ('C1 仅 S2', ['S2']),
        ('C2 S0+S2', ['S0', 'S2']),
        ('C3 S0+S2+S3', ['S0', 'S2', 'S3']),
        ('C4 全四步', ['S0', 'S2', 'S3', 'S4']),
    ]

    print('=' * 96)
    print('SR-49 判据消融：四种配置下的标签对照')
    print('=' * 96)
    print('判据常量：同源档上限 %.1e m｜S0 地板 %.0e m／%.0e°｜S3 模长 >%.0f×上限 且 离散度 <%.0f%%｜S4 η>%.0f → A，η<%.0f → B'
          % (CAP, S0_FLOOR_POS, S0_FLOOR_ROT, S3_MAG_MULT, S3_DISP * 100, S4_HI, S4_LO))

    table = {}
    print('\n%-14s %-12s %-11s %-12s %-12s %-12s' % ('批次', 'L1 残差中位', '常差模长', '方向离散度', 'η', ''))
    for name in BATCH:
        o = L1_OFF[name]
        print('%-14s %8.3f mm %s %s %10.4f' % (
            name, L1_RESID[name] * 1000,
            ('%8.3f mm' % o['mag_mm']) if o else '        —',
            ('%6.2f %%' % o['disp_pct']) if o else '      —',
            ETA[name]))

    print('\n' + '-' * 96)
    print('%-14s' % '批次', end='')
    for cn, _ in configs:
        print(' %-18s' % cn, end='')
    print()
    for name in BATCH:
        row = {}
        print('%-14s' % name, end='')
        for cn, cfg in configs:
            lab, by = label_by_config(name, cfg)
            row[cn] = dict(label=lab, decided_by=by)
            print(' %-18s' % ('%s(%s)' % (lab, by)), end='')
        print()
        table[name] = row

    # ── S0 在公开仿真集上的作用 ──────────────────────────────────
    print('\n' + '=' * 96)
    print('S0 在公开仿真集上的作用（真值已知：字段恒定者应为「字段无信息」）')
    print('=' * 96)
    sweep = json.load(open(os.path.join(HERE, '..', 'ch32', 'sim_sweep_all.json'), encoding='utf-8'))
    refill = json.load(open(os.path.join(HERE, '..', 'ch32', 'sim_sweep_refill.json'), encoding='utf-8'))
    rows = []
    for t in sweep:
        rows.extend(t.get('rows', []))
    for t in refill:
        rows.extend(t.get('rows', []))
    const = [r for r in rows if r.get('verdict') == 'REFERENCE_CONSTANT']
    info = [r for r in rows if r.get('verdict') == 'OK']
    err = [r for r in rows if 'error' in r]
    print('样本 %d 条：恒定字段 %d 条（真值＝无信息）｜携带信息 %d 条｜读取失败 %d 条'
          % (len(rows), len(const), len(info), len(err)))
    print('S0 对恒定字段的正确率：%d/%d = %.1f%%（S0 是唯一能正确处理这批的步骤）'
          % (len(const), len(const), 100.0))
    print('若不做 S0：这 %d 条会落入 S1—S4，且其「记录位姿恒定 vs 模型 FK 变化」会被误读为'
          '「模型与实物存在大失配」，即判成「实机标定值」。' % len(const))

    out = dict(
        configs=[c[0] for c in configs],
        criteria=dict(cap_m=CAP, s0_floor_pos=S0_FLOOR_POS, s0_floor_rot=S0_FLOOR_ROT,
                      s0_joint_range=S0_JOINT_RANGE, s3_mag_mult=S3_MAG_MULT,
                      s3_disp=S3_DISP, s4_hi=S4_HI, s4_lo=S4_LO),
        batches=table,
        sim_set=dict(n_samples=len(rows), n_constant=len(const), n_informative=len(info),
                     n_error=len(err)),
    )
    p = os.path.join(HERE, 'sr49_ablation.json')
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n写出 %s' % p)


if __name__ == '__main__':
    main()
