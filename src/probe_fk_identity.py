#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L1 残差核算：RoboMIND v1.0 的记录末端位姿是否「就是」官方模型的 FK 输出。

结论性发现（2026-11-03）：
  1. 位置：以 puppet 关节角驱动 MuJoCo 的 Franka 模型，`hand` 体位置与记录
     `end_effector` 位置的最大偏差为 **2.97e-08 m**（浮点精度级）。即记录位置
     由同一套 FK 算出，**不含任何外部测量成分**。
  2. 朝向：两者相差一个**恒定的 45° 绕 z 轴旋转**（逐帧标准差 1.2e-08）。
     在 sim 侧右乘 Rz(45°) 后，偏差降为 **0.0000°**（最大 0.0000°）。
     故这是坐标约定差，不是物理差。

对本文的含义（重要）：
  → **L1 不是「与真机测量比对的残差」，而是「模型自洽性检查」。**
    它的数值必然接近 0，只能用于发现数据与模型不匹配（例如 URDF 版本不符、
    关节顺序错、单位错），**不能作为「模型有多准」的证据**。
  → 因此 Δ = e_L2 − e_L1 中的 e_L1 实质上是数值基线而非真实误差项；
    Δ 的解释必须相应改写，否则会被审稿人认定为循环论证。
"""
import json
import sys
import numpy as np

sys.path.insert(0, '.')
from replay_l1 import load_robomind_v1, rpy_to_matrix, geodesic_deg  # noqa: E402

import mujoco  # noqa: E402
import mujoco_menagerie as mm  # noqa: E402

RZ45 = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4), 0.0],
                 [np.sin(np.pi / 4), np.cos(np.pi / 4), 0.0],
                 [0.0, 0.0, 1.0]])


def probe(h5, model_name, body, n_joints, stride=1):
    ep = load_robomind_v1(h5, 'puppet')
    m = mm.load(model_name)
    d = mujoco.MjData(m)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)

    idx = list(range(n_joints))
    pos, rot_raw, rot_fix = [], [], []
    for t in range(0, ep.n_frames, stride):
        d.qpos[idx] = ep.joints[t, :n_joints]
        mujoco.mj_forward(m, d)
        p = d.xpos[bid].copy()
        R = d.xmat[bid].reshape(3, 3).copy()
        pos.append(np.linalg.norm(p - ep.ee_xyz[t]))
        Rrec = rpy_to_matrix(ep.ee_rpy[t])[0]
        rot_raw.append(geodesic_deg(R[None], Rrec[None])[0])
        rot_fix.append(geodesic_deg((R @ RZ45)[None], Rrec[None])[0])

    pos = np.array(pos); rot_raw = np.array(rot_raw); rot_fix = np.array(rot_fix)
    return {
        'file': h5.split('/')[-1],
        'model': model_name,
        'body': body,
        'n_frames': ep.n_frames,
        'n_sampled': len(pos),
        'pos_abs_max_m': float(pos.max()),
        'pos_rms_m': float(np.sqrt((pos ** 2).mean())),
        'rot_raw_mean_deg': float(rot_raw.mean()),
        'rot_raw_max_deg': float(rot_raw.max()),
        'rot_after_45deg_fix_mean_deg': float(rot_fix.mean()),
        'rot_after_45deg_fix_max_deg': float(rot_fix.max()),
    }


def main():
    out = []
    # 成功样本（Franka，7 关节）
    out.append(probe('../v1probe/traj_0923_164719.hdf5', 'franka_emika_panda', 'hand', 7))
    print(json.dumps(out[-1], ensure_ascii=False, indent=1))

    # 失败样本（UR5，6 关节）——用 UR5e 模型，验证 schema 分支
    try:
        r = probe('../v1probe/fail_ur_stack_apples_0312_110951.hdf5',
                  'universal_robots_ur5e', 'wrist_3_link', 6)
        out.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=1))
    except Exception as e:
        print('失败样本探测失败:', type(e).__name__, e)

    json.dump(out, open('l1_fk_identity.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n已写出 l1_fk_identity.json')


if __name__ == '__main__':
    main()