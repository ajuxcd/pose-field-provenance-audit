#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1.9 · FK 同一性普遍性核查（多本体 × 多轨迹）＋ 逐本体常差标定。

改判背景（2026-11-03，用户确认）：1.8 发现 RoboMIND 的记录末端位姿就是官方模型的
FK 输出（位置差 4e-8 m、朝向恒定 45° 约定差）。原 1.9「检查残差是否随关节角系统漂移」
因此失效——位姿由同一套 FK 算出，零位若错也会同错，残差里不会出现漂移信号。
故 1.9 改为回答：

  Q1 同一性是否普遍成立？（跨本体、跨轨迹、跨数据集）
  Q2 每个本体的常差是什么？稳定（可一次性扣除）还是随位姿变化（不可扣除）？

常差一律用**可判定的假设对**拟合，取残差小者，不预设结论：
  位置 A：世界系平移        p_rec = p_sim + t
  位置 B：体坐标系工具偏移  p_rec = p_sim + R_sim · t_b
  朝向 C：右乘              R_rec = R_sim · C
  朝向 D：左乘              R_rec = C · R_sim

判据：扣除常差后位置 < 1e-5 m 且朝向 < 1e-3° → 同一性成立。

已知的旋转变换坑（本轮实测，务必带上）：
  · RoboMIND v1.0 的 end_effector 后 3 列是 **RPY 欧拉角（弧度）**（官方
    all_robot_h5_info_v1.2.md 明写 `[x, y, z, r, p, y]`），不可直接做差。
  · RoboDojo 的 ee_poses 后 4 列是 **wxyz 序四元数**（不是文档暗示的 xyzw）。
    按 xyzw 读会得到恰好 180° 的假偏差，按 wxyz 读则 C=I、残差 5e-4°。
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from replay_l1 import load_robomind_v1, rpy_to_matrix, quat_to_matrix, geodesic_deg  # noqa: E402

import h5py  # noqa: E402
import mujoco  # noqa: E402
import mujoco_menagerie as mm  # noqa: E402

X5_URDF = os.path.join(HERE, '..', 'assets17', 'X5A.urdf')

# 逐本体：模型、末端参照体、臂关节数。参照体必须与数据方的定义一致，
# 否则会得到「恒定但很大」的偏移（1.8 的 UR5 就是这样被判出来的）。
CFG = {
    'franka':  dict(model='franka_emika_panda',    body='hand',         n_arm=7),
    'ur5':     dict(model='universal_robots_ur5e',  body='wrist_3_link', n_arm=6),
    'piper_x': dict(model='agilex_piper',          body='link6',        n_arm=6),
    'piper':   dict(model='agilex_piper',          body='link6',        n_arm=6),
    'arx_x5':  dict(urdf=X5_URDF,                  body='link6',        n_arm=6),
}


def get_model(kind):
    c = CFG[kind]
    return mujoco.MjModel.from_xml_path(c['urdf']) if c.get('urdf') else mm.load(c['model'])


def proj_so3(M):
    """投影到最近的 SO(3)。"""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
    return R


def analyse(Ps, Rs, Precs, Rrecs):
    Ps, Rs = np.asarray(Ps), np.asarray(Rs)
    Precs, Rrecs = np.asarray(Precs), np.asarray(Rrecs)
    n = len(Ps)

    # ---- 位置 A：世界系常平移
    tA = (Precs - Ps).mean(axis=0)
    resA = np.linalg.norm(Precs - (Ps + tA), axis=1)

    # ---- 位置 B：体坐标系常偏移（线性最小二乘）
    A = Rs.reshape(-1, 3)
    b = (Precs - Ps).reshape(-1)
    tB, *_ = np.linalg.lstsq(A, b, rcond=None)
    resB = np.linalg.norm(Precs - (Ps + np.einsum('nij,j->ni', Rs, tB)), axis=1)

    # ---- 朝向 C：右乘  R_rec = R_sim · C
    C_right = proj_so3(np.einsum('nji,njk->nik', Rs, Rrecs).mean(axis=0))
    resC = geodesic_deg(np.einsum('nij,jk->nik', Rs, C_right), Rrecs)
    # ---- 朝向 D：左乘  R_rec = C · R_sim
    C_left = proj_so3(np.einsum('nij,nkj->nik', Rrecs, Rs).mean(axis=0))
    resD = geodesic_deg(np.einsum('ij,njk->nik', C_left, Rs), Rrecs)

    pos_best = 'B' if resB.max() < resA.max() else 'A'
    rot_best = 'C' if resC.max() < resD.max() else 'D'

    return {
        'n': int(n),
        'raw': {
            'pos_max_m': float(np.linalg.norm(Precs - Ps, axis=1).max()),
            'rot_max_deg': float(geodesic_deg(Rs, Rrecs).max()),
        },
        'pos': {
            'best': pos_best,
            'A_t_m': [float(x) for x in tA],
            'A_res_max_m': float(resA.max()),
            'B_tb_m': [float(x) for x in tB],
            'B_res_max_m': float(resB.max()),
            'res_after_max_m': float((resB if pos_best == 'B' else resA).max()),
        },
        'rot': {
            'best': rot_best,
            'C_matrix': [[float(x) for x in r] for r in C_right],
            'C_res_max_deg': float(resC.max()),
            'D_matrix': [[float(x) for x in r] for r in C_left],
            'D_res_max_deg': float(resD.max()),
            'res_after_max_deg': float((resC if rot_best == 'C' else resD).max()),
            'best_matrix': [[float(x) for x in r] for r in (C_right if rot_best == 'C' else C_left)],
        },
        'identical': bool((resB if pos_best == 'B' else resA).max() < 1e-5
                          and (resC if rot_best == 'C' else resD).max() < 1e-3),
    }


def fk_series(kind, joints, body, n_arm=None):
    c = CFG[kind]
    n_arm = n_arm or c['n_arm']
    m = get_model(kind)
    d = mujoco.MjData(m)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise RuntimeError(f'{kind}: 无 body {body}')
    Ps, Rs = [], []
    for t in range(len(joints)):
        d.qpos[:n_arm] = joints[t, :n_arm]
        mujoco.mj_forward(m, d)
        Ps.append(d.xpos[bid].copy())
        Rs.append(d.xmat[bid].reshape(3, 3).copy())
    return np.array(Ps), np.array(Rs)


def run_robomind():
    out = []
    root = os.path.join(HERE, '..')
    files = [('franka', os.path.join(root, 'v1probe', 'traj_0923_164719.hdf5')),
             ('ur5',    os.path.join(root, 'v1probe', 'fail_ur_stack_apples_0312_110951.hdf5'))]
    for kind, path in files:
        if not os.path.exists(path):
            print(f'[skip] {path}')
            continue
        ep = load_robomind_v1(path, 'puppet')
        Rrec = np.array([rpy_to_matrix(r)[0] for r in ep.ee_rpy])
        Ps, Rs = fk_series(kind, ep.joints, CFG[kind]['body'])
        a = analyse(Ps, Rs, ep.ee_xyz, Rrec)
        a.update(kind=kind, dataset='RoboMIND-v1', file=os.path.basename(path),
                 side='-', n_arm=CFG[kind]['n_arm'], body=CFG[kind]['body'], quat_order='RPY')
        out.append(a)
        _print(a)
    return out


def run_robodojo():
    out = []
    for path in sorted(glob.glob(os.path.join(HERE, '..', 'fk19', '*.hdf5'))):
        if os.path.getsize(path) < 1e7:
            continue
        emb = os.path.basename(path).split('__')[0]
        if emb not in CFG:
            continue
        with h5py.File(path, 'r') as f:
            for side in ('left', 'right'):
                try:
                    j = f[f'state/{side}_arm_joint_states'][()].astype(np.float64)
                    pz = f[f'state/{side}_ee_poses'][()].astype(np.float64)
                except KeyError:
                    continue
                Rrec = np.array([quat_to_matrix(q, order='wxyz')[0] for q in pz[:, 3:7]])
                Ps, Rs = fk_series(emb, j, CFG[emb]['body'])
                a = analyse(Ps, Rs, pz[:, :3], Rrec)
                a.update(kind=emb, dataset='RoboDojo-real', file=os.path.basename(path),
                         side=side, n_arm=CFG[emb]['n_arm'], body=CFG[emb]['body'],
                         quat_order='wxyz')
                out.append(a)
                _print(a)
    return out


def _print(a):
    print(f'[{a["dataset"]:14s}] {a["kind"]:8s}/{a["side"]:5s} n={a["n"]:5d} '
          f'raw: pos={a["raw"]["pos_max_m"]:.3e} m rot={a["raw"]["rot_max_deg"]:.4f}° | '
          f'pos模型{a["pos"]["best"]} → {a["pos"]["res_after_max_m"]:.3e} m | '
          f'rot模型{a["rot"]["best"]} → {a["rot"]["res_after_max_deg"]:.3e}° | '
          f'同={"Y" if a["identical"] else "N"}')


def main():
    results = run_robomind() + run_robodojo()
    out = os.path.join(HERE, 'fk_identity_matrix.json')
    json.dump(results, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n共 {len(results)} 组 → {os.path.basename(out)}')

    print('\n=== 逐本体常差（位置：世界平移 / 体偏移；朝向：右乘 C）===')
    print(f'{"本体":9s} {"侧":5s} {"N":>5s} {"原始位置":>10s} {"原始朝向":>9s} '
          f'{"常差后位置":>11s} {"常差后朝向":>11s} {"同":>2s}')
    for r in results:
        print(f'{r["kind"]:9s} {r["side"]:5s} {r["n"]:5d} '
              f'{r["raw"]["pos_max_m"]:10.3e} {r["raw"]["rot_max_deg"]:9.4f} '
              f'{r["pos"]["res_after_max_m"]:11.3e} {r["rot"]["res_after_max_deg"]:11.3e} '
              f'{"Y" if r["identical"] else "N":>2s}')
    ok = sum(bool(r['identical']) for r in results)
    print(f'\n同一性成立 {ok} / {len(results)} 组')

    # 按本体聚合常差，看跨轨迹是否稳定（这是 1.10 阈值的输入）
    print('\n=== 常差跨轨迹稳定性（同本体同侧的常差是否一致）===')
    key = {}
    for r in results:
        key.setdefault((r['kind'], r['side']), []).append(r)
    for (k, s), rs in key.items():
        if len(rs) < 2:
            continue
        tA = np.array([x['pos']['A_t_m'] for x in rs])
        tB = np.array([x['pos']['B_tb_m'] for x in rs])
        print(f'  {k:8s}/{s:5s} 组数={len(rs)}')
        print(f'    世界平移 t 跨轨迹极差 = {np.round(tA.max(0)-tA.min(0), 5)} m')
        print(f'    体偏移   t 跨轨迹极差 = {np.round(tB.max(0)-tB.min(0), 5)} m')


if __name__ == '__main__':
    main()