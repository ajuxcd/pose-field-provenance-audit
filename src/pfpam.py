#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PFPAM — Pose-Field Provenance Audit Method.

Reference implementation of Algorithm 1 of the manuscript

    (HREAI) Field-Level Verification of Robot Manipulation Data:
    A Method for End-Effector Pose Field Provenance
    with Per-Embodiment Threshold Tiers

The method decides, for a batch of recorded trajectories of one embodiment,
whether the recorded end-effector pose field is (A) the output of the model's
own forward kinematics, (B) a calibrated real-robot value, (C) a field that
carries no information, or (D) doubtful. It needs no ground truth: only the
recorded joint angles, the recorded poses and the official model.

Decision order (fixed by the manuscript): S0 > S2 > S3 > S4.

    S0  zero-variance pre-check   pose range at the numerical floor while the
                                  joints move  -> field carries no information
    S2  magnitude                 r_L1 at the same-source level -> model output
    S3  offset stability          large and directionally stable Kabsch offset
                                  -> calibrated value
    S4  noise sensitivity         eta(1e-6 rad) > 5 -> model output,
                                  eta < 2 -> calibrated, otherwise doubtful

Reporting convention (matches Section 6.1 of the manuscript): the L1
representative value is the 95th percentile of the frame-wise residual after
Kabsch removal of the constant offset, taken over the whole trajectory, and
then the median over trajectories. Note that the S1 line of the Algorithm box
refers to Eq. (3); in the implementation the whole-trajectory P95 is used, as
in Section 3.3, and the end-of-segment convention is reserved for the
non-kinematic residual Delta.

Usage
-----
    python pfpam.py --body franka --batch /path/to/npz --json out.json
    python pfpam.py --selftest            # no dataset needed

The batch directory must contain one ``.npz`` per trajectory in a layout that
``replay_l2.load_episode`` understands (RoboMIND v1.0 or RoboDojo).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import e1_residuals as E

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {
    # thresholds of the same-source tier (Section 4.1)
    'same_source_pos_m': 1.0e-6,
    'same_source_rot_deg': 1.0e-3,
    # S0 numerical floor
    's0_pos_floor_m': 1.0e-9,
    's0_rot_floor_deg': 1.0e-6,
    's0_arm_range_rad': 0.1,
    # S3 two bounds
    's3_mag_mult': 10.0,
    's3_disp_max': 0.50,
    # S4
    's4_sigma_rad': 1.0e-6,
    's4_eta_model': 5.0,
    's4_eta_calib': 2.0,
    # reproducible noise
    'seed': 20261111,
}

LABELS = {
    'fk': 'model-forward-kinematics',
    'calib': 'calibrated',
    'none': 'no-information',
    'doubt': 'doubtful',
}


# --------------------------------------------------------------------- S0
def s0_check(joints, poses, cfg):
    """Zero-variance pre-check. Returns (triggered, diagnostics)."""
    pose_range = float(np.abs(poses.max(0) - poses.min(0)).max())
    arm_range = float(np.abs(joints.max(0) - joints.min(0)).max())
    triggered = (pose_range <= cfg['s0_pos_floor_m']) and (arm_range > cfg['s0_arm_range_rad'])
    return triggered, {'pose_range_m': pose_range, 'arm_range_rad': arm_range}


# --------------------------------------------------------------------- S1
def l1_residual(fk_pos, rec_pos):
    """Frame-wise L1 position residual after Kabsch removal of the offset.

    Returns (p95_m, p95_frame_residuals, kabsch_translation).
    """
    R, t = E.kabsch(fk_pos, rec_pos)
    d = np.linalg.norm((R @ fk_pos.T).T + t - rec_pos, axis=1)
    return float(np.percentile(d, 95)), d, np.asarray(t, dtype=float)


def batch_baseline(trajs, body):
    """Per-trajectory P95 and Kabsch offsets for the unperturbed batch."""
    p95s, offsets = [], []
    for q, rec in trajs:
        fk = E.fk_positions(body, q)
        if fk.shape != rec.shape:
            continue
        p95, _, t = l1_residual(fk, rec)
        p95s.append(p95)
        offsets.append(t)
    if not p95s:
        return None
    offsets = np.asarray(offsets)
    mags = np.linalg.norm(offsets, axis=1)
    med = float(np.median(mags))
    sd = np.array([float(np.std(offsets[:, k])) for k in range(3)])
    disp = float(sd.max() / max(med, 1e-12))
    return {
        'n': len(p95s),
        'median_p95_m': float(np.median(p95s)),
        'p95_m_all': [float(x) for x in p95s],
        'offset_median_m': med,
        'offset_dispersion': disp,
    }


# --------------------------------------------------------------------- S4
def eta_noise(trajs, body, cfg, base_median):
    """Median P95 under sigma=1e-6 rad joint noise, divided by the baseline."""
    if not base_median or base_median <= 0:
        return None
    vals = []
    for q, rec in trajs:
        rng = np.random.default_rng(cfg['seed'])
        qq = q + rng.normal(0.0, cfg['s4_sigma_rad'], size=q.shape)
        fk = E.fk_positions(body, qq)
        if fk.shape != rec.shape:
            continue
        p95, _, _ = l1_residual(fk, rec)
        vals.append(p95)
    if not vals:
        return None
    return float(np.median(vals) / base_median)


# --------------------------------------------------------------------- decide
def decide(trajs, body, cfg):
    """Run Algorithm 1 on a batch. Returns a dict with label and all diagnostics."""
    if not trajs:
        return {'label': LABELS['doubt'], 'reason': 'no usable trajectory'}

    # S0 is a property of the batch: constant pose field while joints move.
    j0, p0 = trajs[0]
    trig, s0diag = s0_check(j0, p0, cfg)
    if trig:
        return {'label': LABELS['none'], 'reason': 'S0 zero-variance',
                's0': s0diag}

    base = batch_baseline(trajs, body)
    if base is None:
        return {'label': LABELS['doubt'], 'reason': 'forward kinematics failed'}

    out = {'s0': s0diag, 's1': base}

    # S2 magnitude
    if base['median_p95_m'] <= cfg['same_source_pos_m']:
        out.update(label=LABELS['fk'], reason='S2 same-source magnitude')
        return out

    # S3 offset stability: both bounds must hold
    c1 = base['offset_median_m'] > cfg['s3_mag_mult'] * cfg['same_source_pos_m']
    c2 = base['offset_dispersion'] < cfg['s3_disp_max']
    out['s3'] = {'magnitude_ok': bool(c1), 'dispersion_ok': bool(c2)}
    if c1 and c2:
        out.update(label=LABELS['calib'], reason='S3 stable constant offset')
        return out

    # S4 noise sensitivity
    eta = eta_noise(trajs, body, cfg, base['median_p95_m'])
    out['s4'] = {'eta': eta, 'sigma_rad': cfg['s4_sigma_rad']}
    if eta is None:
        out.update(label=LABELS['doubt'], reason='S4 not computable')
    elif eta > cfg['s4_eta_model']:
        out.update(label=LABELS['fk'], reason='S4 eta > %.1f' % cfg['s4_eta_model'])
    elif eta < cfg['s4_eta_calib']:
        out.update(label=LABELS['calib'], reason='S4 eta < %.1f' % cfg['s4_eta_calib'])
    else:
        out.update(label=LABELS['doubt'], reason='S4 in undecidable band')
    return out


# --------------------------------------------------------------------- IO
def load_batch(batch_dir, body, limit=0):
    """Read one .npz per trajectory through replay_l2.load_episode."""
    import replay_l2 as R2L
    files = sorted(f for f in os.listdir(batch_dir)
                   if f.endswith('.npz') and not f.startswith('._'))
    if limit:
        files = files[:limit]
    trajs = []
    for f in files:
        ep = R2L.load_episode(os.path.join(batch_dir, f), body, 'left')
        if ep is None:
            continue
        q, rec = ep['joints'], ep['ee_xyz']
        if q is None or rec is None or len(q) < 2:
            continue
        trajs.append((np.asarray(q, float), np.asarray(rec, float)))
    return trajs


# --------------------------------------------------------------------- selftest
def selftest():
    """Build three synthetic batches with known ground truth and check the chain.

    A: pose := forward kinematics of q                      -> must be 'fk'
    B: pose := FK(q) + a constant rigid offset              -> must be 'calib'
    C: pose := a constant, joints moving                    -> must be 'none'
    """
    body = 'franka'
    rng = np.random.default_rng(7)
    T, N, NJ = 120, 3, 7

    def make_q():
        base = rng.uniform(-0.6, 0.6, size=NJ)
        return base + np.cumsum(rng.normal(0, 0.004, size=(T, NJ)), axis=0)

    cases = {}
    # A: FK output
    trajs_a = []
    for _ in range(N):
        q = make_q()
        trajs_a.append((q, E.fk_positions(body, q)))
    cases['A_fk'] = (trajs_a, LABELS['fk'])

    # B: FK plus a constant offset *and* per-trajectory measurement noise.
    # A pure constant offset is absorbed by Kabsch and must still be labelled
    # 'fk' (class B1 of Table 10) — that is the correct outcome, not a failure.
    # A calibrated field differs by carrying a residual above the same-source
    # tier while the offset stays directionally stable across trajectories.
    trajs_b = []
    off = np.array([0.080, -0.035, 0.052])
    for _ in range(N):
        q = make_q()
        noise = rng.normal(0.0, 8.0e-4, size=(T, 3))
        trajs_b.append((q, E.fk_positions(body, q) + off + noise))
    cases['B_calib'] = (trajs_b, LABELS['calib'])

    # C: constant field
    trajs_c = []
    for _ in range(N):
        q = make_q()
        trajs_c.append((q, np.tile(np.array([0.45, 0.0, 0.30]), (T, 1))))
    cases['C_none'] = (trajs_c, LABELS['none'])

    # D: independent measurement with a directionally unstable offset.
    # Six offsets along +-x, +-y, +-z: the magnitude is large (0.1 m) but the
    # direction is maximally dispersed, so S3 must fail on the dispersion bound
    # and S4 must decide. Corresponds to the failure side of Section 6.1.
    trajs_d = []
    dirs = [np.array(v, float) for v in
            ([0.1, 0, 0], [-0.1, 0, 0], [0, 0.1, 0], [0, -0.1, 0], [0, 0, 0.1], [0, 0, -0.1])]
    for v in dirs:
        q = make_q()
        noise = rng.normal(0.0, 1.0e-3, size=(T, 3))
        trajs_d.append((q, E.fk_positions(body, q) + v + noise))
    cases['D_s4_calib'] = (trajs_d, LABELS['calib'])

    ok = True
    for name, (trajs, expect) in cases.items():
        res = decide(trajs, body, DEFAULTS)
        good = res['label'] == expect
        ok &= good
        print('%-9s expect=%-32s got=%-32s %s | %s'
              % (name, expect, res['label'], 'OK' if good else 'FAIL', res.get('reason', '')))
    print('\nSELFTEST', 'PASSED' if ok else 'FAILED')
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description='PFPAM — reference implementation of Algorithm 1')
    ap.add_argument('--body', default='franka', help='embodiment key: franka | piper | arx_x5 | ur5')
    ap.add_argument('--batch', help='directory holding one .npz per trajectory')
    ap.add_argument('--limit', type=int, default=0, help='use only the first N trajectories')
    ap.add_argument('--config', help='JSON with threshold overrides')
    ap.add_argument('--json', help='write the full decision record to this path')
    ap.add_argument('--selftest', action='store_true', help='run the synthetic checks; needs no dataset')
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    cfg = dict(DEFAULTS)
    if a.config and os.path.exists(a.config):
        cfg.update(json.load(open(a.config, encoding='utf-8')))

    if not a.batch:
        ap.error('--batch is required unless --selftest is given')

    trajs = load_batch(a.batch, a.body, a.limit)
    res = decide(trajs, a.body, cfg)
    res['body'] = a.body
    res['batch'] = os.path.abspath(a.batch)

    print('PFPAM decision for %s (%d trajectories)' % (a.body, len(trajs)))
    print('  label  :', res['label'])
    print('  reason :', res.get('reason'))
    s1 = res.get('s1') or {}
    if s1:
        print('  r_L1   : median P95 = %.3e m over %d trajectories' % (s1['median_p95_m'], s1['n']))
        print('  offset : median |t| = %.3e m, dispersion = %.4f'
              % (s1['offset_median_m'], s1['offset_dispersion']))
    if 's4' in res and res['s4'].get('eta') is not None:
        print('  eta    : %.4f at sigma = %.1e rad' % (res['s4']['eta'], res['s4']['sigma_rad']))

    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1, ensure_ascii=False, default=float)
        print('  wrote  :', a.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
