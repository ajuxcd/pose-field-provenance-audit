# Pose-Field Provenance Audit (PFPA)

Code, threshold configuration and result files accompanying the manuscript:

> **(HREAI) Field-Level Verification of Robot Manipulation Data: A Method for End-Effector Pose Field Provenance with Per-Embodiment Threshold Tiers**

Submitted to the *Chinese Journal of Mechanical Engineering*, column "Humanoid Robotics and Embodied Intelligence".

## What this repository provides

- The replay pipeline: kinematic replay (L1), open-loop physics replay in MuJoCo (L2), and the non-kinematic residual Δ = e_L2 − e_L1.
- The provenance criteria: variance pre-check (S0), residual magnitude (S2), cross-trajectory offset stability (S3) and joint-noise sensitivity (S4).
- The per-embodiment threshold calibration and the resulting configuration (`config/l1_thresholds.json`).
- The result files behind every reported number, table and figure (`results/`).

## What it does not provide

- **No data.** All inputs are public datasets (RoboMIND v1.0, RoboDojo, FMB, RH20T) and the officially released URDF/MJCF models. Point `PFPA_DATA_ROOT` at your local copy.
- **No single entry point for the four-step decision.** S0–S4 are implemented inside the analysis scripts listed below rather than as one callable CLI. A consolidated `pfpam.py` is planned; until then, `sr42_s3_layer_check.py` and `sr43_fail_side_s4.py` are the closest thing to a reference implementation of S3 and S4.

## Environment

```
python >= 3.10
pip install -r requirements.txt
```

`mujoco-menagerie` supplies the embodiment models used by the kinematic probes.

## Data layout

```bash
export PFPA_DATA_ROOT=/path/to/your/datasets
```

The scripts expect, under `$PFPA_DATA_ROOT`:

```
state_e1/                 RoboMIND v1.0 real trajectories (state .npz)
fmb/                      FMB release (insert-only episodes)
robomind_sim/             RoboMIND public simulation batches (v1.0–v1.2)
```

Every path in the code is resolved through `PFPA_DATA_ROOT`; nothing is hard-coded to a machine.

## Script ↔ paper mapping

| Script | Paper location | Purpose |
|---|---|---|
| `replay_l1.py` | Sec. 3.1, 3.3 | L1 kinematic replay, residual primitives, data loaders |
| `replay_l2.py` | Sec. 3.2, 3.4 | L2 open-loop physics replay in MuJoCo; Δ |
| `probe_fk_identity.py`, `probe_fk_identity_multi.py` | Sec. 3.1 | Forward-kinematics self-consistency of the recorded pose field |
| `sr42_s3_layer_check.py` | Sec. 6.1 | S3 decision on the L1 layer |
| `sr43_fail_side_s4.py` | Sec. 6.1 | S4 decision for the failure side (η = 0.9985) |
| `sr49_ablation.py` | Sec. 6.6, Table 9 | Ablation over the four-step chain |
| `sr50_injection.py` | Sec. 6.6, Table 10 | Injection experiment and confusion matrix |
| `sr50_public_truth.py` | Sec. 6.1, Table 11 | Known-truth check on the public simulation batches |
| `sr49_51_judge_eval.py` | Sec. 6.3 | Discrimination (AUC) and its bootstrap interval |
| `crossbody_delta.py` | Sec. 6.2, Fig. 5 | Δ magnitude and normalised shape across embodiments |
| `sr35_solver_sensitivity.py` | Sec. 6.7 | Twelve solver configurations |
| `sr37_asset_sensitivity.py` | Sec. 7.1 | Mass/inertia perturbation, 4% upper bound |
| `sr52_54_analysis.py` | Sec. 6.7 | Holdout check and convention sensitivity |
| `replication_check.py` | Sec. 6.2 | Re-run on three non-overlapping tasks |
| `judge_fmb.py` | Sec. 6.1 | Contact-side recomputation on 39 FMB insertions |
| `calib_l1_thresholds.py` | Sec. 4.1 | Threshold calibration (writes `config/l1_thresholds.json`) |
| `cluster_stats.py`, `fetch_stream.py` | Sec. 5.3, 5.1 | Statistics helpers, data fetching |

## Results

Each file in `results/` is the raw output of the corresponding script; the paper's tables and figures are rendered from them without manual editing.

## License

MIT — see `LICENSE`.
