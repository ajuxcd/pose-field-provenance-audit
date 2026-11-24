#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用取数器：分片 tar.gz 的**可断点续传**下载 + 流式解包 + slim。

## 为什么单独写一个

已有的 `fetch_e1.py` 是「一次下完再解」，中断就得从头来；而本次要取的
双臂成功集（6 任务 × 24–59 GiB，合计 194 GiB）与孪生集（9 任务 × 约 45 GiB）
都远超单次可完成的体量，必须能断点续传。

## 三个设计要点

1. **续传**：每个任务缓存为 `<task>.tar.gz.part`，每次用 `curl -C -` 追加。
   已下够（能解出目标条数）就跳过，不重复下。
2. **增量推进**：按 `--step-gib` 逐步加量（下→解→数条数→不够再下），
   而不是一次下满。**这是省流量的关键**——多数任务的首段就够 30 条。
3. **流式解包**：`tarfile.open(mode='r|gz')` 读截断的 gzip 会在断点抛异常，
   但**断点前的成员已全部读出**，属预期行为，捕获后照常返回。

## 只取 part-aa

RoboMIND 的任务包是 `split` 切的，`part-aa` 是**完整的前缀**，故只需它。
若目标条数在 part-aa 内取不完（理论上不会，part-aa 至少 10 GiB），
再追加 part-ab。
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import subprocess
import sys
import tarfile
import urllib.parse

BASE = os.environ.get('ROBOMIND_BASE', 'data')
API = ('https://www.modelscope.cn/api/v1/datasets/X-Humanoid/RoboMIND'
       '/repo?Revision=master&FilePath=')
TREE = ('https://www.modelscope.cn/api/v1/datasets/X-Humanoid/RoboMIND'
        '/repo/tree?Revision=master&Root=')
GIB = 2 ** 30


def tree(path):
    r = subprocess.run(['curl', '-sL', '--max-time', '45',
                        TREE + urllib.parse.quote(path, safe='')],
                       capture_output=True, text=True)
    try:
        return json.loads(r.stdout)['Data']['Files']
    except Exception:
        return []


def shards_of(repo_dir, task):
    """返回**指定任务**的分片名清单（已排序）。

    ⚠ 任务名只是**文件名前缀**，不是子目录——早期版本把
    `{repo_dir}/{task}` 当路径去 tree，必然返回空（本轮实际踩到，
    报「无分片」）。正确做法是列父目录再按前缀过滤。
    """
    fs = tree(repo_dir)
    names = [x['Name'] for x in fs if x['Type'] == 'blob']
    pre = task + '.tar.gz'
    return sorted(n for n in names if n.startswith(pre))


def url_of(repo_dir, shard):
    """⚠ repo_dir 是**父目录**，shard 已是完整文件名（内含任务名）。
    早期版本传 `{repo_dir}/{task}`，导致路径里任务名重复两次而 404（本轮实际踩到）。
    """
    return API + urllib.parse.quote(f'{repo_dir}/{shard}', safe='')


def download_to(repo_dir, shard, dst, want_bytes):
    """把 shard 的 [已有长度, want_bytes) 区间追加到 dst —— 即**断点续传**。

    用 Range 精确请求到 want_bytes，而不是下整片（整片 10 GiB，远超所需）。
    重复调用幂等：已有长度 ≥ want_bytes 时直接返回。
    """
    have = os.path.getsize(dst) if os.path.exists(dst) else 0
    if have >= want_bytes:
        return have
    rng = f'{have}-{want_bytes - 1}'
    # **低速中止**：只给 --max-time 是不够的——服务器挂住连接时 curl 会一直等满一小时。
    # 加 --speed-time/--speed-limit：低于 1 KB/s 持续 60 s 即放弃本轮，
    # 已下的部分仍会被追加进 .part，下一轮从新长度续传（本轮实际踩到：孪生首个任务
    # 卡在 1.14 GiB 三十余分钟不动）。
    subprocess.run(['curl', '-sL', '--max-time', '3600',
                    '--speed-time', '60', '--speed-limit', '1024',
                    '-r', rng, '-o', dst + '.chunk', url_of(repo_dir, shard)])
    if os.path.exists(dst + '.chunk'):
        with open(dst, 'ab') as fo, open(dst + '.chunk', 'rb') as fi:
            while True:
                b = fi.read(1 << 22)
                if not b:
                    break
                fo.write(b)
        os.remove(dst + '.chunk')
    return os.path.getsize(dst) if os.path.exists(dst) else 0


def hdf5_members(part_path):
    """流式列出 tar 内 hdf5 成员；截断包的末尾成员会损坏，忽略其后的异常。"""
    out = []
    try:
        with tarfile.open(part_path, mode='r|gz') as tf:
            for m in tf:
                if m.isfile() and m.name.endswith('.hdf5'):
                    out.append(m)
    except (tarfile.TarError, EOFError, OSError):
        pass                       # 截断点，预期
    return out


def episode_id(member_name):
    """从 `task/success_episodes/train/<EP>/data/trajectory.hdf5` 取 `<EP>`。

    ⚠ **不能取 basename**：所有成员都叫 `trajectory.hdf5`，取 basename 会让
    19 个成员写到同一个文件名上，只有第 1 个真正落盘（本轮实际踩到，
    显示「落盘 1 条」而实际应 19 条）。
    """
    parts = [x for x in member_name.split('/') if x]
    for i, x in enumerate(parts):
        if x == 'data' and i >= 1:
            return parts[i - 1]
    return parts[-2] if len(parts) >= 2 else parts[0]


def iter_and_slim(part_path, out_dir, task, per_task, manifest):
    """**单次扫描** tar：边扫边 slim，凑够 per_task 就停。

    早前版本是「先列成员、再逐个 extract_episode」，两者都用 `r|gz` 流式模式，
    于是每个成员都要把 2 GiB 从头解压一遍——19 个成员 = 19 次全量扫描，
    慢到不可用，且多次调用 extract_episode 大量返回 None。改为单次流式。
    """
    got = 0
    try:
        with tarfile.open(part_path, mode='r|gz') as tf:
            for m in tf:
                if not (m.isfile() and m.name.endswith('.hdf5')):
                    continue
                eid = episode_id(m.name)
                dst = os.path.join(out_dir, f'{task}__{eid}.npz')
                if os.path.exists(dst):
                    got += 1
                    if got >= per_task:
                        break
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                raw = f.read()
                n = slim_one(raw, dst)
                if n:
                    got += 1
                    manifest.append({'task': task, 'member': m.name,
                                     'frames': n, 'npz': os.path.basename(dst)})
                if got >= per_task:
                    break
    except (tarfile.TarError, EOFError, OSError):
        pass                                   # 截断点，预期
    return got


STATE_KEYS_ARM = ('arm_joint_position', 'joint_position')


def slim_one(raw, dst_npz):
    """从 hdf5 字节流取状态量并压成 npz。

    ⚠ **必须捕获所有异常**：截断包末尾的成员是不完整的 hdf5，打开时会抛
    `bad heap free list` 之类的运行时错误。不捕获会让整批任务中断
    （本轮实际踩到，第 19 个成员在 2 GiB 边界处损坏）。
    返回 0 表示该成员不可用，由调用方跳过。
    """
    import h5py
    import numpy as np
    try:
        return _slim_one_inner(raw, dst_npz, h5py, np)
    except Exception:
        return 0


def _slim_one_inner(raw, dst_npz, h5py, np):
    """**按顶层组自动识别**，不写死 `puppet`/`master`。

    ⚠ 首版只认 `puppet/` 与 `master/`（真机侧），而**仿真侧的状态挂在 `franka/` 下**，
    于是孪生数据下了 3 GiB、落盘 0 条——`slim_one` 静默返回 0，调用方只看到「已解 0 条」。
    改为遍历顶层组，任一组含 `joint_position` 即按该组前缀保存。
    """
    with h5py.File(io.BytesIO(raw), 'r') as h:
        for src in list(h.keys()):
            if not isinstance(h[src], h5py.Group):
                continue
            arm = None
            used = None
            for k in STATE_KEYS_ARM:
                key = f'{src}/{k}'
                if key in h:
                    arm = h[key][()]
                    used = k
                    break
            if arm is None:
                continue
            save = {f'{src}__{used}': arm}
            for extra in ('hand_joint_position', 'joint_effort',
                          'joint_velocity', 'end_effector',
                          'end_effector_left'):
                ke = f'{src}/{extra}'
                if ke in h:
                    save[f'{src}__{extra}'] = h[ke][()]
            np.savez_compressed(dst_npz, **save)
            return int(arm.shape[0])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-dir', required=True, help='如 benchmark1_1_compressed/h5_franka_fr3_dual')
    ap.add_argument('--tasks', default='', help='逗号分隔的任务名前缀；空=全部')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tmp', default=None)
    ap.add_argument('--per-task', type=int, default=30)
    ap.add_argument('--step-gib', type=float, default=4.0, help='每轮加量')
    ap.add_argument('--max-gib', type=float, default=15.0, help='单任务上限')
    ap.add_argument('--plan', action='store_true')
    a = ap.parse_args()

    tmp = a.tmp or os.path.join(BASE, 'work', 'stream_tmp')
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)

    # 任务清单
    fs = tree(a.repo_dir)
    tasks = sorted({x['Name'].split('.tar.gz')[0] for x in fs
                    if x['Type'] == 'blob' and '.tar.gz' in x['Name']})
    if a.tasks:
        want = [t.strip() for t in a.tasks.split(',') if t.strip()]
        tasks = [t for t in tasks if any(t.startswith(w) for w in want)]

    print(f'{len(tasks)} 个任务｜目标 {a.per_task} 条/任务｜每轮 +{a.step_gib} GiB｜上限 {a.max_gib} GiB')
    if a.plan:
        for t in tasks:
            print(f'  {t}')
        return 0

    manifest = []
    for ti, task in enumerate(tasks, 1):
        shards = shards_of(a.repo_dir, task)
        if not shards:
            print(f'[{ti}/{len(tasks)}] {task}: 无分片，跳过')
            continue
        part = os.path.join(tmp, task + '.tar.gz.part')
        done = [x for x in glob.glob(os.path.join(a.out, task + '__*.npz'))]
        if len(done) >= a.per_task:
            print(f'[{ti}/{len(tasks)}] {task}: 已有 {len(done)} 条，跳过')
            continue
        print(f'\n[{ti}/{len(tasks)}] {task}  ({len(shards)} 分片)')

        got = 0
        for step in range(1, int(a.max_gib / a.step_gib) + 2):
            want = int(min(step * a.step_gib, a.max_gib) * GIB)
            si = min(step - 1, len(shards) - 1)
            download_to(a.repo_dir, shards[si], part, want)
            have = os.path.getsize(part) / GIB
            print(f'    轮 {step}: {have:.2f} GiB', end=' ')
            got = iter_and_slim(part, a.out, task, a.per_task, manifest)
            print(f'｜已解 {got} 条')
            if got >= a.per_task:
                print(f'    达标 {got} 条')
                break
        if got < a.per_task:
            print(f'    ⚠ 只取到 {got} 条（上限 {a.max_gib} GiB 内，末尾成员可能落在截断点）')

    mp = os.path.join(a.out, 'manifest.json')
    old = json.load(open(mp, encoding='utf-8')) if os.path.exists(mp) else []
    json.dump(old + manifest, open(mp, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'\n落盘 {len(manifest)} 条 → {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())