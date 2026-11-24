# -*- coding: utf-8 -*-
"""样本外复现检验：用另外 3 个任务检验三条结论是否成立。

设计（2026-11-09）：
  **主集**＝原 3 任务 × 30 条/本体（表 7 的阈值、6.2 的 Δ、3.3 的 KS 均基于此，已冻结）；
  **复现集**＝另 3 个任务 × 30 条/本体（同一数据集、同一本体、不同任务）。

为什么不把两批合并：
  合并会让已冻结的 R1 阈值与全部已报数字一起变（阈值由主集 P95 标定），
  两周内改动面过大；而**样本外复现**在方法上更强——它检验的不是「同一批数据算得准不准」，
  而是「换一批同本体的任务，结论还在不在」。

判定三项：
  ① 来源判定：复现集的 L1 量级与主集同档（同源档 <1e-6 m；近似档 mm 级）；
  ② 阈值可用：复现集在**冻结阈值**下的通过率（不重新标定）；
  ③ Δ 的可分离性：复现集的 Δ 中位数、负值比例与主集同向。

用法：python3 replication_check.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CH21 = os.path.join(HERE, '..', 'ch21')

MAIN = {'piper': ['fill_pen_holder', 'insert_charger', 'put_objects_into_basket'],
        'arx_x5': ['cover_blocks', 'insert_tubes', 'make_bread']}
REP = {'piper': ['stack_and_cover_blocks', 'stack_bowls', 'stand_up_bottles'],
       'arx_x5': ['make_food', 'pack_and_pour_fruit', 'store_in_safe']}

THR = json.load(open(os.path.join(HERE, '..', 'ch110', 'l1_thresholds.json'),
                     encoding='utf-8'))['bodies']


def task_of(fn):
    return fn.split('__')[0]


def split(rows, tasks):
    return [r for r in rows if task_of(r['file']) in tasks]


def main():
    report = {}
    print(f'{"本体":<8s} {"集合":<6s} {"n":>3s} {"L1 P95 中位":>12s} '
          f'{"R1 通过":>10s} {"L2 段末中位":>11s} {"Δ 中位":>9s} {"Δ<0":>6s}')
    print('-' * 74)
    for body in ('arx_x5', 'piper'):
        # L1 文件名按 E1 脚本的命名（arx_x5 → e1_arx_180）
        tag = 'arx' if body == 'arx_x5' else body
        l1 = [r for r in json.load(open(os.path.join(CH21, f'e1_{tag}_180.json'),
                                        encoding='utf-8')) if r.get('verdict') == 'OK']
        l2tag = 'arx' if body == 'arx_x5' else body
        l2 = [r for r in json.load(open(os.path.join(HERE, f'e2_{l2tag}_180.json'),
                                        encoding='utf-8')) if r.get('verdict') == 'OK']
        m2 = {r['file']: r for r in l2}
        thr_p = THR[body]['threshold_pos_m']
        report[body] = {}
        for tag, tasks in (('主集', MAIN[body]), ('复现', REP[body])):
            a = split(l1, tasks)
            p = np.array([r['pos_after_p95_m'] for r in a]) * 1000.0     # mm
            pass_rate = np.mean([r['pos_after_p95_m'] <= thr_p for r in a])
            pairs = [(r['file'], m2[r['file']]) for r in a if r['file'] in m2]
            l2v = np.array([x['pos_end_median_m'] for _, x in pairs]) * 1000.0
            delta = np.array([x['pos_end_median_m'] - rr['pos_after_p95_m']
                              for rr, x in zip(a, [y for _, y in pairs])]) * 1000.0
            frac_neg = float(np.mean(delta < 0))
            print(f'{body if tag == "主集" else "":<8s} {tag:<6s} {len(a):>3d} '
                  f'{np.median(p):>11.3f}mm {pass_rate:>9.0%} '
                  f'{np.median(l2v):>10.2f}mm {np.median(delta):>8.2f} {frac_neg:>5.0%}')
            report[body][tag] = {
                'n': len(a),
                'l1_p95_median_mm': float(np.median(p)),
                'r1_pass_rate': float(pass_rate),
                'l2_end_median_mm': float(np.median(l2v)),
                'delta_median_mm': float(np.median(delta)),
                'delta_neg_frac': frac_neg,
                'l1_per_task_mm': {t: float(np.median([r['pos_after_p95_m'] * 1000
                                                       for r in a if task_of(r['file']) == t]))
                                   for t in tasks},
            }
        print()

    json.dump(report, open(os.path.join(HERE, 'replication_check.json'), 'w',
                           encoding='utf-8'), ensure_ascii=False, indent=1)
    print('→ replication_check.json')


if __name__ == '__main__':
    main()
