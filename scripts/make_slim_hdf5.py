#!/usr/bin/env python3
"""生成 slim 版数据集: 把每个 hdf5 里 4 路触觉 depth (tactile/*/depth, 占 ~82%) 去掉, 其余全保留。
   data/<task>/<task>_new  ->  data/<task>/<task>_slim   (hdf5/ + metadata.json + suc_map.txt)
   跳过 depth 时根本不读它, 所以只搬 ~18% 的数据, 很快。用法:
     conda activate UniVTAC
     python3 scripts/make_slim_hdf5.py [task ...]      # 不给参数=全部4个
"""
import os, sys, shutil, h5py
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = '/home/ubuntu/Documents/UniVTAC-NeoSim'
ALL = ['dual_cup_handover_place', 'dual_cup_place_stack',
       'dual_bowl_place_stack', 'dual_plate_place_stack']
TASKS = sys.argv[1:] or ALL

def is_depth(name):
    return name.startswith('tactile/') and name.endswith('/depth')

def slim_one(args):
    src, dst = args
    tmp = dst + '.tmp'
    try:
        with h5py.File(src, 'r') as s, h5py.File(tmp, 'w') as d:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    if is_depth(name):
                        return                       # 丢弃 depth (不读不写)
                    ds = d.create_dataset(name, data=obj[()], dtype=obj.dtype,
                                          compression=obj.compression)
                    for k, v in obj.attrs.items():
                        ds.attrs[k] = v
            s.visititems(visit)
            for k, v in s.attrs.items():
                d.attrs[k] = v
        os.replace(tmp, dst)
        return (os.path.basename(src), os.path.getsize(dst))
    except Exception as e:
        if os.path.exists(tmp): os.remove(tmp)
        return (os.path.basename(src), f'ERR {e}')

def main():
    jobs = []
    for t in TASKS:
        srcd = f'{REPO}/data/{t}/{t}_new'
        dstd = f'{REPO}/data/{t}/{t}_slim'
        if not os.path.isdir(f'{srcd}/hdf5'):
            print(f'skip {t}: no {srcd}/hdf5'); continue
        os.makedirs(f'{dstd}/hdf5', exist_ok=True)
        for f in ['metadata.json', 'suc_map.txt']:
            if os.path.exists(f'{srcd}/{f}'):
                shutil.copy2(f'{srcd}/{f}', f'{dstd}/{f}')
        for fn in sorted(os.listdir(f'{srcd}/hdf5')):
            if fn.endswith('.hdf5'):
                jobs.append((f'{srcd}/hdf5/{fn}', f'{dstd}/hdf5/{fn}'))
    print(f'slimming {len(jobs)} files across {len(TASKS)} tasks with 12 workers...', flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(slim_one, j) for j in jobs]
        for fu in as_completed(futs):
            name, res = fu.result(); done += 1
            if isinstance(res, str):
                print(f'  [{done}/{len(jobs)}] {name}: {res}', flush=True)
            elif done % 25 == 0 or done == len(jobs):
                print(f'  [{done}/{len(jobs)}] latest {name} -> {res/1e6:.0f}M', flush=True)
    print('DONE. sizes:')
    for t in TASKS:
        d = f'{REPO}/data/{t}/{t}_slim'
        if os.path.isdir(d):
            os.system(f'du -sh {d} {d}/hdf5 2>/dev/null')

if __name__ == '__main__':
    main()
