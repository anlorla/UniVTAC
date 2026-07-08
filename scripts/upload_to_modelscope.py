#!/usr/bin/env python3
"""把本会话采集的 4 个数据集(各 100 条)上传到 ModelScope 数据集 nybchen/NeoSim。

每个任务作为仓库里的一个子文件夹(path_in_repo=<task>), 只传 hdf5/ + metadata.json + suc_map.txt
(排除 scene/ 工作区、video/、日志)。全程【直连不走代理】。断点续传: 重跑即可继续(use_cache=True)。

运行(必须在装了 modelscope 的 base 环境):
    conda activate base
    MS_TOKEN=ms-你的有效令牌  python3 scripts/upload_to_modelscope.py

可选: 只传某几个任务(重试用):
    MS_TOKEN=ms-xxx python3 scripts/upload_to_modelscope.py dual_plate_place_stack dual_bowl_place_stack

拿令牌: modelscope.cn -> 头像 -> 访问令牌(Access Token), 复制 ms- 开头那串。
下载(别人用):
    from modelscope.msdatasets import MsDataset
    ds = MsDataset.load('nybchen/NeoSim', subset_name='dual_cup_place_stack')
"""
import os, sys, time

# ---- 强制直连: 清掉所有代理, 全部主机 no_proxy ----
for k in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    os.environ.pop(k, None)
os.environ['no_proxy'] = '*'
os.environ['NO_PROXY'] = '*'

REPO = '/home/ubuntu/Documents/UniVTAC-NeoSim'
DATASET = 'nybchen/NeoSim'
# 数据版本后缀: 默认 _slim(去掉 depth 的瘦身版); 可用 DATA_SUFFIX=_new 传原始全量版
SUFFIX = os.environ.get('DATA_SUFFIX', '_slim')
ALL_TASKS = [
    'dual_cup_handover_place',
    'dual_cup_place_stack',
    'dual_bowl_place_stack',
    'dual_plate_place_stack',
]
# 只传数据本体, 排除工作区/视频/日志
IGNORE = ['scene/*', 'scene/**', 'video/*', 'video/**', '*.log']

TASKS = sys.argv[1:] or ALL_TASKS

TOKEN = os.environ.get('MS_TOKEN', '').strip()
if not TOKEN:
    sys.exit('ERROR: 请用环境变量提供令牌:  MS_TOKEN=ms-xxxx python3 scripts/upload_to_modelscope.py')

from modelscope.hub.api import HubApi

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

api = HubApi()
try:
    api.login(TOKEN)
    log('登录成功 (直连)')
except Exception as e:
    sys.exit(f'ERROR 登录失败, 令牌无效或过期: {e}')

# 确保数据集仓库存在(不存在则创建为私有; 已存在则忽略报错)
try:
    api.create_repo(DATASET, repo_type='dataset', visibility='private',
                    chinese_name='NeoSim dual-arm tactile demos')
    log(f'已创建数据集仓库 {DATASET} (private)')
except Exception as e:
    log(f'仓库已存在或创建跳过: {str(e)[:80]}')

ok, failed = [], []
for t in TASKS:
    folder = f'{REPO}/data/{t}/{t}{SUFFIX}'
    if not os.path.isdir(f'{folder}/hdf5'):
        log(f'跳过 {t}: 找不到 {folder}/hdf5'); failed.append(t); continue
    n = len([f for f in os.listdir(f'{folder}/hdf5') if f.endswith('.hdf5')])
    log(f'==== 上传 {t}  ({n} 个 hdf5) -> {DATASET}/{t} ====')
    try:
        api.upload_folder(
            repo_id=DATASET,
            repo_type='dataset',
            folder_path=folder,
            path_in_repo=t,
            ignore_patterns=IGNORE,
            commit_message=f'add {t} (100 demos, no-weld, fixed stacking)',
            disable_tqdm=False,
        )
        log(f'==== {t} 上传完成 ====')
        ok.append(t)
    except Exception as e:
        log(f'!!!! {t} 上传失败: {e}')
        failed.append(t)

log(f'完成: 成功={ok}  失败={failed}')
if failed:
    log('失败的可单独重试(会断点续传): '
        f'MS_TOKEN=... python3 scripts/upload_to_modelscope.py {" ".join(failed)}')
    sys.exit(1)
