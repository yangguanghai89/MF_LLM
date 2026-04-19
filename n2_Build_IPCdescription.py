import pandas as pd
from tqdm import tqdm

# ---------- 1. 读取 IPC 表 ----------
ipc_df = pd.read_csv('ipc.tsv', sep='\t', header=None, names=['ipc', 'desc_cn', 'desc_en'])
ipc_map = {}
for _, row in ipc_df.iterrows():
    code = row['ipc'].strip().upper()
    desc_en = str(row['desc_en']).strip()
    ipc_map[code] = desc_en

# ---------- 2. 读取待处理文件 ----------
test_df = pd.read_csv('data/train_with_entity.tsv', sep='\t')

# ---------- 3. 定义 IPC→描述函数 ----------
def ipc_to_desc(ipc_str: str) -> str:
    if pd.isna(ipc_str):
        return ''
    parts = ipc_str.strip().split()
    desc_parts = []
    for p in parts:
        p_upper = p.upper()
        found = None
        for l in range(len(p_upper), 0, -1):
            maybe = p_upper[:l]
            if maybe in ipc_map:
                found = ipc_map[maybe]
                break
        if found is None:
            found = p
        desc_parts.append(found)
    return ' '.join(desc_parts)

# ---------- 4. 逐行处理 + 进度条 ----------
tqdm.pandas(desc='IPC描述生成')          # 对 pandas 的 apply 注入进度条
test_df['#1 IPC_Description'] = test_df['#1 IPC'].progress_apply(ipc_to_desc)
test_df['#2 IPC_Description'] = test_df['#2 IPC'].progress_apply(ipc_to_desc)

# ---------- 5. 保存 ----------
test_df.to_csv('train_with_entity_with_desc.tsv', sep='\t', index=False)
print('全部完成！结果已写入 train_with_entity_with_desc.tsv')
