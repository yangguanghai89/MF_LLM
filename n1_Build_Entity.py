import sys
import os
import json
import time, subprocess
from tqdm import tqdm
import csv
import pandas as pd


# 添加父目录到路径以导入相关模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from prompt import *
from utils import load_data
from utils import load_data_new
from arguments import get_args
#from call_openai import call_with_messages
from call_qwen import  call_with_messages
from model_inference import model_load, model_inference

def normalize_ipc(ipc1: str, ipc2: str) -> str:
    """
    按规则返回「应该参考的 IPC 片段」
    规则：
    1. 完全一致 → 返回 ipc1
    2. 完全不一致（部都不同）→ 返回 ipc1
    3. 部分一致 → 从后往前找最长公共前缀（部→大类→小类）
    """
    ipc1, ipc2 = ipc1.strip(), ipc2.strip()
    if ipc1 == ipc2:
        return ipc1          # 规则1

    # 拆成部、大类、小类
    def parts(ipc):
        # A22C  →  ['A', '22', 'C']
        return [ipc[0], ipc[1:3], ipc[3]] if len(ipc) >= 4 else [ipc[0], ipc[1:3], '']
    p1, p2 = parts(ipc1), parts(ipc2)

    # 从部开始往后找最长一致前缀
    same = []
    for a, b in zip(p1, p2):
        if a == b:
            same.append(a)
        else:
            break
    if not same:            # 部都不同 → 规则2
        return ipc1
    # 拼回字符串
    return ''.join(same)    # 规则3：如只部一致返回 'A'

def generate_text(args, instruction, model=None, tokenizer=None):
    """使用指定的模型生成文本。"""
    gen = None

    if args.model_import_type == "hf" and model and tokenizer:
        gen = model_inference(args.model_type, instruction, model, tokenizer)
        if gen is None:
            print("生成失败")
            gen = "null"

    elif args.model_import_type == "api":
        gen = call_with_messages(args.api_model_name, args.api_key, instruction, 5, 2)
        if gen is None:
            print("生成失败，使用默认答案")
            gen = "null"

    return gen



def process_entity_generation(args, model=None, tokenizer=None):
    in_file = "/home/wangfei/study/dataset/sxc/V6_IPC/train.tsv"
    out_file = args.output_path

    # 读取原始数据
    df = pd.read_csv(in_file, sep='\t', dtype=str, keep_default_na=False)

    # 添加实体列（如果不存在）
    for col in ["#1 entity", "#2 entity"]:
        if col not in df.columns:
            df[col] = ""

    # 去重：只保留唯一 ID 对应的行
    unique_df = df.drop_duplicates(subset=["#1 ID", "#2 ID"]).copy()

    # 缓存：ID -> 实体
    entity_cache = {}

    prompt_tpl = gen_entity_zh if args.language == "zh" else gen_entity_en

    def _build_prompt(abs_text, ipc1, ipc2):
        ref_ipc = normalize_ipc(ipc1, ipc2)
        return prompt_tpl.format(abs=abs_text, ipc=ref_ipc)

    def _gen_entities(abs_text, ipc1, ipc2):
        prompt = _build_prompt(abs_text, ipc1, ipc2)
        raw = generate_text(args, prompt, model, tokenizer) or "null"
        return raw.replace("\n", " ")

    # 只处理唯一 ID 对应的行
    for _, row in tqdm(unique_df.iterrows(), total=len(unique_df), desc="生成实体"):
        id1 = row["#1 ID"]
        id2 = row["#2 ID"]

        # 生成实体并缓存
        if id1 not in entity_cache:
            entity_cache[id1] = _gen_entities(row["#1 abstract"], row["#1 IPC"], row["#2 IPC"])
        if id2 not in entity_cache:
            entity_cache[id2] = _gen_entities(row["#2 abstract"], row["#1 IPC"], row["#2 IPC"])

    # 映射回原始数据
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="映射实体"):
        df.at[idx, "#1 entity"] = entity_cache.get(row["#1 ID"], "")
        df.at[idx, "#2 entity"] = entity_cache.get(row["#2 ID"], "")

        # 每 1000 行保存一次
        if (idx + 1) % 1000 == 0:
            df.to_csv(out_file, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL)

    # 最终保存
    df.to_csv(out_file, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"实体生成完成，结果已保存至: {out_file}")

    uniq_file = os.path.join(os.path.dirname(out_file), "id_entity_uniq.tsv")
    pd.DataFrame.from_dict(entity_cache, orient='index', columns=['entity']) \
               .rename_axis('ID').reset_index() \
               .to_csv(uniq_file, sep='\t', index=False)
    print(f"去重后的 ID-Entity 文件已保存至: {uniq_file}")

def main():
    # 获取命令行参数
    args = get_args()

    # 检查生成类型参数
    if not hasattr(args, 'generation_type'):
        print("错误: 请使用 --generation_type 参数指定生成类型 (entity 或 ontology)")
        sys.exit(1)

    # 初始化模型
    model, tokenizer = None, None

    if args.model_import_type == "hf":
        try:
            model, tokenizer = model_load(args.hf_model_name, args.model_type)
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)

    # 根据生成类型执行不同的处理
    if args.generation_type == "entity":
        process_entity_generation(args, model, tokenizer)
    else:
        print(f"错误: 不支持的生成类型: {args.generation_type}，请使用 --generation_type entity 或 ontology")
        sys.exit(1)


if __name__ == "__main__":
    main()
