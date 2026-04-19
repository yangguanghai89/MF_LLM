# -*- coding: utf-8 -*-
"""
专利重排与评估脚本 (修正版)
功能：
  1. 读取 test_with_entity.tsv
  2. 使用 vectors_bge_small_en_v1.5.npy 中的向量矩阵和 id_mapping_output_final.txt 映射计算余弦相似度
  3. 修正逻辑：按 #1 ID 分组，对每个主题的全部 1000 个候选进行重排
  4. 修正逻辑：在完整排序后，分别截取 Top-100, Top-500, Top-1000 进行评估
  5. 保存每组的重排结果到单独文件
  6. 从 test_qrels.txt 读取真实相关集（QRELS）
  7. 调用评估函数计算 Recall, Accuracy, MAP, PRES
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import codecs
import datetime
import os


# =============================
# 评估函数 (保持不变)
# =============================

def readQRELS(fname):
    """从文件读取 QRELS（格式：topic_id \t candidate_id）"""
    result = {}
    reader = codecs.open(filename=fname, mode='r', encoding='utf-8')
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        ss = line.split('\t')
        if len(ss) < 2:
            continue
        tid = ss[0]
        cid = ss[1]
        if tid not in result:
            result[tid] = []
        result[tid].append(cid)
    reader.close()
    return result


def computePerformanceForOnePatent(sids_ret, sids_qrel):
    """计算单个主题专利的性能"""
    nCount = 0
    map_val = 0.0
    sumRank = 0.0  # 用于计算PRES

    for i in range(len(sids_ret)):
        sid_ret = sids_ret[i]
        if sid_ret in sids_qrel:
            nCount += 1
            map_val += nCount / (i + 1)
            sumRank += i + 1

    # 召回率
    recall = float(nCount) / len(sids_qrel) if len(sids_qrel) > 0 else 0.0
    # 准确率
    accuracy = float(nCount) / len(sids_ret) if len(sids_ret) > 0 else 0.0
    # MAP
    map_val = map_val / nCount if nCount > 0 else 0.0
    # PRES
    n = len(sids_qrel)
    nMax = len(sids_ret)
    pres = 0.0
    if n * nMax != 0:
        nCollection = nMax + n
        remain = n - nCount
        # 剩余未匹配的伪排名估算
        sumRank += remain * (nCollection - (remain - 1) / 2.0)
        denominator = n * (nCollection - n)
        if denominator != 0:
            pres = 1 - (sumRank - (n * (n + 1) / 2.0)) / denominator
        else:
            pres = 0.0
        if pres < 0.0 or pres > 1.0:
            print(f'Warning: PRES out of range: {pres} for n={n}, nMax={nMax}, nCount={nCount}')

    return recall, accuracy, map_val, pres


def computePerformance(results, QRELS):
    """计算整体性能指标"""
    n = 0
    sum_recall = 0.0
    sum_accuracy = 0.0
    sum_map = 0.0
    sum_pres = 0.0

    for tid, sids_ret in results.items():
        if tid not in QRELS:
            continue
        n += 1
        sids_qrel = QRELS[tid]
        recall, accuracy, map_val, pres = computePerformanceForOnePatent(sids_ret, sids_qrel)
        sum_recall += recall
        sum_accuracy += accuracy
        sum_map += map_val
        sum_pres += pres

    if n == 0:
        print("Error: No topic patents found in QRELS.")
        return 0.0, 0.0, 0.0, 0.0

    return sum_recall / n, sum_accuracy / n, sum_map / n, sum_pres / n


def evalute(ret_results, QRELS, tid='All Patents'):
    """主评估入口函数"""
    Recall, Accuracy, MAP, PRES = computePerformance(ret_results, QRELS)

    datetime_object = datetime.datetime.now()
    outstr = 'Current Time:{}\n'.format(str(datetime_object))
    outstr += 'adding patent:{}\n'.format(tid)
    outstr += 'Average Recall of {} topic patents:{}\n'.format(len(ret_results), str(Recall))
    outstr += 'Average Accuracy of {} topic patents:{}\n'.format(len(ret_results), str(Accuracy))
    outstr += 'Average MAP of {} topic patents:{}\n'.format(len(ret_results), str(MAP))
    outstr += 'Average PRES of {} topic patents:{}\n\n'.format(len(ret_results), str(PRES))
    print(outstr)

    # 写入 result.txt
    path = 'result.txt'
    writer = codecs.open(filename=path, mode='a+', encoding='utf-8')
    writer.write(outstr)
    writer.close()

    return Recall, Accuracy, MAP, PRES


# =============================
# 主程序开始
# =============================

if __name__ == "__main__":
    print("🚀 开始专利重排与评估任务...")

    # 创建保存结果的目录
    output_dir = "reordered_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"📁 创建输出目录: {output_dir}")

    # Step 1: 加载向量矩阵和 ID 映射
    print("📁 加载向量矩阵: vectors_bge_small_en_v1_5_title_abstract_ipcDescription.npy")
    try:
        vectors_matrix = np.load('vectors_bge_small_en_v1_5_title_abstract_ipcDescription.npy', allow_pickle=True)
        print(f"✅ 向量矩阵加载成功，形状: {vectors_matrix.shape}")
    except Exception as e:
        print(f"❌ 加载向量矩阵失败: {e}")
        exit(1)

    print("📁 加载 ID 映射文件: id_mapping_output_final.txt")
    try:
        df_mapping = pd.read_csv('id_mapping_output_final.txt', sep='\t', dtype=str)
        df_mapping['id'] = pd.to_numeric(df_mapping['id'], errors='coerce')

        id_to_index = {}
        for _, row in df_mapping.iterrows():
            combined_id = row['combined_ID']
            numeric_id = row['id']
            index = int(numeric_id) - 1
            if index < vectors_matrix.shape[0]:
                id_to_index[combined_id] = index
            else:
                print(f"⚠️ 数值 ID {numeric_id} 超出向量矩阵范围，跳过 {combined_id}")
        print(f"✅ 成功构建 {len(id_to_index)} 个专利ID到向量索引的映射。")
    except Exception as e:
        print(f"❌ 加载或解析 ID 映射文件失败: {e}")
        exit(1)

    # Step 2: 加载测试数据
    print("📁 加载测试数据: test_with_entity_with_desc.tsv")
    try:
        df = pd.read_csv('test_with_entity_with_desc.tsv', sep='\t', dtype=str)
        print(f"✅ 成功加载 {len(df)} 条数据。")
    except Exception as e:
        print(f"❌ 加载TSV文件失败: {e}")
        exit(1)

    # Step 3: 从 test_qrels.txt 读取 QRELS
    print("📁 从 test_qrels.txt 加载 QRELS...")
    try:
        QRELS = readQRELS('test_qrels.txt')
        print(f"✅ 成功加载 QRELS，共 {len(QRELS)} 个主题专利的相关集合。")
    except Exception as e:
        print(f"❌ 加载 test_qrels.txt 失败: {e}")
        exit(1)

    # =============================
    # Step 4: 修正逻辑 - 按主题专利 ID 分组进行全量重排
    # =============================
    print("🔍 开始按主题专利 ID 分组计算余弦相似度并重排...")

    # 用于存储所有主题的完整排序结果
    full_ranked_results = {}

    # 按 #1 ID (主题专利) 分组
    grouped = df.groupby('#1 ID')

    for topic_id, group_df in grouped:
        print(f"🔄 正在处理主题专利: {topic_id} (候选数量: {len(group_df)})")

        # 确保候选数量正确
        if len(group_df) != 1000:
            print(f"⚠️ 警告: 主题专利 {topic_id} 的候选数量为 {len(group_df)}，不是 1000。")

        # 获取主题专利向量
        if topic_id not in id_to_index:
            print(f"⚠️ 主题专利 {topic_id} 无向量，跳过")
            continue
        vec_t = vectors_matrix[id_to_index[topic_id]].reshape(1, -1)  # (1, 384)

        # 收集候选专利ID和向量
        candidate_ids = []
        valid_candidate_ids = []
        vec_c_list = []
        candidate_qualities = []

        for _, row in group_df.iterrows():
            cid = row['#2 ID']
            quality = pd.to_numeric(row['Quality'], errors='coerce')
            if pd.isna(quality):
                quality = 0
            quality = int(quality)

            candidate_ids.append(cid)
            candidate_qualities.append(quality)

            if cid in id_to_index:
                vec_c = vectors_matrix[id_to_index[cid]]
                vec_c_list.append(vec_c)
                valid_candidate_ids.append(cid)
            # 如果没有向量，不加入计算列表，后续会排到最后

        if len(vec_c_list) == 0:
            print(f"⚠️ 主题专利 {topic_id} 无有效候选向量，跳过")
            continue

        # 计算相似度
        vec_c_matrix = np.array(vec_c_list)
        similarities = cosine_similarity(vec_t, vec_c_matrix)[0]

        # 创建映射
        cid_to_sim = {}
        for i, cid in enumerate(valid_candidate_ids):
            cid_to_sim[cid] = similarities[i]

        # 按原始顺序排列所有候选（包括无向量的），无向量的相似度设为-999（排在最后）
        all_candidates_with_info = []
        for i, cid in enumerate(candidate_ids):
            quality = candidate_qualities[i]
            sim = cid_to_sim.get(cid, -999)
            all_candidates_with_info.append((cid, sim, quality))

        # 按相似度降序排序
        sorted_candidates = sorted(all_candidates_with_info, key=lambda x: x[1], reverse=True)

        # 保存排序结果到单独文件
        output_file = os.path.join(output_dir, f"topic_{topic_id}_reordered.txt")
        with codecs.open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"主题专利: {topic_id}\n")
            f.write(f"重排结果 (按相似度降序排列):\n")
            f.write("排名\t候选专利ID\t相似度\tQuality\n")
            f.write("-" * 60 + "\n")
            for i, (cid, sim, quality) in enumerate(sorted_candidates):
                quality_str = "相关" if quality == 1 else "不相关"
                f.write(f"{i + 1}\t{cid}\t{sim:.6f}\t{quality}({quality_str})\n")

        # 保存排序结果用于评估 (完整的1000条)
        sorted_cids = [cid for cid, _, _ in sorted_candidates]
        full_ranked_results[topic_id] = sorted_cids

    print(f"✅ 全量重排完成，共处理 {len(full_ranked_results)} 个主题专利")

    # =============================
    # Step 5: 修正逻辑 - 在完整排序基础上截断并评估
    # =============================
    WINDOW_SIZES = [100, 500, 1000]
    all_metrics = {}

    print("📊 开始多窗口评估...")

    for ws in WINDOW_SIZES:
        print(f"\n🔍 正在评估 Top-{ws}...")

        # 关键操作：从完整排序列表中切片
        truncated_results = {}
        for tid, sorted_cids in full_ranked_results.items():
            truncated_results[tid] = sorted_cids[:ws]

        # 调用评估函数
        Recall, Accuracy, MAP, PRES = evalute(truncated_results, QRELS, tid=f'Patent Similarity Top-{ws}')
        all_metrics[ws] = (Recall, Accuracy, MAP, PRES)

    # Step 6: 打印汇总结果
    print("\n" + "=" * 60)
    print("📊 基线最终汇总结果:")
    print("=" * 60)
    print(f"{'Window':<10} {'Recall':<12} {'Accuracy':<12} {'MAP':<12} {'PRES':<12}")
    print("-" * 60)
    for ws in WINDOW_SIZES:
        Recall, Accuracy, MAP, PRES = all_metrics[ws]
        print(f"Top@{ws:<6} {Recall:<12.4f} {Accuracy:<12.4f} {MAP:<12.4f} {PRES:<12.4f}")

    print(f"\n✅ 所有结果已追加保存至 'result.txt'")
    print(f"📁 每个主题专利的详细重排结果已保存到 '{output_dir}' 目录")
