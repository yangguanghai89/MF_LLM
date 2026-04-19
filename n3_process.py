import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, LayerNorm
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# 设置 Hugging Face 镜像（加速下载）
# -------------------------------
# 第一步：读取原始数据，合并并去重文本
# -------------------------------
print("正在读取并处理原始数据...")
df = pd.read_csv('test_with_entity_with_desc.tsv', sep='\t')
print("原始数据行数：", len(df))
print("#2 ID 唯一值数量：", df['#2 ID'].nunique())
# 合并标题和摘要，处理缺失值（英文文本）
# df['text_1'] = df['#1 title'].fillna('') + ' ' + df['#1 entity'].fillna('')
# df['text_2'] = df['#2 title'].fillna('') + ' ' + df['#2 entity'].fillna('')
# df['text_1'] = df['#1 entity']
# df['text_2'] = df['#2 entity']
df['text_1'] = (
    df['#1 title'].fillna('') + ' ' +
    df['#1 abstract'].fillna('') + ' ' +
    df['#1 IPC_Description'].fillna('')
    # df['#1 entity'].fillna('')
).str.strip()
df['text_2'] = (
    df['#2 title'].fillna('') + ' ' +
    df['#2 abstract'].fillna('') + ' ' +
    df['#2 IPC_Description'].fillna('')
    # df['#2 entity'].fillna('')
).str.strip()
combined_text = pd.concat([df['text_1'], df['text_2']], ignore_index=True)
combined_ID = pd.concat([df['#1 ID'], df['#2 ID']], ignore_index=True)
new_df = pd.DataFrame({'combined_ID': combined_ID,
                       'combined_text': combined_text})
print(" ID 唯一值数量：", new_df['combined_ID'].nunique())
# 按  ID 分组，合并同一 ID 的所有文本，并去重
grouped = new_df.groupby('combined_ID')['combined_text'].agg(
    lambda x: ' '.join(x.dropna().unique())
).reset_index()
print(" grouped的长度：",len(grouped))
# 为每个唯一 ID 分配一个从 1 开始的整数 id
grouped['id'] = range(1, len(grouped) + 1)

# 调整列顺序
grouped = grouped[['id', 'combined_ID', 'combined_text']]

# 保存 ID 映射文件
output_mapping_file = 'id_mapping_output2.txt'
grouped.to_csv(output_mapping_file, sep='\t', index=False, header=True)
print(f"✅ ID 映射文件已保存至: {output_mapping_file}")

# -------------------------------
# 第二步：加载 BGE 英文模型并生成向量
# -------------------------------



# print("\n正在加载 BGE 英文 Sentence-BERT 模型 (bge-small-en-v1.5)...")
#
# # ✅ 使用专为英文优化的 BGE 模型（比 all-MiniLM-L6-v2 更好）
# print("CUDA 可用:", torch.cuda.is_available())
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# model = SentenceTransformer('./bge-small-en-v1.5')
# model = model.to(device)  # 显式移动到 GPU
# print("✅ 模型加载完成，并已移至设备:", device)
# print("✅ 模型加载完成！")
#
# # 提取文本列表
# sentences = grouped['combined_text'].tolist()
# print(f"共 {len(sentences)} 条英文文本，正在生成嵌入向量...")
#
# # 生成向量（BGE 推荐归一化，便于后续计算余弦相似度）
# vectors = model.encode(
#     sentences,
#     batch_size=8,
#     show_progress_bar=True,
#     convert_to_numpy=True,
#     normalize_embeddings=True,  # BGE 推荐
#     device='cuda'
# )
# print("✅ 向量生成完成！")

# -------------------------------
# 第二步：加载 BGE 模型 + 引入层次化提示与自注意力模块
# -------------------------------

print("\n正在加载 BGE 英文 Sentence-BERT 模型 (bge-small-en-v1.5) 并构建层次化提示模块...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("CUDA 可用:", torch.cuda.is_available())

# 加载 BGE 模型
model = SentenceTransformer('./bge-small-en-v1.5')
model = model.to(device)
tokenizer = model.tokenizer  # 用于获取 attention mask

# 定义轻量自注意力模块（单层 Transformer Encoder）


class HierarchicalPromptEnhancer(nn.Module):
    def __init__(self, hidden_dim=384, nhead=4, dim_feedforward=768, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 域级提示：通过 mean pooling 得到，无需参数
        # 任务级提示：将域提示拼接到每个 token（隐式调制）
        # 自注意力层
        self.transformer_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.norm = LayerNorm(hidden_dim)

    def forward(self, token_embeddings, attention_mask):
        # token_embeddings: [B, L, D]
        # attention_mask: [B, L]
        attention_mask = attention_mask.float()

        # Step 1: 域级提示 (Domain Prompt) —— 全局平均池化
        # 忽略 padding
        masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
        domain_prompt = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, D]

        # Step 2: 任务级提示 = token + domain_prompt（广播拼接后投影回原维度）
        # 这里我们不拼接，而是将 domain_prompt 作为 bias 或 condition
        # 更高效做法：直接将 domain_prompt 加到每个 token 上（类似 prefix tuning）
        enhanced_tokens = token_embeddings + domain_prompt.unsqueeze(1)  # [B, L, D]

        # Step 3: 自注意力机制（模拟 PMN 调制）
        # 扩展 attention mask for transformer
        src_key_padding_mask = ~attention_mask.bool()  # Transformer expects True for padding
        attended = self.transformer_layer(
            enhanced_tokens,
            src_key_padding_mask=src_key_padding_mask
        )  # [B, L, D]

        # Step 4: 最终池化（可用 mean 或 [CLS]，这里用 mean 保持一致性）
        final_masked = attended * attention_mask.unsqueeze(-1)
        final_vector = final_masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, D]

        return final_vector


# 初始化增强模块
enhancer = HierarchicalPromptEnhancer(hidden_dim=384).to(device)
enhancer.eval()  # 不训练，仅推理（你也可选择微调）

# 提取文本
sentences = grouped['combined_text'].tolist()
print(f"共 {len(sentences)} 条英文文本，正在生成带层次化提示的嵌入向量...")

all_vectors = []
batch_size = 8

# 计算总批次数
num_batches = (len(sentences) + batch_size - 1) // batch_size

for i in tqdm(range(0, len(sentences), batch_size), total=num_batches, desc="生成层次化提示向量"):
    batch_sents = sentences[i:i + batch_size]

    with torch.no_grad():
        # Step 1: Get token embeddings as list of tensors
        token_emb_list = model.encode(
            batch_sents,
            batch_size=len(batch_sents),
            show_progress_bar=False,  # 关闭内部进度条，避免嵌套
            convert_to_tensor=True,
            output_value='token_embeddings',
            device=device
        )

        # Step 2: Pad to same length
        token_embs_padded = pad_sequence(token_emb_list, batch_first=True, padding_value=0.0)

        # Step 3: Create attention mask from actual lengths
        lengths = [t.size(0) for t in token_emb_list]
        max_len = token_embs_padded.size(1)
        attn_mask = torch.zeros(len(lengths), max_len, dtype=torch.long, device=device)
        for idx, seq_len in enumerate(lengths):
            attn_mask[idx, :seq_len] = 1

        # Step 4: Apply enhancer
        enhanced_vec = enhancer(token_embs_padded, attn_mask)
        enhanced_vec = torch.nn.functional.normalize(enhanced_vec, p=2, dim=1)
        all_vectors.append(enhanced_vec.cpu().numpy())


# -------------------------------
# 第三步：保存结果
# -------------------------------
vectors = np.vstack(all_vectors)
np.save('vectors_bge_small_en_v1_5_title_abstract_ipcDescription.npy', vectors)
grouped[['id', 'combined_ID']].to_csv('id_mapping_output_final.txt', sep='\t', index=False, header=True)
print("✅ 向量和 ID 映射已保存！")
