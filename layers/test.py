# 测试代码（单独运行，验证attn是否非None）
import torch
from SelfAttention_Family import AttentionLayer, FullAttention, QueryProjection, KeyProjection, ValueProjection

# 初始化模型
d_model = 512
n_heads = 8
attention = FullAttention(mask_flag=True, output_attention=True)
attention_layer = AttentionLayer(
    attention=attention,
    d_model=d_model,
    n_heads=n_heads,
    alpha=2.0,
    top_k_ratio=0.2
)

# 生成测试数据
B = 32
L = 12  # query长度
S = 24  # key/value长度
queries = torch.randn(B, L, d_model)
keys = torch.randn(B, S, d_model)
values = torch.randn(B, S, d_model)

# 前向传播
out, attn = attention_layer(queries, keys, values)
print(f"最终attn是否非None: {attn is not None}")