# import torch
# import torch.nn as nn
# import numpy as np
# from math import sqrt
# from utils.masking import TriangularCausalMask, ProbMask
# from reformer_pytorch import LSHSelfAttention
# from einops import rearrange, repeat


# class DSAttention(nn.Module):
#     '''De-stationary Attention'''

#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(DSAttention, self).__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / sqrt(E)

#         tau = 1.0 if tau is None else tau.unsqueeze(
#             1).unsqueeze(1)  # B x 1 x 1 x 1
#         delta = 0.0 if delta is None else delta.unsqueeze(
#             1).unsqueeze(1)  # B x 1 x 1 x S

#         # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
#         scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

#         if self.mask_flag:
#             if attn_mask is None:
#                 attn_mask = TriangularCausalMask(B, L, device=queries.device)

#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)

#         if self.output_attention:
#             return V.contiguous(), A
#         else:
#             return V.contiguous(), None


# class FullAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(FullAttention, self).__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / sqrt(E)

#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)

#         if self.mask_flag:
#             if attn_mask is None:
#                 attn_mask = TriangularCausalMask(B, L, device=queries.device)

#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)

#         if self.output_attention:
#             return V.contiguous(), A
#         else:
#             return V.contiguous(), None


# class ProbAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(ProbAttention, self).__init__()
#         self.factor = factor
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

#     def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
#         # Q [B, H, L, D]
#         B, H, L_K, E = K.shape
#         _, _, L_Q, _ = Q.shape

#         # calculate the sampled Q_K
#         K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
#         # real U = U_part(factor*ln(L_k))*L_q
#         index_sample = torch.randint(L_K, (L_Q, sample_k))
#         K_sample = K_expand[:, :, torch.arange(
#             L_Q).unsqueeze(1), index_sample, :]
#         Q_K_sample = torch.matmul(
#             Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

#         # find the Top_k query with sparisty measurement
#         M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
#         M_top = M.topk(n_top, sorted=False)[1]

#         # use the reduced Q to calculate Q_K
#         Q_reduce = Q[torch.arange(B)[:, None, None],
#                    torch.arange(H)[None, :, None],
#                    M_top, :]  # factor*ln(L_q)
#         Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

#         return Q_K, M_top

#     def _get_initial_context(self, V, L_Q):
#         B, H, L_V, D = V.shape
#         if not self.mask_flag:
#             # V_sum = V.sum(dim=-2)
#             V_sum = V.mean(dim=-2)
#             contex = V_sum.unsqueeze(-2).expand(B, H,
#                                                 L_Q, V_sum.shape[-1]).clone()
#         else:  # use mask
#             # requires that L_Q == L_V, i.e. for self-attention only
#             assert (L_Q == L_V)
#             contex = V.cumsum(dim=-2)
#         return contex

#     def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
#         B, H, L_V, D = V.shape

#         if self.mask_flag:
#             attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

#         context_in[torch.arange(B)[:, None, None],
#         torch.arange(H)[None, :, None],
#         index, :] = torch.matmul(attn, V).type_as(context_in)
#         if self.output_attention:
#             attns = (torch.ones([B, H, L_V, L_V]) /
#                      L_V).type_as(attn).to(attn.device)
#             attns[torch.arange(B)[:, None, None], torch.arange(H)[
#                                                   None, :, None], index, :] = attn
#             return context_in, attns
#         else:
#             return context_in, None

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L_Q, H, D = queries.shape
#         _, L_K, _, _ = keys.shape

#         queries = queries.transpose(2, 1)
#         keys = keys.transpose(2, 1)
#         values = values.transpose(2, 1)

#         U_part = self.factor * \
#                  np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
#         u = self.factor * \
#             np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

#         U_part = U_part if U_part < L_K else L_K
#         u = u if u < L_Q else L_Q

#         scores_top, index = self._prob_QK(
#             queries, keys, sample_k=U_part, n_top=u)

#         # add scale factor
#         scale = self.scale or 1. / sqrt(D)
#         if scale is not None:
#             scores_top = scores_top * scale
#         # get the context
#         context = self._get_initial_context(values, L_Q)
#         # update the context with selected top_k queries
#         context, attn = self._update_context(
#             context, values, scores_top, index, L_Q, attn_mask)

#         return context.contiguous(), attn


# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None):
#         super(AttentionLayer, self).__init__()

#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)

#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads

#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)

#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask,
#             tau=tau,
#             delta=delta
#         )
#         out = out.view(B, L, -1)

#         return self.out_projection(out), attn


# class ReformerLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None, causal=False, bucket_size=4, n_hashes=4):
#         super().__init__()
#         self.bucket_size = bucket_size
#         self.attn = LSHSelfAttention(
#             dim=d_model,
#             heads=n_heads,
#             bucket_size=bucket_size,
#             n_hashes=n_hashes,
#             causal=causal
#         )

#     def fit_length(self, queries):
#         # inside reformer: assert N % (bucket_size * 2) == 0
#         B, N, C = queries.shape
#         if N % (self.bucket_size * 2) == 0:
#             return queries
#         else:
#             # fill the time series
#             fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
#             return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

#     def forward(self, queries, keys, values, attn_mask, tau, delta):
#         # in Reformer: defalut queries=keys
#         B, N, C = queries.shape
#         queries = self.attn(self.fit_length(queries))[:, :N, :]
#         return queries, None


# class TwoStageAttentionLayer(nn.Module):
#     '''
#     The Two Stage Attention (TSA) Layer
#     input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
#     '''

#     def __init__(self, configs,
#                  seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
#         super(TwoStageAttentionLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                                            output_attention=False), d_model, n_heads)
#         self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                                        output_attention=False), d_model, n_heads)
#         self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                                          output_attention=False), d_model, n_heads)
#         self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

#         self.dropout = nn.Dropout(dropout)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.norm4 = nn.LayerNorm(d_model)

#         self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
#                                   nn.GELU(),
#                                   nn.Linear(d_ff, d_model))
#         self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
#                                   nn.GELU(),
#                                   nn.Linear(d_ff, d_model))

#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         # Cross Time Stage: Directly apply MSA to each dimension
#         batch = x.shape[0]
#         time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
#         time_enc, attn = self.time_attention(
#             time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
#         )
#         dim_in = time_in + self.dropout(time_enc)
#         dim_in = self.norm1(dim_in)
#         dim_in = dim_in + self.dropout(self.MLP1(dim_in))
#         dim_in = self.norm2(dim_in)

#         # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
#         dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
#         batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
#         dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
#         dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
#         dim_enc = dim_send + self.dropout(dim_receive)
#         dim_enc = self.norm3(dim_enc)
#         dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
#         dim_enc = self.norm4(dim_enc)

#         final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

#         return final_out




# import torch
# import torch.nn as nn
# import numpy as np
# from math import sqrt
# from utils.masking import TriangularCausalMask, ProbMask
# from reformer_pytorch import LSHSelfAttention
# from einops import rearrange, repeat


# # 新增：自定义键投影函数，支持注意力权重加权梯度
# import torch
# import torch.nn as nn
# import numpy as np
# from math import sqrt
# from utils.masking import TriangularCausalMask, ProbMask
# from reformer_pytorch import LSHSelfAttention
# from einops import rearrange, repeat


# # 新增：自定义键投影函数，支持注意力权重加权梯度
# # 功能：通过自定义autograd函数，实现键投影的前向计算和带注意力权重的反向梯度计算
# class KeyProjectionFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, attn):
#         """
#         前向传播：计算键的线性投影
#         参数：
#             ctx: 上下文对象，用于保存反向传播所需变量
#             input: 输入特征，形状为[B, S, d_model]（B：批量大小，S：key的时间步长度，d_model：模型维度）
#             weight: 投影权重，形状为[d_keys*n_heads, d_model]（键投影的权重矩阵）
#             bias: 偏置项，形状为[d_keys*n_heads]（可选，键投影的偏置）
#             attn: 注意力矩阵，形状为[B, H, L, S]（H：注意力头数，L：query的时间步长度）
#         返回：
#             output: 投影后的键特征，形状为[B, S, d_keys*n_heads]
#         """
#         # 保存反向传播所需的变量
#         ctx.save_for_backward(input, weight, bias, attn)
#         # 计算线性投影：input * weight^T（广播后矩阵乘法, outpout形状为[B, S, d_keys*n_heads]）
#         output = torch.matmul(input, weight.t())
#         # 若有偏置，添加偏置项
#         if bias is not None:
#             output += bias
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         反向传播：计算输入、权重、偏置的梯度，并用注意力权重加权梯度
#         参数：
#             ctx: 保存的前向传播变量
#             grad_output: 上游传来的梯度，形状为[B, S, d_keys*n_heads]
#         返回：
#             grad_input: 计算后输入的梯度，形状为[B, S, d_model]
#             grad_weight: 权重的梯度，形状为[d_keys*n_heads, d_model]
#             grad_bias: 偏置的梯度，形状为[d_keys*n_heads]
#             None: attn无梯度（不需要计算）
#         """
#         # 恢复前向传播保存的变量
#         input, weight, bias, attn = ctx.saved_tensors
#         # 初始化梯度变量（默认None，若不需要梯度则保持None）
#         grad_input = grad_weight = grad_bias = None

#         # 计算时间步注意力权重（在batch和head维度取平均）
#         if attn is not None:
#             # attn形状: [B, H, L, S]，其中L是query的时间步长度, S是key的时间步长度
#             B, H, L, S = attn.shape
#             # 计算每个时间步的平均注意力权重：在batch、head、query维度取平均，time_weights形状为[S]
#             # 含义：每个key时间步不同维度被各个query和head关注的平均程度
#             time_weights = attn.mean(dim=[0, 1, 2])  
#             # 扩展到与输入匹配的形状 [B, S, 1]：为每个样本的每个时间步分配对应的权重
#             time_weights = time_weights.unsqueeze(0).unsqueeze(-1).repeat(input.size(0), 1, 1)
#             # 应用时间权重到梯度：注意力高的时间步，其梯度被放大（增强关键信息的梯度影响）
#             weighted_grad_output = grad_output * time_weights
#         else:
#             # 若没有注意力权重，直接使用原始梯度
#             weighted_grad_output = grad_output

#         # 计算输入梯度（若需要）
#         if ctx.needs_input_grad[0]:
#             # input梯度 = 加权后的输出梯度 * weight（矩阵乘法）
#             grad_input = torch.matmul(weighted_grad_output, weight)

#         # 计算权重梯度（若需要）
#         if ctx.needs_input_grad[1]:
#             # weight梯度 = (加权后的输出梯度^T * input) 在batch维度取平均
#             # 形状：[d_keys*n_heads, d_model]
#             grad_weight = torch.matmul(weighted_grad_output.transpose(1, 2), input).mean(dim=0)

#         # 计算偏置梯度（若需要且存在偏置）
#         if bias is not None and ctx.needs_input_grad[2]:
#             # bias梯度 = 加权后的输出梯度在batch和时间步维度求和
#             grad_bias = weighted_grad_output.sum(dim=[0, 1])

#         return grad_input, grad_weight, grad_bias, None


# # 新增：自定义键投影层
# # 功能：封装键投影的参数（权重和偏置），并存储注意力权重用于反向传播
# class KeyProjection(nn.Module):
#     def __init__(self, in_features, out_features, bias=False):
#         """
#         参数：
#             in_features: 输入特征维度（d_model）
#             out_features: 输出特征维度（d_keys * n_heads）
#             bias: 是否使用偏置
#         """
#         super().__init__()
#         # 初始化权重参数：[out_features, in_features]
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         if bias:
#             # 初始化偏置参数：[out_features]
#             self.bias = nn.Parameter(torch.Tensor(out_features))
#         else:
#             # 不使用偏置时，注册为None
#             self.register_parameter('bias', None)
#         self.attn = None  # 用于存储当前批次的注意力权重（反向传播时使用）
#         self.reset_parameters()  # 初始化参数

#     def reset_parameters(self):
#         """初始化权重和偏置（遵循PyTorch线性层默认初始化方式）"""
#         # 权重使用kaiming均匀分布初始化
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             # 偏置根据输入特征维度初始化
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             nn.init.uniform_(self.bias, -bound, bound)

#     def set_attention(self, attn):
#         """设置当前批次的注意力权重（供反向传播使用）
#         参数：
#             attn: 注意力矩阵，形状为[B, H, L, S]
#         """
#         self.attn = attn

#     def forward(self, input):
#         """前向传播：调用自定义的KeyProjectionFunction计算键投影
#         参数：
#             input: 输入特征，形状为[B, S, d_model]
#         返回：
#             投影后的键特征，形状为[B, S, d_keys*n_heads]
#         """
#         return KeyProjectionFunction.apply(input, self.weight, self.bias, self.attn)


# # 以下为原有注意力类（DSAttention/FullAttention等），此处仅保留注释框架
# class DSAttention(nn.Module):
#     '''De-stationary Attention：含去平稳化因子的注意力机制'''
#     # ...（原有代码逻辑不变）


# class FullAttention(nn.Module):
#     '''Full Attention：标准全注意力机制'''
#     # ...（原有代码逻辑不变）


# # 修改AttentionLayer，使用自定义KeyProjection
# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None):
#         """
#         参数：
#             attention: 注意力机制（如FullAttention/DSAttention）
#             d_model: 模型维度
#             n_heads: 注意力头数
#             d_keys: 每个头的key维度（默认d_model//n_heads）
#             d_values: 每个头的value维度（默认d_model//n_heads）
#         """
#         super(AttentionLayer, self).__init__()

#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)

#         self.inner_attention = attention  # 内部注意力机制
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # query投影（标准线性层）
#         # 使用自定义的KeyProjection替代默认Linear：支持注意力加权梯度
#         self.key_projection = KeyProjection(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)  # value投影（标准线性层）
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)  # 输出投影（标准线性层）
#         self.n_heads = n_heads  # 注意力头数

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         """
#         前向传播：计算注意力输出，并将注意力权重传递给key_projection
#         参数：
#             queries: 查询特征，形状[B, L, d_model]
#             keys: 键特征，形状[B, S, d_model]
#             values: 值特征，形状[B, S, d_model]
#             attn_mask: 注意力掩码（用于遮挡无效位置）
#             tau/delta: 去平稳化参数（DSAttention使用）
#         返回：
#             out: 注意力层输出，形状[B, L, d_model]
#             attn: 注意力矩阵，形状[B, H, L, S]（可选，若output_attention=True）
#         """
#         B, L, _ = queries.shape  # B：批量大小，L：query时间步长度
#         _, S, _ = keys.shape     # S：key时间步长度
#         H = self.n_heads         # H：注意力头数

#         # 投影并拆分多头：[B, L, d_keys*n_heads] -> [B, L, H, d_keys]
#         queries = self.query_projection(queries).view(B, L, H, -1)
#         # 用自定义KeyProjection投影key并拆分多头
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         # 投影并拆分多头：[B, S, d_values*n_heads] -> [B, S, H, d_values]
#         values = self.value_projection(values).view(B, S, H, -1)

#         # 计算注意力并获取注意力矩阵（A）
#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask,
#             tau=tau,
#             delta=delta
#         )
        
#         # 将注意力矩阵传递给key_projection，供反向传播时加权梯度使用
#         self.key_projection.set_attention(attn)

#         # 合并多头并投影输出：[B, L, H*d_values] -> [B, L, d_model]
#         out = out.view(B, L, -1)
#         return self.out_projection(out), attn


# # 其余类（ProbAttention/ReformerLayer/TwoStageAttentionLayer）保持原有逻辑不变
# # ...（原有代码）


# # 其余类（ProbAttention、ReformerLayer、TwoStageAttentionLayer）保持不变
# class ProbAttention(nn.Module):
#     # 保持不变
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(ProbAttention, self).__init__()
#         self.factor = factor
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

#     def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
#         # 保持不变
#         B, H, L_K, E = K.shape
#         _, _, L_Q, _ = Q.shape

#         K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
#         index_sample = torch.randint(L_K, (L_Q, sample_k))
#         K_sample = K_expand[:, :, torch.arange(
#             L_Q).unsqueeze(1), index_sample, :]
#         Q_K_sample = torch.matmul(
#             Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

#         M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
#         M_top = M.topk(n_top, sorted=False)[1]

#         Q_reduce = Q[torch.arange(B)[:, None, None],
#                    torch.arange(H)[None, :, None],
#                    M_top, :]
#         Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

#         return Q_K, M_top

#     def _get_initial_context(self, V, L_Q):
#         # 保持不变
#         B, H, L_V, D = V.shape
#         if not self.mask_flag:
#             V_sum = V.mean(dim=-2)
#             contex = V_sum.unsqueeze(-2).expand(B, H,
#                                                 L_Q, V_sum.shape[-1]).clone()
#         else:
#             assert (L_Q == L_V)
#             contex = V.cumsum(dim=-2)
#         return contex

#     def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
#         # 保持不变
#         B, H, L_V, D = V.shape

#         if self.mask_flag:
#             attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         attn = torch.softmax(scores, dim=-1)

#         context_in[torch.arange(B)[:, None, None],
#         torch.arange(H)[None, :, None],
#         index, :] = torch.matmul(attn, V).type_as(context_in)
#         if self.output_attention:
#             attns = (torch.ones([B, H, L_V, L_V]) /
#                      L_V).type_as(attn).to(attn.device)
#             attns[torch.arange(B)[:, None, None], torch.arange(H)[
#                                                   None, :, None], index, :] = attn
#             return context_in, attns
#         else:
#             return context_in, None

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         # 保持不变
#         B, L_Q, H, D = queries.shape
#         _, L_K, _, _ = keys.shape

#         queries = queries.transpose(2, 1)
#         keys = keys.transpose(2, 1)
#         values = values.transpose(2, 1)

#         U_part = self.factor * \
#                  np.ceil(np.log(L_K)).astype('int').item()
#         u = self.factor * \
#             np.ceil(np.log(L_Q)).astype('int').item()

#         U_part = U_part if U_part < L_K else L_K
#         u = u if u < L_Q else L_Q

#         scores_top, index = self._prob_QK(
#             queries, keys, sample_k=U_part, n_top=u)

#         scale = self.scale or 1. / sqrt(D)
#         if scale is not None:
#             scores_top = scores_top * scale
#         context = self._get_initial_context(values, L_Q)
#         context, attn = self._update_context(
#             context, values, scores_top, index, L_Q, attn_mask)

#         return context.contiguous(), attn


# class ReformerLayer(nn.Module):
#     # 保持不变
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None, causal=False, bucket_size=4, n_hashes=4):
#         super().__init__()
#         self.bucket_size = bucket_size
#         self.attn = LSHSelfAttention(
#             dim=d_model,
#             heads=n_heads,
#             bucket_size=bucket_size,
#             n_hashes=n_hashes,
#             causal=causal
#         )

#     def fit_length(self, queries):
#         B, N, C = queries.shape
#         if N % (self.bucket_size * 2) == 0:
#             return queries
#         else:
#             fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
#             return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

#     def forward(self, queries, keys, values, attn_mask, tau, delta):
#         B, N, C = queries.shape
#         queries = self.attn(self.fit_length(queries))[:, :N, :]
#         return queries, None


# class TwoStageAttentionLayer(nn.Module):
#     # 保持不变
#     def __init__(self, configs,
#                  seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
#         super(TwoStageAttentionLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                                            output_attention=False), d_model, n_heads)
#         self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                                        output_attention=False), d_model, n_heads)
#         self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                                          output_attention=False), d_model, n_heads)
#         self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

#         self.dropout = nn.Dropout(dropout)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.norm4 = nn.LayerNorm(d_model)

#         self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
#                                   nn.GELU(),
#                                   nn.Linear(d_ff, d_model))
#         self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
#                                   nn.GELU(),
#                                   nn.Linear(d_ff, d_model))

#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         batch = x.shape[0]
#         time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
#         time_enc, attn = self.time_attention(
#             time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
#         )
#         dim_in = time_in + self.dropout(time_enc)
#         dim_in = self.norm1(dim_in)
#         dim_in = dim_in + self.dropout(self.MLP1(dim_in))
#         dim_in = self.norm2(dim_in)

#         dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
#         batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
#         dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
#         dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
#         dim_enc = dim_send + self.dropout(dim_receive)
#         dim_enc = self.norm3(dim_enc)
#         dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
#         dim_enc = self.norm4(dim_enc)

#         final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

#         return final_out







# import sys
# import os
# import torch
# import torch.nn as nn
# import numpy as np
# from math import sqrt
# from utils.masking import TriangularCausalMask, ProbMask
# from reformer_pytorch import LSHSelfAttention
# from einops import rearrange, repeat


# # --------------------------
# # 核心：跨文件夹导入visualization.py（自动处理路径）
# # --------------------------
# # 获取当前文件（SelfAttention_Family.py）的绝对路径
# current_file_path = os.path.abspath(__file__)
# # 获取layers文件夹路径（当前文件的父目录）
# layers_dir = os.path.dirname(current_file_path)
# # 获取项目根目录（layers文件夹的父目录）
# project_root = os.path.dirname(layers_dir)
# # 将项目根目录添加到Python搜索路径（确保能找到utils/visualization.py）
# if project_root not in sys.path:
#     sys.path.append(project_root)

# # 跨文件夹导入可视化函数（根据实际visualization.py路径调整导入语句）
# try:
#     from models.visualization import visualize_backward_attn, global_backward_attn_without_weight
# except ImportError:
#     # 若visualization.py在单独的 visualization/ 下：
#     from models.visualization import visualize_backward_attn, global_backward_attn_without_weight


# try:
#     from models.visualization import visualize_forward_attn, global_forward_attn_without_weight
# except ImportError:
#     # 若visualization.py在单独的 visualization/ 下：
#     from models.visualization import visualize_forward_attn, global_forward_attn_without_weight


# # # 自定义键投影函数：在单个batch和head内用注意力权重加权梯度
# # class KeyProjectionFunction(torch.autograd.Function):
# #     @staticmethod
# #     def forward(ctx, input, weight, bias, attn):
# #         """
# #         前向传播：计算键的线性投影
# #         参数：
# #             ctx: 上下文对象，保存反向传播所需变量
# #             input: 输入特征，形状[B, S, d_model] 
# #                    （B：批量大小，S：key的时间步长度，d_model：模型维度）
# #             weight: 投影权重，形状[H*d_keys, d_model] 
# #                    （H：头数，d_keys：每个头的key维度）
# #             bias: 偏置项，形状[H*d_keys]（可选）
# #             attn: 注意力矩阵，形状[B, H, L, S] 
# #                    （L：query的时间步长度，H：头数, S：key的时间步长度）
# #         返回：
# #             output: 投影后的键特征，形状[B, S, H*d_keys]
# #         """
# #         ctx.save_for_backward(input, weight, bias, attn)
# #         output = torch.matmul(input, weight.t())  # [B, S, d_model] × [d_model, H*d_keys] → [B, S, H*d_keys]
# #         if bias is not None:
# #             output += bias  # 加偏置
# #         return output

# #     @staticmethod
# #     def backward(ctx, grad_output):
# #         """
# #         反向传播：在单个batch和head内用注意力权重加权梯度
# #         参数：
# #             ctx: 保存的前向变量
# #             grad_output: 上游梯度，形状[B, S, H*d_keys]
# #         返回：
# #             各参数梯度（形状与输入对应）
# #         """
# #         input, weight, bias, attn = ctx.saved_tensors  # 恢复变量
# #         grad_input = grad_weight = grad_bias = None

# #         # --------------------------
# #         # 1. 计算单个batch和head内的注意力权重
# #         # --------------------------
# #         if attn is not None:
# #             B, H, L, S = attn.shape  # 注意力矩阵形状
# #             # 只在query维度（L）取平均，保留batch（B）和head（H）维度
# #             # 结果形状：[B, H, S] → 每个batch、每个头、每个key时间步的平均注意力权重
# #             time_weights = attn.mean(dim=2)  # 对L维度求平均：[B, H, L, S] → [B, H, S]
            
# #             # --------------------------
# #             # 2. 调整权重形状以匹配梯度
# #             # --------------------------
# #             # grad_output形状：[B, S, H*d_keys] → 需拆分为头维度
# #             d_keys = weight.shape[0] // H  # 计算每个头的key维度：H*d_keys → d_keys
# #             # 将grad_output按头拆分：[B, S, H*d_keys] → [B, S, H, d_keys]
# #             grad_output_split = grad_output.view(B, S, H, d_keys)
            
# #             # 将time_weights扩展为[B, H, S, 1]，与拆分后的梯度广播相乘
# #             # [B, H, S] → [B, H, S, 1]
# #             time_weights_expanded = time_weights.unsqueeze(-1)  # [B, H, S, 1]
            
# #             # 应用注意力权重：每个batch、每个头、每个时间步单独加权
# #             # [B, S, H, d_keys] × [B, H, S, 1] → 广播后逐元素相乘
# #             # 交换S和H维度以匹配权重的维度顺序
# #             weighted_grad_split = grad_output_split.transpose(1, 2) * time_weights_expanded  # [B, H, S, d_keys]
            
# #             # 合并头维度：[B, H, S, d_keys] → [B, S, H*d_keys]（恢复原始梯度形状）
# #             weighted_grad_output = weighted_grad_split.transpose(1, 2).contiguous().view(B, S, -1)
# #         else:
# #             weighted_grad_output = grad_output  # 无注意力权重时直接使用原始梯度

# #         # --------------------------
# #         # 3. 计算输入梯度（可选）
# #         # --------------------------
# #         if ctx.needs_input_grad[0]:
# #             # input梯度 = 加权梯度 × 权重矩阵
# #             # [B, S, H*d_keys] × [H*d_keys, d_model] → [B, S, d_model]
# #             grad_input = torch.matmul(weighted_grad_output, weight)

# #         # --------------------------
# #         # 4. 计算权重梯度（可选）
# #         # --------------------------
# #         if ctx.needs_input_grad[1]:
# #             # 权重梯度 = (加权梯度^T × input) 在batch维度取平均
# #             # 步骤1：转置加权梯度 → [B, H*d_keys, S]
# #             weighted_grad_t = weighted_grad_output.transpose(1, 2)
# #             # 步骤2：矩阵乘法 → [B, H*d_keys, S] × [B, S, d_model] → [B, H*d_keys, d_model]
# #             grad_weight_batch = torch.matmul(weighted_grad_t, input)
# #             # 步骤3：在batch维度取平均 → [H*d_keys, d_model]
# #             grad_weight = grad_weight_batch.mean(dim=0)

# #         # --------------------------
# #         # 5. 计算偏置梯度（若需要）
# #         # --------------------------
# #         if bias is not None and ctx.needs_input_grad[2]:
# #             # 偏置梯度 = 加权梯度在batch和时间步维度求和
# #             # [B, S, H*d_keys] → 对B和S维度求平均 → [H*d_keys]
# #             grad_bias = weighted_grad_output.mean(dim=[0, 1])

# #         return grad_input, grad_weight, grad_bias, None

# # 新增QueryProjection（与KeyProjection逻辑一致，仅类名和变量名调整）
# class QueryProjection(nn.Module):
#     def __init__(self, in_features, out_features, bias=False, alpha=20.0, top_k_ratio=0.1):
#         super().__init__()
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
#         self.alpha = alpha
#         self.top_k_ratio = top_k_ratio
#         self.attn = None
#         self.sample_loss = None
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
#         if self.bias is not None:
#             fan_in = nn.init._calculate_fan_in_and_fan_out(self.weight)[0]
#             bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
#             nn.init.uniform_(self.bias, -bound, bound)

#     def set_attention(self, attn, sample_loss=None):
#         self.attn = attn
#         self.sample_loss = sample_loss

#     def forward(self, input):
#         return KeyProjectionFunction.apply(
#             input, 
#             self.weight, 
#             self.bias, 
#             self.attn, 
#             self.sample_loss, 
#             self.alpha, 
#             self.top_k_ratio
#         )


# # 自定义键投影的自动求导函数
# class KeyProjectionFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, attn, sample_loss, alpha, top_k_ratio):
#         """
#         前向传播：计算键的线性投影并保存反向传播所需变量
#         参数：
#             ctx: 上下文对象，用于保存反向传播变量
#             input: 输入特征，形状[B, S, d_model]
#             weight: 投影权重，形状[H*d_keys, d_model]
#             bias: 偏置项，形状[H*d_keys]（可选）
#             attn: 注意力矩阵，形状[B, H, L, S]（可选）
#             sample_loss: 样本损失，形状[B,]（可选）
#             alpha: 注意力锐化指数
#             top_k_ratio: Top-K筛选比例
#         返回：
#             output: 投影后的键特征，形状[B, S, H*d_keys]
#         """
#         # 保存张量变量（用于反向传播）
#         ctx.save_for_backward(input, weight, bias, attn, sample_loss)
#         # 保存非张量超参数
#         ctx.alpha = alpha
#         ctx.top_k_ratio = top_k_ratio
        
#         # 计算线性投影：input × weight^T
#         output = torch.matmul(input, weight.t())  # [B, S, d_model] × [d_model, H*d_keys] → [B, S, H*d_keys]
#         if bias is not None:
#             output += bias  # 加偏置
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         反向传播：计算梯度并应用注意力权重加权
#         """
#         # 恢复前向传播保存的变量
#         input, weight, bias, attn, sample_loss = ctx.saved_tensors
#         alpha = ctx.alpha
#         top_k_ratio = ctx.top_k_ratio
        
#         # 初始化梯度变量
#         grad_input = grad_weight = grad_bias = None
#         B = input.shape[0]  # 批量大小（input必不为空）

#         # 仅当attn非空时执行加权逻辑
#         if attn is not None:
#             # 解析注意力矩阵维度
#             B_attn, H, L_attn, S = attn.shape
#             assert B_attn == B, f"attn批量大小({B_attn})与input({B})不匹配"
#             d_keys = weight.shape[0] // H  # 单个头的key维度

#             # 1. 注意力权重锐化（增强区分度）
#             attn_sharp = attn ** alpha
#             attn_sharp = attn_sharp / attn_sharp.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # 归一化

#             # 2. Top-K筛选（保留关键时间步）
#             k = max(1, int(S * top_k_ratio))  # 至少保留1个时间步
#             top_k_vals, top_k_idx = torch.topk(attn_sharp, k, dim=-1)
#             attn_topk = torch.zeros_like(attn_sharp)
#             attn_topk.scatter_(-1, top_k_idx, top_k_vals)  # 仅保留Top-K权重
#             attn_sharp = attn_topk / attn_topk.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # 重新归一化


                
#             # --------------------------
#             # 新增：保存反向传播的注意力权重（供可视化）
#             # --------------------------
#             # 将attn_sharp保存到投影层实例（如key_projection）
#             # 需在KeyProjection类中添加backward_attn属性
#             ctx.key_projection.backward_attn = attn_sharp  # 后续会在KeyProjection中定义该属性

#             # 3. 计算时间步权重（对query维度取平均）
#             time_weights = attn_sharp.mean(dim=2)  # [B, H, L, S] → [B, H, S]

#             # 4. 样本损失加权（高损失样本权重放大）
#             if sample_loss is not None:
#                 sample_loss_norm = (sample_loss - sample_loss.min()) / (
#                     (sample_loss.max() - sample_loss.min()).clamp(min=1e-8)
#                 )  # 归一化到[0,1]
#                 time_weights = time_weights * (1.0 + sample_loss_norm.unsqueeze(1).unsqueeze(2))  # 广播匹配

#             # 5. 梯度加权（按头和时间步适配维度）
#             grad_output_split = grad_output.view(B, S, H, d_keys).transpose(1, 2)  # [B, H, S, d_keys]
#             weighted_grad_split = grad_output_split * time_weights.unsqueeze(-1)  # 广播加权
#             weighted_grad_output = weighted_grad_split.transpose(1, 2).contiguous().view(B, S, -1)  # 恢复形状
#         else:
#             # 无注意力矩阵时使用原始梯度
#             weighted_grad_output = grad_output

#         # 计算各参数梯度
#         if ctx.needs_input_grad[0]:
#             grad_input = torch.matmul(weighted_grad_output, weight)  # 输入梯度
#         if ctx.needs_input_grad[1]:
#             # 权重梯度（batch维度取平均）
#             grad_weight = torch.matmul(weighted_grad_output.transpose(1, 2), input).mean(dim=0)
#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = weighted_grad_output.sum(dim=[0, 1])  # 偏置梯度
        
#         # 返回梯度（后四个为None，对应非参数输入）
#         return grad_input, grad_weight, grad_bias, None, None, None, None


# # # 自定义键投影层：封装参数并存储当前batch的注意力权重
# # class KeyProjection(nn.Module):
# #     def __init__(self, in_features, out_features, bias=False):
# #         """
# #         参数：
# #             in_features: 输入维度（d_model）
# #             out_features: 输出维度（H*d_keys）
# #             bias: 是否使用偏置
# #         """
# #         super().__init__()
# #         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))  # [H*d_keys, d_model]
# #         if bias:
# #             self.bias = nn.Parameter(torch.Tensor(out_features))  # [H*d_keys]
# #         else:
# #             self.register_parameter('bias', None)
# #         self.attn = None  # 存储当前batch的注意力矩阵[B, H, L, S]
# #         self.reset_parameters()

# #     def reset_parameters(self):
# #         # 权重初始化（与PyTorch线性层一致）
# #         nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
# #         if self.bias is not None:
# #             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
# #             bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
# #             nn.init.uniform_(self.bias, -bound, bound)

# #     def set_attention(self, attn):
# #         """设置当前batch的注意力矩阵（供反向传播使用）"""
# #         self.attn = attn  # 形状[B, H, L, S]

# #     def forward(self, input):
# #         """前向传播：调用自定义函数计算键投影"""
# #         return KeyProjectionFunction.apply(input, self.weight, self.bias, self.attn)
    

# class ValueProjection(nn.Module):
#     def __init__(self, in_features, out_features, bias=False, alpha=20.0, top_k_ratio=0.1):
#         super().__init__()
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
#         self.alpha = alpha
#         self.top_k_ratio = top_k_ratio
#         self.attn = None
#         self.sample_loss = None
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
#         if self.bias is not None:
#             fan_in = nn.init._calculate_fan_in_and_fan_out(self.weight)[0]
#             bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
#             nn.init.uniform_(self.bias, -bound, bound)

#     def set_attention(self, attn, sample_loss=None):
#         self.attn = attn
#         self.sample_loss = sample_loss

#     def forward(self, input):
#         return KeyProjectionFunction.apply(
#             input, 
#             self.weight, 
#             self.bias, 
#             self.attn, 
#             self.sample_loss, 
#             self.alpha, 
#             self.top_k_ratio
#         )
# # 键投影层（使用自定义自动求导函数）
# class KeyProjection(nn.Module):
#     def __init__(self, in_features, out_features, bias=False, alpha=20.0, top_k_ratio=0.1):
#         """
#         参数：
#             in_features: 输入特征维度（d_model）
#             out_features: 输出特征维度（H*d_keys）
#             bias: 是否使用偏置
#             alpha: 注意力锐化指数（>1增强区分度）
#             top_k_ratio: Top-K筛选比例（0~1）
#         """
#         super().__init__()
#         # 定义权重和偏置参数
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))  # [H*d_keys, d_model]
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_features))  # [H*d_keys]
#         else:
#             self.bias = None  # 无偏置
        
#         # 关键修复：显式声明alpha和top_k_ratio为实例属性
#         self.alpha = alpha
#         self.top_k_ratio = top_k_ratio
        
#         # 存储注意力矩阵和样本损失（动态更新）
#         self.attn = None  # 注意力矩阵[B, H, L, S]
#         self.sample_loss = None  # 样本损失[B,]
        
#         # 初始化参数
#         self.reset_parameters()

#         self.backward_attn = None  # 新增：存储反向传播中锐化后的注意力权重

#     def reset_parameters(self):
#         """使用Kaiming均匀分布初始化权重"""
#         nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
#         if self.bias is not None:
#             fan_in = nn.init._calculate_fan_in_and_fan_out(self.weight)[0]
#             bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
#             nn.init.uniform_(self.bias, -bound, bound)

#     def set_attention(self, attn, sample_loss=None):
#         """更新当前批次的注意力矩阵和样本损失"""
#         self.attn = attn
#         self.sample_loss = sample_loss

#     def forward(self, input):
#         """前向传播：调用自定义自动求导函数"""
#         return KeyProjectionFunction.apply(
#             input, 
#             self.weight, 
#             self.bias, 
#             self.attn, 
#             self.sample_loss, 
#             self.alpha,  # 此处已确保self.alpha存在
#             self.top_k_ratio  # 此处已确保self.top_k_ratio存在
#         )



# # 原有注意力机制类（保持不变）
# class DSAttention(nn.Module):
#     '''De-stationary Attention：含去平稳化因子的注意力机制'''
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
#         super(DSAttention, self).__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / sqrt(E)

#         tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # [B,1,1,1]
#         delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # [B,1,1,S]

#         scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta  # [B,H,L,S]
#         if self.mask_flag:
#             if attn_mask is None:
#                 attn_mask = TriangularCausalMask(B, L, device=queries.device)
#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B,H,L,S]
#         V = torch.einsum("bhls,bshd->blhd", A, values)  # [B,L,H,D]
#         return (V.contiguous(), A) if self.output_attention else (V.contiguous(), None)


# class FullAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
#         super(FullAttention, self).__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention  # 强制默认True
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         # 强制验证output_attention状态
#         # print(f"FullAttention: self.output_attention = {self.output_attention}")  # 必须打印True
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / sqrt(E)

#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # [B,H,L,S]
#         if self.mask_flag:
#             if attn_mask is None:
#                 attn_mask = TriangularCausalMask(B, L, device=queries.device)
#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B,H,L,S]
#         # 验证A的有效性
#         # print(f"FullAttention: A形状={A.shape}, 均值={A.mean().item():.4f}")  # 应打印有效形状和均值
#         V = torch.einsum("bhls,bshd->blhd", A, values)  # [B,L,H,D]
        
#         return (V.contiguous(), A) if self.output_attention else (V.contiguous(), None)


# # 修改后的注意力层：适配自定义键投影
# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
#         """
#         参数：
#             attention: 注意力机制（如FullAttention/DSAttention）
#             d_model: 模型维度
#             n_heads: 注意力头数（H）
#             d_keys: 每个头的key维度（默认d_model//n_heads）
#             d_values: 每个头的value维度（默认d_model//n_heads）
#         """
#         super(AttentionLayer, self).__init__()
#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)

#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # [d_model, H*d_keys]
#         self.key_projection = KeyProjection(d_model, d_keys * n_heads)  # 自定义键投影
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)  # [d_model, H*d_values]
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)  # [H*d_values, d_model]
#         self.n_heads = n_heads  # 头数H

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         """
#         前向传播：计算注意力输出并传递注意力矩阵
#         参数：
#             queries: [B, L, d_model]，查询特征
#             keys: [B, S, d_model]，键特征
#             values: [B, S, d_model]，值特征
#             attn_mask: 注意力掩码
#         返回：
#             out: [B, L, d_model]，注意力层输出
#             attn: [B, H, L, S]，注意力矩阵（可选）
#         """
#         B, L, _ = queries.shape  # B：批量，L：query时间步
#         _, S, _ = keys.shape     # S：key时间步
#         H = self.n_heads         # H：头数

#         # 投影并拆分多头
#         queries = self.query_projection(queries).view(B, L, H, -1)  # [B,L,H,d_keys]
#         keys = self.key_projection(keys).view(B, S, H, -1)          # [B,S,H,d_keys]
#         values = self.value_projection(values).view(B, S, H, -1)    # [B,S,H,d_values]

#         # 计算注意力，得到输出和注意力矩阵
#         out, attn = self.inner_attention(queries, keys, values, attn_mask, tau, delta)
#         # attn形状：[B, H, L, S]（每个batch、每个头的注意力分布）

#         # 将注意力矩阵传递给key_projection，供反向传播使用
#         self.key_projection.set_attention(attn)

#         # 合并多头并投影输出
#         out = out.view(B, L, -1)  # [B,L,H*d_values]
#         return self.out_projection(out), attn  # [B,L,d_model]
    
# import sys
# import os
# import torch
# import torch.nn as nn
# import numpy as np
# from math import sqrt

# # --------------------------
# # 路径配置：确保跨文件夹导入正常
# # --------------------------
# current_file_path = os.path.abspath(__file__)
# layers_dir = os.path.dirname(current_file_path)
# project_root = os.path.dirname(layers_dir)
# if project_root not in sys.path:
#     sys.path.append(project_root)

# # --------------------------
# # 可视化相关导入（兼容路径问题）
# # --------------------------
# try:
#     from models.visualization import (
#         visualize_forward_attn,
#         visualize_backward_attn,
#         global_forward_attn_without_weight,
#         global_backward_attn_without_weight
#     )
# except ImportError as e:
#     print(f"[可视化导入警告] 导入失败：{str(e)}，不影响训练但无可视化")
#     # 定义占位函数避免训练中断
#     def visualize_forward_attn(*args, **kwargs):
#         pass
#     def visualize_backward_attn(*args, **kwargs):
#         pass
#     global_forward_attn_without_weight = None
#     global_backward_attn_without_weight = None

# # --------------------------
# # 补充缺失的依赖类（原代码导入但未定义）
# # --------------------------
# class TriangularCausalMask(nn.Module):
#     """三角因果掩码（防止Transformer解码器未来信息泄露）"""
#     def __init__(self, B, L, device="cpu"):
#         super().__init__()
#         mask_shape = [B, 1, L, L]
#         self.mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1)

# class ProbMask(nn.Module):
#     """概率注意力掩码（用于ProbAttention）"""
#     def __init__(self, B, H, L, index, scores, device="cpu"):
#         super().__init__()
#         _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1)
#         _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
#         indicator = _mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :]
#         self.mask = indicator.view(scores.shape).to(device)

# # --------------------------
# # 兼容缺失的外部依赖（LSHSelfAttention和einops）
# # --------------------------
# try:
#     from reformer_pytorch import LSHSelfAttention
# except ImportError:
#     class LSHSelfAttention(nn.Module):
#         """占位类：未安装reformer_pytorch时避免报错"""
#         def __init__(self, dim, heads, bucket_size, n_hashes, causal):
#             super().__init__()
#             self.dim = dim
#             self.heads = heads
#             self.bucket_size = bucket_size
#             self.n_hashes = n_hashes
#             self.causal = causal
#             self.linear = nn.Linear(dim, dim)

#         def forward(self, x):
#             out = self.linear(x)
#             attn = torch.ones(x.shape[0], self.heads, x.shape[1], x.shape[1], device=x.device) / x.shape[1]
#             return out, attn

# try:
#     from einops import rearrange, repeat
# except ImportError:
#     """占位函数：未安装einops时提供基础实现"""
#     def rearrange(x, pattern):
#         if 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model' in pattern:
#             return x.reshape(-1, x.shape[2], x.shape[3])
#         elif '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model' in pattern:
#             b = int(pattern.split('b=')[1].split(')')[0])
#             return x.reshape(b, -1, x.shape[1], x.shape[2]).transpose(1, 2).reshape(-1, x.shape[1], x.shape[2])
#         elif '(b seg_num) ts_d d_model -> b ts_d seg_num d_model' in pattern:
#             b = int(pattern.split('b=')[1].split(')')[0])
#             seg_num = int(x.shape[0] / b)
#             return x.reshape(b, seg_num, x.shape[1], x.shape[2]).transpose(1, 2)
#         else:
#             return x

#     def repeat(x, pattern):
#         repeat_num = int(pattern.split('repeat=')[1].split(')')[0])
#         return x.repeat(repeat_num, 1, 1)

# # --------------------------
# # 核心：自定义自动求导函数（实现注意力加权和反向权重保存）
# # --------------------------
# class KeyProjectionFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, attn, sample_loss, alpha, top_k_ratio, key_proj):
#         # 关键修复：只保存张量到save_for_backward，非张量用ctx属性存储
#         ctx.save_for_backward(input, weight, bias, attn, sample_loss)  # 移除key_proj
#         ctx.key_proj = key_proj  # 用ctx属性存储模块实例（非张量）
#         ctx.alpha = alpha
#         ctx.top_k_ratio = top_k_ratio
            
#         # 原有投影计算逻辑不变
#         output = torch.matmul(input, weight.t())
#         if bias is not None:
#             output += bias
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         # 恢复保存的张量（注意：这里不再包含key_proj）
#         input, weight, bias, attn, sample_loss = ctx.saved_tensors
#         key_proj = ctx.key_proj  # 从ctx属性获取模块实例
#         alpha = ctx.alpha
#         top_k_ratio = ctx.top_k_ratio
        
#         # 后续逻辑不变（梯度计算、反向注意力权重保存等）
#         grad_input = grad_weight = grad_bias = None
#         B = input.shape[0]

#         if attn is not None:
#             B_attn, H, L_attn, S = attn.shape
#             assert B_attn == B, f"attn批量({B_attn})与输入({B})不匹配"
#             d_keys = weight.shape[0] // H

#             # 注意力锐化与Top-K稀疏化
#             attn_sharp = attn ** alpha
#             attn_sharp = attn_sharp / attn_sharp.sum(dim=-1, keepdim=True).clamp(min=1e-8)
#             k = max(1, int(S * top_k_ratio))
#             top_k_vals, top_k_idx = torch.topk(attn_sharp, k, dim=-1)
#             attn_topk = torch.zeros_like(attn_sharp)
#             attn_topk.scatter_(-1, top_k_idx, top_k_vals)
#             attn_sharp = attn_topk / attn_topk.sum(dim=-1, keepdim=True).clamp(min=1e-8)

#             # 保存反向注意力权重（仍使用key_proj）
#             key_proj.backward_attn = attn_sharp
#             print(f"[反向权重保存] 投影层={key_proj.__class__.__name__}，形状={attn_sharp.shape}")

#             # 时间权重计算（不变）
#             time_weights = attn_sharp.mean(dim=2)
#             if sample_loss is not None:
#                 sample_loss_norm = (sample_loss - sample_loss.min()) / (
#                     (sample_loss.max() - sample_loss.min()).clamp(min=1e-8)
#                 )
#                 time_weights = time_weights * (1.0 + sample_loss_norm.unsqueeze(1).unsqueeze(2))
#         else:
#             time_weights = 1.0

#         # 加权梯度计算（不变）
#         if attn is not None:
#             grad_output_split = grad_output.view(B, S, H, d_keys).transpose(1, 2)
#             weighted_grad_split = grad_output_split * time_weights.unsqueeze(-1)
#             weighted_grad_output = weighted_grad_split.transpose(1, 2).contiguous().view(B, S, -1)
#         else:
#             weighted_grad_output = grad_output

#         # 梯度计算（不变）
#         if ctx.needs_input_grad[0]:
#             grad_input = torch.matmul(weighted_grad_output, weight)
#         if ctx.needs_input_grad[1]:
#             grad_weight = torch.matmul(weighted_grad_output.transpose(1, 2), input).mean(dim=0)
#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = weighted_grad_output.sum(dim=[0, 1])
        
#         return grad_input, grad_weight, grad_bias, None, None, None, None, None

# # --------------------------
# # 自定义投影层（Query/Key/Value）：替换原有nn.Linear，支持注意力加权
# # --------------------------
# class QueryProjection(nn.Module):
#     def __init__(self, in_features, out_features, bias=False, alpha=5.0, top_k_ratio=0.2):
#         super().__init__()
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
#         self.alpha = alpha  # 注意力锐化系数
#         self.top_k_ratio = top_k_ratio  # Top-K稀疏比例
#         self.attn = None  # 存储前向注意力矩阵
#         self.sample_loss = None  # 存储样本损失（可选）
#         self.backward_attn = None  # 存储反向注意力权重
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
#         if self.bias is not None:
#             fan_in = nn.init._calculate_fan_in_and_fan_out(self.weight)[0]
#             bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
#             nn.init.uniform_(self.bias, -bound, bound)

#     def set_attention(self, attn, sample_loss=None):
#         """外部设置前向注意力矩阵和样本损失"""
#         self.attn = attn
#         self.sample_loss = sample_loss

#     def forward(self, input):
#         return KeyProjectionFunction.apply(
#             input, self.weight, self.bias, self.attn, self.sample_loss,
#             self.alpha, self.top_k_ratio, self  # 传递自身作为key_proj
#         )

# class KeyProjection(nn.Module):
#     def __init__(self, in_features, out_features, bias=False, alpha=5.0, top_k_ratio=1.0):
#         super().__init__()
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
#         self.alpha = alpha
#         self.top_k_ratio = top_k_ratio
#         self.attn = None
#         self.sample_loss = None
#         self.backward_attn = None
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
#         if self.bias is not None:
#             fan_in = nn.init._calculate_fan_in_and_fan_out(self.weight)[0]
#             bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
#             nn.init.uniform_(self.bias, -bound, bound)

#     def set_attention(self, attn, sample_loss=None):
#         self.attn = attn
#         self.sample_loss = sample_loss

#     def forward(self, input):
#         return KeyProjectionFunction.apply(
#             input, self.weight, self.bias, self.attn, self.sample_loss,
#             self.alpha, self.top_k_ratio, self
#         )

# class ValueProjection(nn.Module):
#     def __init__(self, in_features, out_features, bias=False, alpha=5.0, top_k_ratio=0.2):
#         super().__init__()
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
#         self.alpha = alpha
#         self.top_k_ratio = top_k_ratio
#         self.attn = None
#         self.sample_loss = None
#         self.backward_attn = None
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
#         if self.bias is not None:
#             fan_in = nn.init._calculate_fan_in_and_fan_out(self.weight)[0]
#             bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
#             nn.init.uniform_(self.bias, -bound, bound)

#     def set_attention(self, attn, sample_loss=None):
#         self.attn = attn
#         self.sample_loss = sample_loss

#     def forward(self, input):
#         return KeyProjectionFunction.apply(
#             input, self.weight, self.bias, self.attn, self.sample_loss,
#             self.alpha, self.top_k_ratio, self
#         )

# # --------------------------
# # 原有注意力机制类（保持核心逻辑，修复output_attention默认值）
# # --------------------------
# class DSAttention(nn.Module):
#     '''De-stationary Attention'''
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
#         super().__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention  # 默认True，返回注意力矩阵
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / sqrt(E)

#         tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
#         delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)

#         scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta
#         if self.mask_flag:
#             if attn_mask is None:
#                 attn_mask = TriangularCausalMask(B, L, device=queries.device)
#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)

#         return (V.contiguous(), A) if self.output_attention else (V.contiguous(), None)

# class FullAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
#         super().__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention  # 默认True
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / sqrt(E)

#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)
#         if self.mask_flag:
#             if attn_mask is None:
#                 attn_mask = TriangularCausalMask(B, L, device=queries.device)
#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)

#         return (V.contiguous(), A) if self.output_attention else (V.contiguous(), None)

# class ProbAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
#         super().__init__()
#         self.factor = factor
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention  # 默认True
#         self.dropout = nn.Dropout(attention_dropout)

#     def _prob_QK(self, Q, K, sample_k, n_top):
#         B, H, L_K, E = K.shape
#         _, _, L_Q, _ = Q.shape
#         K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
#         index_sample = torch.randint(L_K, (L_Q, sample_k))
#         K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
#         Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
#         M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
#         M_top = M.topk(n_top, sorted=False)[1]
#         Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
#         Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
#         return Q_K, M_top

#     def _get_initial_context(self, V, L_Q):
#         B, H, L_V, D = V.shape
#         if not self.mask_flag:
#             V_sum = V.mean(dim=-2)
#             contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
#         else:
#             assert (L_Q == L_V)
#             contex = V.cumsum(dim=-2)
#         return contex

#     def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
#         B, H, L_V, D = V.shape
#         if self.mask_flag:
#             attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
#             scores.masked_fill_(attn_mask.mask, -np.inf)
#         attn = torch.softmax(scores, dim=-1)
#         context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)
#         if self.output_attention:
#             attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
#             attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
#             return context_in, attns
#         else:
#             return context_in, None

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L_Q, H, D = queries.shape
#         _, L_K, _, _ = keys.shape
#         queries = queries.transpose(2, 1)
#         keys = keys.transpose(2, 1)
#         values = values.transpose(2, 1)

#         U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
#         u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
#         U_part = U_part if U_part < L_K else L_K
#         u = u if u < L_Q else L_Q

#         scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
#         scale = self.scale or 1. / sqrt(D)
#         if scale is not None:
#             scores_top = scores_top * scale
#         context = self._get_initial_context(values, L_Q)
#         context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

#         return context.contiguous(), attn

# # --------------------------
# # 核心注意力层封装（整合加权和可视化）
# # --------------------------
# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, alpha=20.0, top_k_ratio=0.1):
#         super().__init__()
#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)

#         self.inner_attention = attention
#         # 替换原有nn.Linear为自定义投影层
#         self.query_projection = QueryProjection(
#             in_features=d_model, out_features=d_keys * n_heads, alpha=alpha, top_k_ratio=top_k_ratio
#         )
#         self.key_projection = KeyProjection(
#             in_features=d_model, out_features=d_keys * n_heads, alpha=alpha, top_k_ratio=top_k_ratio
#         )
#         self.value_projection = ValueProjection(
#             in_features=d_model, out_features=d_values * n_heads, alpha=alpha, top_k_ratio=top_k_ratio
#         )
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads
#         self.d_keys = d_keys
#         self.d_values = d_values

#         # 可视化配置
#         self.visualize_freq = 1  # 每1轮可视化一次
#         self.current_epoch = 0  # 当前训练轮次
#         self.use_weight = True  # 是否启用注意力加权
#         self.visualize_count = 0  # 避免同轮多次可视化

#     def set_epoch(self, epoch):
#         """训练循环中调用：更新轮次，重置可视化计数"""
#         self.current_epoch = epoch
#         self.visualize_count = 0

#     def forward(self, queries, keys, values, attn_mask=None, sample_loss=None, tau=None, delta=None):
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads

#         # --------------------------
#         # 1. 计算前向注意力矩阵（用于可视化）
#         # --------------------------
#         # 临时投影层（仅计算注意力矩阵，不参与参数更新）
#         temp_query_proj = nn.Linear(queries.shape[-1], self.d_keys * H).to(queries.device)
#         temp_key_proj = nn.Linear(keys.shape[-1], self.d_keys * H).to(keys.device)
#         temp_value_proj = nn.Linear(values.shape[-1], self.d_values * H).to(values.device)
        
#         temp_queries = temp_query_proj(queries).view(B, L, H, -1)
#         temp_keys = temp_key_proj(keys).view(B, S, H, -1)
#         temp_values = temp_value_proj(values).view(B, S, H, -1)
        
#         _, attn = self.inner_attention(temp_queries, temp_keys, temp_values, attn_mask, tau, delta)
#         self.last_forward_attn = attn[0].cpu().detach().numpy()  # 保存前向注意力

#         # --------------------------
#         # 2. 前向/反向注意力可视化
#         # --------------------------
#         if self.training and (self.current_epoch % self.visualize_freq == 0) and (self.visualize_count == 0):
#             try:
#                 # 前向注意力可视化
#                 attn_sample = attn[0].cpu().detach().numpy()
#                 global global_forward_attn_without_weight

#                 if not self.use_weight and global_forward_attn_without_weight is None:
#                     # 第0轮生成无加权基准图
#                     global_forward_attn_without_weight = attn_sample
#                     visualize_forward_attn(
#                         attn_matrix=attn_sample, epoch=self.current_epoch, prefix="forward_without_weight"
#                     )
#                 elif self.use_weight:
#                     # 后续轮次生成加权图
#                     visualize_forward_attn(
#                         attn_matrix=attn_sample, epoch=self.current_epoch, prefix="forward_with_weight"
#                     )

#                 # 反向注意力可视化（从key_projection提取）
#                 if hasattr(self.key_projection, 'backward_attn') and self.key_projection.backward_attn is not None:
#                     backward_attn = self.key_projection.backward_attn[0].cpu().detach().numpy()
#                     global global_backward_attn_without_weight

#                     if not self.use_weight and global_backward_attn_without_weight is None:
#                         global_backward_attn_without_weight = backward_attn
#                         visualize_backward_attn(
#                             attn_sharp_matrix=backward_attn, epoch=self.current_epoch, prefix="backward_without_weight"
#                         )
#                     elif self.use_weight:
#                         visualize_backward_attn(
#                             attn_sharp_matrix=backward_attn, epoch=self.current_epoch, prefix="backward_with_weight"
#                         )

#                 self.visualize_count += 1
#             except Exception as e:
#                 print(f"[可视化警告] 生成失败：{str(e)}，训练继续")

#         # --------------------------
#         # 3. 正式计算加权注意力输出
#         # --------------------------
#         self.query_projection.set_attention(attn, sample_loss)
#         self.key_projection.set_attention(attn, sample_loss)
#         self.value_projection.set_attention(attn, sample_loss)

#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)

#         out, _ = self.inner_attention(queries, keys, values, attn_mask, tau, delta)
#         out = out.view(B, L, -1)
#         return self.out_projection(out), attn

# # --------------------------
# # 原有辅助类（保持逻辑，适配新AttentionLayer）
# # --------------------------
# class ReformerLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, causal=False, bucket_size=4, n_hashes=4):
#         super().__init__()
#         self.bucket_size = bucket_size
#         self.attn = LSHSelfAttention(
#             dim=d_model, heads=n_heads, bucket_size=bucket_size, n_hashes=n_hashes, causal=causal
#         )

#     def fit_length(self, queries):
#         B, N, C = queries.shape
#         if N % (self.bucket_size * 2) == 0:
#             return queries
#         else:
#             fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
#             return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

#     def forward(self, queries, keys, values, attn_mask, tau, delta):
#         B, N, C = queries.shape
#         out, attn = self.attn(self.fit_length(queries))  # 适配LSHSelfAttention返回(attn)
#         out = out[:, :N, :]
#         return out, attn

# class TwoStageAttentionLayer(nn.Module):
#     '''Two Stage Attention (TSA) Layer'''
#     def __init__(self, configs, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
#         super().__init__()
#         d_ff = d_ff or 4 * d_model
#         # 适配新AttentionLayer，添加alpha和top_k_ratio参数
#         self.time_attention = AttentionLayer(
#             FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=True),
#             d_model, n_heads, alpha=20.0, top_k_ratio=0.1
#         )
#         self.dim_sender = AttentionLayer(
#             FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=True),
#             d_model, n_heads, alpha=20.0, top_k_ratio=0.1
#         )
#         self.dim_receiver = AttentionLayer(
#             FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=True),
#             d_model, n_heads, alpha=20.0, top_k_ratio=0.1
#         )
#         self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
#         self.dropout = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.norm4 = nn.LayerNorm(d_model)
#         self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
#         self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         batch = x.shape[0]
#         time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
#         time_enc, attn = self.time_attention(time_in, time_in, time_in, attn_mask=None, tau=None, delta=None)
#         dim_in = time_in + self.dropout(time_enc)
#         dim_in = self.norm1(dim_in)
#         dim_in = dim_in + self.dropout(self.MLP1(dim_in))
#         dim_in = self.norm2(dim_in)

#         dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
#         batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
#         dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
#         dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
#         dim_enc = dim_send + self.dropout(dim_receive)
#         dim_enc = self.norm3(dim_enc)
#         dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
#         dim_enc = self.norm4(dim_enc)

#         final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
#         return final_out, attn  # 统一返回(output, attn)，适配上层调用























import sys
import os
import torch
import torch.nn as nn
import numpy as np
from math import sqrt

# --------------------------
# 路径配置：确保跨文件夹导入正常
# --------------------------
current_file_path = os.path.abspath(__file__)
layers_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(layers_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --------------------------
# 可视化相关导入（兼容路径问题）
# --------------------------

from models.visualization import (
        visualize_forward_attn,
        visualize_backward_attn,
        global_forward_attn_without_weight,
        global_backward_attn_without_weight
    )


class TriangularCausalMask(nn.Module):
    """三角因果掩码（防止Transformer解码器未来信息泄露）"""
    def __init__(self, B, L, device="cpu"):
        super().__init__()
        mask_shape = [B, 1, L, L]
        self.mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1)

class ProbMask(nn.Module):
    """概率注意力掩码（用于ProbAttention）"""
    def __init__(self, B, H, L, index, scores, device="cpu"):
        super().__init__()
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :]
        self.mask = indicator.view(scores.shape).to(device)


from reformer_pytorch import LSHSelfAttention
        
from einops import rearrange, repeat

# --------------------------
# 核心：自定义自动求导函数（逐通道加权）
# --------------------------
class KeyProjectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, attn, sample_loss, alpha, top_k_ratio, key_proj):
        ctx.save_for_backward(input, weight, bias, attn, sample_loss)
        ctx.key_proj = key_proj
        ctx.alpha = alpha
        ctx.top_k_ratio = top_k_ratio
            
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, attn, sample_loss = ctx.saved_tensors
        key_proj = ctx.key_proj
        alpha = ctx.alpha
        top_k_ratio = ctx.top_k_ratio
        
        grad_input = grad_weight = grad_bias = None
        B = input.shape[0]

        if attn is not None:
            B_attn, H, L_attn, S = attn.shape
            assert B_attn == B, f"attn批量({B_attn})与输入({B})不匹配"
            d_keys = weight.shape[0] // H

            # # 1. 注意力锐化（保留所有权重，不做稀疏化）
            # attn_sharp = attn ** alpha
            # attn_sharp = attn_sharp / attn_sharp.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # 2. 计算需要保留的top-k时间步索引（按比例选择）
            k = max(1, int(S * top_k_ratio))  # top-k时间步数量
            _, top_k_idx = torch.topk(attn, k, dim=-1)  # 获取top-k时间步的索引

            # 3. 创建top-k时间步的掩码（仅保留这些时间步的梯度）
            mask = torch.zeros_like(attn)
            mask.scatter_(-1, top_k_idx, 1.0)  # 在top-k索引位置置为1，其他为0

            # 4. 保存完整的反向注意力权重（保留所有值）
            key_proj.backward_attn = attn

            # 6. 仅对top-k时间步计算梯度（其他时间步梯度被掩码清零）
            grad_output_split = grad_output.view(B, S, H, d_keys).transpose(1, 2)  # [B, H, S, d_keys]
            mask_agg = mask.mean(dim=2).unsqueeze(-1)  # 聚合序列维度，扩展到d_keys维度
            weighted_grad_split = (grad_output_split ** alpha) * mask_agg  # 仅保留top-k时间步的梯度
            weighted_grad_output = weighted_grad_split.transpose(1, 2).contiguous().view(B, S, -1)
        else:
            weighted_grad_output = grad_output

        # 常规梯度计算（使用过滤后的梯度）
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(weighted_grad_output, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(weighted_grad_output.transpose(1, 2), input).mean(dim=0)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = weighted_grad_output.mean(dim=0).sum(dim=1)
        
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

# --------------------------
# 自定义投影层（支持逐通道加权）
# --------------------------
class QueryProjection(nn.Module):
    def __init__(self, in_features, out_features, bias=False, alpha=5.0, top_k_ratio=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.alpha = alpha  # 注意力锐化系数
        self.top_k_ratio = top_k_ratio  # Top-K稀疏比例
        self.attn = None  # 存储前向注意力矩阵（含通道维度）
        self.sample_loss = None  # 存储样本损失
        self.backward_attn = None  # 存储反向注意力权重（含通道维度）
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in = nn.init._calculate_fan_in_and_fan_out(self.weight)[0]
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def set_attention(self, attn, sample_loss=None):
        """外部设置前向注意力矩阵（含通道维度）和样本损失"""
        self.attn = attn
        self.sample_loss = sample_loss

    def forward(self, input):
        return KeyProjectionFunction.apply(
            input, self.weight, self.bias, self.attn, self.sample_loss,
            self.alpha, self.top_k_ratio, self
        )

class KeyProjection(nn.Module):
    def __init__(self, in_features, out_features, bias=False, alpha=5.0, top_k_ratio=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.alpha = alpha
        self.top_k_ratio = top_k_ratio
        self.attn = None
        self.sample_loss = None
        self.backward_attn = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in = nn.init._calculate_fan_in_and_fan_out(self.weight)[0]
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def set_attention(self, attn, sample_loss=None):
        self.attn = attn
        self.sample_loss = sample_loss

    def forward(self, input):
        return KeyProjectionFunction.apply(
            input, self.weight, self.bias, self.attn, self.sample_loss,
            self.alpha, self.top_k_ratio, self
        )

class ValueProjection(nn.Module):
    def __init__(self, in_features, out_features, bias=False, alpha=5.0, top_k_ratio=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.alpha = alpha
        self.top_k_ratio = top_k_ratio
        self.attn = None
        self.sample_loss = None
        self.backward_attn = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in = nn.init._calculate_fan_in_and_fan_out(self.weight)[0]
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def set_attention(self, attn, sample_loss=None):
        self.attn = attn
        self.sample_loss = sample_loss

    def forward(self, input):
        return KeyProjectionFunction.apply(
            input, self.weight, self.bias, self.attn, self.sample_loss,
            self.alpha, self.top_k_ratio, self
        )

# --------------------------
# 注意力机制类（适配逐通道处理）
# --------------------------
class DSAttention(nn.Module):
    '''De-stationary Attention'''
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # queries: [B, L, H, E]，包含通道维度信息
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)

        # 逐通道计算注意力分数
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return (V.contiguous(), A) if self.output_attention else (V.contiguous(), None)

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # 适配PatchTST的通道维度：[B, L, H, E]
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # 逐通道计算注意力分数
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return (V.contiguous(), A) if self.output_attention else (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn

# --------------------------
# 核心注意力层封装（整合逐通道加权）
# --------------------------
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, alpha=5.0, top_k_ratio=0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        # 替换原有nn.Linear为自定义投影层（支持逐通道加权）
        self.query_projection = QueryProjection(
            in_features=d_model, out_features=d_keys * n_heads, alpha=alpha, top_k_ratio=top_k_ratio
        )
        self.key_projection = KeyProjection(
            in_features=d_model, out_features=d_keys * n_heads, alpha=alpha, top_k_ratio=top_k_ratio
        )
        self.value_projection = ValueProjection(
            in_features=d_model, out_features=d_values * n_heads, alpha=alpha, top_k_ratio=top_k_ratio
        )
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values

        # 可视化配置
        self.visualize_freq = 1
        self.current_epoch = 0
        self.use_weight = True
        self.visualize_count = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.visualize_count = 0

    def forward(self, queries, keys, values, attn_mask=None, sample_loss=None, tau=None, delta=None):
        # 适配PatchTST的输入形状：[B, seg_num, d_model]（已包含通道信息）
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 1. 计算前向注意力矩阵（含通道维度）
        temp_query_proj = nn.Linear(queries.shape[-1], self.d_keys * H).to(queries.device)
        temp_key_proj = nn.Linear(keys.shape[-1], self.d_keys * H).to(keys.device)
        temp_value_proj = nn.Linear(values.shape[-1], self.d_values * H).to(values.device)
        
        temp_queries = temp_query_proj(queries).view(B, L, H, -1)  # [B, L, H, d_keys]
        temp_keys = temp_key_proj(keys).view(B, S, H, -1)          # [B, S, H, d_keys]
        temp_values = temp_value_proj(values).view(B, S, H, -1)    # [B, S, H, d_values]
        
        _, attn = self.inner_attention(temp_queries, temp_keys, temp_values, attn_mask, tau, delta)
        self.last_forward_attn = attn[0].cpu().detach().numpy()

        # 2. 注意力可视化（保持通道信息）
        if self.training and (self.current_epoch % self.visualize_freq == 0) and (self.visualize_count == 0):
            try:
                attn_sample = attn[0].cpu().detach().numpy()
                global global_forward_attn_without_weight

                if not self.use_weight and global_forward_attn_without_weight is None:
                    global_forward_attn_without_weight = attn_sample
                    visualize_forward_attn(
                        attn_matrix=attn_sample, epoch=self.current_epoch, prefix="forward_without_weight"
                    )
                elif self.use_weight:
                    visualize_forward_attn(
                        attn_matrix=attn_sample, epoch=self.current_epoch, prefix="forward_with_weight"
                    )

                if hasattr(self.key_projection, 'backward_attn') and self.key_projection.backward_attn is not None:
                    backward_attn = self.key_projection.backward_attn[0].cpu().detach().numpy()
                    global global_backward_attn_without_weight

                    if not self.use_weight and global_backward_attn_without_weight is None:
                        global_backward_attn_without_weight = backward_attn
                        visualize_backward_attn(
                            attn_sharp_matrix=backward_attn, epoch=self.current_epoch, prefix="backward_without_weight"
                        )
                    elif self.use_weight:
                        visualize_backward_attn(
                            attn_sharp_matrix=backward_attn, epoch=self.current_epoch, prefix="backward_with_weight"
                        )

                self.visualize_count += 1
            except Exception as e:
                print(f"[可视化警告] 生成失败：{str(e)}，训练继续")

        # 3. 正式计算逐通道加权注意力输出
        self.query_projection.set_attention(attn, sample_loss)
        self.key_projection.set_attention(attn, sample_loss)
        self.value_projection.set_attention(attn, sample_loss)

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, _ = self.inner_attention(queries, keys, values, attn_mask, tau, delta)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

# --------------------------
# 辅助类（适配新AttentionLayer）
# --------------------------
class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model, heads=n_heads, bucket_size=bucket_size, n_hashes=n_hashes, causal=causal
        )

    def fit_length(self, queries):
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        B, N, C = queries.shape
        out, attn = self.attn(self.fit_length(queries))
        out = out[:, :N, :]
        return out, attn

class TwoStageAttentionLayer(nn.Module):
    '''Two Stage Attention (TSA) Layer'''
    def __init__(self, configs, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=True),
            d_model, n_heads, alpha=5.0, top_k_ratio=0.1
        )
        self.dim_sender = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=True),
            d_model, n_heads, alpha=5.0, top_k_ratio=0.1
        )
        self.dim_receiver = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=True),
            d_model, n_heads, alpha=5.0, top_k_ratio=0.1
        )
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(time_in, time_in, time_in, attn_mask=None, tau=None, delta=None)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
        return final_out, attn




















