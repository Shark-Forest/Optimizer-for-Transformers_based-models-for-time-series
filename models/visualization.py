import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 缓存无加权基准（前向和反向分别缓存）
global_forward_attn_without_weight = None  # 前向无加权基准
global_backward_attn_without_weight = None  # 反向无加权基准

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def visualize_forward_attn(attn_matrix, epoch, save_dir="attn_visualizations", prefix="forward_with_weight"):
    """可视化前向传播的注意力权重"""
    os.makedirs(save_dir, exist_ok=True)
    H, L, S = attn_matrix.shape
    attn_mean = attn_matrix.mean(axis=0)  # 多头平均

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn_mean,
        cmap="viridis",
        xticklabels=10,
        yticklabels=10,
        cbar_kws={"label": "前向注意力权重"}
    )
    plt.xlabel("Key时间步")
    plt.ylabel("Query时间步")
    plt.title(f"{prefix} - 前向注意力热力图（epoch {epoch}）")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"前向注意力热力图已保存至：{save_path}")


# 关键：确保该函数被正确定义，名称无拼写错误
def visualize_backward_attn(attn_sharp_matrix, epoch, save_dir="attn_visualizations", prefix="backward_with_weight"):
    """可视化反向传播中用于梯度加权的注意力权重（锐化后）"""
    os.makedirs(save_dir, exist_ok=True)
    # attn_sharp_matrix形状：[B, H, L, S]（反向传播中保存的权重）
    # 取第一个样本、多头平均，简化可视化
    attn_sharp_mean = attn_sharp_matrix[0].mean(axis=0)  # [L, S]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn_sharp_mean,
        cmap="plasma",  # 用不同色系区分前向和反向
        xticklabels=10,
        yticklabels=10,
        cbar_kws={"label": "反向梯度加权权重（锐化后）"}
    )
    plt.xlabel("Key时间步")
    plt.ylabel("Query时间步")
    plt.title(f"{prefix} - 反向传播注意力热力图（epoch {epoch}）")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"反向传播注意力热力图已保存至：{save_path}")