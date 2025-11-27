
import torch
import torch.nn as nn

from cross_entropy_loss import cross_entropy
from get_batch import get_batch
def evaluate_model(model: nn.Module, dataset, device, batch_size, context_length, num_batches=10):
    """
    在验证集上评估模型性能
    
    Args:
        model: 要评估的模型
        dataset: 验证数据集，一维token序列
        device: 计算设备
        batch_size: 批次大小
        context_length: 上下文长度
        num_batches: 要评估的批次数量
        
    Returns:
        float: 验证集上的平均损失
    """
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    with torch.no_grad():  # 不计算梯度以节省内存
        for _ in range(num_batches):
            # 从验证集中获取一批数据
            inputs, targets = get_batch(
                dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device
            )
            # 前向传播
            logits = model(inputs)
            # 计算损失
            loss = cross_entropy(logits, targets)
            total_loss += loss.item()
    
    model.train()  # 恢复为训练模式
    return total_loss / num_batches  # 返回平均损失