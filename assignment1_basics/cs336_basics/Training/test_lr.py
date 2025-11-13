from collections.abc import Callable, Iterable 
from typing import Optional 
import torch 
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        """初始化SGD优化器
        
        参数:
            params: 需要优化的参数(可迭代对象)
            lr: 初始学习率(默认1e-3)
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """执行单次参数更新
        
        参数:
            closure: 用于重新计算损失的可调用函数(可选)
        返回:
            计算出的损失值(如果提供了closure)
        """
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]  # 获取当前学习率
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                state = self.state[p]  # 获取参数p的状态字典
                t = state.get("t", 0)  # 从状态中获取迭代次数，默认为0
                grad = p.grad.data  # 获取损失函数关于p的梯度
                
                # 执行带学习率衰减的更新(核心公式)
                p.data -= lr / math.sqrt(t + 1) * grad  
                
                state["t"] = t + 1  # 更新迭代计数器

        return loss
    

if __name__ == "__main__":
    # 创建可训练参数 (10x10随机矩阵，初始标准差为5)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))

    # 初始化SGD优化器(学习率设为1)
    opt = SGD([weights], lr=100)

    # 训练循环
    for t in range(100):
        opt.zero_grad()                  # 重置所有可训练参数的梯度
        loss = (weights**2).mean()       # 计算标量损失值(示例使用L2正则)
        print(loss.cpu().item())         # 打印当前损失值
        loss.backward()                  # 反向传播计算梯度
        opt.step()                       # 执行优化器更新