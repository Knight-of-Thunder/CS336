import torch
import config
from evaluate_model import evaluate_model
from get_batch import get_batch
from AdamW import AdamW
from gradient_clip import gradient_clip, compute_grad_norm
from lr_cosine_schedule import lr_cosine_schedule
from cross_entropy_loss import cross_entropy
from check_point import save_checkpoint, load_checkpoint

from ..Model.TransformerLM import TransformerLM

# def train(model, optimizer, train_data, val_data, paths):

import numpy as np
import numpy.typing as npt
import wandb
from tqdm import tqdm
from loguru import logger

if __name__ == "__main__":
    logger.add("./data/log/train_v0.log", rotation="1 day", retention="7 days", level="INFO")

    # # 设置所有超参数
    # # 模型参数
    # model_config = {
    #     "vocab_size": 10000,      # 词汇表大小
    #     "context_length": 256,    # 上下文长度
    #     "num_layers": 4,          # Transformer Block数
    #     "num_heads": 16,          # 注意力头数
    #     "d_model": 512,           # 嵌入空间维度
    #     "d_ff": 1344,             # 前馈网络维度
    #     "rope_theta": 10000,      # RoPE参数
    # }
    
    # # 优化器参数
    # optim_config = {
    #     "lr": 3e-4,               # 学习率
    #     "weight_decay": 1e-2,     # 权重衰减
    #     "betas": (0.9, 0.999),    # AdamW的beta参数
    #     "max_norm": 1.0,          # 梯度裁剪的最大范数
    # }
    
    # # 训练参数
    # train_config = {
    #     "batch_size": 16,         # 批次大小
    #     "total_epochs": 0.5,      # 训练轮数
    #     "checkpoint_freq": 2000,  # 每隔多少步保存一次检查点
    #     "log_freq": 10,           # 每隔多少步记录一次日志
    #     "val_freq": 400,          # 每隔多少步在验证集上评估
    #     "val_batch_size": 16,     # 验证时的批次大小
    #     "val_batches": 20,        # 验证时使用的批次数量
    # }
    
    # # 数据路径
    # data_paths = {
    #     "training_dataset_path": "./data/token/TinyStories_train_10000_token_ids.npy",
    #     "validation_dataset_path": "./data/token/TinyStories_valid_10000_token_ids.npy",  # 验证集路径
    #     "checkpoint_load_path": None,  # 模型检查点路径
    #     "checkpoint_save_format": "./data/model/checkpoint_v0_{}.pt",  # 检查点保存路径格式
    #     "final_model_path": "./data/model/final_model_v0.pt",  # 最终模型保存路径
    # }
    
    # 初始化wandb
    run = wandb.init(
        project="cs336-assignment-1",
        name="train_v1",
        config={
            "model": config.model,
            "optimizer": config.optimizer,
            "training": config.train,
        }
    )
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    logger.info("开始初始化模型...")
    model = TransformerLM(
        vocab_size=config.model["vocab_size"],
        context_length=config.model["context_length"],
        num_layers=config.model["num_layers"],
        num_heads=config.model["num_heads"],
        d_model=config.model["d_model"],
        d_ff=config.model["d_ff"],
        rope_theta=config.model["rope_theta"],
        # device=device,
    )
    logger.info("模型初始化完成。")

    # 初始化优化器
    logger.info("开始初始化优化器...")
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer["lr"],
        weight_decay=config.optimizer["weight_decay"],
        betas=config.optimizer["betas"],
    )
    logger.info("优化器初始化完成。")

    # 如果有checkpoint，则加载checkpoint
    start_iter = 1
    if config.paths["checkpoint_load_path"]:
        logger.info(f"开始加载模型检查点: {config.paths['checkpoint_load_path']}")
        start_iter = load_checkpoint(
            config.paths["checkpoint_load_path"],
            model=model,
            optimizer=optimizer
        )
        start_iter += 1
        logger.info(f"模型检查点加载成功，当前迭代次数: {start_iter}")
    else:
        logger.info("没有提供模型检查点，开始从头训练。")
    
    # 加载数据集
    logger.info(f"开始加载数据集，训练集：{config.paths['training_dataset_path']}, 验证集：{config.paths['validation_dataset_path']}")
    training_dataset = np.load(config.paths['training_dataset_path'], mmap_mode='r+') # 使用内存映射
    validation_dataset = None
    if config.paths['validation_dataset_path']:
        validation_dataset = np.load(config.paths['validation_dataset_path'], mmap_mode='r+')
    logger.info("数据集加载完成")

    # 计算训练所需step
    total_tokens = training_dataset.shape[0]
    total_steps = int(config.train["total_epochs"] * total_tokens) // (config.train["batch_size"] * config.train["context_length"])
    logger.info(f"总token数: {total_tokens}, 训练轮数: {config.train['total_epochs']}, batch大小: {config.train['batch_size']}, 上下文长度: {config.train['context_length']}")
    logger.info(f"总训练步数: {total_steps}")

    # step循环开始
    logger.info("开始训练模型...")
    for step in tqdm(range(start_iter, total_steps + 1), desc="训练进度", unit="step"):
        # 清空梯度
        optimizer.zero_grad()

        # 使用余弦退火更新学习率
        it=step,
        max_learning_rate=config.optimizer["lr"],
        min_learning_rate=config.optimizer["lr"] * 0.01,
        warmup_iters=int(0.05 * total_steps),
        cosine_cycle_iters=total_steps,
        lr_now = lr_cosine_schedule(
            t = it,
            a_max=max_learning_rate,
            a_min=min_learning_rate,
            T_w=warmup_iters,
            T_c=cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now
        
        # 获取batch数据
        inputs, targets = get_batch(
            training_dataset,
            batch_size=config.train["batch_size"],
            context_length=config.train["context_length"],
            device="cuda:0" if device.type == "cuda" else "cpu"
        )

        # 前向传播
        logits = model(inputs)

        # 计算损失
        loss = cross_entropy(logits, targets)

        # 反向传播和优化参数
        loss.backward()
        
        # 计算梯度的 L2 范数
        if step % config.train["log_freq"] == 0:
            grad_norm = compute_grad_norm(model.parameters())
        
        gradient_clip(model.parameters(), max_l2_norm=config.train["max_norm"]) # 梯度裁剪
        optimizer.step()

        # 日志记录
        if step % config.train["log_freq"] == 0:
            logger.info(f"Step {step}, Loss: {loss.item()}, Grad L2 Norm: {grad_norm}")

            # 使用wandb记录损失和梯度范数
            wandb.log({"train_loss": loss.item(), "lr": lr_now, "grad_l2_norm": grad_norm, "step": step})
        
        # 在验证集上评估模型
        if validation_dataset is not None and step % config.train["val_freq"] == 0:
            logger.info(f"在验证集上评估模型...")
            val_loss = evaluate_model(
                model=model,
                dataset=validation_dataset,
                device=device,
                batch_size=config.train["val_batch_size"],
                context_length=config.train["context_length"],
                num_batches=config.train["val_batches"]
            )
            logger.info(f"验证集损失: {val_loss}")
            wandb.log({"val_loss": val_loss, "step": step})
        
        # 保存检查点
        if step % config.train["checkpoint_freq"] == 0:
            checkpoint_save_path = config.paths["checkpoint_save_format"].format(step)
            logger.info(f"正在保存模型检查点到: {checkpoint_save_path}")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=checkpoint_save_path
            )
            logger.info("模型检查点保存成功。")
    logger.info("模型训练完成。")
    
    # 保存最终模型
    logger.info(f"正在保存最终模型到: {config.paths["final_model_path"]}")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=total_steps,
        out=config.paths["final_model_path"],
    )
    logger.info("最终模型保存成功。")
    
    # 关闭wandb
    wandb.finish()