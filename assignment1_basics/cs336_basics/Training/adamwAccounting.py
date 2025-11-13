"""
AdamW 训练内存占用计算

根据以下假设计算峰值内存：
- 所有张量均为 FP32（4 字节）
- Embedding 和 lm_head 权重共享
- d_ff = 4 * d_model
"""

def calculate_adamw_memory(
    vocab_size,
    context_length,
    num_layers,
    d_model,
    num_heads,
    batch_size,
    d_ff=None,
    verbose=True
):
    """
    计算 AdamW 训练所需的峰值内存。
    
    Args:
        vocab_size: 词表大小 (V)
        context_length: 序列长度 (S)
        num_layers: Transformer 层数 (L)
        d_model: 模型维度 (D)
        num_heads: 注意力头数 (H)
        batch_size: 批量大小 (B)
        d_ff: FFN 中间维度，默认为 4 * d_model (F)
        verbose: 是否打印详细信息
    
    Returns:
        dict: 包含各部分内存占用的字典
    """
    
    # 设置默认值
    if d_ff is None:
        d_ff = 4 * d_model
    
    V = vocab_size
    S = context_length
    L = num_layers
    D = d_model
    H = num_heads
    B = batch_size
    F = d_ff
    
    # 验证参数合理性
    if D % H != 0:
        raise ValueError(f"d_model ({D}) 必须能被 num_heads ({H}) 整除")
    
    d_h = D // H  # 每头维度
    
    # ========== 参数内存 ==========
    # P_total = L(2D + 4D^2 + 3DF) + D + VD
    
    # 每层参数
    params_per_layer = 2*D + 4*D**2 + 3*D*F
    
    # RMSNorm scales (2 per layer): 2D per layer
    params_rmsnorm_per_layer = 2 * D
    
    # MHSA (W_Q, W_K, W_V, W_O): 4D^2
    params_mhsa_per_layer = 4 * D**2
    
    # FFN (linear1, linear3, linear2): 3DF
    params_ffn_per_layer = 3 * D * F
    
    # 所有层参数
    params_all_layers = L * params_per_layer
    
    # 最终 RMSNorm scale
    params_ln_final = D
    
    # Embedding 和 lm_head 共享权重
    params_embed_lm = V * D
    
    # 总参数数
    params_total = params_all_layers + params_ln_final + params_embed_lm
    
    # 参数内存（字节）
    bytes_per_param = 4  # FP32
    memory_params = params_total * bytes_per_param
    
    # ========== 激活值内存 ==========
    # M_activations = L(16BSD + 2BHS^2) + BSD + 2BSV
    
    # 每层激活值
    activations_per_layer_elem = 16*B*S*D + 2*B*H*S**2
    
    # 分解：
    # - 每层 RMSNorm、MHSA、FFN 的激活值：16BSD
    activations_linear_per_layer = 16 * B * S * D
    
    # - 注意力矩阵 Q^T K 和 softmax 权重：2BHS^2
    activations_attention_matrix_per_layer = 2 * B * H * S**2
    
    # 所有层激活值
    activations_all_layers = L * activations_per_layer_elem
    
    # 最终 RMSNorm 激活值
    activations_ln_final = B * S * D
    
    # Logits（lm_head 输出）
    activations_logits = B * S * V
    
    # 交叉熵计算中的激活值
    activations_ce = B * S * V
    
    # 总激活值元素数
    activations_total_elem = activations_all_layers + activations_ln_final + activations_logits + activations_ce
    
    memory_activations = activations_total_elem * bytes_per_param
    
    # ========== 梯度内存 ==========
    # 梯度与参数形状相同
    memory_gradients = memory_params
    
    # ========== 优化器状态内存（AdamW） ==========
    # AdamW 为每个参数维护 m (一阶动量) 和 v (二阶动量)
    # 共 2 倍参数大小
    memory_optimizer = 2 * memory_params
    
    # ========== 总峰值内存 ==========
    memory_total = memory_params + memory_activations + memory_gradients + memory_optimizer
    
    # ========== 输出结果 ==========
    if verbose:
        print("=" * 80)
        print("AdamW 训练峰值内存计算")
        print("=" * 80)
        print(f"\n【模型配置】")
        print(f"  词表大小 (V)           : {V:,}")
        print(f"  序列长度 (S)           : {S:,}")
        print(f"  Transformer 层数 (L)   : {L}")
        print(f"  模型维度 (D)           : {D}")
        print(f"  注意力头数 (H)         : {H}")
        print(f"  每头维度 (d_h = D/H)   : {d_h}")
        print(f"  FFN 中间维度 (F = 4D)  : {F}")
        print(f"  批量大小 (B)           : {B}")
        
        print(f"\n【参数内存】")
        print(f"  {'-' * 76}")
        print(f"  组件                           参数数量              内存 (GB)")
        print(f"  {'-' * 76}")
        
        # 每层参数分解
        print(f"  每层 RMSNorm scales")
        print(f"    (2D per layer, L layers)     {params_rmsnorm_per_layer * L:>15,}      {params_rmsnorm_per_layer * L * bytes_per_param / 1e9:>10.4f}")
        
        print(f"  每层 MHSA 投影")
        print(f"    (W_Q, W_K, W_V, W_O)         {params_mhsa_per_layer * L:>15,}      {params_mhsa_per_layer * L * bytes_per_param / 1e9:>10.4f}")
        
        print(f"  每层 FFN 线性层")
        print(f"    (linear1, linear3, linear2)  {params_ffn_per_layer * L:>15,}      {params_ffn_per_layer * L * bytes_per_param / 1e9:>10.4f}")
        
        print(f"  最终 RMSNorm scale             {params_ln_final:>15,}      {params_ln_final * bytes_per_param / 1e9:>10.4f}")
        
        print(f"  Embedding + lm_head (共享)     {params_embed_lm:>15,}      {params_embed_lm * bytes_per_param / 1e9:>10.4f}")
        
        print(f"  {'-' * 76}")
        print(f"  参数总数                       {params_total:>15,}      {memory_params / 1e9:>10.4f}")
        print(f"  {'-' * 76}")
        
        print(f"\n【激活值内存】")
        print(f"  {'-' * 76}")
        print(f"  组件                           元素数量              内存 (GB)")
        print(f"  {'-' * 76}")
        
        # 每层激活值分解
        print(f"  每层线性激活值 (RMSNorm, MHSA, FFN)")
        print(f"    (16BSD per layer, L layers)  {activations_linear_per_layer * L:>15,}      {activations_linear_per_layer * L * bytes_per_param / 1e9:>10.4f}")
        
        print(f"  每层注意力矩阵激活值")
        print(f"    (Q^T K, softmax weights)     {activations_attention_matrix_per_layer * L:>15,}      {activations_attention_matrix_per_layer * L * bytes_per_param / 1e9:>10.4f}")
        
        print(f"  最终 RMSNorm 激活值            {activations_ln_final:>15,}      {activations_ln_final * bytes_per_param / 1e9:>10.4f}")
        
        print(f"  Logits (lm_head 输出)          {activations_logits:>15,}      {activations_logits * bytes_per_param / 1e9:>10.4f}")
        
        print(f"  交叉熵计算激活值               {activations_ce:>15,}      {activations_ce * bytes_per_param / 1e9:>10.4f}")
        
        print(f"  {'-' * 76}")
        print(f"  激活值总数                     {activations_total_elem:>15,}      {memory_activations / 1e9:>10.4f}")
        print(f"  {'-' * 76}")
        
        print(f"\n【梯度内存】")
        print(f"  {'-' * 76}")
        print(f"  梯度与参数形状相同             {params_total:>15,}      {memory_gradients / 1e9:>10.4f}")
        print(f"  {'-' * 76}")
        
        print(f"\n【优化器状态内存 (AdamW)】")
        print(f"  {'-' * 76}")
        print(f"  一阶动量 (m)                   {params_total:>15,}      {memory_params / 1e9:>10.4f}")
        print(f"  二阶动量 (v)                   {params_total:>15,}      {memory_params / 1e9:>10.4f}")
        print(f"  {'-' * 76}")
        print(f"  优化器状态总数                 {2 * params_total:>15,}      {memory_optimizer / 1e9:>10.4f}")
        print(f"  {'-' * 76}")
        
        print(f"\n【峰值内存总计】")
        print(f"  {'-' * 76}")
        print(f"  参数                                                 {memory_params / 1e9:>10.4f} GB")
        print(f"  激活值                                               {memory_activations / 1e9:>10.4f} GB")
        print(f"  梯度                                                 {memory_gradients / 1e9:>10.4f} GB")
        print(f"  优化器状态                                           {memory_optimizer / 1e9:>10.4f} GB")
        print(f"  {'-' * 76}")
        print(f"  总计                                                 {memory_total / 1e9:>10.4f} GB")
        print(f"  {'-' * 76}\n")
    
    # 返回结果字典
    return {
        "params": {
            "rmsnorm": params_rmsnorm_per_layer * L,
            "mhsa": params_mhsa_per_layer * L,
            "ffn": params_ffn_per_layer * L,
            "ln_final": params_ln_final,
            "embed_lm_head": params_embed_lm,
            "total": params_total,
            "bytes": memory_params
        },
        "activations": {
            "linear_per_layer": activations_linear_per_layer * L,
            "attention_matrix": activations_attention_matrix_per_layer * L,
            "ln_final": activations_ln_final,
            "logits": activations_logits,
            "ce": activations_ce,
            "total_elem": activations_total_elem,
            "bytes": memory_activations
        },
        "gradients": {
            "total": params_total,
            "bytes": memory_gradients
        },
        "optimizer": {
            "m": params_total,
            "v": params_total,
            "total": 2 * params_total,
            "bytes": memory_optimizer
        },
        "total_bytes": memory_total,
        "total_gb": memory_total / 1e9
    }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("示例 1: 你的测试配置")
    print("=" * 80)
    result1 = calculate_adamw_memory(
        vocab_size=10_000,
        context_length=16,
        num_layers=3,
        d_model=64,
        num_heads=4,
        batch_size=4,
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("示例 2: GPT-2 XL 配置")
    print("=" * 80)
    result2 = calculate_adamw_memory(
        vocab_size=50_257,
        context_length=1_024,
        num_layers=48,
        d_model=1_600,
        num_heads=25,
        batch_size=4,
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("示例 3: 小模型配置（快速测试）")
    print("=" * 80)
    result3 = calculate_adamw_memory(
        vocab_size=5_000,
        context_length=512,
        num_layers=6,
        d_model=256,
        num_heads=8,
        batch_size=8,
        verbose=True
    )
