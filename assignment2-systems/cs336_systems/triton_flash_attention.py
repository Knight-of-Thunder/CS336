import math
import triton
import triton.language as tl
import torch
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,  # 输入矩阵指针
    O_ptr, L_ptr,         # 输出矩阵指针
    # 各张量步长参数
    stride_qb, stride_qq, stride_qd,  # Q的批次/查询/特征维度步长
    stride_kb, stride_kk, stride_kd,  # K的批次/键/特征维度步长  
    stride_vb, stride_vk, stride_vd,  # V的批次/键/特征维度步长
    stride_ob, stride_oq, stride_od,  # O的批次/查询/特征维度步长
    stride_lb, stride_lq,             # L的批次/查询维度步长
    N_QUERIES, N_KEYS,               # 查询数和键值数
    scale,                            # 缩放因子1/sqrt(d)
    D: tl.constexpr,                  # 特征维度（编译期常量）
    Q_TILE_SIZE: tl.constexpr,        # 查询分块尺寸B_q
    K_TILE_SIZE: tl.constexpr,        # 键分块尺寸B_k 
):
    # 获取程序索引
    query_tile_index = tl.program_id(0)  # 查询区块索引
    batch_index = tl.program_id(1)       # 批次索引
    
    # 根据批次偏移量调整指针
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,  # 批次偏移后的指针
        shape=(N_QUERIES, D),            # 矩阵整体形状
        strides=(stride_qq, stride_qd),   # 行/列步长
        offsets=(query_tile_index * Q_TILE_SIZE, 0),  # 当前区块偏移
        block_shape=(Q_TILE_SIZE, D),     # 区块尺寸
        order=(1, 0)                      # 内存布局顺序（列优先）
    )

    # K, V, O block ptr
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load Qi
    Qi = tl.load(Q_block_ptr)  # (B_q, D)
    Qi = Qi.to(tl.float32)  # 转为float32以提高数值稳定性

    # Initialize m, l, O
    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)  # (B_q,)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)                 # (B_q,)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)              # (B_q, D)
    
    # Iterate over K, V tiles
    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)  # 键区块总数
    for k_tile_index in range(Tk):
        # Load Kj, Vj
        Kj = tl.load(K_block_ptr)  # (B_k, D)
        Vj = tl.load(V_block_ptr)  # (B_k, D)
        # Kj = Kj.to(tl.float32)
        # Vj = Vj.to(tl.float32)

        # Compute scaled dot-product scores S_ij
        # shapes: Qi (B_q, D), Kj (B_k, D) -> scores (B_q, B_k)
        S_ij = tl.dot(Qi, tl.trans(Kj)) * scale  # (B_q, B_k)

        m_ij = tl.maximum(m_i, tl.max(S_ij, axis=1))  # (B_q,)

        # Compute exp(S_ij - m_ij)
        P_ij = tl.exp(S_ij - m_ij[:, None])  # (B_q, B_k)

        # Compute rowsum(tildeP)
        rowsum_tildeP = tl.sum(P_ij, axis=1)  # (B_q,)
        # Compute scale factors: exp(m_i - m_ij)
        exp_mi_minus_mij = tl.exp(m_i - m_ij)  # (B_q,)
        # Update l_i
        l_ij = exp_mi_minus_mij * l_i + rowsum_tildeP
        # Update O_i
        O_i = (exp_mi_minus_mij[:, None] * O_i)  # (B_q, D)
        O_i = tl.dot(P_ij.to(Vj.dtype), Vj, acc=O_i)
        # Update m_i, l_i for next iteration
        m_i = m_ij
        l_i = l_ij 
        # Advance K, V block pointers to next tile
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    O_i = O_i.to(tl.float16)  # 转回float16以节省存储
    L_i = m_i + tl.log(l_i)  # (B_q,)
    
    # Store O_i and l_i

    tl.store(L_block_ptr, L_i.to(L_block_ptr.type.element_ty))  # reshape to (1, Q_TILE_SIZE)

    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty))
    # tl.store(L_ptr_out, L_i)


class TritonFlashAttention2Algorithm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, Nq, D = Q.shape
        _, Nk, _ = K.shape

        # Output tensors
        O = torch.empty((B, Nq, D), device=Q.device, dtype=Q.dtype)
        L = torch.empty((B, Nq), device=Q.device, dtype=Q.dtype)

        # Define tile sizes
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64

        # Calculate grid dimensions
        grid_q = (Nq + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        grid_b = B

        # Launch Triton kernel
        flash_fwd_kernel[(grid_q, grid_b)](
            Q_ptr=Q,
            K_ptr=K,
            V_ptr=V,
            O_ptr=O,
            L_ptr=L,
            stride_qb=Q.stride(0),
            stride_qq=Q.stride(1),
            stride_qd=Q.stride(2),
            stride_kb=K.stride(0),
            stride_kk=K.stride(1),
            stride_kd=K.stride(2),
            stride_vb=V.stride(0),
            stride_vk=V.stride(1),
            stride_vd=V.stride(2),
            stride_ob=O.stride(0),
            stride_oq=O.stride(1),
            stride_od=O.stride(2),
            stride_lb=L.stride(0),
            stride_lq=L.stride(1),
            N_QUERIES=Nq,
            N_KEYS=Nk,
            scale=1.0 / math.sqrt(D),
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )

        ctx.save_for_backward(Q, K, V, L)
        ctx.scale = 1.0 / math.sqrt(D)
        ctx.is_causal = is_causal

        return O