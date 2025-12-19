import math
import torch


@torch.compile
def _flash_backward_core(Q, K, V, O, L, dO, is_causal):
    B, Nq, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    D = torch.sum(O * dO, dim=-1)

    S = torch.matmul(Q, K.transpose(-1, -2)) * scale

    if is_causal:
        mask = torch.tril(torch.ones_like(S), diagonal=0)
        S = S.masked_fill(mask == 0, -1e6)

    P = torch.exp(S - L.unsqueeze(-1))

    dV = torch.matmul(P.transpose(-1, -2), dO)
    dP = torch.matmul(dO, V.transpose(-1, -2))
    dS = P * (dP - D.unsqueeze(-1))
    dQ = torch.matmul(dS, K) * scale
    dK = torch.matmul(dS.transpose(-1, -2), Q) * scale

    return dQ, dK, dV

class FlashAttention2Algorithm(torch.autograd.Function):
    """
    Strict Algorithm-1 style FlashAttention-2 forward (single-head).
    Inputs:
        Q, K, V: (B, L, D)
    Returns:
        O: (B, L, D)
        L: (B, L)   # logsumexp per query row
    Backward is not implemented (raises NotImplementedError).
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False, *, Bq: int = 16, Bk: int = 16):
        if is_causal:
            raise NotImplementedError("is_causal=True not supported in this implementation.")

        # Basic shape checks
        assert Q.dim() == 3 and K.dim() == 3 and V.dim() == 3, "Q/K/V must be (B, L, D)"
        B, Nq, D = Q.shape
        _, Nk, Dk = K.shape
        assert D == Dk and V.shape == (B, Nk, D), "K/V shape mismatch"

        device = Q.device
        dtype = Q.dtype
        scale = 1.0 / math.sqrt(D)

        # Outputs
        O = torch.zeros((B, Nq, D), dtype=dtype, device=device)
        Lvec = torch.empty((B, Nq), dtype=dtype, device=device)

        # Number of tiles (for readability; we'll iterate by ranges)
        # Tq = ceil(Nq / Bq), Tk = ceil(Nk / Bk)

        # Outer loop: for each query tile i
        for i_start in range(0, Nq, Bq):
            i_end = min(i_start + Bq, Nq)
            Qi = Q[:, i_start:i_end, :]                # (B, Bq_i, D)
            Bq_i = Qi.size(1)

            # Initialize O_i^(0), l_i^(0), m_i^(0)
            O_i_prev = torch.zeros((B, Bq_i, D), dtype=dtype, device=device)   # O^{(0)}_i
            l_i_prev = torch.zeros((B, Bq_i), dtype=dtype, device=device)      # l^{(0)}_i
            m_i_prev = torch.full((B, Bq_i), -float('inf'), dtype=dtype, device=device)  # m^{(0)}_i = -inf

            # Inner loop: scan over key tiles j
            for j_start in range(0, Nk, Bk):
                j_end = min(j_start + Bk, Nk)
                K_j = K[:, j_start:j_end, :]   # (B, Bk_j, D)
                V_j = V[:, j_start:j_end, :]   # (B, Bk_j, D)
                Bk_j = K_j.size(1)

                # Compute S_i^{(j)} = Qi @ K_j^T / sqrt(d)
                # shapes: Qi (B, Bq_i, D), K_j (B, Bk_j, D) -> scores (B, Bq_i, Bk_j)
                scores = torch.matmul(Qi, K_j.transpose(-1, -2)) * scale  # (B, Bq_i, Bk_j)

                # mj = rowmax(S_i^{(j)}) over keys (last dim)
                mj = scores.amax(dim=-1)  # (B, Bq_i)

                # m^{(j)}_i = max(m^{(j-1)}_i, mj)
                m_i_new = torch.maximum(m_i_prev, mj)  # (B, Bq_i)

                # compute tilde P^{(j)}_i = exp(S - m^{(j)}_i)
                # subtract m_i_new along last dim
                exp_scores = torch.exp(scores - m_i_new.unsqueeze(-1))  # (B, Bq_i, Bk_j)

                # rowsum(tildeP)
                rowsum_tildeP = exp_scores.sum(dim=-1)  # (B, Bq_i)

                # compute scale factors: exp(m^{(j-1)} - m^{(j)})
                # note: when m_i_prev == -inf, exp(-inf) -> 0, that's correct for initialization
                exp_mprev_minus_mnew = torch.exp(m_i_prev - m_i_new)  # (B, Bq_i)

                # update l: l^{(j)} = exp(m_prev - m_new) * l_prev + rowsum(tildeP)
                l_i_new = exp_mprev_minus_mnew * l_i_prev + rowsum_tildeP  # (B, Bq_i)

                # update O accumulator:
                # O^{(j)} = diag(exp(m_prev-m_new)) * O^{(j-1)} + tildeP @ V_j
                # compute tildeP @ V_j -> (B, Bq_i, D)
                # exp factor needs unsqueeze for broadcasting to last dim
                weighted_v = torch.matmul(exp_scores, V_j)  # (B, Bq_i, D)
                O_i_new = O_i_prev * exp_mprev_minus_mnew.unsqueeze(-1) + weighted_v  # (B, Bq_i, D)

                # move to next j: set prev <- new
                m_i_prev = m_i_new
                l_i_prev = l_i_new
                O_i_prev = O_i_new

            # after finishing all key tiles j, finalize:
            # O_i = O^{(Tk)}_i / l^{(Tk)}_i
            # L_i = m^{(Tk)}_i + log(l^{(Tk)}_i)
            # watch out for numerical issues: l_i_prev should be > 0 (unless degenerate)
            O_i = O_i_prev / l_i_prev.unsqueeze(-1)       # (B, Bq_i, D)
            L_i = m_i_prev + torch.log(l_i_prev)          # (B, Bq_i)

            # write outputs into global O and Lvec
            O[:, i_start:i_end, :] = O_i
            Lvec[:, i_start:i_end] = L_i

        # Save for backward (as assignment requests)
        ctx.save_for_backward(Q, K, V, O, Lvec)
        ctx.Bq = Bq
        ctx.Bk = Bk
        ctx.scale = scale
        ctx.is_causal = is_causal

        return O


    @staticmethod
    def backward(ctx, *grad_outputs):
        Q, K, V, O, L = ctx.saved_tensors
        dO = grad_outputs[0]
        dQ, dK, dV = _flash_backward_core(
            Q, K, V, O, L, dO, ctx.is_causal
        )
        return dQ, dK, dV, None
