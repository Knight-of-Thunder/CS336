import math
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                m = state.get('m', torch.zeros_like(p.data))
                v = state.get('v', torch.zeros_like(p.data))
                t = state.get('t', 0)
                t += 1
                
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                lr = group['lr']

                # update first moment estimate
                m = beta1 * m + (1 - beta1) * grad
                # update second moment estimate
                v = beta2 * v + (1 - beta2) * (grad * grad)
                # bias correction
                lr_t = lr * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
                # parameter update
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                # weight decay
                p.data -= lr * weight_decay * p.data
                
                state['m'] = m
                state['v'] = v
                state['t'] = t


        return loss