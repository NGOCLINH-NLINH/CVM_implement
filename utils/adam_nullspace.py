import math
import torch
from torch.optim.optimizer import Optimizer


class Adam_NullSpace(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, svd=False, thres=0.98, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, svd=svd, thres=thres)
        super(Adam_NullSpace, self).__init__(params, defaults)

        self.covariances = {}
        self.projectors = {}

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            svd = group['svd']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                update = self.get_update(group, grad, p)

                if svd and p in self.projectors:
                    update_ = torch.mm(update, self.projectors[p])
                else:
                    update_ = update

                p.data.add_(update_)
        return loss

    def update_null_space(self, fea_in):
        for group in self.param_groups:
            if not group['svd']:
                continue
            thres = group['thres']
            for p in group['params']:
                if not p.requires_grad:
                    continue

                if p not in self.covariances:
                    self.covariances[p] = fea_in[p].clone().to(p.device)
                else:
                    self.covariances[p] += fea_in[p].to(p.device)

                _, S, V = torch.svd(self.covariances[p])

                cumulative_sum = S.cumsum(dim=0) / S.sum()
                idxs = (cumulative_sum >= thres).nonzero(as_tuple=True)[0]
                num_vectors = idxs[0].item() + 1 if len(idxs) > 0 else S.shape[0]
                basis = V[:, :num_vectors]

                identity = torch.eye(basis.shape[0], device=basis.device)
                P = identity - torch.mm(basis, basis.transpose(1, 0))

                self.projectors[p] = P.detach()
                print(f"      -> Lock {num_vectors}/{S.shape[0]} dims space for this param")

    def get_update(self, group, grad, p):
        state = self.state[p]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)
            state['exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']
        state['step'] += 1

        if group['weight_decay'] != 0:
            grad.add_(p.data, alpha=float(group['weight_decay']))

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
        update = - step_size * exp_avg / denom
        return update