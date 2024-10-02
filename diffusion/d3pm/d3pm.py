# Largely Adapted from Google's JAX Implementation: https://github.com/google-research/google-research/blob/master/d3pm/images/diffusion_categorical.py
from typing import List, Literal

import pydantic
from torch import nn
import torch
import torch.nn.functional as F

class D3PMConfig(pydantic.BaseModel):

    sched_type: str = 'jsd'
    transition_mat_type: str = 'absorbing'
    num_states: int = 5
    beta_min: float = 0.01
    beta_max: float = 1.0
    loss_lambda: float = 0.01

    diffusion_steps: int = 256


class D3PM(nn.Module):

    def __init__(self, config: D3PMConfig):
        super().__init__()

        self.config = config
        cfg = config

        self.betas = self.get_noise_schedule()
        self.eps = 1e-10



        if cfg.transition_mat_type == 'uniform':
            Q_mat_t = [self.get_unif_trans_mat(t) for t in range(cfg.diffusion_steps)]
        elif cfg.transition_mat_type == 'gaussian':
            Q_mat_t = [self.get_gaussian_trans_mat(t) for t in range(cfg.diffusion_steps)]
        elif cfg.transition_mat_type == 'absorbing':
            Q_mat_t = [self.get_absorbing_trans_mat(t) for t in range(cfg.diffusion_steps)]
        else:
            raise ValueError("Invalid Transition Matrix Type")

        self.Q_mat_t = torch.stack(Q_mat_t, dim=0)
        self.Q_mat_t_transposed = self.Q_mat_t.mT
        Q = self.Q_mat_t[0]
        Q_mats = [Q]
        for i in range(1, cfg.diffusion_steps):
            Q = Q @ self.Q_mat_t[i]
            Q_mats.append(Q)

        self.Q_mats = torch.stack(Q_mats, dim=0)
    def get_noise_schedule(self):
        cfg = self.config
        if cfg.sched_type == 'linear':
            return torch.linspace(cfg.beta_min, cfg.beta_max, cfg.diffusion_steps)
        elif cfg.sched_type == 'cosine':
            # use with 'uniform' transition_mat_type
            steps = torch.arange(cfg.diffusion_steps + 1, dtype=torch.float64) / cfg.diffusion_steps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
            return torch.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], torch.tensor(0.999, dtype=torch.float64))
        elif cfg.sched_type == 'jsd':
            # use with 'absorbing' transition_mat_type
            return 1. / torch.linspace(cfg.diffusion_steps, 1., cfg.diffusion_steps)
        else:
            raise NotImplementedError(cfg.sched_type)

    def get_unif_trans_mat(self, t):
        cfg = self.config
        beta_t = self.betas[t].item()
        Q = torch.full((cfg.num_states, cfg.num_states), beta_t / cfg.num_states)
        Q.fill_diagonal_(1 - beta_t + beta_t / cfg.num_states)

        return Q

    def get_gaussian_trans_mat(self, t):
        raise NotImplementedError

    def get_absorbing_trans_mat(self, t):
        cfg = self.config
        beta_t = self.betas[t].item()
        Q = torch.zeros((cfg.num_states, cfg.num_states))
        Q.fill_diagonal_(1 - beta_t)
        Q[:, cfg.num_states - 1] += beta_t

        return Q

    def extract_rows(self, A, x, t):
        # print(t.unsqueeze(-1))
        A = A.to(x.device)
        return A[t.unsqueeze(-1), x]

    def extract_rows_onehot(self, A, x, t):
        A = A.to(x.device)
        return (x.unsqueeze(-2) @ A[t.unsqueeze(-1).expand(t.shape[0], x.shape[1])]).squeeze()

    def q_marginal(self, x_start, t):
        # x_start is NOT one-hot
        # x_start has shape (bsize, seq len)
        # t has shape (bsize,)

        # self.Q_mats has shape (diffusion_steps, num_states, num_states)

        # output has shape (bsize, seq len, num_states)

        return self.extract_rows(self.Q_mats, x_start, t)

    def q_sample(self, x_start, t):
        q_probs = self.q_marginal(x_start, t)
        q_logprobs = torch.log(q_probs + self.eps)
        samples = F.gumbel_softmax(q_logprobs)
        return torch.argmax(samples, dim=-1)

    def q_posterior_logits(self, x_start, x_t, t, x_start_logits=False):
        cfg = self.config
        fact1 = torch.log(self.extract_rows(self.Q_mat_t_transposed, x_t, t) + self.eps)
        if x_start_logits:
            fact2 = self.extract_rows_onehot(self.Q_mats, torch.softmax(x_start, dim=-1), t-1)
            t0_logits = x_start
        else:
            fact2 = self.extract_rows(self.Q_mats, x_start, t-1)
            t0_logits = torch.log(F.one_hot(x_start, num_classes=cfg.num_states) + self.eps)
        
        fact2 = torch.log(fact2 + self.eps)
        posterior = fact1 + fact2
        
        t_broadcast = t.view(*t.shape, 1, 1).expand(t.shape[0], x_start.shape[1], -1) 
        return torch.where(t_broadcast == 0, t0_logits, posterior)

    def p_logits(self, model, x, t, mask):
        x = x * mask
        pred_x_start_logits = model(x, t)
        return self.q_posterior_logits(pred_x_start_logits, x, t, x_start_logits=True)

    def p_sample_one_tstep(self, model, x, t, mask):
        model_logits, pred_x_start_logits = self.p_logits(model, x, t, mask)
        t_broadcast = t.view(*t.shape, 1, 1).expand(t.shape[0], x.shape[1], -1) 
        samples = F.gumbel_softmax(model_logits)
        samples = torch.where(t_broadcast == 0, model_logits, samples)

        return torch.argmax(samples, dim=-1), torch.softmax(pred_x_start_logits, dim=-1)

    def p_sample_full(self, model, x, shape, mask):
        cfg = self.config
        if cfg.transition_mat_type in ['gaussian', 'uniform']:
            x_T = torch.randint(cfg.num_states, shape)
        elif cfg.transition_mat_type == 'absorbing':
            x_T = torch.full(shape, cfg.num_states - 1)
        else:
            raise ValueError("Invalid Transition Matrix Type")

        x = x_T
        for i in range(cfg.diffusion_steps):
            t = torch.full((shape[0],), cfg.diffusion_steps - 1 - i)
            x, _ = self.p_sample_one_tstep(model, x, t, mask)

        return x

    def v_bound_L_t(self, model, x_start, x_t, t, mask):
        cfg = self.config
        # convert x_start to logits
        # x_start_logits = torch.log(F.one_hot(x_start, cfg.num_states) + self.eps)
        # print("x_start_logits:", x_start_logits)
        target_logprobs = self.q_posterior_logits(x_start, x_t, t).log_softmax(dim=-1)
        pred_logprobs = self.p_logits(model, x_t, t, mask).log_softmax(dim=-1)

        kl = (F.kl_div(pred_logprobs, target_logprobs, reduction='none', log_target=True).sum(dim=-1) * mask).sum() / mask.sum()
        pred_logprobs = pred_logprobs.view(pred_logprobs.shape[1], pred_logprobs.shape[2], -1)
        nll = F.cross_entropy(pred_logprobs, (x_start - 1 + mask).T, ignore_index=-1)

        return torch.where(t == 0, nll, kl)

    def v_bound_prior(self, x_start, mask):
        cfg = self.config
        q_probs = self.q_marginal(x_start, torch.full((x_start.shape[0],), cfg.diffusion_steps - 1))

        if cfg.transition_mat_type in ['gaussian', 'uniform']:
            prior_probs = torch.full_like(q_probs, 1 / cfg.num_states)

        elif cfg.transition_mat_type == 'absorbing':
            prior_probs = torch.zeros_like(q_probs)
            prior_probs[:,:,-1] += 1.0

        else:
            raise ValueError("Invalid Transition Matrix Type")

        return (F.kl_div(prior_probs.log(), q_probs, reduction='none').sum(dim=-1) * mask).sum() / mask.sum()

    def full_v_bound(self, model, x_start, mask):
        cfg = self.config
        vb = 0
        for i in range(cfg.diffusion_steps):
            t = torch.full((x_start.shape[0],), i, device=x_start.device)
            x_t = self.q_sample(x_start, t)
            vb += self.v_bound_L_t(model, x_start, x_t, t, mask).mean()

        vb_prior = self.v_bound_prior(x_start, mask)

        full_vb = vb + vb_prior

        return vb_prior, vb, full_vb
