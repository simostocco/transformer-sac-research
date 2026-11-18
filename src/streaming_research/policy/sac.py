"""Extracted research definitions; see docs/source_map.json for provenance."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimBaBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 2):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * int(expansion))
        self.fc2 = nn.Linear(dim * int(expansion), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        return x + y


class SimBaMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, depth: int = 4, expansion: int = 2):
        super().__init__()
        self.input_fc = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([SimBaBlock(hidden, expansion=expansion) for _ in range(int(depth))])
        self.out_ln = nn.LayerNorm(hidden)
        self.output_fc = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_fc(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.out_ln(x)
        return self.output_fc(x)


class SimBaStateEncoder(nn.Module):
    def __init__(self, s_dim: int, k: int = 128, hidden: int = 512, depth: int = 2, expansion: int = 2):
        super().__init__()
        self.net = SimBaMLP(s_dim, k, hidden=hidden, depth=depth, expansion=expansion)
        self.ln = nn.LayerNorm(k)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.ln(self.net(s))


class SimBaActionEncoder(nn.Module):
    def __init__(self, a_dim: int = 1, k: int = 128, hidden: int = 128, depth: int = 2, expansion: int = 2):
        super().__init__()
        self.net = SimBaMLP(a_dim, k, hidden=hidden, depth=depth, expansion=expansion)
        self.ln = nn.LayerNorm(k)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.ln(self.net(a))


class QNetInteraction(nn.Module):
    """
    SimBa-based critic:
      s_feat = SEnc(s)
      a_feat = AEnc(a)
      comb   = [s_feat, a_feat, s_feat*a_feat, s_feat-a_feat]
      q      = SimBaMLP(comb -> 1)
    """
    def __init__(
        self,
        s_dim: int,
        a_dim: int = 1,
        k: int = 128,
        s_hidden: int = 512,
        s_depth: int = 2,
        a_hidden: int = 128,
        a_depth: int = 2,
        q_hidden: int = 512,
        q_depth: int = 4,
        expansion: int = 2,
    ):
        super().__init__()
        self.s_enc = SimBaStateEncoder(s_dim, k=k, hidden=s_hidden, depth=s_depth, expansion=expansion)
        self.a_enc = SimBaActionEncoder(a_dim=a_dim, k=k, hidden=a_hidden, depth=a_depth, expansion=expansion)
        comb_dim = k * 4
        self.q = SimBaMLP(comb_dim, 1, hidden=q_hidden, depth=q_depth, expansion=expansion)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        s_feat = self.s_enc(s)
        a_feat = self.a_enc(a)
        comb = torch.cat([s_feat, a_feat, s_feat * a_feat, s_feat - a_feat], dim=-1)
        return self.q(comb)


class BetaSignedPolicy(nn.Module):
    """
    Internal latent u ~ Beta(alpha, beta) in (0,1)
    Actor latent z = 2*u - 1 in (-1,1)

    z is not sent directly to the environment. SAC maps z to executable_u
    using current inventory: positive z scales buy room, negative z scales
    sell room, and z=0 always means no trade.

    Exact log-prob with change of variables:
      u = (z + 1) / 2
      log p(z) = log p(u) - log|du/dz| = log p(u) - log(1/2)
    """
    def __init__(
        self,
        s_dim: int,
        hidden: int = 512,
        depth: int = 4,
        min_conc: float = 0.5,
        max_conc: float = 20.0,
        log_eps: float = 1e-6,
    ):
        super().__init__()
        self.base = SimBaMLP(s_dim, 2, hidden=hidden, depth=depth)
        self.min_conc = float(min_conc)
        self.max_conc = float(max_conc)
        self.log_eps = float(log_eps)
        self.log_abs_det_jac = math.log(0.5)

        last = self.base.output_fc
        with torch.no_grad():
            last.bias[0].fill_(+0.5)
            last.bias[1].fill_(0.0)

    def _params(self, s: torch.Tensor):
        out = self.base(s)
        alpha = F.softplus(out[:, 0:1]) + self.min_conc
        beta = F.softplus(out[:, 1:2]) + self.min_conc
        if self.max_conc > 0:
            alpha = alpha.clamp(max=self.max_conc)
            beta = beta.clamp(max=self.max_conc)
        return alpha, beta

    def _u_to_a(self, u: torch.Tensor) -> torch.Tensor:
        return 2.0 * u - 1.0

    def _a_to_u(self, a: torch.Tensor) -> torch.Tensor:
        return 0.5 * (a + 1.0)

    def sample(self, s: torch.Tensor):
        alpha, beta = self._params(s)
        dist = torch.distributions.Beta(alpha, beta)

        u = dist.rsample().clamp(self.log_eps, 1.0 - self.log_eps)
        a = self._u_to_a(u).clamp(-1.0 + 2.0 * self.log_eps, 1.0 - 2.0 * self.log_eps)

        logp_u = dist.log_prob(u).sum(dim=-1, keepdim=True)
        logp_a = logp_u - self.log_abs_det_jac
        return a, logp_a

    def sample_with_stats(self, s: torch.Tensor):
        a, logp = self.sample(s)
        stats = {
            "a_std": float(a.std(unbiased=False).item()),
            "logp_std": float(logp.std(unbiased=False).item()),
            "frac_near_neg1": float((a < -0.98).float().mean().item()),
            "frac_near_pos1": float((a > 0.98).float().mean().item()),
        }
        aux = {}
        return a, logp, stats, aux

    @torch.no_grad()
    def act_torch(self, s: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        alpha, beta = self._params(s)
        if deterministic:
            u = alpha / (alpha + beta)
        else:
            u = torch.distributions.Beta(alpha, beta).sample()

        a = self._u_to_a(u)
        return a.clamp(-1.0 + 2.0 * self.log_eps, 1.0 - 2.0 * self.log_eps)


class EmbReducer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


ARCHITECTURE_VERSION = "SAC_V4_CONFIGURABLE_NET_V1"


NETWORK_PRESETS = ("legacy_v4", "large_deep_v1")


@dataclass(frozen=True)
class NetworkArchCfg:
    architecture_version: str = ARCHITECTURE_VERSION
    network_preset: str = "legacy_v4"
    actor_hidden_dims: Tuple[int, ...] = (256, 256, 256, 256, 256)
    critic_state_hidden_dims: Tuple[int, ...] = (384, 384, 384, 384, 384, 384, 384)
    critic_action_hidden_dims: Tuple[int, ...] = (64, 64, 64, 64)
    critic_hidden_dims: Tuple[int, ...] = (256, 256, 256, 256, 256, 256, 256)
    critic_k: int = 96
    expansion: int = 2
    actor_min_conc: float = 0.5
    actor_max_conc: float = 10.0
    actor_log_eps: float = 1e-6
    emb_reducer_type: str = "layernorm_linear_relu"


def _validate_simba_dims(dims: Tuple[int, ...], *, name: str) -> None:
    if len(dims) < 1:
        raise ValueError(f"{name} must contain at least one positive integer")
    if any(int(d) <= 0 for d in dims):
        raise ValueError(f"{name} must contain only positive integers: {dims}")
    if len(set(int(d) for d in dims)) != 1:
        raise ValueError(
            f"{name}={dims} is incompatible with the current SimBa residual MLP; "
            "all hidden widths must match because each residual block is fixed-width"
        )


def _hidden_depth_from_dims(dims: Tuple[int, ...], *, name: str) -> Tuple[int, int]:
    _validate_simba_dims(dims, name=name)
    return int(dims[0]), max(0, len(dims) - 1)


def network_arch_preset(name: str) -> NetworkArchCfg:
    name = str(name or "legacy_v4")
    if name == "legacy_v4":
        return NetworkArchCfg(network_preset="legacy_v4")
    if name == "large_deep_v1":
        return NetworkArchCfg(
            network_preset="large_deep_v1",
            actor_hidden_dims=(320, 320, 320, 320, 320, 320),
            critic_state_hidden_dims=(576, 576, 576, 576, 576, 576, 576, 576),
            critic_action_hidden_dims=(96, 96, 96, 96, 96),
            critic_hidden_dims=(384, 384, 384, 384, 384, 384, 384, 384),
            critic_k=128,
            expansion=2,
        )
    raise ValueError(f"unknown --network_preset={name!r}; expected one of {NETWORK_PRESETS}")


class ReplayBuffer:
    def __init__(self, s_dim: int, capacity: int = 2_000_000, pin_memory: bool = True):
        self.cap = int(capacity)
        self.pin = bool(pin_memory)

        self.s = torch.empty((self.cap, s_dim), dtype=torch.float32, device="cpu", pin_memory=self.pin)
        self.a = torch.empty((self.cap, 1), dtype=torch.float32, device="cpu", pin_memory=self.pin)
        self.r = torch.empty((self.cap, 1), dtype=torch.float32, device="cpu", pin_memory=self.pin)
        self.sp = torch.empty((self.cap, s_dim), dtype=torch.float32, device="cpu", pin_memory=self.pin)
        self.d = torch.empty((self.cap, 1), dtype=torch.float32, device="cpu", pin_memory=self.pin)
        self.reg = torch.empty((self.cap, 1), dtype=torch.float32, device="cpu", pin_memory=self.pin)
        self.pre = torch.empty((self.cap, 1), dtype=torch.float32, device="cpu", pin_memory=self.pin)

        self.n = 0
        self.i = 0

    def add(self, s_np, a_np, r_np, sp_np, d_np, reg_np, pre_np):
        idx = self.i
        self.s[idx].copy_(torch.from_numpy(s_np), non_blocking=self.pin)
        self.a[idx].copy_(torch.as_tensor(a_np, dtype=torch.float32), non_blocking=self.pin)
        self.r[idx].copy_(torch.as_tensor(r_np, dtype=torch.float32), non_blocking=self.pin)
        self.sp[idx].copy_(torch.from_numpy(sp_np), non_blocking=self.pin)
        self.d[idx].copy_(torch.as_tensor(d_np, dtype=torch.float32), non_blocking=self.pin)
        self.reg[idx].copy_(torch.as_tensor(reg_np, dtype=torch.float32), non_blocking=self.pin)
        self.pre[idx].copy_(torch.as_tensor(pre_np, dtype=torch.float32), non_blocking=self.pin)

        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch_size: int, device: torch.device):
        idx = torch.randint(0, self.n, (int(batch_size),), device="cpu")
        s = self.s[idx].to(device, non_blocking=self.pin)
        a = self.a[idx].to(device, non_blocking=self.pin)
        r = self.r[idx].to(device, non_blocking=self.pin)
        sp = self.sp[idx].to(device, non_blocking=self.pin)
        d = self.d[idx].to(device, non_blocking=self.pin)
        return {"s": s, "a": a, "r": r, "sp": sp, "d": d}

    def sample_states_only(self, batch_size: int, device: torch.device):
        idx = torch.randint(0, self.n, (int(batch_size),), device="cpu")
        s = self.s[idx].to(device, non_blocking=self.pin)
        return s

    def sample_mixed(self, batch_size: int, device: torch.device, active_frac: float = 0.5):
        n = int(self.n)
        if n <= 0:
            raise RuntimeError("ReplayBuffer is empty")

        active_frac = float(np.clip(active_frac, 0.0, 1.0))
        n_active = int(round(batch_size * active_frac))
        n_uniform = int(batch_size) - n_active

        reg_cpu = self.reg[:n, 0]
        active_idx_all = torch.nonzero(reg_cpu > 0.5, as_tuple=False).squeeze(-1)

        if active_idx_all.numel() == 0:
            idx = torch.randint(0, n, (int(batch_size),), device="cpu")
        else:
            idx_parts = []

            if n_active > 0:
                pick_a = active_idx_all[torch.randint(0, active_idx_all.numel(), (n_active,), device="cpu")]
                idx_parts.append(pick_a)

            if n_uniform > 0:
                pick_u = torch.randint(0, n, (n_uniform,), device="cpu")
                idx_parts.append(pick_u)

            idx = torch.cat(idx_parts, dim=0)
            perm = torch.randperm(idx.shape[0], device="cpu")
            idx = idx[perm]

        s = self.s[idx].to(device, non_blocking=self.pin)
        a = self.a[idx].to(device, non_blocking=self.pin)
        r = self.r[idx].to(device, non_blocking=self.pin)
        sp = self.sp[idx].to(device, non_blocking=self.pin)
        d = self.d[idx].to(device, non_blocking=self.pin)
        reg = self.reg[idx].to(device, non_blocking=self.pin)

        return {"s": s, "a": a, "r": r, "sp": sp, "d": d, "reg": reg}

    def sample_stratified(
        self,
        batch_size: int,
        device: torch.device,
        frac_pre_reg: float = 0.5,
        frac_pre_only: float = 0.2,
    ):
        n = int(self.n)
        if n <= 0:
            raise RuntimeError("ReplayBuffer is empty")

        frac_pre_reg = float(np.clip(frac_pre_reg, 0.0, 1.0))
        frac_pre_only = float(np.clip(frac_pre_only, 0.0, 1.0))
        frac_uniform = max(0.0, 1.0 - frac_pre_reg - frac_pre_only)

        n_a = int(round(batch_size * frac_pre_reg))
        n_b = int(round(batch_size * frac_pre_only))
        n_c = int(batch_size) - n_a - n_b

        reg_cpu = self.reg[:n, 0]
        pre_cpu = self.pre[:n, 0]

        idx_a_all = torch.nonzero((pre_cpu > 0.5) & (reg_cpu > 0.5), as_tuple=False).squeeze(-1)
        idx_b_all = torch.nonzero((pre_cpu > 0.5) & (reg_cpu <= 0.5), as_tuple=False).squeeze(-1)

        idx_parts = []

        def sample_from(pool, k):
            if k <= 0:
                return None
            if pool.numel() == 0:
                return torch.randint(0, n, (k,), device="cpu")
            return pool[torch.randint(0, pool.numel(), (k,), device="cpu")]

        part_a = sample_from(idx_a_all, n_a)
        if part_a is not None:
            idx_parts.append(part_a)

        part_b = sample_from(idx_b_all, n_b)
        if part_b is not None:
            idx_parts.append(part_b)

        if n_c > 0:
            idx_parts.append(torch.randint(0, n, (n_c,), device="cpu"))

        idx = torch.cat(idx_parts, dim=0)
        idx = idx[torch.randperm(idx.shape[0], device="cpu")]

        s = self.s[idx].to(device, non_blocking=self.pin)
        a = self.a[idx].to(device, non_blocking=self.pin)
        r = self.r[idx].to(device, non_blocking=self.pin)
        sp = self.sp[idx].to(device, non_blocking=self.pin)
        d = self.d[idx].to(device, non_blocking=self.pin)
        reg = self.reg[idx].to(device, non_blocking=self.pin)
        pre = self.pre[idx].to(device, non_blocking=self.pin)

        return {"s": s, "a": a, "r": r, "sp": sp, "d": d, "reg": reg, "pre": pre}


class NStepAdder:
    def __init__(self, n_step: int, gamma: float):
        self.n = int(n_step)
        self.gamma = float(gamma)
        self.buf = deque()

    def reset(self):
        self.buf.clear()

    def push(self, s, a, r, sp, done, reg, pre, rb_out):
        self.buf.append((s, a, float(r), sp, bool(done), float(reg), float(pre)))

        if len(self.buf) < self.n:
            if done:
                self._flush(rb_out)
            return

        R = 0.0
        done_n = False
        sp_n = None

        for i in range(self.n):
            _s, _a, _r, _sp, _done, _reg, _pre = self.buf[i]
            R += (self.gamma ** i) * _r
            sp_n = _sp
            if _done:
                done_n = True
                break

        s0, a0, _, _, _, reg0, pre0 = self.buf[0]
        rb_out.add(
            s0,
            np.asarray(a0, dtype=np.float32),
            np.asarray([R], dtype=np.float32),
            sp_n,
            np.asarray([1.0 if done_n else 0.0], dtype=np.float32),
            np.asarray([reg0], dtype=np.float32),
            np.asarray([pre0], dtype=np.float32),
        )
        self.buf.popleft()

        if done:
            self._flush(rb_out)

    def _flush(self, rb_out):
        while len(self.buf) > 0:
            R = 0.0
            done_n = False
            sp_n = None
            L = len(self.buf)

            for i in range(L):
                _s, _a, _r, _sp, _done, _reg, _pre = self.buf[i]
                R += (self.gamma ** i) * _r
                sp_n = _sp
                if _done:
                    done_n = True
                    break

            s0, a0, _, _, _, reg0, pre0 = self.buf[0]
            rb_out.add(
                s0,
                np.asarray(a0, dtype=np.float32),
                np.asarray([R], dtype=np.float32),
                sp_n,
                np.asarray([1.0 if done_n else 0.0], dtype=np.float32),
                np.asarray([reg0], dtype=np.float32),
                np.asarray([pre0], dtype=np.float32),
            )
            self.buf.popleft()


@dataclass
class SacCfg:
    gamma: float = 0.97
    tau: float = 0.005
    lr_q: float = 5e-5
    lr_pi: float = 2e-4
    lr_alpha: float = 2e-4
    batch_size: int = 512

    alpha: float = 0.1
    autotune_alpha: bool = True
    target_entropy: float = -0.5

    device: str = "cuda"
    warmup_steps: int = 10_000

    lam_flat: float = 0.0
    lam_gate: float = 0.0
    lam_actor_inertia: float = 0.0
    lam_invalid_action: float = 0.0

    y_clip: float = 10.0
    n_step: int = 1
    policy_delay: int = 2
    pi_updates_emb: bool = True


class SAC:
    def __init__(
        self,
        stack_dim: int,
        emb_dim: int,
        prev_u_idx_in_obs: int,
        cfg: SacCfg,
        emb_out_dim: int = 192, # previous 128
        arch: Optional[NetworkArchCfg] = None,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.arch = arch or network_arch_preset("legacy_v4")

        self.stack_dim = int(stack_dim)
        self.emb_dim = int(emb_dim)
        self.emb_out_dim = int(emb_out_dim)
        self.prev_u_idx_in_obs = int(prev_u_idx_in_obs)

        self.emb_reducer = EmbReducer(self.emb_dim, self.emb_out_dim).to(self.device) if self.emb_dim > 0 else None
        eff_s_dim = self.stack_dim + (self.emb_out_dim if self.emb_dim > 0 else 0)
        self.effective_state_dim = int(eff_s_dim)

        actor_hidden, actor_depth = _hidden_depth_from_dims(self.arch.actor_hidden_dims, name="actor_hidden_dims")
        critic_s_hidden, critic_s_depth = _hidden_depth_from_dims(self.arch.critic_state_hidden_dims, name="critic_state_hidden_dims")
        critic_a_hidden, critic_a_depth = _hidden_depth_from_dims(self.arch.critic_action_hidden_dims, name="critic_action_hidden_dims")
        critic_q_hidden, critic_q_depth = _hidden_depth_from_dims(self.arch.critic_hidden_dims, name="critic_hidden_dims")

        self.q1 = QNetInteraction(
            eff_s_dim, a_dim=1, k=int(self.arch.critic_k),
            s_hidden=critic_s_hidden, s_depth=critic_s_depth,
            a_hidden=critic_a_hidden, a_depth=critic_a_depth,
            q_hidden=critic_q_hidden, q_depth=critic_q_depth, expansion=int(self.arch.expansion)
        ).to(self.device)
        
        # previous self.q1 = QNetInteraction(
        #    eff_s_dim, a_dim=1, k=64, s_hidden=256, s_depth=2,
        #    a_hidden=32, a_depth=1, q_hidden=192, q_depth=3, expansion=2
        #).to(self.device)

        self.q2 = QNetInteraction(
            eff_s_dim, a_dim=1, k=int(self.arch.critic_k),
            s_hidden=critic_s_hidden, s_depth=critic_s_depth,
            a_hidden=critic_a_hidden, a_depth=critic_a_depth,
            q_hidden=critic_q_hidden, q_depth=critic_q_depth, expansion=int(self.arch.expansion)
        ).to(self.device)

        self.q1_t = QNetInteraction(
            eff_s_dim, a_dim=1, k=int(self.arch.critic_k),
            s_hidden=critic_s_hidden, s_depth=critic_s_depth,
            a_hidden=critic_a_hidden, a_depth=critic_a_depth,
            q_hidden=critic_q_hidden, q_depth=critic_q_depth, expansion=int(self.arch.expansion)
        ).to(self.device)

        self.q2_t = QNetInteraction(
            eff_s_dim, a_dim=1, k=int(self.arch.critic_k),
            s_hidden=critic_s_hidden, s_depth=critic_s_depth,
            a_hidden=critic_a_hidden, a_depth=critic_a_depth,
            q_hidden=critic_q_hidden, q_depth=critic_q_depth, expansion=int(self.arch.expansion)
        ).to(self.device)

        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        for p in self.q1_t.parameters():
            p.requires_grad = False
        for p in self.q2_t.parameters():
            p.requires_grad = False

        self.pi = BetaSignedPolicy(
            eff_s_dim,
            hidden=actor_hidden,
            depth=actor_depth,
            min_conc=float(self.arch.actor_min_conc),
            max_conc=float(self.arch.actor_max_conc),
            log_eps=float(self.arch.actor_log_eps),
        ).to(self.device)

        q_params = list(self.q1.parameters()) + list(self.q2.parameters())
        pi_params = list(self.pi.parameters())

        if self.emb_reducer is not None:
            q_params += list(self.emb_reducer.parameters())
            if bool(self.cfg.pi_updates_emb):
                pi_params += list(self.emb_reducer.parameters())

        self.opt_q = torch.optim.Adam(q_params, lr=cfg.lr_q)
        self.opt_pi = torch.optim.Adam(pi_params, lr=cfg.lr_pi)

        self._update_i = 0
        self.autotune = bool(cfg.autotune_alpha)
        if self.autotune:
            self.log_alpha = torch.tensor(math.log(cfg.alpha), device=self.device, requires_grad=True)
            self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        else:
            self.log_alpha = torch.tensor(math.log(cfg.alpha), device=self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _encode_s(self, s: torch.Tensor) -> torch.Tensor:
        if self.emb_dim <= 0:
            return s
        s_stack = s[:, :self.stack_dim]
        s_emb = s[:, self.stack_dim:self.stack_dim + self.emb_dim]
        e = self.emb_reducer(s_emb)
        return torch.cat([s_stack, e], dim=-1)

    def _latent_to_executable_action(self, s_raw: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Map actor latent z in [-1, 1] to the currently executable trade action.

        The critic is trained on executed_u, so actor/target critic queries
        and env interactions must use executable actions. z=0 means no trade;
        positive z scales available buy room and negative z scales available
        sell room. For stack_len=1/include_close_feat=True
        the extra feature layout is:
            [pos_frac, inv, last_exec, prev_exposure, p_feat, hold_feat]

        prev_u_idx_in_obs points to prev_exposure, so inv is two slots before it.
        """
        inv_idx = int(self.prev_u_idx_in_obs) - 2
        inv = s_raw[:, inv_idx:inv_idx + 1].clamp(0.0, 1.0)

        sell_room = inv
        buy_room = 1.0 - inv

        z_clamped = z.clamp(-1.0, 1.0)
        a_exec = torch.where(
            z_clamped >= 0.0,
            z_clamped * buy_room,
            z_clamped * sell_room,
        ).clamp(-1.0, 1.0)

        aux = {
            "inv": inv,
            "sell_room": sell_room,
            "buy_room": buy_room,
            "sell_limit": -sell_room,
            "buy_limit": buy_room,
            "z": z_clamped,
            "a_exec": a_exec,
        }
        return a_exec, aux

    def _project_raw_action_proxy(self, s_raw: torch.Tensor, a_raw: torch.Tensor) -> torch.Tensor:
        """
        Compatibility wrapper for the old raw-action projection name.

        New actor code interprets policy output as latent z and maps it to the
        valid executable range. Keep this method for older diagnostic call sites.
        """
        a_exec, _ = self._latent_to_executable_action(s_raw, a_raw)
        return a_exec

    @torch.no_grad()
    def latent_to_exec_np(self, s_np: np.ndarray, z_float: float) -> float:
        s_raw = torch.from_numpy(s_np).to(self.device).unsqueeze(0)
        z = torch.tensor([[float(z_float)]], dtype=torch.float32, device=self.device)
        a_exec, _ = self._latent_to_executable_action(s_raw, z)
        return float(a_exec.item())

    @torch.no_grad()
    def act(self, s_np: np.ndarray, deterministic: bool = False, hard_gate: bool = False, return_latent: bool = False):
        s_raw = torch.from_numpy(s_np).to(self.device).unsqueeze(0)
        s_enc = self._encode_s(s_raw)
        z = self.pi.act_torch(s_enc, deterministic=deterministic)
        a_exec, _ = self._latent_to_executable_action(s_raw, z)
        if return_latent:
            return float(a_exec.item()), float(z.item())
        return float(a_exec.item())

    @torch.no_grad()
    def soft_update(self):
        tau = self.cfg.tau
        for net, targ in [(self.q1, self.q1_t), (self.q2, self.q2_t)]:
            for p, pt in zip(net.parameters(), targ.parameters()):
                pt.data.mul_(1 - tau).add_(tau * p.data)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            sp_enc = self._encode_s(batch["sp"])
            a2_z, logp2 = self.pi.sample(sp_enc)
            a2_exec, _ = self._latent_to_executable_action(batch["sp"], a2_z)
            q_t = torch.min(self.q1_t(sp_enc, a2_exec), self.q2_t(sp_enc, a2_exec))
            gamma_n = self.cfg.gamma ** int(self.cfg.n_step)
            y_unclipped = batch["r"] + gamma_n * (1.0 - batch["d"]) * (q_t - self.alpha * logp2)

            if self.cfg.y_clip > 0:
                y = torch.clamp(y_unclipped, -self.cfg.y_clip, self.cfg.y_clip)
                y_clip_frac = (y_unclipped.abs() > float(self.cfg.y_clip)).float().mean()
            else:
                y = y_unclipped
                y_clip_frac = torch.zeros((), device=self.device)

        if not torch.isfinite(y).all():
            raise RuntimeError("Non-finite Q target y detected")

        s_enc = self._encode_s(batch["s"])
        a = batch["a"]

        q1 = self.q1(s_enc, a)
        q2 = self.q2(s_enc, a)

        if not torch.isfinite(q1).all() or not torch.isfinite(q2).all():
            raise RuntimeError("Non-finite Q values detected")

        with torch.no_grad():
            q_mean_batch = 0.5 * (q1 + q2)
            q_min_batch = torch.min(q1, q2)
            td1 = q1 - y
            td2 = q2 - y

            a_neg1 = torch.full_like(a, -1.0)
            a_neg75 = torch.full_like(a, -0.75)
            a_neg50 = torch.full_like(a, -0.50)
            a_neg25 = torch.full_like(a, -0.25)
            a0 = torch.zeros_like(a)
            a25 = torch.full_like(a, 0.25)
            a50 = torch.full_like(a, 0.50)
            a75 = torch.full_like(a, 0.75)
            a100 = torch.full_like(a, 1.00)

            a_neg1_exec, _ = self._latent_to_executable_action(batch["s"], a_neg1)
            a_neg75_exec, _ = self._latent_to_executable_action(batch["s"], a_neg75)
            a_neg50_exec, _ = self._latent_to_executable_action(batch["s"], a_neg50)
            a_neg25_exec, _ = self._latent_to_executable_action(batch["s"], a_neg25)
            a0_exec, _ = self._latent_to_executable_action(batch["s"], a0)
            a25_exec, _ = self._latent_to_executable_action(batch["s"], a25)
            a50_exec, _ = self._latent_to_executable_action(batch["s"], a50)
            a75_exec, _ = self._latent_to_executable_action(batch["s"], a75)
            a100_exec, _ = self._latent_to_executable_action(batch["s"], a100)

            qmin_aneg1 = torch.min(self.q1(s_enc, a_neg1_exec), self.q2(s_enc, a_neg1_exec))
            qmin_aneg75 = torch.min(self.q1(s_enc, a_neg75_exec), self.q2(s_enc, a_neg75_exec))
            qmin_aneg50 = torch.min(self.q1(s_enc, a_neg50_exec), self.q2(s_enc, a_neg50_exec))
            qmin_aneg25 = torch.min(self.q1(s_enc, a_neg25_exec), self.q2(s_enc, a_neg25_exec))
            qmin_a0 = torch.min(self.q1(s_enc, a0_exec), self.q2(s_enc, a0_exec))
            qmin_a25 = torch.min(self.q1(s_enc, a25_exec), self.q2(s_enc, a25_exec))
            qmin_a50 = torch.min(self.q1(s_enc, a50_exec), self.q2(s_enc, a50_exec))
            qmin_a75 = torch.min(self.q1(s_enc, a75_exec), self.q2(s_enc, a75_exec))
            qmin_a100 = torch.min(self.q1(s_enc, a100_exec), self.q2(s_enc, a100_exec))
            q_grid = torch.stack(
                [
                    qmin_aneg1.squeeze(-1),
                    qmin_aneg75.squeeze(-1),
                    qmin_aneg50.squeeze(-1),
                    qmin_aneg25.squeeze(-1),
                    qmin_a0.squeeze(-1),
                    qmin_a25.squeeze(-1),
                    qmin_a50.squeeze(-1),
                    qmin_a75.squeeze(-1),
                    qmin_a100.squeeze(-1),
                ],
                dim=1,
            )
            q_action_range = q_grid.max(dim=1).values - q_grid.min(dim=1).values
            td_abs_mean = 0.5 * (td1.abs().mean() + td2.abs().mean())
            q_gap_best = q_grid.max(dim=1).values - qmin_a0.squeeze(-1)
            q_gap_td_ratio = q_gap_best.mean() / (td_abs_mean + 1e-12)

        q_loss = F.smooth_l1_loss(q1, y) + F.smooth_l1_loss(q2, y)

        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()

        to_clip = list(self.q1.parameters()) + list(self.q2.parameters())
        if self.emb_reducer is not None:
            to_clip += list(self.emb_reducer.parameters())
        torch.nn.utils.clip_grad_norm_(to_clip, 10.0)
        self.opt_q.step()

        self._update_i += 1
        do_policy = (self._update_i % int(self.cfg.policy_delay) == 0)

        pi_loss_val = float("nan")
        q_pi_mean = float("nan")
        logp_mean = float("nan")
        logp_std = float("nan")
        flat_pen_val = float("nan")
        a_pi_std = float("nan")
        a_pi_near_neg1 = float("nan")
        a_pi_near_pos1 = float("nan")
        inertia_pen_val = float("nan")
        invalid_projection_pen_val = float("nan")
        a_pi_raw_mean = float("nan")
        a_pi_exec_mean = float("nan")
        a_pi_projection_gap = float("nan")
        a_pi_z_mean = float("nan")
        a_pi_z_std = float("nan")
        a_pi_transform_gap = float("nan")
        alpha_loss_val = 0.0

        if do_policy:
            s_enc2 = self._encode_s(batch["s"])
            z_pi, logp, pi_stats, _ = self.pi.sample_with_stats(s_enc2)
            a_pi_exec, _ = self._latent_to_executable_action(batch["s"], z_pi)
            q_pi = torch.min(self.q1(s_enc2, a_pi_exec), self.q2(s_enc2, a_pi_exec))

            with torch.no_grad():
                q_det_z = self.pi.act_torch(s_enc2, deterministic=True)
                q_det_a_exec, _ = self._latent_to_executable_action(batch["s"], q_det_z)
                q_det = torch.min(self.q1(s_enc2, q_det_a_exec), self.q2(s_enc2, q_det_a_exec))

            flat_pen = (a_pi_exec ** 2).mean()

            last_exec_idx = int(self.prev_u_idx_in_obs) - 1
            prev_exec_obs = batch["s"][:, last_exec_idx:last_exec_idx + 1].clamp(-1.0, 1.0)
            inertia_pen = ((a_pi_exec - prev_exec_obs) ** 2).mean()
            invalid_projection_pen = torch.zeros((), device=self.device)

            pi_loss = (
                (self.alpha * logp - q_pi).mean()
                + float(self.cfg.lam_flat) * flat_pen
                + float(self.cfg.lam_actor_inertia) * inertia_pen
                + float(self.cfg.lam_invalid_action) * invalid_projection_pen
            )

            if not torch.isfinite(pi_loss):
                raise RuntimeError("Non-finite pi_loss detected")

            self.opt_pi.zero_grad(set_to_none=True)
            pi_loss.backward()

            to_clip_pi = list(self.pi.parameters())
            if self.emb_reducer is not None and bool(self.cfg.pi_updates_emb):
                to_clip_pi += list(self.emb_reducer.parameters())
            torch.nn.utils.clip_grad_norm_(to_clip_pi, 10.0)
            self.opt_pi.step()

            inertia_pen_val = float(inertia_pen.item())
            invalid_projection_pen_val = float(invalid_projection_pen.item())
            pi_loss_val = float(pi_loss.item())
            q_pi_mean = float(q_pi.mean().item())
            logp_mean = float(logp.mean().item())
            logp_std = float(logp.std(unbiased=False).item())
            flat_pen_val = float(flat_pen.item())
            a_pi_std = float(pi_stats["a_std"])
            a_pi_near_neg1 = float(pi_stats["frac_near_neg1"])
            a_pi_near_pos1 = float(pi_stats["frac_near_pos1"])
            a_pi_raw_mean = float(z_pi.mean().item())
            a_pi_exec_mean = float(a_pi_exec.mean().item())
            a_pi_projection_gap = float((z_pi - a_pi_exec).abs().mean().item())
            a_pi_z_mean = float(z_pi.mean().item())
            a_pi_z_std = float(z_pi.std(unbiased=False).item())
            a_pi_transform_gap = float((z_pi - a_pi_exec).abs().mean().item())

            if self.autotune:
                alpha_loss = -(self.log_alpha * (logp.detach() + self.cfg.target_entropy)).mean()
                self.opt_alpha.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.opt_alpha.step()

                with torch.no_grad():
                    self.log_alpha.clamp_(math.log(0.005), math.log(0.40))

                alpha_loss_val = float(alpha_loss.item())

        self.soft_update()

        return {
            "q_loss": float(q_loss.item()),
            "pi_loss": float(pi_loss_val),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss_val) if self.autotune else 0.0,

            "q_y_mean": float(y.mean().item()),
            "q_y_std": float(y.std(unbiased=False).item()),
            "q_y_unclipped_mean": float(y_unclipped.mean().item()),
            "q_y_unclipped_std": float(y_unclipped.std(unbiased=False).item()),
            "q_y_clip_frac": float(y_clip_frac.item()),
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
            "q_mean_batch": float(q_mean_batch.mean().item()),
            "q_min_batch": float(q_min_batch.mean().item()),
            "td1_abs_mean": float(td1.abs().mean().item()),
            "td2_abs_mean": float(td2.abs().mean().item()),
            "td_abs_mean": float(td_abs_mean.item()),
            "q_action_range_mean": float(q_action_range.mean().item()),
            "q_gap_best_mean": float(q_gap_best.mean().item()),
            "q_gap_td_ratio": float(q_gap_td_ratio.item()),

            "qmin_aneg1_batch": float(qmin_aneg1.mean().item()),
            "qmin_aneg75_batch": float(qmin_aneg75.mean().item()),
            "qmin_aneg50_batch": float(qmin_aneg50.mean().item()),
            "qmin_aneg25_batch": float(qmin_aneg25.mean().item()),
            "qmin_a0_batch": float(qmin_a0.mean().item()),
            "qmin_a25_batch": float(qmin_a25.mean().item()),
            "qmin_a50_batch": float(qmin_a50.mean().item()),
            "qmin_a75_batch": float(qmin_a75.mean().item()),
            "qmin_a100_batch": float(qmin_a100.mean().item()),
            "qmin_zneg1_batch": float(qmin_aneg1.mean().item()),
            "qmin_zneg75_batch": float(qmin_aneg75.mean().item()),
            "qmin_zneg50_batch": float(qmin_aneg50.mean().item()),
            "qmin_zneg25_batch": float(qmin_aneg25.mean().item()),
            "qmin_z0_batch": float(qmin_a0.mean().item()),
            "qmin_z25_batch": float(qmin_a25.mean().item()),
            "qmin_z50_batch": float(qmin_a50.mean().item()),
            "qmin_z75_batch": float(qmin_a75.mean().item()),
            "qmin_z100_batch": float(qmin_a100.mean().item()),

            "q_pi_mean": float(q_pi_mean),
            "q_det_mean": float(q_det.mean().item()) if do_policy else float("nan"),
            "logp_mean": float(logp_mean),
            "logp_std": float(logp_std),

            "flat_pen": float(flat_pen_val),
            "a_pi_std": float(a_pi_std),
            "a_pi_near_neg1": float(a_pi_near_neg1),
            "a_pi_near_pos1": float(a_pi_near_pos1),
            "gate_pen": 0.0,
            "gate_prob_mean": 0.0,
            "inertia_pen": float(inertia_pen_val),
            "invalid_projection_pen": float(invalid_projection_pen_val),
            "a_pi_raw_mean": float(a_pi_raw_mean),
            "a_pi_exec_mean": float(a_pi_exec_mean),
            "a_pi_projection_gap": float(a_pi_projection_gap),
            "a_pi_z_mean": float(a_pi_z_mean),
            "a_pi_z_std": float(a_pi_z_std),
            "a_pi_transform_gap": float(a_pi_transform_gap),
        }
