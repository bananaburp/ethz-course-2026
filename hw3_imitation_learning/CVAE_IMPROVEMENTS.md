# CVAE Improvement Plan
**Current baseline:** ~10% success rate on Exercise 3 (multicube)
**Target:** Maximize success rate beyond 10%
**Files to modify:** `hw3/model.py`, `scripts/train.py`

---

## Fix 1 — Learned Conditional Prior `p_θ(z|s)` [VERY HIGH ROI]

### Why
The current model samples `z ~ N(0,I)` at inference — a fixed prior that ignores all state information. This is suboptimal: the robot's current state strongly constrains which latent behaviors make sense (e.g., if the red cube is already placed, the prior for z should reflect "go pick up blue"). Lecture slides 35–40 (Play-LMP architecture) show a learned prior network `p_θ(z|s)` conditioned on state. During training, the KL becomes `KL(posterior || prior(state))` instead of `KL(posterior || N(0,I))`. At inference, sample from the learned prior `p_θ(z|s)` instead of N(0,I).

### Changes to `hw3/model.py`

**1a. Add `prior_net` and `prior_out` to `_build()`**

In `CVAEPolicy._build()`, after the decoder block (after `self.dec_out = ...`), add:

```python
# Prior network: state → (mu_prior, log_var_prior)
prior_layers: list[nn.Module] = [nn.Linear(self.state_dim, self.d_model)]
if layer_norm:
    prior_layers.append(nn.LayerNorm(self.d_model))
prior_layers.append(nn.ReLU())
self.prior_net = nn.Sequential(*prior_layers)
self.prior_hidden = nn.ModuleList([
    _MLPBlock(self.d_model, layer_norm=layer_norm, residual=residual)
    for _ in range(max(1, self.depth - 2))
])
self.prior_out = nn.Linear(self.d_model, 2 * self.latent_dim)
```

**1b. Add `_prior()` method**

Add this method to `CVAEPolicy`, between `_encode` and `_decode`:

```python
def _prior(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = self.prior_net(state)
    for block in self.prior_hidden:
        x = block(x)
    mu_p, log_var_p = self.prior_out(x).chunk(2, dim=-1)
    log_var_p = log_var_p.clamp(-10, 4)
    return mu_p, log_var_p
```

**1c. Update `compute_loss()` — use `KL(posterior || prior(state))`**

Replace the current KL line in `compute_loss`:

```python
# OLD:
kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()

# NEW:
mu_p, log_var_p = self._prior(state)
var_p = log_var_p.exp()
kl = 0.5 * (
    log_var_p - log_var
    + (log_var.exp() + (mu - mu_p).pow(2)) / var_p
    - 1
).sum(dim=-1).mean()
```

**1d. Update `sample_actions()` — sample from learned prior**

Replace the current `sample_actions` body:

```python
def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
    mu_p, log_var_p = self._prior(state)
    std_p = (0.5 * log_var_p).exp()
    z = mu_p + std_p * torch.randn_like(std_p)
    return self._decode(state, z)
```

**1e. Update `load_state_dict()` — rebuild prior network on load**

The existing `load_state_dict` calls `self._build(needs_ln, needs_res)`, which now includes `prior_net`, so no changes needed there. However, if loading old checkpoints (without `prior_net` weights), strict=True will fail. The harness likely passes `strict=False` or handles this — verify by running eval once. If it errors, add `strict=False` to the super() call.

**1f. Update the class docstring** to reflect the new prior:

```python
"""CVAE policy with learned conditional prior p_θ(z|s).

Training: encoder q_φ(z|s,a) → z, KL(q_φ || p_θ(z|s)) + recon loss.
Inference: sample z ~ p_θ(z|s), decode (s, z) → action_chunk.
"""
```

---

## Fix 2 — Use Prior Mean at Inference (Deterministic z) [HIGH ROI]

### Why
Stochastic sampling from the prior at inference introduces variance that can destabilize execution across the chunk horizon. For evaluation (not training), using `z = mu_prior(state)` eliminates this noise and gives the most likely action under the learned prior. This is equivalent to best-of-1 but without the variance. Lecture slide 27 notes that deterministic inference is valid at test time.

This fix depends on Fix 1 being implemented first.

### Changes to `hw3/model.py`

Add a `deterministic` flag to `sample_actions`, defaulting to `True` at inference:

```python
def sample_actions(self, state: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
    mu_p, log_var_p = self._prior(state)
    if deterministic:
        z = mu_p
    else:
        std_p = (0.5 * log_var_p).exp()
        z = mu_p + std_p * torch.randn_like(std_p)
    return self._decode(state, z)
```

No changes needed to training (training always uses the posterior z, not `sample_actions`).

---

## Fix 3 — KL Reduction: `.mean()` Instead of `.sum()` [MEDIUM ROI]

### Why
The current KL uses `.sum(dim=-1)` over the `latent_dim=64` dimensions, then `.mean()` over the batch. This means the KL term scales linearly with `latent_dim` (×64 vs a single scalar). With `beta=1.0` and `latent_dim=64`, the effective KL weight is 64× larger than intended relative to the per-timestep MSE. The ELBO derivation in lecture slide 39 shows KL as a single scalar expectation — use `.mean(dim=-1)` to normalize by latent dimension.

### Changes to `hw3/model.py`

In `compute_loss`, change the KL reduction (applies to the N(0,I) version or the learned prior version from Fix 1):

```python
# If NOT doing Fix 1, change:
kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean(dim=-1).mean()
#                                                       ^^^^^ was .sum()

# If doing Fix 1, apply the same: replace .sum(dim=-1) with .mean(dim=-1) in the KL formula
kl = 0.5 * (
    log_var_p - log_var
    + (log_var.exp() + (mu - mu_p).pow(2)) / var_p
    - 1
).mean(dim=-1).mean()
```

If you apply this fix, you may also want to tune `beta` back down slightly (try `beta=0.5` or `beta=0.1`) since the KL magnitude is now smaller.

---

## Fix 4 — Best-of-N Sampling at Inference [MEDIUM ROI]

### Why
Even with a learned prior, a single z sample may decode to a suboptimal action chunk. Sampling N candidates and picking the one with lowest reconstruction cost (measured by re-encoding and checking consistency) improves robustness. This is the "Best-of-N" inference trick (lecture slide 27).

### Changes to `hw3/model.py`

Add a `n_samples` parameter to `sample_actions`:

```python
def sample_actions(self, state: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
    if n_samples == 1:
        # Fast path: use prior mean (Fix 2)
        mu_p, _ = self._prior(state)
        return self._decode(state, mu_p)

    # Sample N candidates from the learned prior
    mu_p, log_var_p = self._prior(state)
    std_p = (0.5 * log_var_p).exp()

    # Expand state for N candidates: (B, state_dim) → (B*N, state_dim)
    B = state.size(0)
    state_exp = state.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
    mu_exp = mu_p.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
    std_exp = std_p.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)

    z = mu_exp + std_exp * torch.randn_like(std_exp)
    chunks = self._decode(state_exp, z)  # (B*N, chunk_size, action_dim)
    chunks = chunks.view(B, n_samples, self.chunk_size, self.action_dim)

    # Pick the candidate closest to the prior mean (lowest z deviation)
    z_dev = (z - mu_exp).pow(2).sum(dim=-1).view(B, n_samples)  # (B, N)
    best_idx = z_dev.argmin(dim=1)  # (B,)
    return chunks[torch.arange(B), best_idx]  # (B, chunk_size, action_dim)
```

To use N=10 at inference, no training changes needed — just modify the eval call or temporarily patch `sample_actions`. Since the eval harness calls `sample_actions(state)` without extra args, this defaults to N=1 (prior mean). To use N>1 you'd need to patch the harness call, which is not possible. So this fix is only useful if you can modify how `sample_actions` is invoked — **skip this fix if using the compiled harness**.

---

## Recommended Training Command (after implementing Fix 1–3)

```bash
cd /Volumes/T9/DevSpace/Github/robot-learning/hw3_imitation_learning

python scripts/train.py \
    --zarr datasets/processed/multi_cube/processed_ee_full.zarr \
    --extra-zarr datasets/raw/multi_cube/dagger/2026-03-17_12-31-59/so100_multicube_dagger.zarr \
    --state-keys state_ee_full state_gripper \
        "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" \
        state_goal goal_pos \
    --action-keys action_ee_full action_gripper \
    --policy cvae --chunk-size 16 --d-model 512 --depth 4 --epochs 200 \
    --latent-dim 64 --beta 1.0 --layer-norm --residual

python student_eval/run_eval.py \
    --exercise 3 \
    --checkpoint ./checkpoints/multi_cube/best_model_ee_full_cvae_dagger12ep.pt
```

If Fix 3 (`.mean()` reduction) is applied, try `--beta 0.5` or `--beta 0.1` in the training command.

---

## Priority Order

| Priority | Fix | Expected Gain |
|----------|-----|---------------|
| 1 | Learned conditional prior (Fix 1) | Large — removes fixed N(0,I) prior mismatch |
| 2 | Deterministic z at inference (Fix 2) | Medium — eliminates inference variance |
| 3 | KL mean reduction (Fix 3) | Medium — fixes over-regularization with latent_dim=64 |
| 4 | Best-of-N sampling (Fix 4) | Skip — not compatible with compiled harness |

Implement Fix 1 + Fix 2 together (they are tightly coupled). Fix 3 is independent and low-risk. Fix 4 is not actionable with the current eval setup.
