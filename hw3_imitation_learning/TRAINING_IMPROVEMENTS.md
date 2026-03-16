# Training Improvement Todos
Ordered by expected ROI on model performance (highest first).

---

- [x] **1. Fix state feature completeness** — Verify `--state-keys` includes every signal the policy needs (obstacle position, cube XYZ, gripper state). This is the single biggest lever: if the model can't observe something relevant, no architecture change compensates. Check what the policy is failing on and add the missing observation first.

- [x] **2. Use episode-aware train/val split** — The current `random_split` lets frames from the same episode bleed across train and val, inflating val metrics and masking overfitting. Split by whole episodes instead (using `episode_ends`). This gives honest feedback that guides every other decision.

- [ ] **3. Tune chunk size (`--chunk-size`)** — The prediction horizon H is task-critical. Too small → jerky, reactive motion. Too large → harder to learn and recover from errors. Sweep values (8, 16, 32) and pick the one with the best episode success rate, not just val loss.

- [x] **4. Add gradient clipping** — Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()` in `train_one_epoch`. Prevents gradient spikes from destabilizing training, especially with deeper nets or noisy demos.

- [ ] **5. Add per-dimension action loss weighting** — Gripper (`action_gripper`) is near-binary while XYZ is continuous. Even after normalization, the MSE gradient can be dominated by the wrong dimensions. Apply a per-dimension weight vector to the loss (e.g. up-weight gripper if it's under-predicted).

- [ ] **6. Add LayerNorm between hidden layers** — Insert `nn.LayerNorm(d_model)` after each `nn.Linear` + activation in `ObstaclePolicy`. Stabilizes gradient flow and improves convergence, especially when scaling up depth.

- [ ] **7. Increase network capacity (`--d-model`, `--depth`)** — The default 128-dim, 2-layer MLP is small. For multi-object scenes or longer chunk sizes, try `--d-model 256 --depth 3` or `--d-model 512 --depth 4`. Only worth doing after fixing the val split so you can actually measure overfitting.

- [ ] **8. Add dropout for regularization** — Add `nn.Dropout(p=0.1)` after each hidden activation in `ObstaclePolicy`. Helps when the dataset is small relative to model size (common with real-robot demos).

- [ ] **9. Switch ReLU → GELU activation** — Replace `nn.ReLU()` with `nn.GELU()` in the MLP layers. Smoother gradient flow; marginal but consistent improvement in practice for action-space MLPs.

- [ ] **10. Increase DataLoader workers** — Change `num_workers=0` to `num_workers=4` in both `train_loader` and `val_loader`. Purely a training speed improvement (no effect on final model quality), but faster iteration means more experiments.
