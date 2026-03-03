# Training & pipeline improvements (to add after current run)

Do **not** apply these while epochs are running. Add in a future training run.

---

## 1. Mixed precision (AMP)

- **What:** Use `torch.cuda.amp.autocast()` and `GradScaler` in the training loop.
- **Why:** ~1.5–2× faster training with almost no quality loss.
- **Where:** In `train.ipynb`, in the batch loop: wrap forward in `with torch.cuda.amp.autocast():`, use `scaler.scale(loss).backward()`, `scaler.step(optimizer)`, `scaler.update()`. Create `scaler = torch.cuda.amp.GradScaler()` before the epoch loop.

---

## 2. Early stopping

- **What:** Stop training when validation metric (e.g. val loss or val top-1) has not improved for N epochs.
- **Why:** Avoid wasting time on extra epochs that don’t help; use a saved best checkpoint.
- **Where:** In `train.ipynb`: track best val metric and `patience` counter; if no improvement for `patience` epochs (e.g. 2 or 3), `break` out of the epoch loop. Optionally make `patience` a constant at the top of the cell.

---

## 3. Dropout

- **What:** Model already has dropout (e.g. 0.1 in `model.py`). Optionally make dropout rate configurable (e.g. in config or notebook) and try 0.1 vs 0.15 for regularization.
- **Why:** Slightly higher dropout can reduce overfitting if val loss starts to rise while train loss keeps falling.
- **Where:** `model.py` — `SharedTransformerBlock` and `RecurrentTransformer` accept `dropout`; ensure it’s passed from config if we add a config field, or set in notebook when building the model.

---

## Optional (later)

- **torch.compile(model)** before training (PyTorch 2+) for extra speed.
- **Pre-tokenize** the dataset and save to disk to speed up data loading (helps when `num_workers=0` on Windows).
