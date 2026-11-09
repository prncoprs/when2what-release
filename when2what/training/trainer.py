"""Step-based training loop.

AdamW, linear warm-up then cosine decay to zero, bf16 autocast on GPU, gradient
clipping, validation loss every ``val_every`` steps on the first ``val_batches``
validation batches; the checkpoint with the lowest validation loss is ``best.pt``.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..data import TrainWindows, ValWindows
from ..models import build_model, count_parameters
from .losses import compute_losses


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def warmup_cosine(warmup: int, total: int):
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return factor


@torch.no_grad()
def validation_loss(model, loader, cfg: Config, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= cfg.val_batches:
            break
        loss, _ = compute_losses(model, batch, cfg, device)
        total += float(loss)
        n += 1
    model.train()
    return total / max(1, n)


def _save(path: Path, **state) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(cfg: Config, run_dir: str, device: torch.device, resume: bool = True) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    set_seed(cfg.seed)

    val_loader = DataLoader(ValWindows(cfg), batch_size=max(2, cfg.batch_size // 2), shuffle=False, num_workers=2)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(cfg.warmup_steps, cfg.max_steps))
    start, best, best_step = 0, float("inf"), -1
    last = run_dir / "last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best, best_step = ck["step"], ck["best"], ck["best_step"]
    print(f"{cfg.variant}: {count_parameters(model) / 1e6:.2f}M parameters, starting at step {start}", flush=True)

    loader = DataLoader(TrainWindows(cfg), batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                        pin_memory=device.type == "cuda", drop_last=True,
                        persistent_workers=cfg.num_workers > 0,
                        prefetch_factor=4 if cfg.num_workers > 0 else None)
    batches = iter(loader)
    model.train()
    t0 = time.time()
    for step in range(start + 1, cfg.max_steps + 1):
        loss, parts = compute_losses(model, next(batches), cfg, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if step % cfg.val_every == 0 or step == cfg.max_steps:
            val = validation_loss(model, val_loader, cfg, device)
            improved = val < best - 1e-7
            if improved:
                best, best_step = val, step
                _save(run_dir / "best.pt", model=model.state_dict(), step=step, val_loss=val)
            _save(last, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                  step=step, best=best, best_step=best_step)
            print(f"step {step}/{cfg.max_steps} loss {parts['total']:.4f} (timing {parts['timing']:.4f} "
                  f"wave {parts['wave']:.4f}) val {val:.4f} best {best:.4f}@{best_step} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
    return run_dir / "best.pt"
