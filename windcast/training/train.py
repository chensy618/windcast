"""Train an operator on (driver forecast + static) -> MET Nordic, or evaluate a checkpoint.

    python -m windcast.training.train --config configs/f1_meps.yaml --run-name f1_fno
    python -m windcast.training.train --config ... --run-name f1_unet --set model.name=unet
    python -m windcast.training.train --config ... --eval RUN/best.pt --split test
    python -m windcast.training.train --config ... --baseline --split test   # F0, no model

Prediction = driver (u10, v10, t2m, ...) + residual_std * model(x). The loss is the Huber
loss of the normalised residual, every target variable weighted alike. Every evaluation
also scores the raw driver and persistence of the analysis at init, per lead.

Outputs in {paths.runs}/<run-name>: config.yaml, run_info.json, metrics.jsonl (per
epoch), last.pt, best.pt (lowest validation loss), eval_<split>.json.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler, Subset

from windcast.data.dataset import TARGET, ForecastDataset, load_stats
from windcast.models import build_model, n_params
from windcast.training.metrics import Scores, huber_loss


def apply_overrides(cfg: dict, items: list[str]) -> dict:
    for it in items:
        key, val = it.split("=", 1)
        d = cfg
        *path, last = key.split(".")
        for p in path:
            d = d.setdefault(p, {})
        v = yaml.safe_load(val)
        if isinstance(v, str):  # YAML 1.1 reads "1e-3" as a string
            try:
                v = float(v)
            except ValueError:
                pass
        d[last] = v
    return cfg


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


class Forecaster(torch.nn.Module):
    """Operator + residual scaling: physical prediction from (x, base)."""

    def __init__(self, net, res_std: np.ndarray):
        super().__init__()
        self.net = net
        self.register_buffer("res_std", torch.as_tensor(res_std, dtype=torch.float32))

    def forward(self, x, base):
        return base + self.res_std * self.net(x).float()


def to_device(b, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}


@torch.no_grad()
def evaluate(model, loader, device, res_std, amp: bool, baseline_only: bool = False) -> tuple:
    sc, loss_sum, n = Scores(), 0.0, 0
    if model is not None:
        model.eval()
    for b in loader:
        b = to_device(b, device)
        if not baseline_only:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                pred = model(b["x"], b["base"])
            loss_sum += huber_loss((pred - b["base"]) / res_std, (b["y"] - b["base"]) / res_std).item()
            n += 1
            sc.add("model", pred, b["y"], b["lead"])
        sc.add("driver", b["base"], b["y"], b["lead"])
        if "persist" in b:
            sc.add("persistence", b["persist"], b["y"], b["lead"])
    return sc.result(), loss_sum / max(n, 1)


def print_table(res: dict, keys=("rmse_wind", "rmse_t2m", "rmse_rh2m", "rmse_msl", "rmse_tcc")):
    for name, per_lead in res.items():
        print(f"== {name}")
        print("lead  " + "  ".join(f"{k:>10}" for k in keys))
        for L, r in per_lead.items():
            print(f"{str(L):>4}  " + "  ".join(f"{r[k]:10.3f}" for k in keys))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-name")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--eval", help="checkpoint to evaluate")
    p.add_argument("--baseline", action="store_true", help="score driver / persistence only")
    p.add_argument("--split", default="val")
    a = p.parse_args()
    cfg = apply_overrides(yaml.safe_load(open(a.config)), a.set)
    if a.eval:
        saved = yaml.safe_load(open(Path(a.eval).parent / "config.yaml"))
        saved["data"]["root"], saved["paths"] = cfg["data"]["root"], cfg["paths"]
        cfg = apply_overrides(saved, a.set)
    tc = cfg["train"]
    torch.manual_seed(tc.get("seed", 0))
    np.random.seed(tc.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(tc.get("amp", True)) and device.type == "cuda"
    stats = load_stats(cfg["data"].get("stats", Path(cfg["data"]["root"]) / "stats.json"))
    res_std = torch.as_tensor(stats["residual_std"], device=device)
    loader_kw = dict(batch_size=tc["batch"], num_workers=tc["workers"], pin_memory=True,
                     persistent_workers=tc["workers"] > 0)

    if a.eval or a.baseline:
        cfg["data"]["persistence"] = True
        ds = ForecastDataset(cfg, a.split, stats)
        model = None
        if a.eval:
            x0 = ds[0]["x"]
            net = build_model(cfg, x0.shape[0], len(TARGET))
            model = Forecaster(net, stats["residual_std"]).to(device)
            model.load_state_dict(torch.load(a.eval, map_location=device)["model"])
        res, loss = evaluate(model, DataLoader(ds, shuffle=False, **loader_kw), device, res_std,
                             amp, baseline_only=a.baseline)
        print_table(res)
        out = (Path(a.eval).parent if a.eval else Path(cfg["paths"]["runs"]) / "baselines")
        out.mkdir(parents=True, exist_ok=True)
        name = f"eval_{a.split}.json" if a.eval else f"baseline_{a.split}.json"
        (out / name).write_text(json.dumps({"loss": loss, "n_samples": len(ds), **res}, indent=1))
        print("wrote", out / name)
        return

    run = Path(cfg["paths"]["runs"]) / a.run_name
    run.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run / "config.yaml", "w"), sort_keys=False)
    train = ForecastDataset(cfg, "train", stats)
    val = ForecastDataset(cfg, "val", stats)
    vs = tc.get("val_stride", 1)
    val_sub = Subset(val, range(0, len(val), vs))
    n_in = train[0]["x"].shape[0]
    net = build_model(cfg, n_in, len(TARGET))
    model = Forecaster(net, stats["residual_std"]).to(device)
    info = {"commit": git_commit(), "host": platform.node(), "n_params": n_params(net),
            "n_in": n_in, "n_train": len(train), "n_val": len(val), "n_val_used": len(val_sub),
            "torch": torch.__version__}
    (run / "run_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps_per_epoch = tc["samples_per_epoch"] // tc["batch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tc["lr"], total_steps=tc["epochs"] * steps_per_epoch,
                                                pct_start=0.05)
    tl = DataLoader(train, sampler=RandomSampler(train, replacement=True,
                                                 num_samples=steps_per_epoch * tc["batch"]),
                    drop_last=True, **loader_kw)
    vl = DataLoader(val_sub, shuffle=False, **loader_kw)
    best = float("inf")
    for ep in range(tc["epochs"]):
        model.train()
        t0, tot, k = time.time(), 0.0, 0
        for b in tl:
            b = to_device(b, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                pred = model(b["x"], b["base"])
            loss = huber_loss((pred - b["base"]) / res_std, (b["y"] - b["base"]) / res_std,
                              cfg["loss"]["huber_delta"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("clip", 1.0))
            opt.step()
            sched.step()
            tot, k = tot + loss.item(), k + 1
        res, vloss = evaluate(model, vl, device, res_std, amp)
        rec = {"epoch": ep, "train_loss": tot / max(k, 1), "val_loss": vloss,
               "val_model": res["model"]["all"], "val_driver": res["driver"]["all"],
               "time_s": time.time() - t0}
        with open(run / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        m, d = res["model"]["all"], res["driver"]["all"]
        print(f"ep {ep:3d} train {rec['train_loss']:.4f} val {vloss:.4f} | wind {m['rmse_wind']:.3f} "
              f"(driver {d['rmse_wind']:.3f}) t2m {m['rmse_t2m']:.3f} ({d['rmse_t2m']:.3f}) "
              f"| {rec['time_s']:.0f}s", flush=True)
        state = {"model": model.state_dict(), "epoch": ep, "val_loss": vloss}
        torch.save(state, run / "last.pt")
        if vloss < best:
            best = vloss
            torch.save(state, run / "best.pt")
    print("best val loss", best)


if __name__ == "__main__":
    main()
