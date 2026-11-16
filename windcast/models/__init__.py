"""Operator zoo. Every model maps x (B, C, H, W) to a normalised residual (B, V, H, W) on
the same grid; its last layer is zero-initialised, so an untrained model returns the
driver. Models handle padding to the sizes they need themselves."""
from windcast.models.fno import FNO
from windcast.models.unet import UNet

MODELS = {"unet", "fno", "ufno"}


def build_model(cfg: dict, n_in: int, n_out: int):
    m = cfg["model"]
    name = m["name"]
    if name == "unet":
        return UNet(n_in, n_out, m.get("width", 32), m.get("depth", 4))
    if name == "fno":
        return FNO(n_in, n_out, m.get("width", 64), m.get("modes", 32), m.get("layers", 4))
    if name == "ufno":
        return FNO(n_in, n_out, m.get("width", 48), m.get("modes", 32), m.get("layers", 6),
                   unet_layers=m.get("unet_layers", 3))
    raise ValueError(f"unknown model {name}; available: {sorted(MODELS)}")


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
