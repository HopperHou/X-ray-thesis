"""Frozen three-input, three-level V6-A2 quality head for YOLO26 O2O."""
from __future__ import annotations

import torch
from torch import nn

from ultralytics.nn.modules.head import Detect


class LevelQualityHead(nn.Module):
    def __init__(self, cls_channels: int, box_channels: int, nc: int = 6):
        super().__init__()
        self.cls_proj = nn.Sequential(nn.Conv2d(cls_channels, 64, 1), nn.SiLU())
        self.box_proj = nn.Sequential(nn.Conv2d(box_channels, 64, 1), nn.SiLU())
        self.raw_proj = nn.Sequential(nn.Conv2d(4, 16, 1), nn.SiLU())
        self.fusion = nn.Sequential(nn.Conv2d(144, 128, 3, padding=1), nn.SiLU(), nn.Conv2d(128, nc, 1))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, h_cls: torch.Tensor, h_box: torch.Tensor, raw_box: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.cls_proj(h_cls.detach()), self.box_proj(h_box.detach()),
                       self.raw_proj(raw_box.detach())), dim=1)
        return self.fusion(x)


def corrected_logits(z0: torch.Tensor, q_logits: torch.Tensor,
                     delta_plus: float, delta_minus: float, b: float, k: float) -> torch.Tensor:
    if delta_plus < 0 or delta_minus < 0 or k <= 0 or not (0 <= b <= 1):
        raise ValueError("Invalid bounded scorer parameters")
    if z0.shape != q_logits.shape:
        raise ValueError("z0 and q_logits shape mismatch")
    if delta_plus == 0 and delta_minus == 0:
        return z0  # exact native identity, including dtype
    q = q_logits.float().sigmoid()
    g_plus = torch.tanh(k * (q - b)).clamp_min(0)
    g_minus = torch.tanh(k * (b - q)).clamp_min(0)
    return z0.float() + delta_plus * g_plus - delta_minus * g_minus


class V6A2Detect(Detect):
    """Native decode and Top300, with correction applied to every raw slot."""

    def forward(self, x: list[torch.Tensor]):
        if not self.end2end or not hasattr(self, "v6a2_quality_heads"):
            raise RuntimeError("V6-A2 requires installed, unfused O2O head")
        x_detach = [xi.detach() for xi in x]
        bs = x_detach[0].shape[0]
        with torch.no_grad():
            one2many = self.forward_head(x_detach, **self.one2many)
        box_penult, cls_penult, raw_maps, cls_maps, q_maps = [], [], [], [], []
        for level in range(self.nl):
            h_box = self.one2one_cv2[level][:-1](x_detach[level])
            raw = self.one2one_cv2[level][-1](h_box)
            with torch.no_grad():
                h_cls = self.one2one_cv3[level][:-1](x_detach[level])
                cls = self.one2one_cv3[level][-1](h_cls)
            q = self.v6a2_quality_heads[level](h_cls, h_box, raw)
            box_penult.append(h_box)
            cls_penult.append(h_cls)
            raw_maps.append(raw)
            cls_maps.append(cls)
            q_maps.append(q)
        raw_boxes = torch.cat([v.view(bs, 4, -1) for v in raw_maps], dim=2)
        z0 = torch.cat([v.view(bs, self.nc, -1) for v in cls_maps], dim=2)
        qlog = torch.cat([v.view(bs, self.nc, -1) for v in q_maps], dim=2)
        plus, minus, b, k = self.v6a2_scorer
        zfinal = corrected_logits(z0, qlog, plus, minus, b, k)
        one2one = {"boxes": raw_boxes, "scores": zfinal, "feats": x_detach,
                   "original_logits": z0, "quality_logits": qlog,
                   "box_penult_shapes": [list(t.shape) for t in box_penult],
                   "cls_penult_shapes": [list(t.shape) for t in cls_penult]}
        preds = {"one2many": one2many, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)


def install_v6a2_head(model, scorer=(0.0, 0.0, 0.35, 4.0)):
    head = model.model[-1]
    if not isinstance(head, Detect) or not head.end2end or head.nc != 6 or head.nl != 3 or head.reg_max != 1:
        raise RuntimeError("Expected unfused three-level six-class YOLO26 O2O reg_max=1")
    if head.cv2 is None or head.cv3 is None or hasattr(head, "v6a2_quality_heads"):
        raise RuntimeError("Head already fused or modified")
    cls_channels = [h[-1].in_channels for h in head.one2one_cv3]
    box_channels = [h[-1].in_channels for h in head.one2one_cv2]
    if cls_channels != [384, 384, 384] or any(h[-1].out_channels != 4 for h in head.one2one_cv2):
        raise RuntimeError(f"Checkpoint incompatible with frozen head: cls={cls_channels}, box={box_channels}")
    head.__class__ = V6A2Detect
    head.v6a2_quality_heads = nn.ModuleList(LevelQualityHead(c, b) for c, b in zip(cls_channels, box_channels))
    head.v6a2_scorer = tuple(float(v) for v in scorer)
    return {"h_cls_channels": cls_channels, "h_box_channels": box_channels}


def set_trainable(model, mode: str) -> None:
    if mode not in {"q_only", "box_native", "box_geo"}:
        raise ValueError(f"Unknown run mode: {mode}")
    for p in model.parameters():
        p.requires_grad_(False)
    head = model.model[-1]
    for p in head.v6a2_quality_heads.parameters():
        p.requires_grad_(True)
    if mode != "q_only":
        for p in head.one2one_cv2.parameters():
            p.requires_grad_(True)
    # Frozen paths keep their running statistics. The trainable box tower may
    # update its own BatchNorm statistics in the two box modes.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    if mode != "q_only":
        for module in head.one2one_cv2.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.train()
