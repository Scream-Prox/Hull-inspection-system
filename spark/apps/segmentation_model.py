from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v2


ARTICLE_CLASSIFIER_CLASS_IDS = [1, 9, 6, 2]
ARTICLE_CLASS_NAME_TO_ID = {
    "normal": 1,
    "ship_hull": 1,
    "marine_growth": 2,
    "paint_peel": 6,
    "corrosion": 9,
}
NORMAL_CLASS_ID = 1


class ConvBnRelu(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
    ):
        if padding is None:
            padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class FlatDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = FlatDoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = FlatDoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class DAMBlock(nn.Module):
    def __init__(self, channels: int, branches: int = 5, branch_channels: int = 204):
        super().__init__()
        dilations = [1, 2, 4, 8, 16][:branches]
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        branch_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for dilation in dilations
            ]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(branch_channels * branches, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.proj(merged)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(self.pool(x))


class RAMBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = FlatDoubleConv(channels, channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 16, 1), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 16, 1), channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined = self.conv(x)
        refined = refined * self.se(refined)
        return refined + x


class FFMBlock(nn.Module):
    def __init__(self, channels: int, mask_channels: int = 1):
        super().__init__()
        self.x_proj = FlatDoubleConv(channels, channels // 2)
        self.y_proj = FlatDoubleConv(channels, channels // 2)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(mask_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
        if mask_logits.shape[2:] != x.shape[2:]:
            mask_logits = nn.functional.interpolate(
                mask_logits,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        x_feat = self.x_proj(x)
        y_feat = self.y_proj(y)
        fused = torch.cat([x_feat, y_feat], dim=1)
        gate = torch.sigmoid(self.mask_proj(mask_logits))
        return fused * gate


class ClassifierHead(nn.Module):
    def __init__(self, in_channels: int = 1024, hidden_channels: int = 512, classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            FlatDoubleConv(in_channels, hidden_channels),
            FlatDoubleConv(hidden_channels, hidden_channels),
            FlatDoubleConv(hidden_channels, hidden_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            FlatDoubleConv(hidden_channels, hidden_channels),
            FlatDoubleConv(hidden_channels, hidden_channels),
            FlatDoubleConv(hidden_channels, hidden_channels),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avg_pool(x).flatten(1)
        return self.fc(x)


class MTHINet(nn.Module):
    def __init__(self, in_channels: int = 3, seg_classes: int = 1, cls_classes: int = 10, base_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.stack1 = nn.ModuleDict(
            {
                "inc": FlatDoubleConv(in_channels, c1),
                "down1": DownBlock(c1, c2),
                "down2": DownBlock(c2, c3),
                "down3": DownBlock(c3, c4),
                "down4": DownBlock(c4, c5),
                "dam": DAMBlock(c5),
                "ram": RAMBlock(c5),
                "up1": DecoderBlock(c5, c4, c4),
                "up2": DecoderBlock(c4, c3, c3),
                "up3": DecoderBlock(c3, c2, c2),
                "out": FlatDoubleConv(c2, c2),
            }
        )

        self.stack2 = nn.ModuleDict(
            {
                "down1": DownBlock(c2, c3),
                "down2": DownBlock(c3, c4),
                "down3": DownBlock(c4, c5),
                "dam": DAMBlock(c5),
                "ram": RAMBlock(c5),
                "up1": DecoderBlock(c5, c4, c4),
                "up2": DecoderBlock(c4, c3, c3),
                "up3": DecoderBlock(c3, c2, c2),
                "up4": DecoderBlock(c2, c1, c1),
                "out": nn.Conv2d(c1, seg_classes, kernel_size=1),
            }
        )

        self.rrm = nn.ModuleDict(
            {
                "pre": FlatDoubleConv(seg_classes, c1),
                "down1": DownBlock(c1, c1),
                "down2": DownBlock(c1, c1),
                "down3": DownBlock(c1, c1),
                "down4": DownBlock(c1, c1),
                "up1": DecoderBlock(c1, c1, c1),
                "up2": DecoderBlock(c1, c1, c1),
                "up3": DecoderBlock(c1, c1, c1),
                "up4": DecoderBlock(c1, c1, c1),
                "post": nn.Sequential(
                    FlatDoubleConv(c1, c1),
                    nn.Conv2d(c1, seg_classes, kernel_size=1),
                ),
            }
        )
        self.ffm = FFMBlock(c5, mask_channels=seg_classes)
        self.classifier = ClassifierHead(c5, hidden_channels=c4, classes=cls_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1 = self.stack1["inc"](x)
        s2 = self.stack1["down1"](s1)
        s3 = self.stack1["down2"](s2)
        s4 = self.stack1["down3"](s3)
        s5 = self.stack1["down4"](s4)
        s5 = self.stack1["dam"](s5)
        s5 = self.stack1["ram"](s5)

        u1 = self.stack1["up1"](s5, s4)
        u2 = self.stack1["up2"](u1, s3)
        u3 = self.stack1["up3"](u2, s2)
        coarse_half = self.stack1["out"](u3)

        t1 = self.stack2["down1"](coarse_half)
        t2 = self.stack2["down2"](t1)
        t3 = self.stack2["down3"](t2)
        t3 = self.stack2["dam"](t3)
        t3 = self.stack2["ram"](t3)

        v1 = self.stack2["up1"](t3, t2)
        v2 = self.stack2["up2"](v1, t1)
        v3 = self.stack2["up3"](v2, coarse_half)
        v4 = self.stack2["up4"](v3, s1)
        seg_logits = self.stack2["out"](v4)

        r0 = self.rrm["pre"](seg_logits)
        r1 = self.rrm["down1"](r0)
        r2 = self.rrm["down2"](r1)
        r3 = self.rrm["down3"](r2)
        r4 = self.rrm["down4"](r3)
        rr1 = self.rrm["up1"](r4, r3)
        rr2 = self.rrm["up2"](rr1, r2)
        rr3 = self.rrm["up3"](rr2, r1)
        rr4 = self.rrm["up4"](rr3, r0)
        seg_logits = self.rrm["post"](rr4) + seg_logits

        cls_features = self.ffm(s5, t3, seg_logits)
        cls_logits = self.classifier(cls_features)
        return {
            "segmentation_logits": seg_logits,
            "classification_logits": cls_logits,
        }


class MobileNetUNet(nn.Module):
    def __init__(self, num_classes: int = 11):
        super().__init__()

        features = mobilenet_v2(weights=None).features
        self.stem = nn.Sequential(OrderedDict((str(i), features[i]) for i in [0, 1]))
        self.block1 = nn.Sequential(OrderedDict((str(i), features[i]) for i in [2, 3]))
        self.block2 = nn.Sequential(OrderedDict((str(i), features[i]) for i in [4, 5, 6]))
        self.block3 = nn.Sequential(OrderedDict((str(i), features[i]) for i in [7, 8, 9, 10, 11, 12, 13]))
        self.block4 = nn.Sequential(OrderedDict((str(i), features[i]) for i in [14, 15, 16, 17]))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(320, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.dec4 = DoubleConv(512 + 96, 256)
        self.dec3 = DoubleConv(256 + 32, 128)
        self.dec2 = DoubleConv(128 + 24, 64)
        self.dec1 = DoubleConv(64 + 16, 32)
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        s1 = self.stem(x)
        s2 = self.block1(s1)
        s3 = self.block2(s2)
        s4 = self.block3(s3)
        s5 = self.block4(s4)

        bottleneck = self.bottleneck(s5)

        d4 = nn.functional.interpolate(bottleneck, size=s4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))

        d3 = nn.functional.interpolate(d4, size=s3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = nn.functional.interpolate(d3, size=s2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = nn.functional.interpolate(d2, size=s1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        logits = self.head(d1)
        return nn.functional.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


@dataclass
class LoadedModelSpec:
    model: nn.Module
    mode: str
    classifier_class_ids: list[int] | None = None
    segmentation_class_ids: list[int] | None = None
    threshold: float = 0.5


def class_ids_from_config_names(names: list[str] | tuple[str, ...] | None) -> list[int]:
    ids: list[int] = []
    for raw_name in names or []:
        normalized = str(raw_name).strip().lower().replace("-", "_").replace(" ", "_")
        class_id = ARTICLE_CLASS_NAME_TO_ID.get(normalized)
        if class_id is not None:
            ids.append(class_id)
    return ids


def predict_with_loaded_model(
    loaded: LoadedModelSpec,
    image_tensor: torch.Tensor,
    *,
    classification_threshold: float = 0.35,
) -> dict:
    with torch.inference_mode():
        raw_output = loaded.model(image_tensor)

    if loaded.mode == "multiclass":
        logits = raw_output
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        unique_ids, counts = np.unique(pred_mask, return_counts=True)
        class_counts = {int(class_id): int(count) for class_id, count in zip(unique_ids, counts)}
        non_void = {class_id: count for class_id, count in class_counts.items() if class_id != 0}
        dominant_class_id = max(non_void, key=non_void.get) if non_void else 0
        predicted_ids = sorted(non_void.keys())
        total_pixels = int(pred_mask.size)
        class_pixel_ratios = {
            class_id: round(count / total_pixels, 6)
            for class_id, count in class_counts.items()
        }
        return {
            "prediction_mode": "multiclass_segmentation",
            "pred_mask": pred_mask,
            "class_pixel_counts": class_counts,
            "class_pixel_ratios": class_pixel_ratios,
            "predicted_class_ids": predicted_ids,
            "dominant_class_id": int(dominant_class_id),
            "classification_scores": {},
        }

    if loaded.mode != "mthinet":
        raise RuntimeError(f"Unsupported loaded model mode: {loaded.mode}")

    seg_logits = raw_output["segmentation_logits"]
    cls_logits = raw_output["classification_logits"]
    threshold = float(loaded.threshold)
    classifier_ids = loaded.classifier_class_ids or []

    if seg_logits.shape[1] > 1:
        seg_probs = torch.sigmoid(seg_logits).squeeze(0).cpu()
        max_prob, max_index = torch.max(seg_probs, dim=0)
        active_mask = max_prob.numpy() >= threshold
        segmentation_ids = loaded.segmentation_class_ids or classifier_ids
        pred_mask = np.zeros(active_mask.shape, dtype=np.uint8)
        for channel_index, class_id in enumerate(segmentation_ids[: seg_probs.shape[0]]):
            channel_mask = (max_index.numpy() == channel_index) & active_mask
            pred_mask[channel_mask] = int(class_id)

        unique_ids, counts = np.unique(pred_mask, return_counts=True)
        class_counts = {int(class_id): int(count) for class_id, count in zip(unique_ids, counts)}
        active_counts = {class_id: count for class_id, count in class_counts.items() if class_id != 0 and count > 0}
        dominant_class_id = max(active_counts, key=active_counts.get) if active_counts else NORMAL_CLASS_ID
        total_pixels = int(pred_mask.size)
        class_pixel_ratios = {
            class_id: round(count / total_pixels, 6)
            for class_id, count in class_counts.items()
        }

        cls_scores = torch.sigmoid(cls_logits).squeeze(0).cpu().tolist()
        score_pairs = list(zip(classifier_ids, [float(score) for score in cls_scores[: len(classifier_ids)]]))
        return {
            "prediction_mode": "multiclass_mthinet_segmentation",
            "pred_mask": pred_mask,
            "class_pixel_counts": class_counts,
            "class_pixel_ratios": class_pixel_ratios,
            "predicted_class_ids": sorted(active_counts.keys()),
            "dominant_class_id": int(dominant_class_id),
            "classification_scores": {int(class_id): round(score, 6) for class_id, score in score_pairs},
        }

    fg_prob = torch.sigmoid(seg_logits).squeeze(0).squeeze(0).cpu().numpy()
    fg_mask = fg_prob >= 0.5
    cls_scores = torch.sigmoid(cls_logits).squeeze(0).cpu().tolist()
    score_pairs = list(zip(classifier_ids, [float(score) for score in cls_scores[: len(classifier_ids)]]))
    score_pairs.sort(key=lambda item: item[1], reverse=True)

    active_pairs = [pair for pair in score_pairs if pair[1] >= classification_threshold]
    if not active_pairs and score_pairs:
        active_pairs = [score_pairs[0]]

    total_pixels = int(fg_mask.size)
    foreground_pixels = int(fg_mask.sum())
    defect_pairs = [pair for pair in active_pairs if pair[0] != NORMAL_CLASS_ID]
    dominant_class_id = int(defect_pairs[0][0]) if defect_pairs else NORMAL_CLASS_ID
    pred_mask = np.zeros(fg_mask.shape, dtype=np.uint8)
    effective_foreground_pixels = foreground_pixels if defect_pairs else 0
    if effective_foreground_pixels > 0 and dominant_class_id and dominant_class_id != NORMAL_CLASS_ID:
        pred_mask[fg_mask] = dominant_class_id

    class_counts = {0: total_pixels - effective_foreground_pixels}
    if effective_foreground_pixels > 0 and defect_pairs:
        positive_total = sum(score for _, score in defect_pairs) or 1.0
        assigned = 0
        for idx, (class_id, score) in enumerate(defect_pairs):
            if idx == len(defect_pairs) - 1:
                count = effective_foreground_pixels - assigned
            else:
                count = int(round(effective_foreground_pixels * (score / positive_total)))
                count = max(0, min(count, effective_foreground_pixels - assigned))
                assigned += count
            class_counts[int(class_id)] = class_counts.get(int(class_id), 0) + int(count)
    else:
        class_counts[NORMAL_CLASS_ID] = total_pixels

    class_pixel_ratios = {
        class_id: round(count / total_pixels, 6)
        for class_id, count in class_counts.items()
    }
    return {
        "prediction_mode": "binary_segmentation_with_classification",
        "pred_mask": pred_mask,
        "class_pixel_counts": class_counts,
        "class_pixel_ratios": class_pixel_ratios,
        "predicted_class_ids": [int(class_id) for class_id, _ in defect_pairs],
        "dominant_class_id": int(dominant_class_id),
        "classification_scores": {int(class_id): round(score, 6) for class_id, score in score_pairs},
    }


def load_checkpoint_payload(checkpoint_path: str, device: torch.device):
    return torch.load(checkpoint_path, map_location=device, weights_only=True)


def build_mthinet_from_payload(payload: dict, device: torch.device) -> LoadedModelSpec:
    config = payload.get("config", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    in_channels = int(model_cfg.get("in_channels", 3))
    seg_classes = int(model_cfg.get("seg_classes", 1))
    cls_classes = int(model_cfg.get("cls_classes", 10))
    base_channels = int(model_cfg.get("base_channels", 64))
    dropout = float(model_cfg.get("dropout", 0.2))

    model = MTHINet(
        in_channels=in_channels,
        seg_classes=seg_classes,
        cls_classes=cls_classes,
        base_channels=base_channels,
        dropout=dropout,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.eval()
    classifier_ids = class_ids_from_config_names(data_cfg.get("classification_classes"))
    if not classifier_ids:
        classifier_ids = ARTICLE_CLASSIFIER_CLASS_IDS[: min(cls_classes, len(ARTICLE_CLASSIFIER_CLASS_IDS))]
    segmentation_ids = class_ids_from_config_names(data_cfg.get("crop_focus_classes"))
    if not segmentation_ids:
        segmentation_ids = classifier_ids[: min(seg_classes, len(classifier_ids))]
    return LoadedModelSpec(
        model=model,
        mode="mthinet",
        classifier_class_ids=classifier_ids[:cls_classes],
        segmentation_class_ids=segmentation_ids[:seg_classes],
        threshold=float(train_cfg.get("threshold", 0.5)),
    )


def build_mobilenet_unet_from_state_dict(state_dict: dict, device: torch.device, num_classes: int = 11) -> LoadedModelSpec:
    model = MobileNetUNet(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return LoadedModelSpec(model=model, mode="multiclass", classifier_class_ids=None)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device, num_classes: int = 11) -> LoadedModelSpec:
    payload = load_checkpoint_payload(checkpoint_path, device)

    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], (dict, OrderedDict)):
        model_cfg = payload.get("config", {}).get("model", {})
        if model_cfg.get("name") == "mthinet" or any(str(key).startswith("stack1.") for key in payload["model"].keys()):
            return build_mthinet_from_payload(payload, device)
        return build_mobilenet_unet_from_state_dict(payload["model"], device=device, num_classes=num_classes)

    if isinstance(payload, (dict, OrderedDict)):
        return build_mobilenet_unet_from_state_dict(payload, device=device, num_classes=num_classes)

    raise RuntimeError(f"Unsupported checkpoint format for {checkpoint_path}")
