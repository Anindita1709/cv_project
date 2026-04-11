from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNetEmbedder(nn.Module):
    def __init__(self, backbone: str = "resnet50") -> None:
        super().__init__()
        if backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.encoder = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = feat.flatten(1)
        feat = F.normalize(feat, dim=1)
        return feat


class AnyLocGlobalExtractor:
    """
    Lightweight stand-in for the repo's DINOv2 + VLAD pipeline.
    If transformers DINOv2 is available, it uses hidden states and mean pooling.
    Otherwise it falls back to a ResNet global descriptor.
    """

    def __init__(self, device: torch.device = DEVICE) -> None:
        self.device = device
        self.kind = "fallback_resnet"
        self.processor = None
        self.model = None
        self.fallback = ResNetEmbedder("resnet50").to(device).eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        try:
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
            self.kind = "dinov2"
        except Exception:
            # keep fallback
            pass

    @torch.no_grad()
    def encode_pil(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.kind == "dinov2" and self.processor is not None and self.model is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            feat = outputs.last_hidden_state.mean(dim=1)
            return F.normalize(feat, dim=1).squeeze(0).cpu()

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        feat = self.fallback(tensor)
        return feat.squeeze(0).cpu()


class CLIPSelector:
    def __init__(self, device: torch.device = DEVICE) -> None:
        self.device = device
        self.kind = "none"
        self.processor = None
        self.model = None
        try:
            from transformers import CLIPModel, CLIPProcessor

            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
            self.kind = "clip"
        except Exception:
            pass

    @torch.no_grad()
    def encode_pil(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.kind == "clip" and self.model is not None and self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feat = self.model.get_image_features(**inputs)
            return F.normalize(feat, dim=1).squeeze(0).cpu()
        # fallback: simple image statistics, deterministic and dependency-free
        tensor = transforms.ToTensor()(image)
        feat = torch.cat([tensor.mean(dim=(1, 2)), tensor.std(dim=(1, 2))], dim=0)
        return F.normalize(feat.unsqueeze(0), dim=1).squeeze(0).cpu()


class SegmentationWrapper:
    """
    Default: Mask R-CNN from torchvision.
    Optional semantic_sam mode can be plugged in by the user later.
    """

    def __init__(self, mode: str = "maskrcnn", device: torch.device = DEVICE) -> None:
        self.mode = mode
        self.device = device
        self.model = None

        if mode == "maskrcnn":
            weights = models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = models.detection.maskrcnn_resnet50_fpn(weights=weights).to(device).eval()
            self.transforms = weights.transforms()
        elif mode == "semantic_sam":
            raise NotImplementedError(
                "semantic_sam is not bundled in this reimplementation. Use mode='maskrcnn' or plug in your local Semantic-SAM."
            )
        else:
            raise ValueError(f"Unsupported segmentation mode: {mode}")

    @torch.no_grad()
    def generate(self, image: Image.Image, score_threshold: float = 0.6) -> list[dict]:
        assert self.model is not None
        tensor = transforms.ToTensor()(image).to(self.device)
        outputs = self.model([tensor])[0]

        scores = outputs["scores"].detach().cpu()
        boxes = outputs["boxes"].detach().cpu()
        masks = outputs["masks"].detach().cpu()
        labels = outputs["labels"].detach().cpu()

        results: list[dict] = []
        for score, box, mask, label in zip(scores, boxes, masks, labels):
            if float(score) < score_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            bbox = [int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))]
            seg = (mask[0] > 0.5).numpy().astype("uint8")
            results.append(
                {
                    "bbox": bbox,
                    "segmentation": seg,
                    "score": float(score),
                    "label": int(label),
                }
            )
        return results


class FineMatcher:
    """Optional LightGlue-based matcher; falls back to ORB matching."""

    def __init__(self, device: torch.device = DEVICE) -> None:
        self.device = device
        self.kind = "orb"
        self.extractor = None
        self.matcher = None
        try:
            from lightglue import LightGlue, SuperPoint

            self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
            self.matcher = LightGlue(features="superpoint").eval().to(device)
            self.kind = "lightglue"
        except Exception:
            pass

    @torch.no_grad()
    def count_matches(self, img0: Image.Image, img1: Image.Image) -> int:
        if self.kind == "lightglue":
            from torchvision.transforms.functional import to_tensor

            im0 = to_tensor(img0.convert("RGB")).to(self.device)
            im1 = to_tensor(img1.convert("RGB")).to(self.device)
            feats0 = self.extractor.extract(im0)
            feats1 = self.extractor.extract(im1)
            matches = self.matcher({"image0": feats0, "image1": feats1})
            ms = matches["matches"]
            return int(ms.shape[0])

        import cv2
        import numpy as np

        a = np.array(img0.convert("L"))
        b = np.array(img1.convert("L"))
        orb = cv2.ORB_create(2000)
        kp1, des1 = orb.detectAndCompute(a, None)
        kp2, des2 = orb.detectAndCompute(b, None)
        if des1 is None or des2 is None:
            return 0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        return int(len(matches))


def default_preprocess(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_resnet(backbone: str = "resnet50") -> nn.Module:
    return ResNetEmbedder(backbone).eval()


def build_anyloc(root: Optional[Path] = None, num_clusters: int = 32, device: torch.device = DEVICE) -> Tuple[AnyLocGlobalExtractor, None]:
    del root, num_clusters
    return AnyLocGlobalExtractor(device=device), None


def build_segmentation(segmentation_type: str = "maskrcnn", device: torch.device = DEVICE) -> SegmentationWrapper:
    return SegmentationWrapper(segmentation_type, device=device)


def build_clip_selector(device: torch.device = DEVICE) -> CLIPSelector:
    return CLIPSelector(device=device)


def build_fine_matcher(device: torch.device = DEVICE) -> FineMatcher:
    return FineMatcher(device=device)
