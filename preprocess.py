from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.cluster import KMeans

from models.build_model import (
    build_anyloc,
    build_clip_selector,
    build_resnet,
    build_segmentation,
    default_preprocess,
)
from utils.geometry import calculate_center, get_adjacent_matrix, get_patches

GREEN = "\033[92m"
RESET = "\033[0m"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def select_reference_images(
    rgb_path: Path,
    ref_rgb_path: Path,
    clip_selector,
    num_references: int = 3,
    mode: str = "closest_to_center",
) -> list[Path]:
    image_files = sorted(
        [p for p in rgb_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}]
    )
    if not image_files:
        raise RuntimeError(f"No RGB images found in {rgb_path}")

    feats = []
    for image_file in image_files:
        image = Image.open(image_file).convert("RGB")
        feats.append(clip_selector.encode_pil(image).numpy())

    features = np.vstack(feats)

    # if room has fewer images than requested references, just use all
    if len(image_files) <= num_references:
        selected_indices = list(range(len(image_files)))
    else:
        kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
        kmeans.fit(features)
        center = kmeans.cluster_centers_[0]

        dists = np.linalg.norm(features - center[None, :], axis=1)
        sorted_indices = np.argsort(dists)

        if mode == "closest_to_center":
            selected_indices = sorted_indices[:num_references].tolist()
        elif mode == "evenly_spread":
            positions = np.linspace(0, len(sorted_indices) - 1, num_references, dtype=int)
            selected_indices = sorted_indices[positions].tolist()
        else:
            raise ValueError(f"Unsupported reference selection mode: {mode}")

    selected_paths = []
    for out_idx, src_idx in enumerate(selected_indices):
        src = image_files[int(src_idx)]
        dst = ref_rgb_path / f"{out_idx}.png"
        shutil.copy(src, dst)
        selected_paths.append(dst)

    return selected_paths


@torch.no_grad()
def segmentation(seg_model, image_path: Path, score_threshold: float = 0.6) -> tuple[list[np.ndarray], np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    outputs = seg_model.generate(image, score_threshold=score_threshold)

    if not outputs:
        return [], np.zeros((0, 4), dtype=int)

    centers = [calculate_center(output["bbox"]) for output in outputs]
    bboxes = [output["bbox"] for output in outputs]
    adjacency = get_adjacent_matrix(centers)
    patches = get_patches(bboxes, adjacency)
    masks = [output["segmentation"] for output in outputs]
    return masks, patches


@torch.no_grad()
def encode_crops(model, crops: list[Image.Image], batch_size: int = 32) -> torch.Tensor | None:
    if not crops:
        return None

    preprocess = default_preprocess(224)
    feats = []

    for start in range(0, len(crops), batch_size):
        batch_imgs = crops[start : start + batch_size]
        batch = torch.stack([preprocess(img.convert("RGB")) for img in batch_imgs], dim=0).to(DEVICE)
        feat = model(batch).cpu()
        feats.append(feat)

    return torch.cat(feats, dim=0)


@torch.no_grad()
def save_object_and_patch_embedding_for_one_ref(
    image_path: Path,
    embed_path: Path,
    ref_idx: int,
    masks: list[np.ndarray],
    patches: np.ndarray,
    model,
) -> None:
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)

    object_crops: list[Image.Image] = []
    for mask in masks:
        masked = image_np * mask[:, :, None]
        object_crops.append(Image.fromarray(masked.astype(np.uint8)))

    patch_crops: list[Image.Image] = []
    h_img, w_img = image_np.shape[:2]
    for patch in patches:
        x, y, w, h = [int(v) for v in patch.tolist()]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        if w <= 0 or h <= 0:
            continue
        crop = image_np[y : y + h, x : x + w]
        patch_crops.append(Image.fromarray(crop.astype(np.uint8)))

    object_feats = encode_crops(model, object_crops)
    patch_feats = encode_crops(model, patch_crops)

    torch.save(object_feats, embed_path / f"objects_{ref_idx}.pt")
    torch.save(patch_feats, embed_path / f"patches_{ref_idx}.pt")


@torch.no_grad()
def save_room_embedding(image_dir: Path, embed_path: Path, global_extractor) -> None:
    feat = []
    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        image = Image.open(img_path).convert("RGB")
        feat.append(global_extractor.encode_pil(image))

    if not feat:
        raise RuntimeError(f"No reference image found in {image_dir}")

    room_feature = torch.stack(feat, dim=0).mean(dim=0)
    torch.save(room_feature, embed_path / "room_feature.pt")


@torch.no_grad()
def main(config: dict) -> None:
    dataset_path = Path(config.get("dataset_root", "datasets")) / config["dataset_name"]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    seg_model = build_segmentation(config.get("segmentation_type", "maskrcnn"), device=DEVICE)
    object_model = build_resnet(config.get("object_backbone", "resnet50")).to(DEVICE).eval()
    clip_selector = build_clip_selector(device=DEVICE)
    global_extractor, _ = build_anyloc(device=DEVICE)

    num_references = int(config.get("num_references", 3))
    selection_mode = config.get("reference_selection_mode", "closest_to_center")

    for scene in sorted(dataset_path.iterdir()):
        if not scene.is_dir():
            continue

        for room in sorted(scene.iterdir()):
            if not room.is_dir():
                continue

            rgb_path = room / "rgb"
            if not rgb_path.exists():
                continue

            ref_path = room / "ref"
            ref_rgb_path = ref_path / "rgb"
            embed_path = ref_path / "embed"

            if ref_path.exists() and config.get("overwrite", True):
                shutil.rmtree(ref_path)

            ref_rgb_path.mkdir(parents=True, exist_ok=True)
            embed_path.mkdir(parents=True, exist_ok=True)

            selected_refs = select_reference_images(
                rgb_path=rgb_path,
                ref_rgb_path=ref_rgb_path,
                clip_selector=clip_selector,
                num_references=num_references,
                mode=selection_mode,
            )

            for ref_idx, selected_ref in enumerate(selected_refs):
                masks, patches = segmentation(
                    seg_model,
                    selected_ref,
                    score_threshold=float(config.get("segmentation_threshold", 0.6)),
                )
                save_object_and_patch_embedding_for_one_ref(
                    selected_ref,
                    embed_path,
                    ref_idx,
                    masks,
                    patches,
                    object_model,
                )

            save_room_embedding(ref_rgb_path, embed_path, global_extractor)
            print(f"{GREEN}Finished processing {scene.name}/{room.name}{RESET}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = load_config(Path("config") / "preprocess.yaml")
    main(config)