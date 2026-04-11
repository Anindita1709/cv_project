from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from data.query_dataset import QueryDataset
from data.reference_dataset import ReferenceDataset
from models.build_model import build_anyloc, build_fine_matcher, build_resnet, build_segmentation, default_preprocess
from utils.geometry import calculate_center, get_adjacent_matrix, get_patches
from utils.scoring import object_aware_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def segment_and_encode(image_path: Path, seg_model, object_model, score_threshold: float = 0.6) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    image = Image.open(image_path).convert("RGB")
    outputs = seg_model.generate(image, score_threshold=score_threshold)

    centers = [calculate_center(output["bbox"]) for output in outputs]
    bboxes = [output["bbox"] for output in outputs]
    adjacency = get_adjacent_matrix(centers) if centers else np.zeros((0, 0), dtype=int)
    patches = get_patches(bboxes, adjacency) if bboxes else np.zeros((0, 4), dtype=int)

    preprocess = default_preprocess(224)
    image_np = np.asarray(image)

    object_crops = []
    for out in outputs:
        mask = out["segmentation"]
        masked = image_np * mask[:, :, None]
        object_crops.append(Image.fromarray(masked.astype(np.uint8)))

    patch_crops = []
    h_img, w_img = image_np.shape[:2]
    for patch in patches:
        x, y, w, h = [int(v) for v in patch.tolist()]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        if w > 0 and h > 0:
            patch_crops.append(Image.fromarray(image_np[y:y+h, x:x+w].astype(np.uint8)))

    def encode(crops):
        if not crops:
            return None
        batch = torch.stack([preprocess(c.convert("RGB")) for c in crops], dim=0).to(DEVICE)
        return object_model(batch).cpu()

    objects_feature = encode(object_crops)
    patches_feature = encode(patch_crops)
    return image, objects_feature, patches_feature


@torch.no_grad()
def load_reference_db(dataset_root: Path, dataset_name: str) -> list[dict]:
    db = []
    reference_dataset = ReferenceDataset(dataset_root, dataset_name)
    for room in reference_dataset:
        room_feature = torch.load(room.embed_dir / "room_feature.pt", map_location="cpu")
        objects = torch.load(room.embed_dir / "objects.pt", map_location="cpu")
        patches = torch.load(room.embed_dir / "patches.pt", map_location="cpu")
        db.append(
            {
                "scene": room.scene,
                "room": room.room,
                "ref_rgb_path": room.ref_rgb_path,
                "room_feature": room_feature.float().view(1, -1),
                "objects": None if objects is None else objects.float(),
                "patches": None if patches is None else patches.float(),
            }
        )
    return db


@torch.no_grad()
def cosine_topk(query_feature: torch.Tensor, reference_db: list[dict], k: int = 5) -> list[dict]:
    q = torch.nn.functional.normalize(query_feature.view(1, -1), dim=1)
    refs = torch.cat([torch.nn.functional.normalize(item["room_feature"], dim=1) for item in reference_db], dim=0)
    sims = (q @ refs.T).squeeze(0)
    indices = torch.argsort(sims, descending=True)[:k].tolist()
    out = []
    for idx in indices:
        item = dict(reference_db[idx])
        item["global_score"] = float(sims[idx].item())
        out.append(item)
    return out


@torch.no_grad()
def run_inference(config: dict) -> list[dict]:
    dataset_root = Path(config.get("dataset_root", "datasets"))
    dataset_name = config["dataset_name"]
    query_dataset = QueryDataset(dataset_root, dataset_name, exclude_reference=False)
    reference_db = load_reference_db(dataset_root, dataset_name)

    global_extractor, _ = build_anyloc(device=DEVICE)
    seg_model = build_segmentation(config.get("segmentation_type", "maskrcnn"), device=DEVICE)
    object_model = build_resnet(config.get("object_backbone", "resnet50")).to(DEVICE).eval()
    fine_matcher = build_fine_matcher(device=DEVICE)

    top5_k = int(config.get("top5_k", 5))
    top2_k = int(config.get("top2_k", 2))
    patch_mode = config.get("patch_score_mode", "max")
    object_mode = config.get("object_score_mode", "mean")
    seg_thr = float(config.get("segmentation_threshold", 0.6))

    results = []
    for query in query_dataset:
        query_image = Image.open(query.image_path).convert("RGB")
        query_global = global_extractor.encode_pil(query_image)
        top5 = cosine_topk(query_global, reference_db, k=top5_k)

        _, q_objects, q_patches = segment_and_encode(query.image_path, seg_model, object_model, score_threshold=seg_thr)
        rescored = []
        for candidate in top5:
            score_parts = object_aware_score(
                global_score=candidate["global_score"],
                q_patches=q_patches,
                r_patches=candidate["patches"],
                q_objects=q_objects,
                r_objects=candidate["objects"],
                patch_mode=patch_mode,
                object_mode=object_mode,
            )
            enriched = dict(candidate)
            enriched.update(score_parts)
            rescored.append(enriched)
        rescored.sort(key=lambda x: x["total"], reverse=True)
        top2 = rescored[:top2_k]

        if len(top2) == 1:
            final = top2[0]
            fine_matches = 0
        else:
            q_img = Image.open(query.image_path).convert("RGB")
            fine_counts = []
            for cand in top2:
                ref_img = Image.open(cand["ref_rgb_path"]).convert("RGB")
                fine_counts.append(fine_matcher.count_matches(q_img, ref_img))
            best_idx = int(np.argmax(fine_counts))
            final = top2[best_idx]
            fine_matches = int(fine_counts[best_idx])

        correct = int(final["room"] == query.room)
        results.append(
            {
                "scene": query.scene,
                "query_room": query.room,
                "query_image": str(query.image_path),
                "pred_room": final["room"],
                "pred_scene": final["scene"],
                "reference_image": str(final["ref_rgb_path"]),
                "correct": correct,
                "global_score": float(final["global"]),
                "patch_score": float(final["patch"]),
                "object_score": float(final["object"]),
                "total_score": float(final["total"]),
                "fine_matches": fine_matches,
            }
        )
        print(f"query={query.image_path.name} gt={query.room} pred={final['room']} correct={correct}")

    return results


def save_results(results: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def summarize(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(r["correct"] for r in results)
    acc = correct / total if total else 0.0
    return {"num_queries": total, "correct": correct, "accuracy": acc}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = load_config(Path("config") / "inference.yaml")
    results = run_inference(config)
    output_csv = Path(config.get("output_csv", "results/inference_results.csv"))
    save_results(results, output_csv)
    summary = summarize(results)
    print(summary)
