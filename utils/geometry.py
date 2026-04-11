from __future__ import annotations

from typing import Iterable, List

import numpy as np
from scipy.spatial import Delaunay


def calculate_center(bbox: list[int] | tuple[int, int, int, int]) -> list[float]:
    x, y, w, h = bbox
    return [x + w / 2.0, y + h / 2.0]


def get_adjacent_matrix(centers: Iterable[Iterable[float]]) -> np.ndarray:
    center_points = np.array(list(centers), dtype=float)
    n = len(center_points)
    adjacency = np.zeros((n, n), dtype=int)
    if n < 2:
        return adjacency
    if n == 2:
        adjacency[0, 1] = adjacency[1, 0] = 1
        return adjacency
    try:
        tri = Delaunay(center_points)
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = int(simplex[i]), int(simplex[j])
                    adjacency[a, b] = 1
                    adjacency[b, a] = 1
    except Exception:
        # fallback to KNN-like adjacency for degenerate geometry
        for i in range(n):
            d = np.sum((center_points - center_points[i]) ** 2, axis=1)
            nn = np.argsort(d)[1 : min(4, n)]
            adjacency[i, nn] = 1
            adjacency[nn, i] = 1
    return adjacency


def get_patches(bboxes: List[list[int]], adjacency_matrix: np.ndarray, iou_threshold: float = 0.7) -> np.ndarray:
    if len(bboxes) == 0:
        return np.zeros((0, 4), dtype=int)

    bboxes_arr = np.array(bboxes, dtype=int)
    expanded = []
    for i in range(len(bboxes_arr)):
        x_min, y_min = bboxes_arr[i, 0], bboxes_arr[i, 1]
        x_max = bboxes_arr[i, 0] + bboxes_arr[i, 2]
        y_max = bboxes_arr[i, 1] + bboxes_arr[i, 3]
        neighbors = np.where(adjacency_matrix[i] == 1)[0]
        for nb in neighbors:
            nx, ny, nw, nh = bboxes_arr[nb]
            x_min = min(x_min, nx)
            y_min = min(y_min, ny)
            x_max = max(x_max, nx + nw)
            y_max = max(y_max, ny + nh)
        expanded.append([x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min)])

    expanded = np.array(expanded, dtype=int)
    return nms_boxes(expanded, iou_threshold=iou_threshold)


def nms_boxes(boxes: np.ndarray, iou_threshold: float = 0.7) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    areas = boxes[:, 2] * boxes[:, 3]
    order = np.argsort(-areas)
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 0] + boxes[i, 2], boxes[rest, 0] + boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 1] + boxes[i, 3], boxes[rest, 1] + boxes[rest, 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = inter / np.clip(union, 1e-6, None)
        order = rest[iou <= iou_threshold]

    return boxes[np.array(keep, dtype=int)]
