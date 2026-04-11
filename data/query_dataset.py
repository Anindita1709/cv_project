from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class QueryImage:
    dataset: str
    scene: str
    room: str
    image_path: Path


class QueryDataset:
    def __init__(self, dataset_root: str | Path, dataset_name: str, exclude_reference: bool = True) -> None:
        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name
        self.root = self.dataset_root / dataset_name
        self.exclude_reference = exclude_reference
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset not found: {self.root}")
        self.queries = list(self._scan())

    def _scan(self) -> Iterator[QueryImage]:
        for scene_dir in sorted(self.root.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name == "room_label.txt":
                continue
            for room_dir in sorted(scene_dir.iterdir()):
                if not room_dir.is_dir():
                    continue
                ref_path = room_dir / "ref" / "rgb" / "0.png"
                ref_resolved = ref_path.resolve() if ref_path.exists() else None
                rgb_dir = room_dir / "rgb"
                if not rgb_dir.exists():
                    continue
                for img_path in sorted(rgb_dir.iterdir()):
                    if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                        continue
                    if self.exclude_reference and ref_resolved is not None and img_path.resolve() == ref_resolved:
                        continue
                    yield QueryImage(
                        dataset=self.dataset_name,
                        scene=scene_dir.name,
                        room=room_dir.name,
                        image_path=img_path,
                    )

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)
