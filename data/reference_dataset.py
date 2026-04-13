from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class ReferenceRoom:
    dataset: str
    scene: str
    room: str
    room_dir: Path
    ref_rgb_paths: list[Path]
    embed_dir: Path


class ReferenceDataset:
    def __init__(self, dataset_root: str | Path, dataset_name: str) -> None:
        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name
        self.root = self.dataset_root / dataset_name

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset not found: {self.root}")

        self.rooms = list(self._scan())

    def _scan(self) -> Iterator[ReferenceRoom]:
        for scene_dir in sorted(self.root.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name == "room_label.txt":
                continue

            for room_dir in sorted(scene_dir.iterdir()):
                if not room_dir.is_dir():
                    continue

                ref_rgb_dir = room_dir / "ref" / "rgb"
                embed_dir = room_dir / "ref" / "embed"

                if not ref_rgb_dir.exists() or not embed_dir.exists():
                    continue

                ref_rgb_paths = sorted(
                    [
                        p for p in ref_rgb_dir.iterdir()
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
                    ]
                )

                if not ref_rgb_paths:
                    continue

                yield ReferenceRoom(
                    dataset=self.dataset_name,
                    scene=scene_dir.name,
                    room=room_dir.name,
                    room_dir=room_dir,
                    ref_rgb_paths=ref_rgb_paths,
                    embed_dir=embed_dir,
                )

    def __len__(self) -> int:
        return len(self.rooms)

    def __iter__(self):
        return iter(self.rooms)