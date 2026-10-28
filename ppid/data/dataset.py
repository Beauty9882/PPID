from __future__ import annotations

import os
from typing import Callable, List, Optional

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class FaceImageDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None, max_images: Optional[int] = None):
        self.root = root
        self.transform = transform
        paths: List[str] = [
            os.path.join(root, n)
            for n in sorted(os.listdir(root))
            if os.path.splitext(n)[1].lower() in IMG_EXTS
        ]
        if max_images is not None:
            paths = paths[:max_images]
        if not paths:
            raise FileNotFoundError(f"No images found under {root}")
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        parent = os.path.basename(os.path.dirname(path))
        root_name = os.path.basename(self.root.rstrip("/"))
        label = parent if parent != root_name else os.path.splitext(os.path.basename(path))[0]
        return image, label, path
