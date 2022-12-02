from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import typing as tp
from os import path
from pathlib import Path
from torchvision.datasets import Cityscapes
from torchvision.transforms import Resize
from rich.progress import track
from .utils import segmentation_to_edge, get_filenames


class BDD10kEdges(Dataset):
    def __init__(self, root: str, holdout: str, pic_size: tp.Tuple[int, int]=(720, 1280)) -> None:
        super().__init__()
        self.root = root
        self.holdout = holdout
        self.pic_size = pic_size

        self.img_path = path.join(root, "images", "10k", holdout)
        self.img_names = []
        for img_path in get_filenames(self.img_path):
            self.img_names.append(path.split(img_path)[1].removesuffix(".jpg"))

        self.edge_path = path.join(root, "labels", f"edges_{pic_size[1]}_{pic_size[0]}", holdout)
        if not path.isdir(self.edge_path):
            self._prepare_edges()
    
    def _prepare_edges(self):
        Path(self.edge_path).mkdir(parents=True, exist_ok=True)
        print(f"Couldn't find edges for BDD10k/{self.holdout}")
        segmentation_path = path.join(self.root, "labels", "sem_seg", "colormaps", self.holdout)
        for img_name in track(self.img_names, "Building edges"):
            seg_path = path.join(segmentation_path, img_name + ".png")
            edge_path = path.join(self.edge_path, img_name + ".png")

            segmentation = cv.imread(seg_path, cv.IMREAD_COLOR)
            segmentation, _, _ = cv.split(segmentation)
            edges = segmentation_to_edge(segmentation)

            cv.imwrite(edge_path, edges)


    def __getitem__(self, index) -> tp.Tuple[np.ndarray, np.ndarray]:
        img_name = self.img_names[index]
        img = cv.imread(path.join(self.img_path, img_name + ".jpg"), cv.IMREAD_COLOR)
        edges = cv.imread(path.join(self.edge_path, img_name + ".png"), cv.IMREAD_GRAYSCALE)
        return img, edges

    def __len__(self) -> int:
        return len(self.img_names)