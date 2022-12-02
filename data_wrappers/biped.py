from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import typing as tp
from .utils import get_filenames


class BIPED(Dataset):
	def __init__(self, path: str, holdout: str) -> None:
		super().__init__()
		self.path = path
		self.data = self._load_holdout(holdout)

	def _load_holdout(self, holdout) -> tp.List[tp.Tuple[np.ndarray, np.ndarray]]:
		main_dir = f"{self.path}/BIPED/edges"
		imgs_dir = f"{main_dir}/imgs/{holdout}/rgbr"
		edges_dir = f"{main_dir}/edge_maps/{holdout}/rgbr"
		if holdout == "train":
			imgs_dir += "/real"
			edges_dir += "/real"
		imgs = [cv.imread(img, cv.IMREAD_COLOR) for img in sorted(get_filenames(imgs_dir))]
		edges = [cv.imread(edge, cv.IMREAD_GRAYSCALE) for edge in sorted(get_filenames(edges_dir))]
		return list(zip(imgs, edges))

	def __getitem__(self, index) -> tp.Tuple[np.ndarray, np.ndarray]:
		return self.data[index]

	def __len__(self) -> int:
		return len(self.data)