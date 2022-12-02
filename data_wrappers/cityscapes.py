from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import typing as tp
from os.path import join, isdir
from pathlib import Path
from torchvision.datasets import Cityscapes
from torchvision.transforms import Resize
from rich.progress import track
from .utils import segmentation_to_edge

class CityScapesEdges(Dataset):
	def __init__(self, root, holdout, pic_size=(512, 1024)) -> None:
		super().__init__()
		if holdout not in ["train", "val"]:
			raise ValueError("This holdout not supported yet")

		self.init_image_loader(root, holdout, pic_size)
		self.path_to_contours = join(root, "gtEdges", f"pic_size_{pic_size[0]}_{pic_size[1]}", holdout)
		if not isdir(self.path_to_contours):
			print("Couldn't find image edges")
		Path(self.path_to_contours).mkdir(parents=True, exist_ok=True)
		self._setup_contours()
	
	def init_image_loader(self, root, holdout, pic_size):
		self.tv_ds = Cityscapes(root,
								split=holdout,
								mode="fine",
								target_type='semantic',
								transform=Resize(size=pic_size),
								target_transform=Resize(size=pic_size))
		
	def get_img_segmentation(self, i) -> tp.Tuple[np.ndarray, np.ndarray]:
		return self.tv_ds[i]
	
	def _setup_contours(self):
		for i in track(range(len(self)), "Building contours"):
			image, segmentation = self.get_img_segmentation(i)

			image = np.array(image)
			segmentation = np.array(segmentation)

			edges = segmentation_to_edge(segmentation)
			cv.imwrite(self._contour_file(i), edges)
  
	def __getitem__(self, index) -> tp.Tuple[np.ndarray, np.ndarray]:
		image, _ = self.get_img_segmentation(index)
		image = np.array(image)
		contours = cv.imread(self._contour_file(index), cv.IMREAD_GRAYSCALE)
		return image, contours
  
	def _contour_file(self, i) -> str:
		return join(self.path_to_contours, f"{i}.png")

	def __len__(self) -> int:
		return len(self.tv_ds)


class CityScapesRain(Dataset):
	def __init__(self, root, holdout, pic_size=(512, 1024)):
		super().__init__()
		
		if holdout not in ["train", "val"]:
			raise ValueError("This holdout not supported yet")
			
		with open(join(root, "rain_trainval_filenames.txt")) as file:
			self.pic_names = file.read().splitlines()
		self.pic_names = [n for n in self.pic_names if n.startswith(holdout)]
		self.pic_suffixes = self.generate_img_suffixes()
		self.root = root
		self.holdout = holdout
		self.pic_size = pic_size
		self.path_to_contours = join(root, "gtEdges", f"pic_size_{pic_size[0]}_{pic_size[1]}", holdout)
		if not isdir(self.path_to_contours):
			print("Couldn't find image edges")
			Path(self.path_to_contours).mkdir(parents=True, exist_ok=True)
			self._setup_contours()
		
	@staticmethod
	def generate_img_suffixes(n_rain_patterns=12, abd_comb=None):
		if abd_comb is None:
			abd_comb = [("0.01", "0.005", "0.01"), 
						("0.02", "0.01", "0.005"), 
						("0.03", "0.015", "0.002")]
		suffixes = []
		for rp in range(1, n_rain_patterns + 1):
			for alpha, beta, dropsize in abd_comb:
				suffix = f"_leftImg8bit_rain_alpha_{alpha}_beta_{beta}_dropsize_{dropsize}_pattern_{rp}.png"
				suffixes.append(suffix)
		return suffixes
	
	def _setup_contours(self):
		for img in self.pic_names:
			segmentation = self._get_segmentation(img)
			segmentation = cv.resize(segmentation, self.pic_size[::-1])
			edges = segmentation_to_edge(segmentation)
			edge_path = self._get_contour_path(img)
			Path(join(*edge_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
			cv.imwrite(self._get_contour_path(img), edges)
	
	def _get_segmentation(self, img_name) -> np.ndarray:
		return cv.imread(join(self.root, "gtFine", img_name + "_gtFine_labelIds.png"), cv.IMREAD_GRAYSCALE)
		
	def _get_contour_path(self, img_name) -> str:
		return join(self.path_to_contours, img_name.removeprefix(self.holdout + "/") + "_edges.png")
	
	def __getitem__(self, index):
		image_name = self.pic_names[index // len(self.pic_names)]
		image_suffix = self.pic_suffixes[index % len(self.pic_suffixes)]
		img = cv.imread(join(self.root, "leftImg8bit_rain", image_name + image_suffix))
		img = cv.resize(img, self.pic_size[::-1])
		edges = cv.imread(self._get_contour_path(image_name))
		return img, edges
		
		
	def __len__(self) -> int:
		return len(self.pic_names) * len(self.pic_suffixes)