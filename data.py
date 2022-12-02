import cv2 as cv
from os import listdir
from os.path import isfile, isdir, join
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from torchvision.transforms import Resize

import typing as tp
import numpy as np

from rich.progress import track


def get_filenames(dir):
  return [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]


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



def segmentation_to_edge(segmentation) -> np.ndarray:
	contour_img = np.zeros_like(segmentation)
	unq_classes = np.unique(segmentation)
	h, w = segmentation.shape[0], segmentation.shape[1]
	for cls in unq_classes:
		cls_img = np.zeros((h, w, 1), dtype=np.uint8)
		cls_img[segmentation == cls] = 255
		contours, _ = cv.findContours(cls_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		contour_img = cv.drawContours(contour_img, contours, -1, (255, 255, 255), 0)
	return contour_img

def apply_roi(img, up_cut=0.39, low_cut=0.77) -> np.ndarray:
	h = img.shape[0]
	up_cut = int(0.39 * h)
	low_cut = int(0.77 * h)
	return img[up_cut:low_cut, :]
	

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