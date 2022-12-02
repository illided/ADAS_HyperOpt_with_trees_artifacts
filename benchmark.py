import typing as tp
from numpy import ndarray
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from rich.progress import track
from abc import ABC, abstractmethod
from metrics import Jaccard, Dice

import json


class Benchmark(ABC):
	def __init__(self) -> None:
		self.img: tp.Optional[ndarray] = None
		self.gt: tp.Optional[ndarray] = None
		self.scores: defaultdict[str, tp.List[float]] = defaultdict(lambda: [])
		self.res = {}

	def run(self, ds: Dataset, save_res: tp.Optional[str]=None) -> tp.Dict[str, tp.List]:
		collate_fn = lambda x: x[0]
		loader = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=collate_fn)
		for batch in track(loader, description="Running benchmark"):
			self.img, self.gt = batch
			metric_dict = self.get_metrics()
			for k, v in metric_dict.items():
				self.scores[k].append(v)
		if save_res is not None:
			self.save(save_res)
		return self.scores

	def result(self) -> tp.Dict[str, float]:
		self.res = {}
		for k, v in self.scores.items():
			self.res[k] = np.mean(v)
		return self.res

	def save(self, filename: str):
		with open(filename, "w") as file:
			file.write(json.dumps(self.result()))
	
	@abstractmethod
	def get_metrics(self) -> tp.Dict[str, float]:
		...

  

def binarize(img: ndarray, v: int) -> ndarray:
	img = np.array(img)
	img[img < v] = 0
	img[img >= v] = 1
	return img


class ImgToEdgeBenchmark(Benchmark):
	def __init__(self, method: tp.Callable[[ndarray], ndarray], bin_v: int=250):
		super().__init__()
		self.method = method
		self.bin_v = bin_v
		self.jaccard_coef = Jaccard()
		self.dice_coef = Dice()
	
	def get_metrics(self) -> tp.Dict[str, float]:
		prediction = binarize(self.method(self.img), self.bin_v)
		gt = binarize(self.gt, self.bin_v)
		return {
			"jaccard": self.jaccard_coef.score(prediction, gt),
			"dice": self.dice_coef.score(prediction, gt)
		}

class OverfittingBenchmark(Benchmark):
	def __init__(self, method: tp.Callable[[ndarray, ndarray], ndarray], bin_v: int=250):
		super().__init__()
		self.method = method
		self.bin_v = bin_v
		self.jaccard_coef = Jaccard()
		self.dice_coef = Dice()

	def get_metrics(self) -> tp.Dict[str, float]:
		imitation = binarize(self.method(self.img, self.gt), self.bin_v)
		gt = binarize(self.gt, self.bin_v)
		return {
			"jaccard": self.jaccard_coef.score(imitation, gt),
			"dice": self.dice_coef.score(imitation, gt)
		}
