import random

from numpy import ndarray
import numpy as np
import typing as tp

from canny import canny
from .common import Individual, Environment
from .crossover import blend_crossover
from benchmark import binarize
from metrics import Jaccard


class CannyIndividual(Individual):
	def __init__(self, thr1: tp.Optional[int]=None, thr2: tp.Optional[int]=None, cross_n: int =2):
		if not thr1:
			thr1 = random.randint(0, 255)
		if not thr2:
			thr2 = random.randint(0, 255)
		self.thr1 = thr1
		self.thr2 = thr2
		self.cross_n = cross_n
	
	def mutate(self, ind_mut_pb: float=0.5):
		if random.random() < ind_mut_pb:
			self.thr1 = random.randint(0, 255)
		if random.random() < ind_mut_pb:
			self.thr2 = random.randint(0, 255)

	def _crossover_one(self, other: "CannyIndividual") -> "CannyIndividual":
		crossed_thr1 = round(blend_crossover(self.thr1, other.thr1)[0])
		crossed_thr2 = round(blend_crossover(self.thr2, other.thr2)[0])
		return CannyIndividual(
			random.choice([crossed_thr1, self.thr1, other.thr1]),
			random.choice([crossed_thr2, self.thr2, other.thr2])
			)
	
	def crossover(self, other: "Individual") -> tp.List["Individual"]:
		if not isinstance(other, CannyIndividual):
			raise TypeError("Cant crossover with generic individual")
		return [self._crossover_one(other) for i in range(self.cross_n)]

	def behave(self, img: ndarray) -> ndarray:
		return canny(img, thr1=self.thr1, thr2=self.thr2)

	def to_func(self):
		def f(img):
			return canny(img, thr1=self.thr1, thr2=self.thr2)
		return f


class SimpleEnvironment(Environment):
	def __init__(self, img: ndarray, edges: ndarray,
				scoring: tp.Optional[tp.Callable[[ndarray, ndarray], float]]=None):
		self.img = img
		self.edges = edges

		if scoring is None:
			def f(pred, gt):
				pred = binarize(pred, 250)
				gt = binarize(gt, 250)
				return Jaccard()(pred, gt) - 0.5 * np.mean(pred)
		scoring = f
		self.scoring: tp.Callable[[ndarray, ndarray], float] = scoring
	
	def fitness(self, individual: Individual) -> float:
		return self.scoring(individual.behave(self.img), self.edges)
