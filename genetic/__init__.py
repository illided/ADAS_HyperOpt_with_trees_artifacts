from .common import Population, Environment, Individual
from .common import select, ellitism, crossover, mutation, best_fitness, run_genetic
from .genetic_canny import CannyIndividual, SimpleEnvironment
import typing as tp
import random
from numpy import ndarray
from canny import canny
from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd
from rich.progress import track

def optimal_canny_param(img: ndarray, edges: ndarray, 
												pop_size: int=20, n_generations: int=7, **kwargs) -> tp.Dict[str, float]:
	population: tp.List[Individual] = [CannyIndividual() for i in range(pop_size)]
	env = SimpleEnvironment(img, edges)
	best = run_genetic(population, env, n_generations=n_generations, **kwargs)
	thr1 = min(best.thr1, best.thr2)
	thr2 = max(best.thr1, best.thr2)
	return {"thr1": thr1, "thr2": thr2}

def optimal_canny_for_ds(ds: Dataset):
	param_ds = defaultdict(lambda: [])
	for img, edge in track(ds, description="Collecting optimal parameters"):
		params = optimal_canny_param(img, edge)
		for k, v in params.items():
			param_ds[k].append(v)
	return pd.DataFrame(param_ds)


def imitate_edge_with_canny(img: ndarray, edges: ndarray, **kwargs) -> ndarray:
	params = optimal_canny_param(img, edges, **kwargs)
	return canny(img, params["thr1"], params["thr2"])
