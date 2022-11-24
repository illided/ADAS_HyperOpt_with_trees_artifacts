import numpy as np
import typing as tp
import random

def blend_crossover(p1: float, p2: float, 
                    alpha: float=0.5, n_offsprings: int=1, 
                    lower_b: tp.Optional[float]=None, upper_b: tp.Optional[float]=None) -> tp.List[float]:
  ofsp_rng = [p1 - alpha * (p2 - p2), p2 + alpha * (p2 - p1)]
  if lower_b is not None and ofsp_rng[0] < lower_b:
    ofsp_rng[0] = lower_b
  if upper_b is not None and ofsp_rng[1] > upper_b:
    ofsp_rng[1] = upper_b
  return [random.uniform(*ofsp_rng) for _ in range(n_offsprings)]

def two_point_crossover(arr1: np.ndarray, arr2: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
  arr1 = np.copy(arr1)
  arr2 = np.copy(arr2)
  p1, p2 = random.sample(range(0, arr1.shape[0]), 2)
  p1 = min(p1, p2)
  p2 = max(p1, p2)
  arr1_cut = np.copy(arr1[p1:p2])
  arr1[p1:p2] = np.copy(arr2[p1:p2])
  arr2[p1:p2] = arr1_cut
  return arr1, arr2