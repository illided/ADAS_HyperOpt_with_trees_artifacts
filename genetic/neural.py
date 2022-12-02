from . import Individual
import typing as tp
import numpy as np
import random

LayersConfig = tp.List[tp.Tuple[int, int]]

def n_parameters(config: LayersConfig) -> int:
    s = 0
    for layer in config:
        s += layer[0] * layer[1]
    return s

def array_to_layers(arr: np.ndarray, config: LayersConfig) -> tp.List[np.ndarray]:
	split_points = []
	i = 0
	for s in config:
		i += s[0] * s[1]
		split_points.append(i)
	layers = []
	arr_splitted =np.split(arr, split_points)
	for m, s in zip(arr_splitted, config):
		layers.append(m.reshape(*s))
	return layers