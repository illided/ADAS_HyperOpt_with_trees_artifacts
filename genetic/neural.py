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

class NeuralNetwork(Individual):
  def __init__(self, layers_structure: tp.List[tp.Tuple[int, int]], weights: tp.Optional[np.ndarray]):
    if weights is None:
        weights = np.random.rand(n_parameters(layers_structure)) * 2 - 1
    self.unflatten(weights)
    self.weights = weights
  
    def mutate(self, ind_mt_pb=0.01):
        for i in range(n_parameters):
            if random.random() <= ind_mt_pb:
        self.weights[i] = random.random() * 2 - 1
        self.unflatten(self.weights)

    def unflatten(self, weights):
        

  def crossover(self, other: "Individual") -> tp.List["Individual"]:
    ...

  def behave(self, *args, **kwargs) -> tp.Any:
    return super().behave(*args, **kwargs)
  

def add_linear(config: LayersConfig, inp: int, out: int) -> LayersConfig:
  config.append((out, inp))
  config.append((out, 1))
  return config