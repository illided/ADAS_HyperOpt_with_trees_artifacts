import random
import typing as tp
import numpy as np
from numpy import ndarray
from abc import abstractmethod, ABC

class Individual(ABC):
    @abstractmethod
    def mutate(self):
        ...
    @abstractmethod
    def crossover(self, other: "Individual") -> tp.List["Individual"]:
        ...
    @abstractmethod
    def behave(self, *args, **kwargs) -> tp.Any:
        ...

Population = tp.List[Individual]

class Environment(ABC):
    @abstractmethod
    def fitness(self, Individual) -> float:
        ...
    

def select(population: Population, target_size: int, 
           env: Environment, tourn_size: int = 2) -> tp.Tuple[Population, tp.List[float]]:
  fitness = np.array([env.fitness(ind) for ind in population])
  cur_pop_size = len(population)
  new_pop = []
  new_fitness = []
  for _ in range(target_size):
    sample_ind = [random.randint(0, cur_pop_size - 1) for _ in range(tourn_size)]
    best = sample_ind[np.argmax(fitness[sample_ind])]
    new_pop.append(population[best])
    new_fitness.append(fitness[best])
  return new_pop, new_fitness


def crossover(population: Population, cros_pb: float) -> Population:
  new_pop: tp.List[Individual] = []
  for ind1, ind2 in zip(population[::2], population[1::2]):
    if random.random() < cros_pb:
      new_pop.extend(ind1.crossover(ind2))
  return new_pop

def mutation(population: Population, mut_pb: float) -> Population:
  for ind in population:
    if random.random() < mut_pb:
      ind.mutate()
  return population

def ellitism(population: Population, fitness: tp.List[float], n: int=5) -> tp.Tuple[Population, Population]:
  elite_ind = np.argpartition(np.array(fitness), -n)[-n:]
  elite = []
  plebs = []
  for i, ind in enumerate(population):
    if i in elite_ind:
      elite.append(ind)
    else:
      plebs.append(ind)
  return elite, plebs

def best_fitness(population: tp.List[Individual], env: Environment) -> Individual:
  fitness = [env.fitness(ind) for ind in population]
  return population[np.argmax(fitness)]

def run_genetic(population: Population,
                env: Environment,
                n_generations: int,
                pop_size: tp.Optional[int]=None,
                tourn_size: int=2,
                p_crossover: float=0.9,
                p_mutation: float=0.1,
                elite_size: int=0,
                trace_hold: tp.Optional[tp.List[tp.Any]]=None) -> Individual:
  random.seed(42)
  if not pop_size:
    pop_size = len(population)
  for generation in range(n_generations):
    population, fitness = select(population, pop_size, env, tourn_size=tourn_size)
    elite, plebs = ellitism(population, fitness, elite_size)
    plebs = crossover(plebs, p_crossover)
    plebs = mutation(plebs, p_mutation)
    population = plebs + elite
  best = best_fitness(population, env)
  return best