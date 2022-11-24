import numpy as np
from abc import ABC, abstractmethod


class BinaryEdgeMetric(ABC):
  def score(self, pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
      raise ValueError(f"Prediction and ground truth shape mismatched. Prediction: {pred.shape}. Gt: {gt.shape}")
    if not self._is_binary(pred):
      raise ValueError("Prediction is not binary image")
    if not self._is_binary(gt):
      raise ValueError("Ground truth is not binary image")
    
    return self._score(pred, gt)


  def _is_binary(self, edge_map: np.ndarray) -> bool:
    return set(np.unique(edge_map)) == {0, 1}

  @abstractmethod
  def _score(self, pred: np.ndarray, gt: np.ndarray) -> float:
    ...


class Jaccard(BinaryEdgeMetric):
  def __init__(self, reg_param=0.):
    self.reg_param = reg_param

  def _score(self, pred: np.ndarray, gt: np.ndarray) -> float:
    d = pred + gt
    d[d > 0] = 1

    n = pred * gt 

    jaccard = np.sum(n) / np.sum(d)

    if self.reg_param != 0:
      jaccard -= self.reg_param * np.mean(pred)

    return jaccard


class Dice(BinaryEdgeMetric):
  def __init__(self, reg_param=.0):
    self.reg_param = reg_param

  def _score(self, pred: np.ndarray, gt: np.ndarray) -> float:
    n = 2 * np.sum(pred * gt)
    d = np.sum(pred) + np.sum(gt)
    dice = n / d
    if self.reg_param != 0:
      dice -= self.reg_param * np.mean(pred)
    return dice

    