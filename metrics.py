import numpy as np
from abc import ABC, abstractmethod
import typing as tp
from scipy.spatial import KDTree


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
  def __init__(self, reg_param: float=.0):
    self.reg_param = reg_param

  def _score(self, pred: np.ndarray, gt: np.ndarray) -> float:
    n = 2 * np.sum(pred * gt)
    d = np.sum(pred) + np.sum(gt)
    dice = n / d
    if self.reg_param != 0:
      dice -= self.reg_param * np.mean(pred)
    return dice


class NormalizedFoM(BinaryEdgeMetric):
  def __init__(self, k_fp: float=0.1, k_fn: float=0.2):
    self.k_fp = k_fp
    self.k_fn = k_fn

  def _score(self, pred: np.ndarray, gt: np.ndarray) -> float:
    FP = np.sum(pred * (1 - gt))  # pred AND not gt
    FN = np.sum(gt * (1 - pred))  # gt AND not pred

    if FP + FN == 0:
      return 1.

    gt_nonzero_points = np.vstack(np.nonzero(gt)).T
    pred_nonzero_points = np.vstack(np.nonzero(pred)).T

    gt_tree = KDTree(gt_nonzero_points)
    pred_tree = KDTree(pred_nonzero_points)
  
    fp_distances = gt_tree.query(pred_nonzero_points)[0]
    fp_distances = 1 / (1 + self.k_fp * fp_distances)
    fp_term = FP * np.sum(fp_distances) / np.sum(pred)

    fn_distances = pred_tree.query(gt_nonzero_points)[0]
    fn_distances = 1 / (1 + self.k_fn * fn_distances)
    fn_term = FN * np.sum(fn_distances) / np.sum(gt)

    return (fp_term + fn_term) / (FP + FN)
