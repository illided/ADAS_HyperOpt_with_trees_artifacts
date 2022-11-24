import cv2 as cv
import numpy as np
import typing as tp
from rich.progress import track
from collections import defaultdict
import pandas as pd

def compression_level(img) -> float:
  _, encimg = cv.imencode('.jpg', img, [cv.IMWRITE_JPEG_QUALITY, 90])
  v_dim = encimg.shape[0]
  return v_dim / np.product(img.shape)

def split_img(img: np.ndarray, n_rows=4, n_columns=4) -> tp.List[np.ndarray]:
  x_dim, y_dim = img.shape[0], img.shape[1]
  x_step, y_step = x_dim // n_rows, y_dim // n_columns
  x_splits = [x_step * i for i in range(n_columns)]
  y_splits = [y_step * i for i in range(n_rows)]
  blocks = []
  for x in x_splits:
    for y in y_splits:
      blocks.append(img[x:x + x_step, y:y + y_step, :])
  return blocks

def jpeg_compression_vector(img: np.ndarray, n_rows=4, n_columns=4) -> np.ndarray:
  v = [compression_level(img)]
  for block in split_img(img, n_rows=n_rows, n_columns=n_columns):
    v.append(compression_level(block))
  return np.array(v)


def gray_level_hist(img: np.ndarray, n_bins=32):
  gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  hist = cv.calcHist([gray_img], [0], None, [n_bins], [0, 256]) / np.product(gray_img.shape)
  return hist.flatten()

def generate_features(ds: tp.Iterable, save_to: str):
    features = defaultdict(lambda: [])
    
    def write_features(v, name):
        for i, el in enumerate(v):
            features[f"{name}_{i}"].append(el)

    for img, _ in track(ds, description="Collecting features"):
        write_features(gray_level_hist(img), "HRF")
        write_features(jpeg_compression_vector(img), "JPG")

    pd.DataFrame(features).to_csv(save_to)