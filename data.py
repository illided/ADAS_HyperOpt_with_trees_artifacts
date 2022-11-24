import cv2 as cv
from os import listdir
from os.path import isfile, isdir, join
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from torchvision.transforms import Resize

import typing as tp
import numpy as np

from rich.progress import track


def get_filenames(dir):
  return [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]


class BIPED(Dataset):
  def __init__(self, path: str, holdout: str) -> None:
    super().__init__()
    self.path = path
    self.data = self._load_holdout(holdout)

  def _load_holdout(self, holdout) -> tp.List[tp.Tuple[np.ndarray, np.ndarray]]:
    main_dir = f"{self.path}/BIPED/edges"
    imgs_dir = f"{main_dir}/imgs/{holdout}/rgbr"
    edges_dir = f"{main_dir}/edge_maps/{holdout}/rgbr"
    if holdout == "train":
      imgs_dir += "/real"
      edges_dir += "/real"
    imgs = [cv.imread(img, cv.IMREAD_COLOR) for img in sorted(get_filenames(imgs_dir))]
    edges = [cv.imread(edge, cv.IMREAD_GRAYSCALE) for edge in sorted(get_filenames(edges_dir))]
    return list(zip(imgs, edges))


  def __getitem__(self, index) -> tp.T_co:
    return self.data[index]

  def __len__(self) -> int:
    return len(self.data)



class CityScapesEdges(Dataset):
  def __init__(self, root, holdout, pic_size=(512, 1024)) -> None:
    super().__init__()
    if holdout not in ["train", "val"]:
      raise ValueError("This holdout not supported yet")

    self.tv_ds = Cityscapes(root,
                            split=holdout,
                            mode="fine",
                            target_type='semantic',
                            transform=Resize(size=pic_size),
                            target_transform=Resize(size=pic_size))
    
    self.path_to_contours = join(root, "gtEdges", f"pic_size_{pic_size[0]}_{pic_size[1]}", holdout)
    if not isdir(self.path_to_contours):
      print("Couldn't found image edges")
      Path(self.path_to_contours).mkdir(parents=True, exist_ok=True)
      self._setup_contours()
  
  def _setup_contours(self):
    for i in track(range(len(self.tv_ds)), "Building contours"):
      image, segmentation = self.tv_ds[i]

      image = np.array(image)
      segmentation = np.array(segmentation)

      contour_img = np.zeros_like(image)
      unq_classes = np.unique(segmentation)
      for cls in unq_classes:
          contour_img = self._draw_contour(segmentation, contour_img, cls)
        
      cv.imwrite(self._contour_file(i), contour_img)

  def _draw_contour(self, src_img, contour_img, target_id):
    h, w = src_img.shape
    cls_img = np.zeros((h, w, 1), dtype=np.uint8)
    cls_img[src_img == target_id] = 255
    contours, _ = cv.findContours(cls_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return cv.drawContours(contour_img, contours, -1, (255, 255, 255), 0)
  
  def __getitem__(self, index) -> tp.T_co:
    image, _ = self.tv_ds[index]
    image = np.array(image)
    contours = cv.imread(self._contour_file(index), cv.IMREAD_GRAYSCALE)
    return image, contours
  
  def _contour_file(self, i) -> str:
    return join(self.path_to_contours, f"{i}.png")

  def __len__(self) -> int:
    return len(self.tv_ds)