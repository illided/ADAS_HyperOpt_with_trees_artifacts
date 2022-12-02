from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import cv2 as cv

def get_filenames(dir):
    return [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]


def segmentation_to_edge(segmentation) -> np.ndarray:
	contour_img = np.zeros_like(segmentation)
	unq_classes = np.unique(segmentation)
	h, w = segmentation.shape[0], segmentation.shape[1]
	for cls in unq_classes:
		cls_img = np.zeros((h, w, 1), dtype=np.uint8)
		cls_img[segmentation == cls] = 255
		contours, _ = cv.findContours(cls_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		contour_img = cv.drawContours(contour_img, contours, -1, (255, 255, 255), 0)
	return contour_img

def apply_roi(img, up_cut=0.39, low_cut=0.77) -> np.ndarray:
	h = img.shape[0]
	up_cut = int(0.39 * h)
	low_cut = int(0.77 * h)
	return img[up_cut:low_cut, :]