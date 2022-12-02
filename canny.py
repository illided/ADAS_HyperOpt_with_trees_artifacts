import cv2 as cv
import numpy as np

def canny(img, thr1, thr2, s=0.0, ks=3):
  if ks % 2 == 0:
    ks += 1
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  blurred = cv.GaussianBlur(gray, (ks, ks), s)
  return cv.Canny(blurred, thr1, thr2)


def const_canny(img):
  return canny(img, 100, 200)


def stat_canny(image, sigma=0.33):
  img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  img = cv.GaussianBlur(img, (3, 3), 0)
  v = np.median(img)
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  return cv.Canny(img, lower, upper)

