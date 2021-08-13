import cv2
import numpy as np
from PIL import Image
from .plot import *

def read_images(filenames):
  '''
    input:
      filenames:  list of strings
    output:
      imgs:       list of cv2 images
  '''
  imgs = []
  for filename in filenames:
    img = cv2.imread(filename)
    imgs.append(img)

  return imgs


def convert_to_grayscale(imgs):
  '''
    input:
      imgs:       list of cv2 images
    output:
      imgs_gray:  list of grayscale cv2 images
  '''
  imgs_gray = []
  for img in imgs:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs_gray.append(img_gray)

  return imgs_gray
