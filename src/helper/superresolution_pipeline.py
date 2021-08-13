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


def extract_features(imgs, num_features=100):
  '''
    function which extracts the SIFT features of the images

    input:
      imgs:         list of cv2 grayscale images
      num_features: number of features to extract
    output:
      (keypoints, descriptor):  tuple containing list of
                                keypoints and list of descriptors
  '''

  # create SIFT feature extractor
  sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_features)

  all_keypoints = []
  all_descriptors = []

  for img in imgs:
    # detect features from the image
    keypoints, descriptors = sift.detectAndCompute(img, None)

    all_keypoints.append(keypoints)
    all_descriptors.append(descriptors)

  return (all_keypoints, all_descriptors)


def match_descriptors_to_first_image(descriptors, num_matches=50):
  '''
    function which matches all keypoints of all images to the first image

    input:
      descriptors: list of descriptors
      num_matches: number of best matches to keep after matching
    output:
      matches:  list of matches between img1 and other images
  '''

  # create feature matcher
  bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

  descriptor1 = descriptors[0]
  matches = []

  for descriptor in descriptors[1:]:
    # match descriptors of both images
    match = bf.match(descriptor1, descriptor)
    # sort matches according to their distance and keep only num_matches matches
    match = sorted(match, key = lambda x:x.distance)
    matches.append(match[:num_matches])

  return matches


def warp_images_to_first_image(matches, keypoints, imgs):
    '''
    function which warps all other images onto the first image

    input:
      matches:    list of matches between img1 and other images
      keypoints:  keypoints of imgs
      imgs:       rgb images
    output:
      aligned_images:  list of images which are warped onto the first image
  '''

  keypoints_img1 = keypoints[0]
  img_src = imgs[0]
  size_src = (img_src.shape[1], img_src.shape[0])

  # first aligned image to first image is first image itself
  aligned_images = [img_src]

  img_counter = 1

  for matches_of_img in matches:
    # extract keypoints of other image
    assert img_counter < len(keypoints)
    keypoints_other_img = keypoints[img_counter]
    img_src = imgs[img_counter]
    img_counter = img_counter + 1

    # reorder keypoints according to the matches and
    # define src and destination keypoints for homography matrix
    keypoints_src = []
    keypoints_dest = []
    for match in matches_of_img:
      keypoints_src.append(keypoints_other_img[match.trainIdx].pt)
      keypoints_dest.append(keypoints_img1[match.queryIdx].pt)

    # convert to np array
    keypoints_src = np.asarray(keypoints_src)
    keypoints_dest = np.asarray(keypoints_dest)

    # compute homography matrix
    h, status = cv2.findHomography(keypoints_src, keypoints_dest)

    # src is beeing warped onto destination
    # => src is other image, destination is first image
    # img_dest is warped img_src
    img_dst = cv2.warpPerspective(img_src, h, size_src)

    aligned_images.append(img_dst)

  return aligned_images


def resize_img(img, scale_factor=1.0):
  '''
    function to resize an image
    -> scale_factor > 1 increase the size
    -> scale_factor < 1 descrease the size

    input:
      img:           rgb image
      scale_factor:  factor with which the resize should happen
    output:
      resized_img:   bigger or smaller image
  '''
  width  = int(img.shape[1] * scale_factor)
  height = int(img.shape[0] * scale_factor)

  new_size = (width, height)

  return cv2.resize(img, new_size)


def compute_average_image(imgs, algorithm='median'):
  '''
    function to compute average img of a list of imgs

    input:
      imgs:       list of rgb image
      algorithm:  'mean' or 'median' => median reproduces sharper imgs
    output:
      avg_img:    average image
  '''
  imgs = np.asarray(imgs)
  if algorithm == 'median':
    avg_img = np.median(imgs, axis=0)
  elif algorithm == 'mean':
    avg_img = np.mean(imgs, axis=0)
  else:
    raise NameError('Algorithm " %s " not supported' % algorithm)

  # convert to image again
  avg_img = Image.fromarray(avg_img.astype(np.uint8))

  return avg_img


def superresolution_pipeline(filenames,
                              output_filename='superresmean.jpg',
                              resize_scale=5,
                              num_features=50,
                              verbose=False):
  '''
    function which computes a superresolution image out of a list of
    unaligned images of the sane object

    input:
      filenames:        list of filenames
      output_filename:  filename of superresolution img
      resize_scale:     amount by which the imgs should be enlarged
      num_features:     number of features which are used for alignement
      verbose:          if some plots should be generated or not
    output:
      superres_img:     superresolution image
  '''
  imgs = read_images(filenames)

  if verbose:
    plot_images(imgs, title='Input Images')

  # resize images
  for i in range(0, len(imgs)):
    imgs[i] = resize_img(imgs[i], scale_factor=resize_scale)

  imgs_gray = convert_to_grayscale(imgs)

  keypoints, descriptors = extract_features(imgs_gray)

  matches = match_descriptors_to_first_image(descriptors, num_keypoints)

  aligned_images = warp_images_to_first_image(matches, keypoints, imgs)

  superres_img = compute_average_image(aligned_images)
  if verbose
    plot_image(superres_img, title='Superres Image')

  cv2.imwrite(output_filename, np.asarray(superres_img))

  return superres_img