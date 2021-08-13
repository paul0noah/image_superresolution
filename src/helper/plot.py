from matplotlib import pyplot as plt
import cv2

def plot_image(img, title='', fig_size=[5, 3]):
  plt.figure(figsize=fig_size)
  plt.title(title, fontsize=12)
  plt.axis('off')
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def plot_matches(img1, keypoints1, img2, keypoints2, matches, fig_size=[15, 12]):
  matches_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, img2, flags=2)
  plot_image(matched_img, title='Related keypoints', fig_size=fig_size)

def plot_image_with_keypoints(img, keypoints, title='', fig_size=[40, 35]):
  plt.figure(figsize=fig_size)
  plt.title(title, fontsize=12)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.drawKeypoints(img_gray, keypoints, img)
  plt.axis('off')
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def plot_images(imgs, title='Images', fig_size=[20, 15]):
  plt.figure(figsize=fig_size)
  plt.title(title, fontsize=12)
  num_images = len(imgs)
  num_rows = math.ceil(num_images / 4)
  img_index = 1
  for img in imgs:
    plt.subplot(num_rows, 4, img_index)
    img_index = img_index + 1
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')