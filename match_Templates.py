# take an image and search its best fit in a directory of images
import numpy as np
import cv2
import glob
from google.colab.patches import cv2_imshow

def get_good_matches(template,img):
  
  orb = cv2.ORB_create(nfeatures=5000)
  kp1, des1 = orb.detectAndCompute(template, None)
  kp2, des2 = orb.detectAndCompute(img, None)

  FLANN_INDEX_KDTREE = 0
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks = 50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32), 2)

  # store all the good matches as per Lowe's ratio test.
  good_matches = []
  for m,n in matches:
      if m.distance < 0.7*n.distance:
          good_matches.append(m)

  return len(good_matches)

template_path = "/random/3FnU3$3V9FQhw$NPAJjce$.png"
images_path = "/random"
template_image = cv2.imread(template_path,0)
print("Template Image:")
cv2_imshow(template_image)
good_matches_list = []
images_path_list = []

for imagePath in glob.glob(images_path +"/*"):
  vertex_image = cv2.imread(imagePath)
  good_match = get_good_matches(template_image,vertex_image)
  good_matches_list.append(good_match)
  images_path_list.append(imagePath)

larger_len=0
index = 0
for i in range(len(good_matches_list)):
  if good_matches_list[i]>larger_len:
    larger_len = good_matches_list[i]
    index = i

match_img = cv2.imread(images_path_list[index])
print("\nImage that best match with the tempelate:")
cv2_imshow(match_img)
cv2.waitKey(0)
print()
print('Image Name:')
print(images_path_list[index])
