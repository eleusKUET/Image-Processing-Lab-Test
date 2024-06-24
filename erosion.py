import cv2
import numpy as np

img = cv2.imread('img_2.png', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3))
eroded = cv2.erode(img, kernel, iterations=1)
border = img - eroded
border = cv2.normalize(border, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow('border', border)
cv2.waitKey(0)
cv2.destroyAllWindows()