import cv2
import numpy as np
import math
from math import acos
from tabulate import tabulate
import random

def calculate_descriptors(img,i):
    #_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=1)
    border = img - eroded
    cv2.imshow('Border'+str(i), border)
    cv2.imshow('Input image'+str(i), img)
    return border

def calculate_features(img, border_img):
    area = np.count_nonzero(img)
    perimeter = np.count_nonzero(border_img)
    xmn = 1e9
    xmx = 0
    ymn = 1e9
    ymx = 0
    for x in range(border_img.shape[0]):
        for y in range(border_img.shape[1]):
            if (border_img[x, y] == 0):
                continue
            xmn = min(xmn, x)
            xmx = max(xmx, x)
            ymn = min(ymn, y)
            ymx = max(ymx, y)
    diam = max(xmx - xmn, ymx - ymn)
    compactness = (perimeter ** 2) / area
    form_factor = 4 * acos(-1) * area / (perimeter ** 2)
    roundness = 4 * area / (acos(-1) * diam ** 2)
    return [form_factor, roundness, compactness]

image_name = ['c1.jpg','t1.jpg','p1.png', 'c2.jpg', 't2.jpg', 'p2.png', 'st.jpg']
features = []
for i in range(len(image_name)):
    img = cv2.imread(image_name[i], 0)
    border = calculate_descriptors(img,i)
    features.append(calculate_features(img, border))

def cal_cos(x, y):
    mod_x = math.sqrt(x[0] ** 2 + x[1]** 2 + x[2] ** 2)
    mod_y = math.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)
    return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]) / (mod_x * mod_y)

dist = []
for i in range(3):
    tmp = []
    for j in range(3, 7):
        tmp.append(cal_cos(features[i], features[j]))
    dist.append(tmp)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(dist)

#Table Creation
distances_matrix = dist.copy()
row_headers = ['c1.jpg','t1.jpg','p1.png']
col_headers = ['c2.jpg', 't2.jpg', 'p2.png', 'st.jpg']

distances_matrix = np.array(distances_matrix)
# Display the distance matrix as a table
print(tabulate(distances_matrix, headers=col_headers, showindex=row_headers, tablefmt='grid'))


#File Creation
im_title = ['c1','t1','s1','c2','t2','s2']
file_path = 'output2.txt'
with open(file_path, 'w') as file:
    # Write the column names
    file.write('\t'.join(map(str, [' ', 'form_factor', 'roundness','compactness'])) + '\n')
    # Write horizontal line
    file.write('-' * 50 + '\n')
    i=0
    for i in range(7):
        row = features[i]
        file.write(im_title[i]+'\t\t')
        line = '\t\t'.join(map(str, row ))
        i=i+1
        # Write the line to the file
        file.write(line + '\n')
        # Write horizontal line
        file.write('-' * 50 + '\n')
cv2.waitKey(0)
cv2.destroyAllWindows()