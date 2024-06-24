import cv2
import math
import numpy as np

def convolution(img, kernel, center=None):
    img = img.copy()
    kernel = kernel.copy()
    cx, cy = kernel.shape[0] // 2, kernel.shape[1] // 2
    if center is not None:
        cx, cy = center

    padded = np.zeros((img.shape[0] + kernel.shape[0] * 2, img.shape[1] + kernel.shape[1] * 2))
    padded[kernel.shape[0]:kernel.shape[0] + img.shape[0], kernel.shape[1]:kernel.shape[1] + img.shape[1]] = img
    output = np.zeros_like(padded)
    for i in range(padded.shape[0]):
        for j in range(padded.shape[1]):
            tot = 0
            if i + kernel.shape[0] < padded.shape[0] and j + kernel.shape[1] < padded.shape[1]:
                for k in range(kernel.shape[0]):
                    for l in range(kernel.shape[1]):
                        tot += padded[i + k, j + l] * kernel[-k-1, -l-1]
            output[i, j] = tot
    return output[kernel.shape[0]:kernel.shape[0] + img.shape[0], kernel.shape[1]:kernel.shape[1] + img.shape[1]]

def derivative_kernel(size, sigma, center=None):
    kernel_x = np.zeros((size, size))
    kernel_y = np.zeros((size, size))
    cx, cy = size // 2, size // 2
    if center is not None:
        cx, cy = center
    for i in range(size):
        for j in range(size):
            x, y = i - cx, j - cy
            kernel_x[i, j] =  - x / (math.pi * 2 * sigma ** 4) * math.exp(- ((x * x + y * y) / (2 * sigma * sigma)))
            kernel_y[i, j] =  - y / (math.pi * 2 * sigma ** 4) * math.exp(- ((x * x + y * y) / (2 * sigma * sigma)))
    return (kernel_x, kernel_y)

def find_threshold(img):
    initial = np.mean(img)
    while True:
        total1, total2 = 0, 0
        cnt1, cnt2 = 0, 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] > initial:
                    total1 += float(img[i, j])
                    cnt1 += 1
                else:
                    total2 += float(img[i, j])
                    cnt2 += 1

        new_val = (total1/cnt1 + total2/cnt2) / 2
        if abs(new_val - initial) < 1e-6:
            return new_val
        initial = new_val

def global_thresholding(img):
    kernel_x, kernel_y = derivative_kernel(7, 1)
    img_x = convolution(img, kernel_x)

    cv2.imshow('img_x', cv2.normalize(img_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.waitKey(0)

    img_y = convolution(img, kernel_y)

    cv2.imshow('img_y', cv2.normalize(img_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.waitKey(0)

    gradient = np.sqrt(img_x ** 2 + img_y ** 2)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow('Gradient', gradient)
    cv2.waitKey(0)

    T = find_threshold(gradient)

    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if gradient[i, j] > T:
                output[i, j] = 255
    return output

img = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('input', img)
cv2.waitKey(0)

output = global_thresholding(img)

output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imshow('output', output)
cv2.waitKey(0)

cv2.destroyAllWindows()