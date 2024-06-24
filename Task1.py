import cv2
import numpy as np
import math

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

def gaussian_kernel(size, sigma, center=None):
    kernel = np.zeros((size, size))
    cx, cy = size // 2, size // 2
    if center is not None:
        cx, cy = center
    for i in range(size):
        for j in range(size):
            x, y = i - cx, j - cy
            kernel[i, j] = 1 / (math.pi * 2 * sigma * sigma) * math.exp(- ((x * x + y * y) / (2 * sigma * sigma)))
    return kernel

def laplacian_kernel():
    return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

def LoG_kernel(size, sigma, center=None):
    kernel = np.zeros((size, size))
    cx, cy = size // 2, size // 2
    if center is not None:
        cx, cy = center
    for i in range(size):
        for j in range(size):
            x, y = i - cx, j - cy
            p = (x * x + y * y) / (2 * sigma * sigma)
            kernel[i, j] = -1 / (math.pi * sigma ** 4) * (1 - p) * math.exp(-p)
    return kernel

def sobel_x_kernel():
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

def sobel_y_kernel():
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('input', img)
cv2.waitKey(0)

output = convolution(img, LoG_kernel(5, 0.5))

output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imshow('output', output)
cv2.waitKey(0)

cv2.destroyAllWindows()


