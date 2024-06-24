import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def erlang_distribution(x, k, miu):
    up = math.pow(x, k - 1) * math.exp(- x / miu)
    down = math.pow(miu, k) * math.factorial(k - 1)
    return up / down

def histogram_equalization(channel):
    hist = np.zeros(256).astype(np.int32)
    pdf = np.zeros(256).astype(np.float64)
    cdf = np.zeros(256).astype(np.float64)
    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            hist[channel[i, j]] += 1
    for i in range(256):
        pdf[i] = float(hist[i]) / (channel.shape[0] * channel.shape[1])
        cdf[i] = pdf[i]
        if i >= 1:
            cdf[i] += cdf[i - 1]
    equalized_hist = np.round(cdf * np.max(channel))
    return (hist, pdf, cdf, equalized_hist)

def histogram_matching(img, k, miu):
    pdf = np.zeros(256).astype(np.float64)
    cdf = np.zeros_like(pdf)
    for i in range(256):
        pdf[i] = erlang_distribution(i, k, miu)
        cdf[i] = pdf[i]
        if i >= 1:
            cdf[i] += cdf[i - 1]
    intensity = np.round(np.max(img) * cdf).astype(np.uint8)
    input_img = histogram_equalization(img)
    matched_hist = np.zeros(256).astype(np.int32)
    for i in input_img[-1]:
        diff = 256
        for j in range(256):
            if abs(i - intensity[j]) < diff:
                diff = abs(i - intensity[j])
                matched_hist[int(i)] = j
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i, j] = matched_hist[int(input_img[-1][img[i, j]])]

    return output

def showfig(img, text):
    hist, pdf, cdf, _ = histogram_equalization(img)
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.plot(hist)
    plt.title(text + " Histogram")
    plt.subplot(2, 3, 2)
    plt.plot(pdf)
    plt.title(text + " PDF")
    plt.subplot(2, 3, 3)
    plt.plot(cdf)
    plt.title(text + " CDF")
    plt.show()

def main(img_name):
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    K = int(input('K = ?'))
    MIU = float(input('miu = ?'))
    output = histogram_matching(img, K, MIU)

    cv2.imshow('input', img)
    cv2.waitKey(0)
    showfig(img, 'input')
    cv2.imshow('output', output)
    cv2.waitKey(0)
    showfig(output, 'output')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    while True:
        main('col1.png')
    #main('col2.png')