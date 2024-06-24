import cv2
import numpy as np

centers = []
Do = None

def HNR(u, v, ideal):
    result = 1
    for (uk, vk) in centers:
        Dk = ((u - uk) ** 2 + (v - vk) ** 2) ** 0.5
        D_k = ((u + uk) ** 2 + (v + vk) ** 2) ** 0.5
        Hk, H_k = 0, 0
        if ideal:
            if Dk > Do:
                Hk = 1
            if D_k > Do:
                H_k = 1
        else:
            if Dk != 0:
                Hk = 1 / (1 + (Do / Dk) ** (2 * n))
            if D_k != 0:
                H_k = 1 / (1 + (Do/D_k) ** (2 * n))
        result *= H_k * Hk
    return result

img = cv2.imread('img_1.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Noisy Image', img)
cv2.waitKey(0)

M = img.shape[0]
N = img.shape[1]

no_centers = int(input('No of centers:'))
Do = int(input('Cutoff frequency:'))
n = int(input('Enter Degree:'))
ideal = int(input('Ideal Notch reject filter?: '))

if ideal == 1:
    ideal = True
else:
    ideal = False

centers = []
for i in range(no_centers):
    uk = int(input('Enter Uk:'))
    vk = int(input('Enter Vk:'))

    centers.append((uk, vk))
    centers.append((M - uk, N - vk))

kernel = np.zeros((M, N))
for i in range(M):
    for j in range(N):
        kernel[i, j] = HNR(i, j, ideal)

kernel_2 = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow('Kernel', kernel_2)

ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)

ft_mag = np.abs(ft_shift)
ft_ang = np.angle(ft_shift)

ft_mag_2 = 20 * np.log(ft_mag + 1)
ft_mag_2 = cv2.normalize(ft_mag_2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

ft_ang_2 = cv2.normalize(ft_ang, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow('Magnitude spectrum', ft_mag_2)
cv2.waitKey(0)
cv2.imshow('Angle', ft_ang_2)
cv2.waitKey(0)

filtered_mag = np.multiply(ft_mag, kernel)
filtered_mag_2 = 20 * np.log(filtered_mag + 1)
filtered_mag_2 = cv2.normalize(filtered_mag_2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow('Multiplied Magnitude', filtered_mag_2)
cv2.waitKey(0)

ift = np.multiply(filtered_mag, np.exp(1j * ft_ang))
ift = np.fft.ifftshift(ift)
ift = np.fft.ifft2(ift)
ift = np.real(ift)
ift = cv2.normalize(ift, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imshow('inverse transform', ift)
cv2.waitKey(0)

cv2.destroyAllWindows()