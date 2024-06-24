import cv2
import numpy as np

def calculate_h_nr(u, v, ideal):
    h_nr = 1
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
                H_k = 1 / (1 + (Do / D_k) ** (2 * n))
        
        h_nr *= (Hk * H_k)
        
    return h_nr
        

image = cv2.imread('img_1.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Noisy Image', image)

M = image.shape[0]
N = image.shape[1]

no_centers = int(input('Enter the number of Centers: '))
Do = int(input('Enter Cutoff frequency: '))
n = int(input('Enter Degree: '))
ideal = int(input('Ideal Notch Reject Filter?: '))

if ideal == 1: 
    ideal = True
else:
    ideal = False

centers = []
for i in range(no_centers):
    uk = int(input('Enter uk: '))
    vk = int(input('Enter vk: '))
    
    centers.append((vk, uk))
    centers.append((N-vk, M-uk))
    
kernel = np.zeros((M, N))
for i in range(M):
    for j in range(N):
        kernel[i, j] = calculate_h_nr(i, j, ideal)
        
kernel_2 = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow('Kernel', kernel_2)

ft = np.fft.fft2(image)
ft_shifted = np.fft.fftshift(ft)

ft_mag = np.abs(ft_shifted)
ft_ang = np.angle(ft_shifted)

ft_mag_2 = 20 * np.log(ft_mag + 1)
ft_mag_2 = cv2.normalize(ft_mag_2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

ft_ang_2 = cv2.normalize(ft_ang, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow('Magnitude Spectrum', ft_mag_2)
cv2.imshow('Angle', ft_ang_2)

filtered_mag = np.multiply(ft_mag, kernel)

filtered_mag_2 = 20 * np.log(filtered_mag + 1)
filtered_mag_2 = cv2.normalize(filtered_mag_2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow('Multiplied Magnitude', filtered_mag_2)

ift = np.multiply(filtered_mag, np.exp(1j * ft_ang))
ift = np.fft.ifftshift(ift)
ift = np.fft.ifft2(ift)
ift = np.real(ift)
ift = cv2.normalize(ift, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow('Inverse Tranform', ift)

cv2.waitKey(0)
cv2.destroyAllWindows()