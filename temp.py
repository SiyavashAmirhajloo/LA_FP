import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import math
import os


def calculate_svd(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert color image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Image is already grayscale
        gray_image = image
    
    # Calculate SVD
    U, S, Vt = svd(gray_image)
    return U, S, Vt


def calculate_eigenvalues_eigenvector(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert color image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Image is already grayscale
        gray_image = image
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(np.cov(gray_image))
    return eigenvalues, eigenvectors


def denoise_image_svd(image):
    # Convert image to grayscale if necessary
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    
    # Calculate SVD
    U, S, Vt = calculate_svd(gray_image)
    
    # Set threshold value for singular values
    threshold = 10
    
    # Reduce the magnitude of singular values below the threshold
    S = np.where(S < threshold, 0, S)
    
    # Reconstruct image using modified SVD
    denoised_image = U @ np.diag(S) @ Vt
    
    # Clip the pixel values to the valid range [0, 255]
    denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)
    
    return denoised_image


def svd(A):
    # Calculate SVD of matrix A
    m, n = A.shape
    U = np.zeros((m, m))
    S = np.zeros((m, n))
    Vt = np.zeros((n, n))
    
    # Calculate A*A.T and its eigenvalues and eigenvectors
    ATA = np.dot(A.T, A)
    eigenvalues, eigenvectors = eig(ATA)
    
    # Calculate U matrix
    for i in range(m):
        U[:,i] = eigenvectors[:,i]
    
    # Calculate Vt matrix
    for i in range(n):
        Vt[i,:] = eigenvectors[:,i].T
    
    # Calculate S matrix
    for i in range(min(m, n)):
        S[i,i] = np.sqrt(eigenvalues[i])
    
    return U, S, Vt

def eig(A):
    n = A.shape[0]
    eigenvalues = np.zeros(n)
    eigenvectors = np.zeros((n, n))

    for i in range(n):
        x0 = np.random.rand(n)
        x1 = np.zeros(n)
        epsilon = 1e-6
        max_iterations = 1000
        iterations = 0

        while np.linalg.norm(x1 - x0) > epsilon and iterations < max_iterations:
            x0 = x1
            x1 = A @ x0
            x1 /= np.linalg.norm(x1)
            iterations += 1

        eigenvalues[i] = x1.T @ A @ x1
        eigenvectors[:, i] = x1

    return eigenvalues, eigenvectors


def add_gaussian_noise(image, var_limit=(10, 50), mean=0, p=0.5):
    if random.random() > p:
        return image

    # Convert image data to floating-point values between 0 and 1
    image = image.astype(float) / 255.0

    # Generate random variance
    var = random.uniform(var_limit[0], var_limit[1])
    sigma = var ** 0.5

    # Generate noise and add it to the image
    noise = np.zeros_like(image)
    for c in range(3):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                u1, u2 = random.uniform(0, 1), random.uniform(0, 1)
                z1 = sigma * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
                noise[i, j, c] = z1

    noisy_image = image + noise

    # Convert the image data back to the range [0, 255]
    noisy_image = (noisy_image * 255.0).clip(0, 255).astype(np.uint8)

    return noisy_image


images_dir = '/kaggle/input/image-classification/images/images/architecure'

# Set the desired size of the displayed images
width = 200
height = 200

# Get a list of all the JPG images in the directory
image_files = [filename for filename in os.listdir(images_dir) if filename.endswith('.jpg')]

# Select a subset of images to display
selected_images = image_files[:5]


for filename in selected_images:
    # Load the image using OpenCV's imread function
    img = cv2.imread(os.path.join(images_dir, filename))
    # Resize the image to the desired size
    img = cv2.resize(img, (width, height))
    # Convert the image from BGR to RGB color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    noisy_img = add_gaussian_noise(img, var_limit=(0, 1), mean=0, p=1)
    noisy_img = np.array(noisy_img)
    img_denoised = denoise_image_svd(noisy_img)

    # Resize the denoised image for display
    denoised_img_resized = cv2.resize(img_denoised, (width, height))
    
    # Display the original and denoised images side by side
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img.squeeze())
    axs[0].set_title("Original Image")
    axs[1].imshow(noisy_img.squeeze())
    axs[1].set_title("Noisy Image")
    axs[2].imshow(denoised_img_resized.squeeze())
    axs[2].set_title("Denoised Image")
    plt.show()