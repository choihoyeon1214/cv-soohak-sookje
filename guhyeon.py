import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def to_grayscale(image_path):
    image = Image.open(image_path).convert("L")
    return np.array(image, dtype=np.float32)

def gaussian_blur(image, kernel_size=5, sigma=1):
    size = kernel_size
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma**2)) * np.exp(
            - ((x - (size - 1) / 2)**2 + (y - (size - 1) / 2)**2) / (2 * sigma**2)
        ), (size, size)
    )
    kernel /= np.sum(kernel)
    
    
    height, width = image.shape
    blurred_image = np.zeros_like(image)
    
    
    for i in range(height):
        for j in range(width):
           
            region = image[max(0, i - size//2):min(i + size//2 + 1, height),
                           max(0, j - size//2):min(j + size//2 + 1, width)]
            kernel_region = kernel[
                max(0, size//2 - i):min(size, height - i + size//2),
                max(0, size//2 - j):min(size, width - j + size//2)
            ]
            blurred_image[i, j] = np.sum(region * kernel_region)
    
    return blurred_image

def compute_derivative(values):
    derivative = np.zeros_like(values, dtype=np.float32)
    
    for i in range(1, len(values) - 1):
        derivative[i] = (values[i + 1] - values[i - 1]) / 2 

    return derivative

def compute_row_gradients(image):
    height, width = image.shape
    row_gradients = np.zeros_like(image, dtype=np.float32)
    
    for i in range(height):
        row = image[i, :]
        row_derivative = compute_derivative(row)
        row_gradients[i, 1:-1] = row_derivative[1:-1]  
    
    return row_gradients

def compute_column_gradients(image):
    height, width = image.shape
    column_gradients = np.zeros_like(image, dtype=np.float32)
    
    for j in range(width):
        column = image[:, j]
        column_derivative = compute_derivative(column)
        column_gradients[1:-1, j] = column_derivative[1:-1]  
    
    return column_gradients


def detect_edges(row_gradients, column_gradients):
    return np.sqrt(row_gradients**2 + column_gradients**2)


def save_images(image, row_gradients, column_gradients, edges):
    Image.fromarray(image.astype(np.uint8)).save("Original_image.png")
    Image.fromarray(np.clip(row_gradients, 0, 255).astype(np.uint8)).save("Row_gradients.png")
    Image.fromarray(np.clip(column_gradients, 0, 255).astype(np.uint8)).save("Column_gradients.png")
    Image.fromarray(np.clip(edges, 0, 255).astype(np.uint8)).save("Detected_edges.png")

def plot_results(image, row_gradients, column_gradients, edges):

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('original')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(row_gradients, cmap='gray')
    plt.title('x')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(column_gradients, cmap='gray')
    plt.title('y')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(edges, cmap='gray')
    plt.title('edge')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

image_path = "cat.jpg"

image = to_grayscale(image_path)

blurred_image = gaussian_blur(image, kernel_size=5, sigma=1)

row_gradients = compute_row_gradients(blurred_image)

column_gradients = compute_column_gradients(blurred_image)

edges = detect_edges(row_gradients, column_gradients)

save_images(blurred_image, row_gradients, column_gradients, edges)

plot_results(blurred_image, row_gradients, column_gradients, edges)
