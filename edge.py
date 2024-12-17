import cv2
from PIL import Image


input_path = 'sanz.png' 
image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE) 

blurred_image = cv2.GaussianBlur(image, (5, 5), 0)


low_threshold = 50  
high_threshold = 150  
edges = cv2.Canny(blurred_image, low_threshold, high_threshold)


edges_pillow = Image.fromarray(edges)


edges_pillow.show()


output_path = 'edges_output.jpg'
edges_pillow.save(output_path)
