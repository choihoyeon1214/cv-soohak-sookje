from PIL import Image
from PIL import ImageFilter
import numpy as np
img = Image.open('test.jpg')
x, y = img.size
img_wb = Image.open('test.jpg').convert("L")
img_blur = img_wb.filter(ImageFilter.GaussianBlur(-10))
img.show()
#img_wb.show()
img_blur.show()
#pip install opencv-contrib-python opencv-contrib-python-headless opencv-python opencv-python-headless
